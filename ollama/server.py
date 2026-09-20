"""
Model Service - FastAPI server

Unified management for all model backends (Ollama + llama-server) on slice-test.
Handles SAMcloud resource leasing, model lifecycle, and health reporting.

Endpoints:
  GET  /health              - Health check
  GET  /status              - Full status (all backends, models, leases, resources)
  GET  /models              - List all managed + available models
  POST /models/load         - Load a model (Ollama or llama-server)
  POST /models/unload       - Unload a model (releases lease)
  POST /v1/chat/completions - OpenAI-compatible chat (routes to correct backend)
  POST /v1/completions      - OpenAI-compatible completion

Backends: Ollama (MLX), llama-server (llama.cpp), mlx-vlm (vision), and the
exo pool. The first three are owned by this gateway, which loads and unloads
them against a byte-metered lease on gpu-0. The pool is not: it is placed out
of band, spans two machines, serves one request at a time, and is held with an
**exclusive** lease taken around each generation and released after. Ask for
it by tier (`model: "think"`), not by model name.
"""

import asyncio
import os
import json
import time
import hashlib
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional

from . import capacity
from . import config
from .manager import (
    ModelManager, Backend, VLM_PORT,
    match_vlm_model, match_gguf_model, match_exo_tier,
)
from .exo_client import ExoUnavailable, ExoRequestFailed
from .samcloud import SamcloudClient
from .ollama_client import OllamaClient
from .llama_client import LlamaServerClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("model-service")

SC_TOKEN = config.SC_TOKEN
SERVICE_PORT = config.SERVICE_PORT
SC_VERIFY_URL = config.SC_VERIFY_URL
SC_REQUIRED_SCOPE = config.SC_REQUIRED_SCOPE
AUTH_CACHE_TTL = config.AUTH_CACHE_TTL
AUTH_ENABLED = config.AUTH_ENABLED

# Paths that don't require auth
AUTH_EXEMPT_PATHS = {"/health", "/service-docs"}


class SamcloudAuthMiddleware(BaseHTTPMiddleware):
    """Verify caller identity via SAMcloud token verification.

    Forwards the caller's Bearer token to SAMcloud GET /auth/verify,
    caches results for AUTH_CACHE_TTL seconds, and rejects
    unauthenticated or out-of-scope requests.
    """

    def __init__(self, app, verify_url: str, required_scope: str):
        super().__init__(app)
        self.verify_url = verify_url
        self.required_scope = required_scope
        self._cache: dict[str, tuple[float, dict]] = {}  # token_hash -> (expiry, identity)

    def _cache_key(self, token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()[:16]

    def _get_cached(self, token: str) -> Optional[dict]:
        key = self._cache_key(token)
        entry = self._cache.get(key)
        if entry and entry[0] > time.time():
            return entry[1]
        if entry:
            del self._cache[key]
        return None

    def _set_cached(self, token: str, identity: dict):
        key = self._cache_key(token)
        self._cache[key] = (time.time() + AUTH_CACHE_TTL, identity)

    async def dispatch(self, request: Request, call_next):
        if not AUTH_ENABLED or request.url.path in AUTH_EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing Authorization: Bearer <token>"},
            )

        token = auth_header  # Forward full "Bearer xxx" header

        # Check cache
        cached = self._get_cached(token)
        if cached:
            request.state.caller = cached
            return await call_next(request)

        # Verify with SAMcloud
        try:
            url = self.verify_url
            if self.required_scope:
                url += f"?scope={self.required_scope}"
            async with httpx.AsyncClient(verify=False, timeout=10) as client:
                resp = await client.get(url, headers={"Authorization": token})
        except Exception as e:
            log.warning(f"SAMcloud verify failed: {e}")
            return JSONResponse(
                status_code=502,
                content={"detail": "Auth service unavailable"},
            )

        if resp.status_code == 401:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})
        if resp.status_code == 403:
            return JSONResponse(status_code=403, content={"detail": "Token valid but out of scope"})
        if resp.status_code != 200:
            return JSONResponse(
                status_code=502,
                content={"detail": f"Auth service returned {resp.status_code}"},
            )

        identity = resp.json()
        self._set_cached(token, identity)
        request.state.caller = identity
        log.info(f"Verified caller: {identity.get('username')} ({identity.get('role')})")

        return await call_next(request)

mgr: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mgr
    sc = SamcloudClient(token=SC_TOKEN)
    mgr = ModelManager(sc=sc)

    # Discover and adopt running models
    adopted = mgr.discover()
    for m in adopted:
        log.info(f"Adopted: {m.name} ({m.backend.value}) ~{m.memory_mb}MB on port {m.port}")

    # Claim leases for everything running
    leases = mgr.claim_leases()
    for l in leases:
        log.info(f"Lease: {l}")

    mgr.start_background_tasks()
    log.info(f"Model Service ready - managing {len(mgr.models)} models")

    yield

    log.info("Shutting down...")
    results = mgr.shutdown()
    for r in results:
        log.info(f"  {r}")


app = FastAPI(
    title="Model Service",
    description="Unified model serving with SAMcloud resource leasing",
    lifespan=lifespan,
)

app.add_middleware(
    SamcloudAuthMiddleware,
    verify_url=SC_VERIFY_URL,
    required_scope=SC_REQUIRED_SCOPE,
)


# -- Request models --

class LoadRequest(BaseModel):
    model: str
    backend: str = "auto"  # "ollama", "llama-server", "mlx-vlm", "exo", or "auto"
    port: int = 8000  # for llama-server
    ctx_size: int = 12288
    gpu_layers: int = 99

class UnloadRequest(BaseModel):
    model: str
    force: bool = False

class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[str | dict] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# -- Endpoints --

@app.get("/health")
async def health():
    return {"status": "ok", "service": "model-service", "models": len(mgr.models)}


@app.get("/service-docs", response_class=JSONResponse)
async def service_docs():
    """Public service documentation for discovery by agents and consumers.
    Returns structured docs with a full markdown guide based on the repo README."""
    loaded = {}
    if mgr:
        loaded = {
            name: {
                "backend": mm.backend.value,
                "memory_mb": mm.memory_mb,
            }
            for name, mm in mgr.models.items()
        }

    # Read the repo docs for the full guide
    guide_md = ""
    docs_path = os.path.join(os.path.dirname(__file__), "README.md")
    try:
        with open(docs_path) as f:
            guide_md = f.read()
    except FileNotFoundError:
        guide_md = "(docs not found on disk)"

    return {
        "name": config.SC_SERVICE_NAME,
        "description": (
            f"Unified inference gateway for {config.SC_DEVICE}. "
            "Wraps Ollama (MLX) and llama-server (llama.cpp Metal) behind "
            "a single OpenAI-compatible API. Manages GPU memory via SAMcloud "
            "resource leasing — models spin up on demand and unload after 5 min idle. "
            "Requires SAMcloud token (Bearer) for authenticated access."
        ),
        "auth": {
            "method": "SAMcloud token verification",
            "header": "Authorization: Bearer <your-sc-token>",
            "required_scope": SC_REQUIRED_SCOPE,
            "exempt_paths": list(AUTH_EXEMPT_PATHS),
        },
        "endpoints": {
            "POST /v1/chat/completions": {
                "description": "OpenAI-compatible chat completion",
                "auth": True,
                "body": {"model": "<name>", "messages": [{"role": "user", "content": "..."}], "stream": True},
                "notes": "Model name uses partial matching (e.g. 'qwen3-32b' matches 'Qwen3-32B-Q6_K')",
            },
            "POST /v1/completions": {
                "description": "OpenAI-compatible text completion",
                "auth": True,
                "body": {"model": "<name>", "prompt": "...", "stream": True},
            },
            "GET /models": {
                "description": "List managed and available models",
                "auth": True,
            },
            "GET /status": {
                "description": "Full status — backends, models, leases, resource utilisation",
                "auth": True,
            },
            "POST /models/load": {
                "description": "Load a model (pulls if needed, requests GPU lease)",
                "auth": True,
                "body": {"model": "<name>", "backend": "auto|ollama|llama-server"},
            },
            "POST /models/unload": {
                "description": "Unload a model (releases GPU lease)",
                "auth": True,
                "body": {"model": "<name>", "force": False},
            },
            "GET /health": {
                "description": "Health check",
                "auth": False,
            },
            "GET /service-docs": {
                "description": "This endpoint — public service documentation",
                "auth": False,
            },
        },
        "backends": {
            "ollama": {
                "version": "0.19.0",
                "features": ["mlx", "apple-silicon", "flash-attention"],
                "port": 11434,
            },
            "llama-server": {
                "version": "b8500",
                "features": ["metal", "flash-attention", "quantized-kv-cache"],
                "port": 8000,
            },
        },
        "loaded_models": loaded,
        "model_matching": "Case-insensitive partial match. Use 'qwen3-32b' or 'qwen3.5' as shortnames.",
        "samcloud": {
            "service_id": config.SC_SERVICE_ID,
            "resource": config.SC_RESOURCE_ID,
            "device": config.SC_DEVICE,
        },
        "guide": guide_md,
    }


@app.get("/status")
async def status():
    return mgr.status()


@app.get("/v1/models")
async def list_models_openai():
    """OpenAI-compatible model list.

    Exists because the rest of this gateway's OpenAI surface lives under `/v1`
    — `/v1/chat/completions`, `/v1/completions` — while the only listing was at
    `/models`. A client that found chat where the standard puts it has every
    reason to look for the list where the standard puts it too, and got a 404.
    That is a discovery failure on an otherwise working backend, which is the
    confusing kind: chat succeeds, so the endpoint is clearly right, but the
    client cannot enumerate anything.

    Deliberately cheap. It reports what this gateway can be *asked* for, not
    what is resident — exactly as `/v1/models` does for every other
    OpenAI-compatible server, where an unloaded model is still listed. The pool
    tier is included whenever EXO is enabled rather than only when the pool is
    ready, for the same reason: it is a configured route, and a request for it
    while the pool is down gets the structured `pool_unavailable` 503 that path
    already returns. Readiness lives on `/models` (`exo_pool.ready`,
    `exo_pool.busy`) where there is somewhere to put it.

    No pool read at all, so a wedged pool cannot make discovery hang — the one
    thing that would turn a listing into the outage it is meant to describe.
    """
    now = int(time.time())
    seen: set = set()
    data: list[dict] = []

    def add(model_id: str, owned_by: str):
        if model_id and model_id not in seen:
            seen.add(model_id)
            data.append({
                "id": model_id,
                "object": "model",
                "created": now,
                "owned_by": owned_by,
            })

    # Tiers first: a caller asking this gateway for the pool asks by tier, and
    # listing the resident model id instead would invite a request naming a
    # model we cannot promise to still hold.
    if config.EXO_ENABLED:
        for tier in config.EXO_TIERS:
            add(tier, "exo")

    for name in mgr.models:
        mm = mgr.models[name]
        if mm.backend != Backend.EXO:      # tiers already added under their tier name
            add(name, mm.backend.value)

    try:
        for m in mgr.ollama.list_models():
            add(m.get("name", ""), "ollama")
    except Exception as e:
        log.warning(f"/v1/models: ollama catalogue unavailable: {e}")

    try:
        for m in mgr.llama.available_models():
            add(m.get("name", ""), "llama-server")
    except Exception as e:
        log.warning(f"/v1/models: gguf catalogue unavailable: {e}")

    return {"object": "list", "data": data}


@app.get("/models")
async def list_models():
    return {
        "managed": {
            name: {
                "backend": mm.backend.value,
                "port": mm.port,
                "memory_mb": mm.memory_mb,
                "lease_id": mm.lease_id,
                "idle_seconds": int(time.time() - mm.last_used),
                "request_count": mm.request_count,
            }
            for name, mm in mgr.models.items()
        },
        "available_ollama": [m["name"] for m in mgr.ollama.list_models()],
        "available_gguf": mgr.llama.available_models(),
        "exo_pool": await _exo_pool_view(),
    }


async def _exo_pool_view() -> dict:
    """What the pool can serve right now, for /models.

    Reports the resident model rather than exo's 121-entry catalogue: the
    catalogue is what a swap *could* reach, and advertising it here would
    invite requests for models that are a 30s-10min swap away. Never raises —
    a pool that is down should make this one key say so, not fail the whole
    listing.

    Async, and the `/state` read goes to a thread, because `ExoClient` talks to
    the pool over synchronous httpx. Calling it directly from this handler
    blocked the event loop for the length of a few-hundred-KB fetch, which on
    `main` also stalls `stats_loop` (15s) and, once the offering hold lifts,
    `offering_loop` (30s) — an offering poll delayed behind a status read looks
    like a memory-pressure change. Not newly broken; newly exposed, because
    those two loops did not exist on the branch this came from.

    `asyncio.to_thread` is the right tool *here* and was the wrong tool on the
    generation path, which is worth stating so the two do not get unified later.
    A thread cannot be cancelled: for a minutes-long generation holding an
    exclusive lease that meant a disconnected client kept the pool, so that path
    uses aiohttp. This is a bounded read of at most `timeout=15`, holds no lease,
    and nothing is harmed by it finishing after the caller has gone.
    """
    if not config.EXO_ENABLED:
        return {"enabled": False}
    view = {
        "enabled": True,
        "tiers": list(config.EXO_TIERS),
        "endpoint": config.EXO_BASE,
        "resource_id": config.EXO_RESOURCE_ID,
        "allocation": "exclusive — one request at a time, leased per generation",
    }
    try:
        status = await asyncio.to_thread(mgr.exo.pool_status_cached)
        view["resident_model"] = status["resident_model"]
        view["ready"] = status["ready"]
        # A generation is in flight. Read from exo's runner states rather than
        # from our own lease bookkeeping, so it is true even when the pool is
        # driven directly rather than through this gateway. `busy` with no
        # active lease on the resource is the wedge signature — a generation
        # nobody is reading — which is what a client dying mid-request leaves.
        view["busy"] = status["busy"]
        # Per-runner states, which carry layer progress while a swap is in
        # flight — the difference between "not ready" and "23/47 layers in".
        view["runners"] = {
            i["model"]: i["runners"] for i in status["instances"]
        }
        if not status["ready"]:
            view["note"] = (
                "reachable but no model resident and ready (mid-swap, or never "
                "placed — the pool cannot restart itself)"
            )
    except Exception as e:
        view["ready"] = False
        view["resident_model"] = None
        view["error"] = str(e)
    return view


def _capacity_503(e: "capacity.InsufficientCapacity") -> HTTPException:
    """A refusal is an answer, not a fault — 503 with the numbers to retry on.

    Every load path funnels through here so a caller cannot tell the three
    backends apart by how they decline.
    """
    log.info(f"Refused on capacity: {e}")
    return HTTPException(status_code=503, detail=e.as_dict())


def _busy_503(e: "capacity.PoolBusy") -> HTTPException:
    """The pool is taken. Same 503 shape as a capacity refusal, plus Retry-After.

    A caller must be able to tell "busy, come back" from "does not fit here"
    and from "the gateway is broken" without parsing prose, and must never get
    a hang or a 500 for either of the first two. `Retry-After` is set as a real
    header as well as in the body, so an HTTP client that already understands
    it backs off correctly without reading our JSON.
    """
    log.info(f"Declined on exclusive lease: {e}")
    headers = {}
    if e.retry_after_s:
        headers["Retry-After"] = str(e.retry_after_s)
    return HTTPException(status_code=503, detail=e.as_dict(), headers=headers or None)


def _pool_unavailable_503(e: Exception) -> HTTPException:
    """The pool is not there — distinct from it being busy.

    Deliberately not a 404: the tier is configured and real, so "no such model"
    would send a caller looking for a typo. And deliberately not a retry hint —
    the pool cannot restart itself after a reboot (a headless launch is denied
    local-network access on macOS), so a swap or a relaunch needs a human at a
    Terminal and a tight retry loop would just spin.
    """
    log.warning(f"Pool unavailable: {e}")
    return HTTPException(status_code=503, detail={
        "error": "pool_unavailable",
        "message": str(e),
        "resource_id": config.EXO_RESOURCE_ID,
        "endpoint": config.EXO_BASE,
        "note": (
            "the exo pool is placed out of band and cannot start itself after a "
            "reboot; it needs to be launched from a Terminal on the host"
        ),
    })


@app.post("/models/load")
async def load_model(req: LoadRequest):
    try:
        # Only probe for a GGUF match when the backend could plausibly be
        # llama-server, so an explicit backend="ollama" still bypasses it.
        gguf_match = None
        if req.backend in ("auto", "llama-server"):
            gguf_match = match_gguf_model(req.model, mgr.llama.available_models())

        if req.backend == "exo" or (
            req.backend == "auto" and match_exo_tier(req.model)
        ):
            # "Loading" the pool only registers the tier and reports what is
            # resident — nothing is placed and no lease is taken, because the
            # pool's lease is per generation. Useful as a readiness probe.
            mm = await asyncio.to_thread(
                mgr.resolve_exo_tier, req.model.strip().lower()
            )
        elif req.backend == "mlx-vlm" or (
            req.backend == "auto" and match_vlm_model(req.model)
        ):
            mm = mgr.load_vlm_model(req.model)
        elif req.backend == "llama-server" or gguf_match or (
            req.backend == "auto" and req.model.endswith(".gguf")
        ):
            # Resolve to full path: known GGUF match, absolute path, or a
            # filename relative to MODELS_DIR.
            if gguf_match:
                model_path = gguf_match["file"]
            elif req.model.startswith("/"):
                model_path = req.model
            else:
                model_path = str(config.MODELS_DIR / req.model)
            mm = mgr.load_llama_model(
                model_path,
                port=req.port,
                ctx_size=req.ctx_size,
                gpu_layers=req.gpu_layers,
            )
        else:
            mm = mgr.load_ollama_model(req.model)

        return {
            "status": "loaded",
            "model": mm.name,
            "backend": mm.backend.value,
            "port": mm.port,
            "memory_mb": mm.memory_mb,
            "lease_id": mm.lease_id,
        }
    except capacity.InsufficientCapacity as e:
        raise _capacity_503(e)
    except capacity.PoolBusy as e:
        raise _busy_503(e)
    except ExoUnavailable as e:
        raise _pool_unavailable_503(e)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log.exception(f"Failed to load {req.model}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/unload")
async def unload_model(req: UnloadRequest):
    # Accept the same partial names as /models/load and inference: exact key
    # first, then case-insensitive substring match against loaded models.
    key = req.model
    if key not in mgr.models:
        lower = req.model.lower()
        key = next((n for n in mgr.models if lower in n.lower()), req.model)
    result = mgr.unload(key, force=req.force)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"{req.model} not found")
    return result


def _fix_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Convert Ollama tool_calls to OpenAI format.
    Ollama returns arguments as a JSON object; OpenAI SDK expects a JSON string."""
    fixed = []
    for tc in tool_calls:
        fc = dict(tc)
        if "function" in fc:
            fn = dict(fc["function"])
            if isinstance(fn.get("arguments"), dict):
                fn["arguments"] = json.dumps(fn["arguments"])
            fn.pop("index", None)  # Ollama includes index, OpenAI doesn't
            fc["function"] = fn
        if "type" not in fc:
            fc["type"] = "function"
        fixed.append(fc)
    return fixed


import re

# Gemma 4 tool call format:
#   <|tool_call>call:func_name{key:<|"|">value<|"|">, key2:<|"|">value2<|"|">}<tool_call|>
_GEMMA_TC_RE = re.compile(
    r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>',
    re.DOTALL,
)
_GEMMA_ARG_RE = re.compile(
    r'(\w+):\s*<\|"\|">(.*?)<\|"\|">',
)
_CALL_COUNTER = 0


def _parse_gemma_tool_calls(content: str) -> tuple[str, list[dict]]:
    """Parse Gemma 4 tool call tags from content into structured tool_calls.
    Returns (remaining_content, tool_calls)."""
    global _CALL_COUNTER
    tool_calls = []
    for match in _GEMMA_TC_RE.finditer(content):
        func_name = match.group(1)
        args_str = match.group(2)
        arguments = {}
        for arg_match in _GEMMA_ARG_RE.finditer(args_str):
            arguments[arg_match.group(1)] = arg_match.group(2)
        _CALL_COUNTER += 1
        tool_calls.append({
            "id": f"call_gemma_{_CALL_COUNTER}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments),
            },
        })
    # Strip tool call tags from content
    remaining = _GEMMA_TC_RE.sub("", content).strip()
    return remaining, tool_calls


def _tools_to_gemma_system(tools: list[dict]) -> str:
    """Convert OpenAI tools array to a system prompt for Gemma 4."""
    lines = ["You have access to the following tools:\n"]
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        param_parts = []
        for pname, pinfo in params.items():
            req = " (required)" if pname in required else ""
            param_parts.append(f"{pname}: {pinfo.get('type', 'string')}{req}")
        lines.append(f"- {name}: {desc}. Parameters: {{{', '.join(param_parts)}}}")
    lines.append("")
    lines.append('When you need to use a tool, output EXACTLY this format:')
    lines.append('<|tool_call>call:function_name{param:<|"|">value<|"|">}<tool_call|>')
    lines.append("")
    lines.append("Do not wrap tool calls in markdown. Call tools directly.")
    return "\n".join(lines)


def _openai_messages_to_ollama(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages to Ollama native format.
    Key differences:
    - tool_calls arguments: OpenAI=string, Ollama=object
    - tool messages: OpenAI has tool_call_id, Ollama doesn't use it
    """
    converted = []
    for msg in messages:
        m = dict(msg)
        # Convert tool_calls arguments from string to object
        if "tool_calls" in m and m["tool_calls"]:
            fixed_tcs = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                if "function" in tc:
                    fn = dict(tc["function"])
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    fn.pop("index", None)
                    tc["function"] = fn
                tc.pop("type", None)
                tc.pop("id", None)
                fixed_tcs.append(tc)
            m["tool_calls"] = fixed_tcs
        # Strip tool_call_id from tool messages (Ollama doesn't use it)
        m.pop("tool_call_id", None)
        converted.append(m)
    return converted


async def _resolve_model(model_name: str):
    """Find a managed model, or auto-load it. Never returns None for known models.

    Async only because of the pool. Resolving a tier reads exo's `/state` over
    synchronous httpx with a 15s timeout, and this runs inside the request
    handlers — so on the happy path that was tens of milliseconds nobody
    noticed, and against a wedged or unreachable pool it froze the **entire**
    gateway (every route, health reporting, the stats loop, every other model's
    traffic) for up to 15s per request before raising. That is the failure this
    backend was built to decline gracefully, so blocking the process while
    detecting it is the wrong way round.

    The reads therefore go to a thread. This is the same rule as
    `_exo_pool_view` and NOT the rule for a generation: a tier resolve takes no
    lease and loads nothing, so it is bounded and strands nothing if it outlives
    its caller. A generation is neither, which is why that path uses aiohttp.

    Only the pool's calls are moved. The three local backends keep their
    existing behaviour deliberately — `load_ollama_model` and `load_vlm_model`
    block this loop today for as long as a model takes to load, which is a
    larger, pre-existing problem than this commit should quietly change. Making
    this function async is the seam that lets it be fixed separately.
    """
    # The pool first, and before the partial-match scan below. Two reasons it
    # cannot be folded in with the others: a tier is matched exactly (see
    # match_exo_tier) where every other backend matches on substrings, and
    # resolving a tier must re-read what the pool currently holds rather than
    # trust a cached name. Putting it after the substring scan would also let a
    # loaded model whose name happens to contain "think" shadow the tier.
    if match_exo_tier(model_name):
        try:
            return await asyncio.to_thread(
                mgr.resolve_exo_tier, model_name.strip().lower()
            )
        except ExoUnavailable as e:
            raise _pool_unavailable_503(e)

    # Check already-loaded models (exact then partial match)
    matched_name = None
    if model_name in mgr.models:
        matched_name = model_name
    else:
        lower = model_name.lower()
        for name in mgr.models:
            if lower in name.lower():
                matched_name = name
                break

    if matched_name:
        mm = mgr.models[matched_name]
        if mm.backend == Backend.EXO:
            # Re-reads /state to pick up a swap; same blocking read as above.
            alive = await asyncio.to_thread(mgr.ensure_running, matched_name)
        else:
            alive = mgr.ensure_running(matched_name)
        if not alive and mm.backend == Backend.EXO:
            # For local backends ensure_running() reloads and a False is worth
            # attempting anyway. For the pool there is nothing to reload: False
            # means it is unreachable or mid-swap, and proxying into that
            # produces a 500 several minutes later instead of an answer now.
            raise _pool_unavailable_503(
                ExoUnavailable(
                    f"the pool is not ready to serve tier '{matched_name}'"
                )
            )
        mgr.touch(matched_name)
        return mm

    # Model not loaded — try to auto-load it transparently.

    # Vision-language models: the gateway owns the mlx-vlm process on demand.
    if match_vlm_model(model_name):
        log.info(f"Auto-loading VLM (requested: {model_name})")
        try:
            return mgr.load_vlm_model(model_name)
        except Exception as e:
            log.error(f"VLM auto-load failed for {model_name}: {e}")
            return None

    # Check if it's a known Ollama model (already pulled).
    lower = model_name.lower()
    for m in mgr.ollama.list_models():
        ollama_name = m["name"]
        if lower in ollama_name.lower() or ollama_name.lower() in lower:
            log.info(f"Auto-loading {ollama_name} (requested: {model_name})")
            try:
                mm = mgr.load_ollama_model(ollama_name)
                return mm
            except capacity.InsufficientCapacity as e:
                # Not a missing model — a capacity answer. Surface it, with the
                # list of what does fit, instead of collapsing to "not pulled".
                raise _capacity_503(e)
            except Exception as e:
                log.error(f"Auto-load failed for {ollama_name}: {e}")
                return None

    # Check GGUF files
    for m in mgr.llama.available_models():
        if lower in m["name"].lower():
            log.info(f"Auto-loading {m['file']} (requested: {model_name})")
            try:
                mm = mgr.load_llama_model(m["file"])
                return mm
            except capacity.InsufficientCapacity as e:
                # Not a missing model — a capacity answer. Surface it, with the
                # list of what does fit, instead of collapsing to "not pulled".
                raise _capacity_503(e)
            except Exception as e:
                log.error(f"Auto-load failed for {m['name']}: {e}")
                return None

    return None


async def _watch_disconnect(request: Optional[Request], poll_s: float = 2.0):
    """Resolve when the caller goes away. Never resolves if we cannot tell.

    Used only on the pool path. Every other backend either streams (where a
    disconnect surfaces as CancelledError in the generator) or finishes fast
    enough that nobody is harmed by running to completion. The pool is the one
    backend where an abandoned request denies the resource to everyone else.
    """
    if request is None:
        await asyncio.Event().wait()      # nothing to watch; never fires
    while True:
        if await request.is_disconnected():
            return
        await asyncio.sleep(poll_s)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, http_request: Request = None):
    mm = await _resolve_model(req.model)
    if not mm:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not available (not pulled or no GGUF file)")

    if mm.backend == Backend.OLLAMA:
        # Route to Ollama native /api/chat with think:false to avoid
        # the Ollama 0.19 bug where /v1/ ignores think parameter and
        # returns empty content with all output in reasoning field.
        # We translate the native response to OpenAI format ourselves.
        # Pass tools through for structured tool_call support.
        # Route through native /api/chat (not /v1/) to avoid the Ollama bug
        # where /v1/ puts content in the reasoning field. Via /api/chat,
        # content is populated correctly alongside thinking.
        # Convert OpenAI-format messages to Ollama native format.
        ollama_messages = _openai_messages_to_ollama(req.messages)
        ollama_kwargs = {}
        if req.tools:
            ollama_kwargs["tools"] = req.tools
        # think:false — this is the documented #69 decision but was never wired
        # in. Without it a reasoning model (qwen3.5) spends unbounded, *uncounted*
        # tokens on a thinking phase that we then discard (we only surface
        # message.content), which both wastes the inference slot for minutes
        # (the #97 hang) and starves any num_predict budget below. Disable it so
        # tokens go to the content we actually return.
        ollama_kwargs["think"] = False
        # Bound generation length: Ollama takes this under options.num_predict,
        # not the OpenAI-style top-level max_tokens. Without this an unbounded
        # generation can occupy the single inference slot for minutes and starve
        # every other route (ticket #97).
        if req.max_tokens is not None:
            ollama_kwargs["options"] = {"num_predict": req.max_tokens}

        if req.stream:
            async def stream():
                saw_tool_calls = False
                try:
                    async for chunk in mgr.ollama.chat_stream(mm.name, ollama_messages, **ollama_kwargs):
                        delta = {}
                        if "message" in chunk and chunk["message"].get("content"):
                            delta["content"] = chunk["message"]["content"]
                        if "message" in chunk and chunk["message"].get("tool_calls"):
                            delta["tool_calls"] = _fix_tool_calls(chunk["message"]["tool_calls"])
                            saw_tool_calls = True
                        if chunk.get("done"):
                            finish = "tool_calls" if saw_tool_calls else "stop"
                            yield "data: " + json.dumps({
                                "choices": [{"delta": {}, "finish_reason": finish}]
                            }) + "\n\n"
                            yield "data: [DONE]\n\n"
                        elif delta:
                            yield "data: " + json.dumps({
                                "choices": [{"delta": delta, "finish_reason": None}],
                                "model": mm.name,
                            }) + "\n\n"
                except asyncio.CancelledError:
                    log.info(f"Client disconnected during stream for {mm.name}")
                except Exception as e:
                    log.warning(f"Stream error for {mm.name}: {e}")
            return StreamingResponse(stream(), media_type="text/event-stream")
        else:
            # Non-streaming: collect full response via native API
            full_content = ""
            tool_calls = None
            usage = {}
            for chunk in mgr.ollama.chat(mm.name, ollama_messages, **ollama_kwargs):
                if "message" in chunk:
                    full_content += chunk["message"].get("content", "")
                    if chunk["message"].get("tool_calls"):
                        tool_calls = chunk["message"]["tool_calls"]
                if chunk.get("done"):
                    usage = {
                        "prompt_tokens": chunk.get("prompt_eval_count", 0),
                        "completion_tokens": chunk.get("eval_count", 0),
                        "total_tokens": (chunk.get("prompt_eval_count", 0) +
                                         chunk.get("eval_count", 0)),
                    }
            message = {"role": "assistant", "content": full_content}
            if tool_calls:
                message["tool_calls"] = _fix_tool_calls(tool_calls)
            finish_reason = "tool_calls" if tool_calls else "stop"
            return {
                "choices": [{
                    "message": message,
                    "finish_reason": finish_reason,
                    "index": 0,
                }],
                "model": mm.name,
                "usage": usage,
            }

    elif mm.backend == Backend.EXO:
        # exo speaks OpenAI already, so this is a proxy and not a translation:
        # no native-format detour, no tool-call reshaping, no system-prompt
        # injection. What this branch adds over a bare reverse proxy is the
        # exclusive lease, held for exactly as long as the generation runs.
        payload = {
            "model": mm.name,   # the resident model id, not the tier the caller named
            "messages": req.messages,
            "stream": req.stream,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.tools:
            payload["tools"] = req.tools
        if req.tool_choice is not None:
            payload["tool_choice"] = req.tool_choice

        # Per-model defaults, with max_tokens treated as a ceiling rather than
        # a default: a caller may ask for less, never for more. On a shared
        # backend an over-long generation is rude; on this one it is a denial
        # of service, because one request owns the whole pool while it runs.
        defaults = mgr.exo_request_defaults(mm.name)
        cap = defaults.pop("max_tokens", None)
        if cap is not None:
            payload["max_tokens"] = min(req.max_tokens, cap) if req.max_tokens else cap
        elif req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        for key, value in defaults.items():
            payload.setdefault(key, value)

        purpose = f"chat:{mm.name}"

        if req.stream:
            # Acquire BEFORE returning the response. Once StreamingResponse is
            # returned the 200 is committed, and a busy pool discovered after
            # that could only be reported inside the stream body — which is
            # precisely the "not a hang, not a 500, a structured decline" case
            # this is supposed to get right.
            try:
                lease_id = mgr.acquire_pool(purpose)
            except capacity.PoolBusy as e:
                raise _busy_503(e)

            async def stream():
                try:
                    async for line in mgr.exo.chat_stream(
                        mm.name, req.messages,
                        **{k: v for k, v in payload.items()
                           if k not in ("model", "messages", "stream")}
                    ):
                        yield line + "\n"
                except asyncio.CancelledError:
                    log.info(f"Client disconnected during pool stream for {mm.name}")
                    raise
                except ExoRequestFailed as e:
                    # Headers are already sent, so this cannot become a 502.
                    # Emit it as a terminal SSE error event rather than just
                    # closing the stream, so the caller can tell a failed
                    # generation from a short one.
                    log.warning(f"Pool failed the stream for {mm.name}: {e}")
                    yield "data: " + json.dumps({
                        "error": {"message": e.detail, "type": "pool_error",
                                  "pool_status": e.status},
                    }) + "\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    log.warning(f"Pool stream error for {mm.name}: {e}")
                finally:
                    # Covers a clean finish, an error, and a client that walked
                    # away mid-generation (CancelledError unwinds through here).
                    mgr.release_pool(lease_id)

            return StreamingResponse(stream(), media_type="text/event-stream")

        try:
            with mgr.pool_lease(purpose):
                # Streamed and aggregated rather than sent with stream:false —
                # exo's non-streaming endpoint returns headers and no body (see
                # ExoClient.chat). This also keeps the call cancellable and off
                # the worker threads, which is what lets the disconnect watch
                # below actually stop the work.
                gen = asyncio.ensure_future(mgr.exo.chat_collect(
                    mm.name,
                    req.messages,
                    **{k: v for k, v in payload.items()
                       if k not in ("model", "messages", "stream")},
                ))
                watch = asyncio.ensure_future(_watch_disconnect(http_request))
                try:
                    done, _ = await asyncio.wait(
                        {gen, watch}, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    watch.cancel()
                if gen not in done:
                    # The client gave up first. Starlette does not cancel a
                    # plain handler on disconnect, so without this the pool
                    # would stay leased for the rest of a generation nobody is
                    # waiting for — on an exclusive resource that is not a
                    # wasted computation, it is a closed pool.
                    gen.cancel()
                    try:
                        await gen
                    except (asyncio.CancelledError, Exception):
                        pass
                    log.info(
                        f"Client disconnected before the pool answered; "
                        f"cancelled the generation and released the lease"
                    )
                    raise HTTPException(status_code=499, detail={
                        "error": "client_disconnected",
                        "message": "caller went away before the pool answered",
                    })
                data = gen.result()
        except capacity.PoolBusy as e:
            raise _busy_503(e)
        except ExoRequestFailed as e:
            # A pool that fails a generation is a bad gateway, not a bad
            # request and not a broken model-service. 502 keeps those apart.
            log.warning(f"Pool failed the generation for {mm.name}: {e}")
            raise HTTPException(status_code=502, detail={
                "error": "pool_error",
                "message": e.detail,
                "pool_status": e.status,
                "body": e.body,
            })

        # Report the tier the caller asked for alongside what actually served
        # it, so a swap on the pool is visible in the answer rather than silent.
        if isinstance(data, dict):
            data.setdefault("model", mm.name)
            data["served_by"] = {
                "backend": Backend.EXO.value,
                "tier": mm.tier,
                "model": mm.name,
                "resource_id": config.EXO_RESOURCE_ID,
            }
        return data

    elif mm.backend == Backend.LLAMA:
        # Forward to llama-server's OpenAI-compatible endpoint
        payload = {
            "model": mm.name,
            "messages": req.messages,
            "stream": req.stream,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.tools:
            payload["tools"] = req.tools
        if req.tool_choice is not None:
            payload["tool_choice"] = req.tool_choice

        if req.stream:
            async def stream():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"http://127.0.0.1:{mm.port}/v1/chat/completions",
                        json=payload,
                        timeout=None,
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line:
                                yield line + "\n"
            return StreamingResponse(stream(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{mm.port}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                )
                return resp.json()

    elif mm.backend == Backend.VLM:
        # Forward to mlx-vlm. mlx-vlm doesn't handle tools natively,
        # so we inject tools into the system prompt and parse Gemma 4's
        # tool call tags from the response content.
        messages = list(req.messages)
        if req.tools:
            tool_system = _tools_to_gemma_system(req.tools)
            # Prepend or merge with existing system message
            if messages and messages[0].get("role") == "system":
                messages[0] = dict(messages[0])
                messages[0]["content"] = tool_system + "\n\n" + messages[0]["content"]
            else:
                messages.insert(0, {"role": "system", "content": tool_system})

        payload = {
            "model": mm.name,
            "messages": messages,
            "stream": req.stream,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens

        if req.stream:
            # For streaming with tools, collect full response then parse
            # (Gemma tool calls come as a single chunk, not incremental)
            if req.tools:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"http://127.0.0.1:{mm.port}/v1/chat/completions",
                        json={**payload, "stream": False},
                        timeout=300,
                    )
                    data = resp.json()
                    content = data["choices"][0]["message"].get("content", "")
                    remaining, tool_calls = _parse_gemma_tool_calls(content)
                    message = {"role": "assistant", "content": remaining}
                    if tool_calls:
                        message["tool_calls"] = tool_calls
                    finish = "tool_calls" if tool_calls else "stop"
                    # Emit as a single SSE event
                    async def single_event():
                        yield "data: " + json.dumps({
                            "choices": [{"delta": message, "finish_reason": finish}],
                            "model": mm.name,
                            "usage": data.get("usage", {}),
                        }) + "\n\n"
                        yield "data: [DONE]\n\n"
                    return StreamingResponse(single_event(), media_type="text/event-stream")
            else:
                async def stream():
                    async with httpx.AsyncClient() as client:
                        async with client.stream(
                            "POST",
                            f"http://127.0.0.1:{mm.port}/v1/chat/completions",
                            json=payload,
                            timeout=None,
                        ) as resp:
                            async for line in resp.aiter_lines():
                                if line:
                                    yield line + "\n"
                return StreamingResponse(stream(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{mm.port}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                )
                data = resp.json()
                # Parse Gemma tool calls from content
                if req.tools:
                    content = data["choices"][0]["message"].get("content", "")
                    remaining, tool_calls = _parse_gemma_tool_calls(content)
                    data["choices"][0]["message"]["content"] = remaining
                    if tool_calls:
                        data["choices"][0]["message"]["tool_calls"] = tool_calls
                        data["choices"][0]["finish_reason"] = "tool_calls"
                return data


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    mm = await _resolve_model(req.model)
    if not mm:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not available (not pulled or no GGUF file)")

    if mm.backend == Backend.EXO:
        # exo serves no /v1/completions route — its surface is
        # /v1/chat/completions, /v1/messages and /v1/responses. Declining
        # plainly beats silently wrapping the prompt in a single user message
        # and handing back a chat-shaped body from a completions endpoint,
        # which would be a worse surprise than a 400. Note this decline costs
        # no lease: it is refused before the pool is touched.
        raise HTTPException(status_code=400, detail={
            "error": "unsupported_route",
            "message": (
                f"the exo pool has no text-completion endpoint; send tier "
                f"'{mm.tier}' to /v1/chat/completions instead"
            ),
            "tier": mm.tier,
            "model": mm.name,
        })

    if mm.backend == Backend.OLLAMA:
        if req.stream:
            def stream():
                for chunk in mgr.ollama.generate(mm.name, req.prompt):
                    yield json.dumps(chunk) + "\n"
            return StreamingResponse(stream(), media_type="application/x-ndjson")
        else:
            chunks = list(mgr.ollama.generate(mm.name, req.prompt))
            return chunks[-1] if chunks else {}

    elif mm.backend == Backend.LLAMA:
        payload = {
            "model": mm.name,
            "prompt": req.prompt,
            "stream": req.stream,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens

        if req.stream:
            async def stream():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"http://127.0.0.1:{mm.port}/v1/completions",
                        json=payload,
                        timeout=None,
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line:
                                yield line + "\n"
            return StreamingResponse(stream(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{mm.port}/v1/completions",
                    json=payload,
                    timeout=300,
                )
                return resp.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
