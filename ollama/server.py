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
"""

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

from .manager import ModelManager, Backend
from .samcloud import SamcloudClient
from .ollama_client import OllamaClient
from .llama_client import LlamaServerClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("model-service")

SC_TOKEN = os.environ.get("SC_TOKEN", "sc_agent_448090436817362f5250c0d0f83bef53")
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", "8800"))
SC_VERIFY_URL = os.environ.get(
    "SC_VERIFY_URL", "https://stg.samtg.xyz:9443/api/v1/auth/verify"
)
SC_REQUIRED_SCOPE = os.environ.get("SC_REQUIRED_SCOPE", "device:slice-test")
AUTH_CACHE_TTL = 300  # 5 minutes
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").lower() == "true"

# Paths that don't require auth
AUTH_EXEMPT_PATHS = {"/health"}


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
    backend: str = "auto"  # "ollama", "llama-server", or "auto"
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


@app.get("/status")
async def status():
    return mgr.status()


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
    }


@app.post("/models/load")
async def load_model(req: LoadRequest):
    try:
        if req.backend == "llama-server" or (
            req.backend == "auto" and req.model.endswith(".gguf")
        ):
            # Resolve to full path if just a filename
            model_path = req.model
            if not model_path.startswith("/"):
                model_path = f"/Users/sam/models/{req.model}"
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
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log.exception(f"Failed to load {req.model}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/unload")
async def unload_model(req: UnloadRequest):
    result = mgr.unload(req.model, force=req.force)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"{req.model} not found")
    return result


def _resolve_model(model_name: str):
    """Find which managed model matches the request and touch it."""
    # Exact match
    if model_name in mgr.models:
        mgr.touch(model_name)
        return mgr.models[model_name]
    # Partial match (e.g. "qwen3-32b" matches "Qwen3-32B-Q6_K")
    lower = model_name.lower()
    for name, mm in mgr.models.items():
        if lower in name.lower():
            mgr.touch(name)
            return mm
    return None


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    mm = _resolve_model(req.model)
    if not mm:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not loaded")

    if mm.backend == Backend.OLLAMA:
        # Route to Ollama
        if req.stream:
            def stream():
                for chunk in mgr.ollama.chat(mm.name, req.messages):
                    yield json.dumps(chunk) + "\n"
            return StreamingResponse(stream(), media_type="application/x-ndjson")
        else:
            chunks = list(mgr.ollama.chat(mm.name, req.messages))
            return chunks[-1] if chunks else {}

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


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    mm = _resolve_model(req.model)
    if not mm:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not loaded")

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
