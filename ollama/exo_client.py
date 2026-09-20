"""exo client for the exo-pool inference instance.

Mirrors `ollama_client.py`, with one structural difference that shapes the whole
file: exo's HTTP API is already OpenAI-compatible, so the chat path is a
**proxy**, not a translation. There is no native-format detour like the Ollama
`/api/chat` + `think:false` workaround — we hand exo the caller's payload and
hand the caller exo's answer back.

Two things about this backend are unlike every other one behind the gateway:

1. **We do not load models here.** ONE exo instance spans slice and wafer and
   serves ONE request at a time, and swapping the resident model costs
   30s-10min. The resident model is therefore a property of the instance, not
   a request parameter — placed out of band by whoever operates the pool.
   Callers ask for a *tier* (`think`) and get whatever is resident.
   `resident_model()` reads that from the pool instead of trusting a constant,
   so an operator swapping the model needs no deploy here.

2. **Memory is not ours to account for.** The pages exo occupies are already
   wired on both boxes and already visible to each box's own `capacity.py`.
   Leasing bytes against this resource would double-count them, and a lease
   written from a pre-load estimate was measured 4.7x wrong. The lease on
   exo-pool means "the pool is TAKEN" and carries `memory_mb: null`. Nothing
   in this file estimates a memory figure, on purpose.
"""

import json
import httpx
from dataclasses import dataclass, field
from typing import Optional

from . import config

EXO_BASE = config.EXO_BASE


class ExoUnavailable(Exception):
    """The pool is unreachable, or has no model resident and ready to serve.

    Distinct from the pool being *busy* — that is a lease conflict and is
    raised as `PoolBusy` by the manager. This one means there is nothing to
    talk to, so a caller should not be told to retry in N seconds.
    """


def _instance_model(entry: dict) -> Optional[str]:
    """Pull the model id out of one `/state` instance entry.

    An entry is tagged with its runner variant — `{"MlxRingInstance": {...}}`.
    The variant name is exo's internal enum and changes with the execution
    strategy (ring vs tensor vs a future one), so read through whatever the
    single key is rather than matching on it.
    """
    for variant in entry.values():
        if isinstance(variant, dict):
            model_id = (variant.get("shardAssignments") or {}).get("modelId")
            if model_id:
                return model_id
    return None


def _instance_runners(entry: dict) -> list[str]:
    """Runner ids serving one instance, in no particular order."""
    for variant in entry.values():
        if isinstance(variant, dict):
            assignments = (variant.get("shardAssignments") or {})
            return list((assignments.get("runnerToShard") or {}).keys())
    return []


def _runner_is_running(runner_state: dict) -> bool:
    """True only for `RunnerRunning`.

    Runner state is variant-tagged like the instance: `{"RunnerRunning": {}}`,
    `{"RunnerShuttingDown": {}}`. A shutting-down runner still appears in
    `/state` and still has its shard assignment, so counting a model as
    servable because it is *mentioned* would route a request into a shard
    that is on its way out.
    """
    return isinstance(runner_state, dict) and "RunnerRunning" in runner_state


@dataclass
class ExoClient:
    base_url: str = EXO_BASE
    _http: httpx.Client = field(default=None, repr=False)

    def __post_init__(self):
        # Short timeout for state/catalogue reads. Generation gets its own,
        # much longer, timeout at the call site — a 180B-class model spread
        # over two boxes over Thunderbolt is not a localhost round trip.
        self._http = httpx.Client(base_url=self.base_url, timeout=15)

    # -- pool state --

    def state(self) -> dict:
        """Raw `/state`. The pool's own view of itself."""
        r = self._http.get("/state")
        r.raise_for_status()
        return r.json()

    def reachable(self) -> bool:
        try:
            self.state()
            return True
        except Exception:
            return False

    def resident_model(self) -> Optional[str]:
        """The model the pool is holding and ready to serve, or None.

        "Ready" means every runner backing the instance reports
        `RunnerRunning`. During a model swap the old instance's runners go
        `RunnerShuttingDown` while the new one's come up, and for that window
        there is genuinely nothing resident — which is the honest answer to
        give a caller, rather than naming a model that cannot serve.
        """
        try:
            state = self.state()
        except Exception as e:
            raise ExoUnavailable(f"exo pool at {self.base_url} unreachable: {e}")

        runners = state.get("runners") or {}
        for entry in (state.get("instances") or {}).values():
            model_id = _instance_model(entry)
            if not model_id:
                continue
            runner_ids = _instance_runners(entry)
            if runner_ids and all(
                _runner_is_running(runners.get(rid) or {}) for rid in runner_ids
            ):
                return model_id
        return None

    def list_models(self) -> list[dict]:
        """exo's model *catalogue* — everything it could run, not what is resident.

        121 entries on this pool. Never use this to decide what can serve a
        request right now; that is `resident_model()`. It is here so the
        gateway can say what a swap could reach.
        """
        r = self._http.get("/v1/models")
        r.raise_for_status()
        return r.json().get("data", [])

    # -- generation --

    # A generation on the pool is minutes, not seconds: two boxes, ring
    # pipeline parallelism, and a reasoning model. Connect fast or fail, then
    # wait a long time for tokens. The read timeout is deliberately shorter
    # than the exclusive lease's TTL — see config.EXO_LEASE_TTL for why the
    # ordering matters.
    _generate_timeout = httpx.Timeout(
        connect=10, read=config.EXO_GENERATE_TIMEOUT, write=30, pool=30
    )

    def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
        """exo's own non-streaming endpoint. NOT used by the gateway — see below.

        Measured 2026-09-20, probing the pool directly and bypassing the gateway
        entirely: `stream: false` returns `200` headers immediately and then
        never sends a body. Two runs, 240s each, `size_download=0`. The gateway
        therefore serves non-streaming callers through `chat_collect()`, which
        streams and aggregates.

        Kept because it is the obvious thing for the next person to reach for,
        and a docstring saying "measured, does not return" is cheaper than them
        rediscovering it against a resource that serves one request at a time.
        """
        payload = {"model": model, "messages": messages, "stream": False, **kwargs}
        r = self._http.post(
            "/v1/chat/completions", json=payload, timeout=self._generate_timeout
        )
        r.raise_for_status()
        return r.json()

    async def chat_collect(self, model: str, messages: list[dict], **kwargs) -> dict:
        """A non-streaming answer, assembled from the streaming endpoint.

        Gives a caller the plain OpenAI response shape they asked for without
        using exo's `stream: false` (see `chat()` — it does not return a body).
        Streaming and aggregating has two further properties that matter
        specifically on an exclusive resource:

        - **It is cancellable.** aiohttp drops the connection when the
          coroutine is cancelled, so a client that walks away actually closes
          the socket to exo rather than leaving a generation running with the
          pool leased behind it. The first implementation of this path used
          `asyncio.to_thread` around the sync call above; a thread cannot be
          cancelled, so a disconnected caller held the pool until the
          generation timed out, and it blocked the gateway's own graceful
          shutdown too.
        - **It pins no worker thread** for what can be a 25-minute generation.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict] = []
        finish_reason = None
        usage: dict = {}
        model_reported = None

        async for line in self.chat_stream(model, messages, **kwargs):
            if line.startswith(":"):          # SSE comment, e.g. ": keep-alive"
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            model_reported = chunk.get("model") or model_reported
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                # Deltas while streaming; a full message if exo ever sends one.
                delta = choice.get("delta") or choice.get("message") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                # Reasoning models put the thinking phase in a side channel and
                # the field name is not standardised. Collect it rather than
                # discard it, so a generation that spends its whole budget
                # thinking does not come back as an empty answer with no clue why.
                for key in ("reasoning_content", "reasoning", "thinking"):
                    if delta.get(key):
                        reasoning_parts.append(delta[key])
                        break
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

        message = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        return {
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
            }],
            "model": model_reported or model,
            "usage": usage,
        }

    async def chat_stream(self, model: str, messages: list[dict], **kwargs):
        """Streaming chat completion, yielding raw SSE lines for passthrough.

        aiohttp rather than httpx for the same reason `ollama_client` uses it:
        it drops the connection when the coroutine is cancelled, so a client
        that disconnects mid-generation does not leave the socket — or, here,
        the pool's single inference slot — pinned.
        """
        import aiohttp
        payload = {"model": model, "messages": messages, "stream": True, **kwargs}
        session = aiohttp.ClientSession()
        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=config.EXO_GENERATE_TIMEOUT, sock_connect=10
                ),
            ) as resp:
                resp.raise_for_status()
                async for raw in resp.content:
                    line = raw.decode(errors="replace").rstrip("\r\n")
                    if line:
                        yield line
        finally:
            await session.close()
