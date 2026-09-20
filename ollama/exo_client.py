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

import asyncio
import json
import logging
import time
import httpx
from dataclasses import dataclass, field
from typing import Optional

from . import config

log = logging.getLogger("exo-client")

EXO_BASE = config.EXO_BASE


class ExoRequestFailed(Exception):
    """The pool accepted a request and then failed it.

    Exists so `server` has one exception to catch for a failed generation. The
    two code paths here use different HTTP libraries for good reasons — sync
    httpx for short state reads, aiohttp for generation because it drops the
    socket on cancellation — and they raise unrelated exception hierarchies.
    Leaving that to the caller meant the `except httpx.*` clauses in the chat
    route silently did not cover the aiohttp path, so an error from the pool
    surfaced as a bare gateway 500: the one outcome this route is specified
    never to produce.
    """

    def __init__(self, detail: str, *, status: Optional[int] = None,
                 body: Optional[str] = None):
        super().__init__(detail)
        self.detail = detail
        self.status = status
        self.body = body


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


# exo's runner lifecycle, from `src/exo/shared/types/worker/runners.py`
# (exo 0.3.70), confirmed by the engine's author rather than inferred from a
# reading of `/state`. Runner state is variant-tagged like the instance:
# `{"RunnerReady": {}}`, `{"RunnerLoading": {...}}`.
#
#   RunnerIdle          pre-load
#   RunnerConnecting    pre-load, ring dialling
#   RunnerConnected     pre-load, ring formed, no weights
#   RunnerLoading       loading weights (carries layers_loaded / total_layers)
#   RunnerLoaded        weights in, not warm
#   RunnerWarmingUp     warming
#   RunnerReady         loaded, warm, idle      <- serviceable
#   RunnerRunning       mid-generation          <- serviceable
#   RunnerShuttingDown  going away
#   RunnerShutdown      gone
#   RunnerFailed        carries error_message and diagnostics
#
# Only `Ready` and `Running` can take a request. `Ready -> Running -> Ready` is
# the runtime toggle and `Loading -> Loaded -> WarmingUp -> Ready` the startup
# progression, so the same name means "idle" in one reading and the pool is
# mid-generation in another — exo's own `BaseRunnerStatus.is_running()` is
# exactly `isinstance(self, RunnerRunning)`.
#
# Getting this set wrong is not a small error in either direction, and it has
# been wrong both ways. The first version accepted only `RunnerRunning` —
# the state the pool happened to be in the first time it was read — so when the
# pool was re-placed with both runners `RunnerReady`, this gateway declined a
# pool that provably answered in 8.8s. Accepting too few refuses a working
# pool; accepting too many routes a generation into a shard that is failed or
# on its way out, which is worse.
#
# One important refinement, from `worker/runner/runner.py` on the wafer node:
# exo's own dispatch accepts a `TextGeneration` **only** under
# `isinstance(self.current_status, RunnerReady)`, and its fall-through is
# `case _: raise ValueError(...outside of state machine...)`. So "serviceable"
# here does NOT mean "will accept a generation this instant" — under
# `RunnerRunning` it would not. It means "this instance is placed and
# functional", and what keeps us from dispatching into a busy runner is the
# **exclusive lease**, not this set: a pool mid-generation is already leased, so
# `acquire_pool()` returns 409 and the caller gets `resource_busy` with a retry
# hint long before anything is sent to exo.
#
# That division of labour is deliberate and worth not "fixing". Dropping
# `RunnerRunning` from this set would make a busy pool report as
# *pool_unavailable* — structurally a worse answer than *resource_busy*, since
# it carries no `retry_after_s` and invites a caller to look for a
# misconfiguration instead of coming back in a moment.
_SERVICEABLE_RUNNER_STATES = frozenset({"RunnerReady", "RunnerRunning"})

# Will be serviceable shortly, is not now. Failing closed on these is right,
# and naming them means a decline during a model swap is an expected event in
# the log rather than an unknown-state warning.
_PENDING_RUNNER_STATES = frozenset({
    "RunnerIdle", "RunnerConnecting", "RunnerConnected",
    "RunnerLoading", "RunnerLoaded", "RunnerWarmingUp",
})

# Will not become serviceable without intervention.
_TERMINAL_RUNNER_STATES = frozenset({
    "RunnerShuttingDown", "RunnerShutdown", "RunnerFailed",
})


def _runner_state_name(runner_state: dict) -> Optional[str]:
    """The variant tag of a runner state, e.g. `"RunnerReady"`."""
    if isinstance(runner_state, dict):
        for key in runner_state:
            return key
    return None


def _runner_state_body(runner_state: dict) -> dict:
    """The payload inside a runner state variant, if it carries one."""
    if isinstance(runner_state, dict):
        for value in runner_state.values():
            return value if isinstance(value, dict) else {}
    return {}


def describe_runner_state(runner_state: dict) -> str:
    """A runner's state as something worth putting in front of a person.

    `RunnerLoading` carries `layers_loaded` / `total_layers`, which turns the
    30-second-to-10-minute model swap from an opaque refusal into a progress
    report, and `RunnerFailed` carries `error_message`, so a pool that dies at
    3am says why in this gateway's log instead of only in exo's.
    """
    name = _runner_state_name(runner_state)
    if name is None:
        return "unknown"
    body = _runner_state_body(runner_state)
    if name == "RunnerLoading":
        loaded, total = body.get("layers_loaded"), body.get("total_layers")
        if loaded is not None and total:
            return f"{name} ({loaded}/{total} layers)"
    if name == "RunnerFailed":
        why = body.get("error_message") or body.get("error")
        if why:
            return f"{name}: {str(why)[:300]}"
    return name


def _runner_is_serviceable(runner_state: dict) -> bool:
    """Can this runner take a request right now?

    A shutting-down or failed runner still appears in `/state` and still holds
    its shard assignment, so treating a model as servable because it is merely
    *mentioned* would route a generation into a shard that cannot run it.
    """
    name = _runner_state_name(runner_state)
    if name in _SERVICEABLE_RUNNER_STATES:
        return True
    if name is not None and name not in (
        _PENDING_RUNNER_STATES | _TERMINAL_RUNNER_STATES
    ):
        log.warning(
            "exo runner state %r is not in this client's known set (exo 0.3.70 "
            "has eleven); treating the shard as not serviceable. If it means the "
            "runner can serve, add it to _SERVICEABLE_RUNNER_STATES — until then "
            "the pool reads as unavailable.",
            name,
        )
    return False


@dataclass
class ExoClient:
    base_url: str = EXO_BASE
    _http: httpx.Client = field(default=None, repr=False)
    # (monotonic_timestamp, status) for pool_status_cached. Status reads only.
    _status_cache: Optional[tuple] = field(default=None, repr=False)

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

    def pool_status_cached(self, ttl: Optional[float] = None) -> dict:
        """`pool_status()`, memoised briefly. For status reads, never for serving.

        `/state` is a few hundred KB on this pool and `GET /models` is polled,
        so an uncached status view re-fetches the whole document per poll. A few
        seconds of staleness is the right trade for a readiness display.

        Deliberately NOT used on the serving path: `resolve_exo_tier()` calls
        `pool_status()` directly, because routing a generation at a pool whose
        resident model changed seconds ago is exactly the mistake this client
        exists to avoid.
        """
        ttl = config.EXO_STATUS_CACHE_S if ttl is None else ttl
        now = time.monotonic()
        cached = self._status_cache
        if cached is not None and (now - cached[0]) < ttl:
            return cached[1]
        status = self.pool_status()
        self._status_cache = (now, status)
        return status

    def pool_status(self) -> dict:
        """Everything the gateway needs about the pool, from one `/state` read.

        One read rather than one per question: `/state` is a few hundred KB on
        this pool, and `resident_model()` plus a separate readiness probe would
        fetch it twice per request and could disagree between the two.

        Returns `resident_model` (the model that can serve a request now, or
        None) alongside a per-runner description of whatever instance was
        examined — so a decline can say *why* and, mid-swap, *how far through*.
        """
        try:
            state = self.state()
        except Exception as e:
            raise ExoUnavailable(f"exo pool at {self.base_url} unreachable: {e}")

        runners = state.get("runners") or {}
        instances = []
        for entry in (state.get("instances") or {}).values():
            model_id = _instance_model(entry)
            if not model_id:
                continue
            runner_ids = _instance_runners(entry)
            described = {
                rid[:8]: describe_runner_state(runners.get(rid) or {})
                for rid in runner_ids
            }
            serviceable = bool(runner_ids) and all(
                _runner_is_serviceable(runners.get(rid) or {}) for rid in runner_ids
            )
            # `handle_generation_tasks` sets RunnerRunning on entry and restores
            # RunnerReady only once `active_tasks` drains, so a Running runner
            # means a generation is genuinely in flight — readable here without
            # issuing one. Correlate it with lease state and it is also the
            # wedge signature that cost an hour on 2026-09-20: Running with no
            # active lease on the pool is a generation nobody is reading, which
            # is the shape a client dying mid-request leaves behind. This client
            # deliberately reports the signal and does not diagnose it — the
            # registry half of that correlation is the manager's to know.
            busy = any(
                _runner_state_name(runners.get(rid) or {}) == "RunnerRunning"
                for rid in runner_ids
            )
            instances.append({
                "model": model_id,
                "serviceable": serviceable,
                "busy": busy,
                "runners": described,
            })

        live = next((i for i in instances if i["serviceable"]), None)
        if live is None and instances:
            log.info(
                "no serviceable exo instance: %s",
                {i["model"]: i["runners"] for i in instances},
            )
        return {
            "resident_model": live["model"] if live else None,
            "ready": live is not None,
            "busy": bool(live and live["busy"]),
            "instances": instances,
        }

    def resident_model(self) -> Optional[str]:
        """The model the pool is holding and ready to serve, or None.

        "Ready" means every runner backing the instance reports a state that can
        take a request (`RunnerReady` or `RunnerRunning`). During a model swap
        the old instance's runners go `RunnerShuttingDown` while the new one's
        climb `Loading -> Loaded -> WarmingUp`, and for that window there is
        genuinely nothing resident — which is the honest answer to give a
        caller, rather than naming a model that cannot serve.
        """
        return self.pool_status()["resident_model"]

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
        """exo's own non-streaming endpoint. Works; the gateway still prefers
        `chat_collect()`, and the reason is no longer the one first recorded here.

        **Retraction.** This docstring used to say `stream: false` returns 200
        headers and then never a body, "measured, two runs, 240s each,
        size_download=0". That measurement was real and the conclusion drawn
        from it was wrong: both probes hit a pool that was already occupied —
        the first while an 11-minute generation was still running, the second
        after a killed client had wedged the slot. A pool with no free slot
        accepts the request and emits keep-alives, which is what was observed.
        Re-measured 2026-09-20 against a healthy re-placed pool: `stream: false`
        returns a complete body with usage in ~17s.

        So this is a working endpoint, and the note against reaching for it was
        an artefact of measuring a busy resource. `chat_collect()` remains the
        gateway's path for reasons that do not depend on the retracted claim:
        it is cancellable, so a client that disconnects actually closes the
        socket to exo instead of leaving a generation running with the pool
        leased behind it, and it pins no worker thread for a generation that can
        run for minutes. Both were measured independently.
        """
        payload = {"model": model, "messages": messages, "stream": False, **kwargs}
        r = self._http.post(
            "/v1/chat/completions", json=payload, timeout=self._generate_timeout
        )
        r.raise_for_status()
        return r.json()

    async def chat_collect(self, model: str, messages: list[dict], **kwargs) -> dict:
        """A non-streaming answer, assembled from the streaming endpoint.

        Gives a caller the plain OpenAI response shape they asked for. Not
        because exo's `stream: false` is broken — see the retraction in `chat()`;
        it works — but for two properties that matter on an exclusive resource
        and were measured on their own:

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
                if resp.status >= 400:
                    # Read the body before raising: exo puts the reason in it,
                    # and once the context manager exits it is gone.
                    body = (await resp.text())[:2000]
                    raise ExoRequestFailed(
                        f"the exo pool returned {resp.status}",
                        status=resp.status, body=body,
                    )
                async for raw in resp.content:
                    line = raw.decode(errors="replace").rstrip("\r\n")
                    if line:
                        yield line
        except aiohttp.ClientError as e:
            raise ExoRequestFailed(f"exo transport error: {e}")
        except asyncio.TimeoutError:
            raise ExoRequestFailed(
                f"the exo pool produced no answer within "
                f"{config.EXO_GENERATE_TIMEOUT}s"
            )
        finally:
            await session.close()
