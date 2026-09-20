"""Tests for the pool's lease verdict and resident-model reading.

The lease verdict is the load-bearing half of Backend.EXO: on an exclusive
resource, mistaking "queued" for "granted" puts two consumers inside one
inference instance. These cases are written against the shapes the live
registry actually returns (captured 2026-09-20), not against the shapes the
API index documents — the two differ, and that difference is the point.

Run: cd ollama && python test_exo_lease.py
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

# The package uses relative imports, so run this file as a script with the
# repo root on sys.path rather than from inside the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama.manager import ModelManager, Backend            # noqa: E402
from ollama.exo_client import ExoClient, ExoUnavailable     # noqa: E402
from ollama import capacity, config                         # noqa: E402
from ollama.samcloud import SamcloudClient                  # noqa: E402


def mgr() -> "ModelManager":
    return ModelManager(sc=MagicMock())


FAILS = []


def check(name, got, want):
    if got != want:
        FAILS.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}")


# --- the verdict ----------------------------------------------------------

def test_grant_200():
    """The live registry grants with a plain 200, not the documented 201."""
    m = mgr()
    o = m._lease_outcome({
        "status_code": 200,
        "id": "lease_abc", "resource_id": "claude-services-slice/exo-pool",
        "service_id": "claude-services-slice/model-service",
        "memory_mb": None, "exclusive": 1, "status": "active",
        "queue_position": None,
        "granted_at": "2026-09-20T02:37:30Z",
        "expires_at": "2026-09-20T03:37:30Z",
    })
    check("grant/200 granted", o.granted, True)
    check("grant/200 state", o.state, "active")
    check("grant/200 id", o.lease_id, "lease_abc")


def test_grant_201_documented():
    """A 201 must also read as granted, in case the registry starts sending one."""
    m = mgr()
    o = m._lease_outcome({"status_code": 201, "id": "lease_x", "status": "active"})
    check("grant/201 granted", o.granted, True)


def test_queued_200_is_not_granted():
    """THE bug. A queued lease has an id, and 200, and must still not proceed."""
    m = mgr()
    o = m._lease_outcome({
        "status_code": 200,
        "lease_id": "lease_q", "status": "queued", "queue_position": 2,
        "available_mb": 0, "requested_mb": 1,
    })
    check("queued/200 granted", o.granted, False)
    check("queued/200 state", o.state, "queued")
    check("queued/200 position", o.queue_position, 2)
    check("queued/200 keeps id for release", o.lease_id, "lease_q")


def test_queued_202_documented():
    m = mgr()
    o = m._lease_outcome({"status_code": 202, "lease_id": "lease_q2",
                          "status": "queued", "queue_position": 1})
    check("queued/202 granted", o.granted, False)


def test_queue_position_without_status():
    """A queue_position alone is enough to refuse, even if status is missing."""
    m = mgr()
    o = m._lease_outcome({"status_code": 200, "lease_id": "l", "queue_position": 3})
    check("queue_position alone refuses", o.granted, False)
    check("queue_position alone state", o.state, "queued")


def test_conflict_409_live_shape():
    """The exact body the live registry returned while admin held the lease."""
    m = mgr()
    o = m._lease_outcome({
        "status_code": 409,
        "detail": {
            "detail": "Exclusive lease held by None",
            "held_by": None,
            "expires_at": "2026-09-20T03:37:30.771635+00:00",
        },
    })
    check("409 granted", o.granted, False)
    check("409 state", o.state, "conflict")
    check("409 no lease id", o.lease_id, None)
    check("409 expires_at surfaced", o.expires_at,
          "2026-09-20T03:37:30.771635+00:00")


def test_conflict_409_with_holder():
    m = mgr()
    o = m._lease_outcome({
        "status_code": 409,
        "detail": {"detail": "Exclusive lease held by wafer-services/model-service",
                   "held_by": "wafer-services/model-service",
                   "expires_at": "2026-09-20T03:37:30Z"},
    })
    check("409 holder named", o.held_by, "wafer-services/model-service")


def test_conflict_409_string_detail():
    """The other 409 in the handler is a bare string, not a dict."""
    m = mgr()
    o = m._lease_outcome({
        "status_code": 409,
        "detail": "Cannot grant exclusive lease — active leases exist",
    })
    check("409 string granted", o.granted, False)
    check("409 string state", o.state, "conflict")


def test_server_error():
    m = mgr()
    o = m._lease_outcome({"status_code": 500, "detail": "boom"})
    check("500 granted", o.granted, False)
    check("500 state", o.state, "error")


def test_2xx_without_id():
    """A 200 carrying no identifier is not something to proceed on."""
    m = mgr()
    o = m._lease_outcome({"status_code": 200, "status": "active"})
    check("200 no id granted", o.granted, False)
    check("200 no id state", o.state, "error")


def test_retry_after():
    from datetime import datetime, timedelta, timezone
    m = mgr()
    soon = (datetime.now(timezone.utc) + timedelta(seconds=120)).isoformat()
    got = m._retry_after_s(soon)
    check("retry_after in range", 110 <= got <= 121, True)
    check("retry_after of None", m._retry_after_s(None), None)
    check("retry_after unparseable", m._retry_after_s("not-a-date"), None)
    past = (datetime.now(timezone.utc) - timedelta(seconds=99)).isoformat()
    check("retry_after floors at 1", m._retry_after_s(past), 1)
    check("retry_after handles Z suffix",
          m._retry_after_s("2999-01-01T00:00:00Z") > 0, True)


def test_acquire_raises_poolbusy_on_conflict():
    """The whole point: a conflict must raise, not return a falsy lease id."""
    m = mgr()
    m.sc.request_lease.return_value = {
        "status_code": 409,
        "detail": {"detail": "Exclusive lease held by someone",
                   "held_by": "someone",
                   "expires_at": "2999-01-01T00:00:00Z"},
    }
    try:
        m.acquire_pool("test")
        FAILS.append("acquire_pool on 409 did not raise")
        print("  FAIL acquire_pool on 409 did not raise")
    except capacity.PoolBusy as e:
        body = e.as_dict()
        check("PoolBusy error key", body["error"], "resource_busy")
        check("PoolBusy has retry_after", body["retry_after_s"] > 0, True)
        check("PoolBusy queue_position present", "queue_position" in body, True)
        check("PoolBusy names resource", body["resource_id"],
              config.EXO_RESOURCE_ID)


def test_acquire_sends_no_memory_and_is_exclusive():
    """memory_mb must not be sent: any byte count queues forever on exo-pool."""
    m = mgr()
    m.sc.request_lease.return_value = {
        "status_code": 200, "id": "lease_ok", "status": "active",
        "expires_at": "2999-01-01T00:00:00Z",
    }
    lease_id = m.acquire_pool("test")
    check("acquire returns id", lease_id, "lease_ok")
    kwargs = m.sc.request_lease.call_args.kwargs
    check("memory_mb is None", kwargs["memory_mb"], None)
    check("exclusive is True", kwargs["exclusive"], True)
    check("resource is the pool", kwargs["resource_id"], config.EXO_RESOURCE_ID)
    check("lease tracked for shutdown", lease_id in m._pool_leases, True)


def test_acquire_queued_releases_and_refuses():
    m = mgr()
    m.sc.request_lease.return_value = {
        "status_code": 200, "lease_id": "lease_q", "status": "queued",
        "queue_position": 1,
    }
    try:
        m.acquire_pool("test")
        FAILS.append("acquire_pool on queued did not raise")
        print("  FAIL acquire_pool on queued did not raise")
    except capacity.PoolBusy:
        check("queued row was released", m.sc.release_lease.called, True)
        check("queued lease not tracked", len(m._pool_leases), 0)


def test_unreachable_registry_refuses():
    """A registry we cannot reach is not a free pool."""
    m = mgr()
    m.sc.request_lease.side_effect = RuntimeError("connection refused")
    try:
        m.acquire_pool("test")
        FAILS.append("acquire_pool with dead registry did not raise")
        print("  FAIL acquire_pool with dead registry did not raise")
    except capacity.PoolBusy as e:
        check("dead registry refuses", "registry unreachable" in str(e), True)


def test_pool_lease_releases_on_exception():
    m = mgr()
    m.sc.request_lease.return_value = {
        "status_code": 200, "id": "lease_ctx", "status": "active",
        "expires_at": "2999-01-01T00:00:00Z",
    }
    try:
        with m.pool_lease("test"):
            raise ValueError("generation blew up")
    except ValueError:
        pass
    check("released after exception", m.sc.release_lease.called, True)
    check("not tracked after release", len(m._pool_leases), 0)


def test_shutdown_sweeps_held_lease():
    """A lease acquired directly (the streaming path) is still swept."""
    m = mgr()
    m.sc.request_lease.return_value = {
        "status_code": 200, "id": "lease_orphan", "status": "active",
        "expires_at": "2999-01-01T00:00:00Z",
    }
    m.acquire_pool("stream")
    m.shutdown()
    check("shutdown released the lease", m.sc.release_lease.called, True)
    check("nothing left held", len(m._pool_leases), 0)


def test_shared_lease_queued_is_not_recorded():
    """The same fix on the shared path: a queued lease is released, not kept."""
    m = mgr()
    m.sc.request_lease.return_value = {
        "status_code": 200, "lease_id": "lease_q", "status": "queued",
        "queue_position": 1,
    }
    got = m._request_lease("some-model", 5000)
    check("shared queued -> no lease id", got, None)
    check("shared queued row released", m.sc.release_lease.called, True)


def test_shared_lease_granted():
    m = mgr()
    m.sc.request_lease.return_value = {
        "status_code": 200, "id": "lease_s", "status": "active",
        "expires_at": "2999-01-01T00:00:00Z",
    }
    check("shared granted", m._request_lease("some-model", 5000), "lease_s")


# --- resident model reading ----------------------------------------------

REAL_STATE = {
    "instances": {
        "607f4df1": {"MlxRingInstance": {
            "instanceId": "607f4df1",
            "shardAssignments": {
                "modelId": "mlx-community/GLM-4.7-Flash-6bit",
                "runnerToShard": {"00c6ed85": {}, "3c7551f4": {}},
            },
        }},
    },
    "runners": {
        "e282ff2d": {"RunnerShuttingDown": {}},
        "0bdd828b": {"RunnerShuttingDown": {}},
        "00c6ed85": {"RunnerRunning": {}},
        "3c7551f4": {"RunnerRunning": {}},
    },
}


def exo_with(state):
    c = ExoClient()
    c.state = lambda: state
    return c


def test_resident_model_real_state():
    check("resident from real /state", exo_with(REAL_STATE).resident_model(),
          "mlx-community/GLM-4.7-Flash-6bit")


def test_resident_model_with_ready_runners():
    """RunnerReady is serviceable. Regression: accepting only RunnerRunning made
    this gateway decline a re-placed pool that answered in 8.8s."""
    state = json.loads(json.dumps(REAL_STATE))
    state["runners"]["00c6ed85"] = {"RunnerReady": {}}
    state["runners"]["3c7551f4"] = {"RunnerReady": {}}
    check("ready runners are serviceable",
          exo_with(state).resident_model(), "mlx-community/GLM-4.7-Flash-6bit")


def test_resident_model_mixed_ready_and_running():
    state = json.loads(json.dumps(REAL_STATE))
    state["runners"]["00c6ed85"] = {"RunnerReady": {}}
    check("ready + running is serviceable",
          exo_with(state).resident_model(), "mlx-community/GLM-4.7-Flash-6bit")


def test_resident_none_when_runner_failed():
    """The state the wedged pool was actually in."""
    state = json.loads(json.dumps(REAL_STATE))
    state["runners"]["00c6ed85"] = {"RunnerFailed": {}}
    check("failed runner is not serviceable",
          exo_with(state).resident_model(), None)


def test_resident_none_on_unknown_runner_state():
    """An unrecognised state is refused, not guessed at."""
    state = json.loads(json.dumps(REAL_STATE))
    state["runners"]["00c6ed85"] = {"RunnerSomethingNew": {}}
    check("unknown state refused",
          exo_with(state).resident_model(), None)


def test_resident_picks_serviceable_instance():
    """A failed instance must not mask a good one for the same model."""
    state = json.loads(json.dumps(REAL_STATE))
    state["instances"]["dead"] = {"MlxRingInstance": {
        "instanceId": "dead",
        "shardAssignments": {"modelId": "mlx-community/Old-Model",
                             "runnerToShard": {"e282ff2d": {}}},
    }}
    check("serviceable instance still found",
          exo_with(state).resident_model(), "mlx-community/GLM-4.7-Flash-6bit")


def test_resident_none_when_runner_shutting_down():
    """Mid-swap: the instance is still listed but its runner is going away."""
    state = json.loads(json.dumps(REAL_STATE))
    state["runners"]["00c6ed85"] = {"RunnerShuttingDown": {}}
    check("mid-swap reports nothing resident",
          exo_with(state).resident_model(), None)


def test_resident_none_when_runner_missing():
    state = json.loads(json.dumps(REAL_STATE))
    del state["runners"]["3c7551f4"]
    check("missing runner reports nothing resident",
          exo_with(state).resident_model(), None)


def test_resident_none_on_empty_pool():
    check("empty pool", exo_with({"instances": {}, "runners": {}}).resident_model(),
          None)


def test_resident_variant_agnostic():
    """The runner variant name is exo's internal enum — don't key on it."""
    state = json.loads(json.dumps(REAL_STATE))
    state["instances"]["607f4df1"] = {
        "MlxTensorInstance": state["instances"]["607f4df1"]["MlxRingInstance"]
    }
    check("unknown instance variant still read",
          exo_with(state).resident_model(), "mlx-community/GLM-4.7-Flash-6bit")


def test_resident_unreachable_raises():
    c = ExoClient()
    def boom(): raise RuntimeError("connection refused")
    c.state = boom
    try:
        c.resident_model()
        FAILS.append("unreachable pool did not raise ExoUnavailable")
    except ExoUnavailable:
        check("unreachable raises ExoUnavailable", True, True)


# --- request defaults -----------------------------------------------------

def test_max_tokens_cap():
    m = mgr()
    d = m.exo_request_defaults("mlx-community/GLM-4.7-Flash-6bit")
    check("GLM gets a max_tokens cap", d["max_tokens"], config.EXO_MAX_TOKENS)
    d2 = m.exo_request_defaults("mlx-community/Some-Other-Model")
    check("unknown model still capped", d2["max_tokens"], config.EXO_MAX_TOKENS)
    check("defaults are a fresh dict",
          m.exo_request_defaults("x") is not m.exo_request_defaults("x"), True)


# --- non-streaming answers assembled from the stream ----------------------

def _collect(lines):
    """Run chat_collect against a canned SSE stream."""
    import asyncio

    c = ExoClient()

    async def fake_stream(model, messages, **kwargs):
        for ln in lines:
            yield ln

    c.chat_stream = fake_stream
    return asyncio.run(c.chat_collect("m", [{"role": "user", "content": "hi"}]))


def test_collect_aggregates_content():
    out = _collect([
        ': keep-alive',
        '',
        'data: {"choices":[{"delta":{"content":"3"},"finish_reason":null}],"model":"glm"}',
        'data: {"choices":[{"delta":{"content":"91"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}',
        'data: [DONE]',
    ])
    check("content joined in order", out["choices"][0]["message"]["content"], "391")
    check("finish_reason carried", out["choices"][0]["finish_reason"], "stop")
    check("usage carried", out["usage"]["total_tokens"], 15)
    check("model carried", out["model"], "glm")
    check("role set", out["choices"][0]["message"]["role"], "assistant")


def test_collect_ignores_keepalive_and_junk():
    """The pool emits `: keep-alive` comments; they are not JSON and not data."""
    out = _collect([
        ': keep-alive', ': keep-alive', '',
        'data: not-json-at-all',
        'data: {"choices":[{"delta":{"content":"ok"}}]}',
        'data: [DONE]',
    ])
    check("keepalives and junk skipped",
          out["choices"][0]["message"]["content"], "ok")


def test_collect_keeps_reasoning_separate():
    """A thinking model's side channel is kept, not silently dropped."""
    out = _collect([
        'data: {"choices":[{"delta":{"reasoning_content":"17*23..."}}]}',
        'data: {"choices":[{"delta":{"content":"391"},"finish_reason":"stop"}]}',
        'data: [DONE]',
    ])
    msg = out["choices"][0]["message"]
    check("content is the answer", msg["content"], "391")
    check("reasoning kept aside", msg["reasoning_content"], "17*23...")


def test_collect_empty_stream_is_empty_not_crash():
    """The pool sending only keep-alives must yield an empty answer, not raise."""
    out = _collect([': keep-alive', ': keep-alive', ''])
    check("empty content", out["choices"][0]["message"]["content"], "")
    check("still well-formed", out["choices"][0]["finish_reason"], "stop")


def test_collect_tool_calls():
    out = _collect([
        'data: {"choices":[{"delta":{"tool_calls":[{"id":"1","function":'
        '{"name":"f","arguments":"{}"}}]}}]}',
        'data: [DONE]',
    ])
    msg = out["choices"][0]["message"]
    check("tool_calls collected", len(msg.get("tool_calls", [])), 1)
    check("finish_reason inferred", out["choices"][0]["finish_reason"], "tool_calls")


def test_collect_accepts_full_message_shape():
    """If exo ever sends `message` instead of `delta`, read it anyway."""
    out = _collect([
        'data: {"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}]}',
        'data: [DONE]',
    ])
    check("message shape read", out["choices"][0]["message"]["content"], "hello")


def test_collect_propagates_pool_failure():
    """A pool that fails the generation must raise, not return an empty answer.

    An empty-but-successful answer would be indistinguishable from a model that
    genuinely had nothing to say, and would be reported to the caller as a 200.
    """
    import asyncio
    from ollama.exo_client import ExoRequestFailed

    c = ExoClient()

    async def failing_stream(model, messages, **kwargs):
        raise ExoRequestFailed("the exo pool returned 500", status=500, body="boom")
        yield  # pragma: no cover - makes this an async generator

    c.chat_stream = failing_stream
    try:
        asyncio.run(c.chat_collect("m", [{"role": "user", "content": "hi"}]))
        FAILS.append("chat_collect swallowed a pool failure")
        print("  FAIL chat_collect swallowed a pool failure")
    except ExoRequestFailed as e:
        check("pool failure propagates", e.status, 500)
        check("pool body kept", e.body, "boom")


def test_collect_cancellation_propagates():
    """Cancellation must not be swallowed — it is how the lease gets released."""
    import asyncio

    c = ExoClient()

    async def hanging_stream(model, messages, **kwargs):
        await asyncio.sleep(3600)
        yield "data: {}"  # pragma: no cover

    c.chat_stream = hanging_stream

    async def run():
        task = asyncio.ensure_future(
            c.chat_collect("m", [{"role": "user", "content": "hi"}])
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
            return "completed"
        except asyncio.CancelledError:
            return "cancelled"

    check("cancellation propagates", asyncio.run(run()), "cancelled")


def test_exo_model_is_not_unloaded():
    """unload() must never stop the pool, even with force=True."""
    import time as _t
    from ollama.manager import ManagedModel

    m = mgr()
    m.models["think"] = ManagedModel(
        name="mlx-community/GLM-4.7-Flash-6bit", backend=Backend.EXO,
        memory_mb=0, lease_id=None, port=0, loaded_at=_t.time(),
        last_used=_t.time(), managed=False, tier="think",
    )
    out = m.unload("think", force=True)
    check("pool is deregistered, not stopped", out["status"], "deregistered")
    check("tier removed from registry", "think" in m.models, False)
    check("no lease was released", m.sc.release_lease.called, False)


def test_cooldown_never_touches_the_pool():
    """The idle loop must not try to reclaim a pool it does not own."""
    import time as _t
    from ollama.manager import ManagedModel

    m = mgr()
    m.models["think"] = ManagedModel(
        name="mlx-community/GLM-4.7-Flash-6bit", backend=Backend.EXO,
        memory_mb=0, lease_id=None, port=0,
        loaded_at=_t.time() - 99999, last_used=_t.time() - 99999,
        managed=False, tier="think",
    )
    m.check_cooldowns()
    check("pool tier survives cooldown", "think" in m.models, True)


def test_claim_leases_skips_the_pool():
    """Claiming a residency lease on an exclusive pool would close it at boot."""
    import time as _t
    from ollama.manager import ManagedModel

    m = mgr()
    m.models["think"] = ManagedModel(
        name="mlx-community/GLM-4.7-Flash-6bit", backend=Backend.EXO,
        memory_mb=0, lease_id=None, port=0, loaded_at=_t.time(),
        last_used=_t.time(), managed=False, tier="think",
    )
    m.claim_leases()
    check("no lease requested for the pool", m.sc.request_lease.called, False)


# --- runner states, against exo 0.3.70's actual enum --------------------------

def test_all_eleven_states_classified():
    """Every state exo defines must be classified, so none trips the warning.

    The point of this test: the previous version of the unserviceable set was
    written from observation and contained `RunnerStarting`, which does not
    exist, while omitting six states that do. Normal startup would have logged
    "not in this client's known set" for each one.
    """
    from ollama.exo_client import (
        _SERVICEABLE_RUNNER_STATES, _PENDING_RUNNER_STATES,
        _TERMINAL_RUNNER_STATES,
    )
    exo_states = {
        "RunnerIdle", "RunnerConnecting", "RunnerConnected", "RunnerLoading",
        "RunnerLoaded", "RunnerWarmingUp", "RunnerReady", "RunnerRunning",
        "RunnerShuttingDown", "RunnerShutdown", "RunnerFailed",
    }
    known = (_SERVICEABLE_RUNNER_STATES | _PENDING_RUNNER_STATES
             | _TERMINAL_RUNNER_STATES)
    check("all 11 exo states classified", sorted(exo_states - known), [])
    check("nothing invented", sorted(known - exo_states), [])
    check("only Ready and Running serve",
          sorted(_SERVICEABLE_RUNNER_STATES), ["RunnerReady", "RunnerRunning"])


def test_pending_states_are_not_serviceable():
    """Loading/Loaded/WarmingUp will serve shortly, but not now."""
    for st in ("RunnerIdle", "RunnerConnecting", "RunnerConnected",
               "RunnerLoading", "RunnerLoaded", "RunnerWarmingUp",
               "RunnerShutdown"):
        state = json.loads(json.dumps(REAL_STATE))
        state["runners"]["00c6ed85"] = {st: {}}
        check(f"{st} not serviceable", exo_with(state).resident_model(), None)


def test_loading_reports_layer_progress():
    """A swap in flight should say how far through, not just 'not ready'."""
    from ollama.exo_client import describe_runner_state
    got = describe_runner_state(
        {"RunnerLoading": {"layers_loaded": 23, "total_layers": 47}}
    )
    check("layer progress surfaced", got, "RunnerLoading (23/47 layers)")
    check("loading without counts still names the state",
          describe_runner_state({"RunnerLoading": {}}), "RunnerLoading")


def test_failed_reports_why():
    from ollama.exo_client import describe_runner_state
    got = describe_runner_state(
        {"RunnerFailed": {"error_message": "metal OOM on shard 1"}}
    )
    check("failure reason surfaced", got, "RunnerFailed: metal OOM on shard 1")
    check("failure without a message still names the state",
          describe_runner_state({"RunnerFailed": {}}), "RunnerFailed")


def test_describe_plain_states():
    from ollama.exo_client import describe_runner_state
    check("ready described", describe_runner_state({"RunnerReady": {}}),
          "RunnerReady")
    check("empty described", describe_runner_state({}), "unknown")


def test_pool_status_reports_progress_for_a_swap():
    """The whole point: mid-swap, the gateway can quote layer progress."""
    state = {
        "instances": {"new": {"MlxRingInstance": {
            "instanceId": "new",
            "shardAssignments": {"modelId": "mlx-community/GLM-5",
                                 "runnerToShard": {"aaaaaaaa": {}, "bbbbbbbb": {}}},
        }}},
        "runners": {
            "aaaaaaaa": {"RunnerLoading": {"layers_loaded": 12,
                                           "total_layers": 47}},
            "bbbbbbbb": {"RunnerReady": {}},
        },
    }
    st = exo_with(state).pool_status()
    check("not ready mid-swap", st["ready"], False)
    check("no resident model mid-swap", st["resident_model"], None)
    descriptions = list(st["instances"][0]["runners"].values())
    check("progress is quotable", "RunnerLoading (12/47 layers)" in descriptions,
          True)


def test_pool_status_reports_busy():
    """Running means a generation is in flight, readable without issuing one."""
    state = json.loads(json.dumps(REAL_STATE))
    state["runners"]["00c6ed85"] = {"RunnerRunning": {}}
    state["runners"]["3c7551f4"] = {"RunnerRunning": {}}
    st = exo_with(state).pool_status()
    check("busy while generating", st["busy"], True)
    check("still serviceable while busy", st["ready"], True)
    check("model still named while busy", st["resident_model"],
          "mlx-community/GLM-4.7-Flash-6bit")


def test_pool_status_idle_is_not_busy():
    state = json.loads(json.dumps(REAL_STATE))
    state["runners"]["00c6ed85"] = {"RunnerReady": {}}
    state["runners"]["3c7551f4"] = {"RunnerReady": {}}
    st = exo_with(state).pool_status()
    check("idle is not busy", st["busy"], False)
    check("idle is ready", st["ready"], True)


def test_busy_pool_still_resolves_so_the_lease_can_decline():
    """A busy pool must reach the lease check, not short-circuit as unavailable.

    exo only dispatches under RunnerReady, but the exclusive lease is what stops
    us dispatching into a busy runner — so a mid-generation pool has to resolve
    far enough for acquire_pool() to return the 409 that becomes a
    `resource_busy` 503 with a retry hint. Declining it as `pool_unavailable`
    here would lose the retry hint entirely.
    """
    state = json.loads(json.dumps(REAL_STATE))
    state["runners"]["00c6ed85"] = {"RunnerRunning": {}}
    state["runners"]["3c7551f4"] = {"RunnerRunning": {}}
    check("busy pool still names a resident model",
          exo_with(state).resident_model(), "mlx-community/GLM-4.7-Flash-6bit")


def test_pool_status_one_read():
    """resident_model() must not fetch /state twice."""
    calls = []
    c = ExoClient()
    def counting_state():
        calls.append(1)
        return REAL_STATE
    c.state = counting_state
    c.resident_model()
    check("one /state read per resolve", len(calls), 1)


def test_status_cache_reuses_within_ttl():
    calls = []
    c = ExoClient()
    def counting():
        calls.append(1); return REAL_STATE
    c.state = counting
    a = c.pool_status_cached(ttl=60)
    b = c.pool_status_cached(ttl=60)
    check("cached within ttl", len(calls), 1)
    check("same reading returned", a["resident_model"], b["resident_model"])


def test_status_cache_refetches_after_ttl():
    calls = []
    c = ExoClient()
    def counting():
        calls.append(1); return REAL_STATE
    c.state = counting
    c.pool_status_cached(ttl=0)
    c.pool_status_cached(ttl=0)
    check("ttl=0 always refetches", len(calls), 2)


def test_serving_path_never_uses_the_cache():
    """resolve_exo_tier must read fresh: a stale resident model routes wrongly."""
    calls = []
    m = mgr()
    def counting():
        calls.append(1); return REAL_STATE
    m.exo.state = counting
    m.resolve_exo_tier("think")
    m.models.pop("think", None)
    m.resolve_exo_tier("think")
    check("two resolves -> two /state reads", len(calls), 2)


# --- the serving path must not block the event loop ------------------------

def test_resolve_model_does_not_block_the_loop():
    """A slow pool must not freeze the gateway while a tier is resolved.

    The regression this pins: `_resolve_model` used to be a plain `def` calling
    sync httpx, so against a wedged pool it blocked every route, health
    reporting and the stats loop for the full 15s timeout before raising —
    while detecting exactly the condition the decline path exists to report.

    Measured by racing the resolve against a ticker on the same loop: if the
    read blocks, the ticker cannot advance.
    """
    import asyncio, time as _t, types

    import ollama.server as srv

    slow = 0.4
    class SlowExo:
        def resident_model(self):
            _t.sleep(slow)          # deliberately blocking, like sync httpx
            return "mlx-community/GLM-4.7-Flash-6bit"
        def pool_status(self):
            _t.sleep(slow)
            return {"resident_model": "mlx-community/GLM-4.7-Flash-6bit",
                    "ready": True, "busy": False, "instances": []}

    real_mgr = getattr(srv, "mgr", None)
    m = mgr()
    m.exo = SlowExo()
    srv.mgr = m
    try:
        async def run():
            ticks = 0
            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.02)
                    ticks += 1
            t = asyncio.ensure_future(ticker())
            mm = await srv._resolve_model("think")
            t.cancel()
            return ticks, mm

        ticks, mm = asyncio.run(run())
        check("resolve is a coroutine fn",
              asyncio.iscoroutinefunction(srv._resolve_model), True)
        check("tier still resolved", mm.name, "mlx-community/GLM-4.7-Flash-6bit")
        # A blocked loop yields ~0 ticks; an unblocked one gets ~slow/0.02.
        check(f"loop kept running during a {slow}s pool read (ticks={ticks})",
              ticks >= 5, True)
    finally:
        srv.mgr = real_mgr


def test_concurrent_tier_resolves_lose_no_count():
    """resolve_exo_tier runs in a thread pool now, so its counter needs a lock.

    The category rather than the instance: moving work off the event loop takes
    the loop's implicit serialisation with it, and `request_count += 1` is
    LOAD/ADD/STORE. Concurrent `think` resolves are the normal case, not a
    corner one — resolve happens before the lease, so they are exactly the pair
    where one caller wins the pool and the other is declined.

    The increment's window is amplified deliberately, and getting that right
    took two attempts worth recording. Running 320 plain increments across 8
    threads passed with the lock removed — CPython rarely switches between those
    three bytecodes unaided. Adding a sleep to the counter's getter also passed
    with the lock removed, because the sleep came *before* the read, so the
    getter still returned a fresh value and there was no window to lose. Only
    reading first and then yielding produces the race. Verified both ways:
    without the lock this reports 5 of 40; with it, 40 of 40.
    """
    import threading as _th
    import time as _t

    class SlowCounter:
        """Stands in for ManagedModel, with a wide read-modify-write window."""
        def __init__(self):
            self.backend = Backend.EXO
            self.name = "mlx-community/GLM-4.7-Flash-6bit"
            self.last_used = 0.0
            self.tier = "think"
            self._count = 0

        @property
        def request_count(self):
            # Read FIRST, then yield, so the value returned is stale by the time
            # it is added to and stored. A first attempt slept *before* reading
            # and the test passed with the lock removed — the getter was still
            # returning a fresh value, so there was no window to lose. Order
            # matters more than duration here.
            value = self._count
            _t.sleep(0.002)
            return value

        @request_count.setter
        def request_count(self, v):
            self._count = v

    m = mgr()

    class StubExo:
        def pool_status(self):
            return {"resident_model": "mlx-community/GLM-4.7-Flash-6bit",
                    "ready": True, "busy": False, "instances": []}

    m.exo = StubExo()
    slow = SlowCounter()
    m.models["think"] = slow

    threads, per_thread = 8, 5
    def worker():
        for _ in range(per_thread):
            m.resolve_exo_tier("think")

    ts = [_th.Thread(target=worker) for _ in range(threads)]
    for t in ts: t.start()
    for t in ts: t.join()

    expected = threads * per_thread
    check(f"no counts lost across {threads} threads (expected {expected})",
          slow._count, expected)


def test_status_code_is_not_shadowed_by_the_body():
    """The HTTP status must win over a body field of the same name.

    `request_lease` returns `{"status_code": ..., **body}`. Merged in that
    order a body carrying its own `status_code` silently replaces the real
    HTTP status, and a 409 conflict read as a 200 grant is precisely the
    two-consumers-in-one-pool failure this module exists to prevent. No
    registry response observed on 2026-09-20 carries the field, so this is
    a guard rather than a repair — but the cost of being wrong is the whole
    bug, and the cost of the guard is a dict ordering.
    """
    sc = SamcloudClient.__new__(SamcloudClient)
    sc._http = MagicMock()
    resp = MagicMock()
    resp.status_code = 409
    resp.json.return_value = {
        "detail": "resource is held", "status_code": 200,
        "held_by": "wafer-services/model-service",
    }
    sc._http.post.return_value = resp

    out = sc.request_lease("claude-services-slice/exo-pool", exclusive=True)
    check("HTTP 409 survives a body status_code=200", out["status_code"], 409)
    check("body fields still present", out["held_by"],
          "wafer-services/model-service")

    # And the verdict built from it must still decline.
    m = mgr()
    o = m._lease_outcome(out)
    check("shadowed conflict still declines", o.granted, False)


# --- /v1/models discovery --------------------------------------------------

def test_v1_models_lists_tier_and_never_touches_the_pool():
    """Discovery must list the tier and must not read the pool to do it.

    The pool read is the point: a wedged pool takes up to EXO_GENERATE_TIMEOUT
    to fail, and a discovery call that hangs on it turns a listing into the
    outage it is supposed to describe.
    """
    import asyncio, types
    import ollama.server as srv

    class ExplodingExo:
        def pool_status(self, *a, **k):
            raise AssertionError("/v1/models must not read the pool")
        def pool_status_cached(self, *a, **k):
            raise AssertionError("/v1/models must not read the pool")
        def resident_model(self, *a, **k):
            raise AssertionError("/v1/models must not read the pool")

    class FakeOllama:
        def list_models(self): return [{"name": "qwen3:1.7b"}, {"name": "qwen3.8:27b-mlx"}]
    class FakeLlama:
        def available_models(self): return [{"name": "qwen2.5-32b-agi-q4_k_m"}]

    real = getattr(srv, "mgr", None)
    srv.mgr = types.SimpleNamespace(
        exo=ExplodingExo(), models={}, ollama=FakeOllama(), llama=FakeLlama())
    try:
        out = asyncio.run(srv.list_models_openai())
    finally:
        srv.mgr = real

    ids = [m["id"] for m in out["data"]]
    check("openai envelope", out["object"], "list")
    check("every entry is a model object",
          all(m["object"] == "model" for m in out["data"]), True)
    check("tier listed", "think" in ids, True)
    check("tier owned_by exo",
          next(m["owned_by"] for m in out["data"] if m["id"] == "think"), "exo")
    check("ollama models listed", "qwen3:1.7b" in ids, True)
    check("gguf listed", "qwen2.5-32b-agi-q4_k_m" in ids, True)
    check("no duplicates", len(ids), len(set(ids)))


def test_v1_models_survives_a_dead_ollama():
    """One catalogue being down must not fail the whole listing."""
    import asyncio, types
    import ollama.server as srv

    class DeadOllama:
        def list_models(self): raise RuntimeError("connection refused")
    class FakeLlama:
        def available_models(self): return []

    real = getattr(srv, "mgr", None)
    srv.mgr = types.SimpleNamespace(
        exo=object(), models={}, ollama=DeadOllama(), llama=FakeLlama())
    try:
        out = asyncio.run(srv.list_models_openai())
    finally:
        srv.mgr = real
    check("still returns a list", out["object"], "list")
    check("tier still listed", "think" in [m["id"] for m in out["data"]], True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} test groups\n")
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all pool lease tests passed")
