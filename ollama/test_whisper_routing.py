#!/usr/bin/env python3
"""The gateway half of /v1/audio/transcriptions. No child, no network, no mlx.

Everything here is about the seam between a request and the whisper child: which
names resolve, which requests are refused before anything is loaded, what the
five response formats render, and what the spool has in it afterwards.

The child's own behaviour is test_whisper_child.py, which needs WHISPER_PYTHON.

    python -m ollama.test_whisper_routing
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ["AUTH_ENABLED"] = "0"          # test process only, before import
os.environ.setdefault("SC_TOKEN", "test")
# Off, so `GET /models` renders `{"enabled": false}` for the pool instead of
# reading `/state` off a live exo node. Without this the test is not offline: it
# passes or fails partly on whether the pool is up, which is not what it is for.
os.environ["EXO_ENABLED"] = "0"

SPOOL = Path(tempfile.mkdtemp(prefix="whisper-spool-test-"))
os.environ["WHISPER_SPOOL_DIR"] = str(SPOOL)
os.environ["WHISPER_MAX_UPLOAD_MB"] = "1"

from fastapi.testclient import TestClient

from . import capacity, config, server
from .manager import (
    Backend, ManagedModel, ModelManager, WHISPER_DEFAULT, WHISPER_MODELS,
    match_whisper_model,
)
from .samcloud import SamcloudClient
from .manager import TranscriberBusy
from .whisper_client import WhisperFailed, WhisperUnavailable

failures = []

# One stubbed transcription, in mlx-whisper's own shape. Floats are plain here
# on purpose: with word_timestamps the real decoder returns np.float64, and that
# it survives JSON is asserted in test_whisper_child.py where numpy exists.
RESULT = {
    "text": " Okay, shed shelf two. Three orange batteries.",
    "language": "en",
    "language_name": "english",
    "duration": 5.5,
    "transcribe_seconds": 0.8,
    "segments": [
        {
            "id": 0, "seek": 0, "start": 0.0, "end": 2.5,
            "text": " Okay, shed shelf two.",
            "tokens": [50364, 1033], "temperature": 0.0,
            "avg_logprob": -0.21, "compression_ratio": 1.1,
            "no_speech_prob": 0.01,
            "words": [
                {"word": " Okay,", "start": 0.0, "end": 0.38, "probability": 0.58},
                {"word": " shed", "start": 0.38, "end": 0.71, "probability": 0.91},
            ],
        },
        {
            "id": 1, "seek": 0, "start": 2.5, "end": 5.5,
            "text": " Three orange batteries.",
            "tokens": [50464, 2049], "temperature": 0.0,
            "avg_logprob": -0.3, "compression_ratio": 1.2,
            "no_speech_prob": 0.02,
            "words": [
                {"word": " Three", "start": 2.5, "end": 2.9, "probability": 0.99},
            ],
        },
    ],
}


def step(n, msg):
    print(f"\n{'='*60}\n  Step {n}: {msg}\n{'='*60}")


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


def _managed(name=WHISPER_DEFAULT, backend=Backend.WHISPER, in_flight=0):
    return ManagedModel(
        name=name, backend=backend, memory_mb=2600, lease_id=None,
        port=config.WHISPER_PORT, loaded_at=0.0, last_used=0.0,
        request_count=0, managed=True, in_flight=in_flight,
    )


def _no_registry(*a, **kw):
    """Stands in for any SamcloudClient call. Raises, because it should not happen.

    `_request_lease` catches everything and carries on unleased, so a load still
    works — this is here to keep the file's "no network" claim true rather than
    approximately true, and to make a new registry call in this path show up as a
    log line instead of a silent HTTPS request to the production plane.
    """
    raise RuntimeError("no registry in this test")


def _manager() -> ModelManager:
    """A manager with every backend catalogue stubbed to empty and no registry."""
    mgr = ModelManager(sc=SamcloudClient(token="test"))
    mgr.ollama.list_models = lambda: []
    mgr.ollama.list_running = lambda: []
    mgr.llama.available_models = lambda: []
    mgr.sc.request_lease = _no_registry
    mgr.sc.release_lease = _no_registry
    return mgr


def _client(mgr, calls=None):
    """A TestClient whose whisper load is stubbed and whose child is a dict."""
    server.mgr = mgr
    calls = calls if calls is not None else []

    def fake_load(model_id):
        calls.append(("load", model_id))
        mm = mgr.models.get(model_id) or _managed(model_id)
        mgr.models[model_id] = mm
        mm.request_count += 1
        return mm

    async def fake_transcribe(**kwargs):
        calls.append(("transcribe", kwargs))
        return dict(RESULT)

    mgr.load_whisper_model = fake_load
    mgr.whisper.transcribe = fake_transcribe
    return TestClient(server.app, raise_server_exceptions=False), calls


def _audio(nbytes=64, name="note.m4a"):
    return {"file": (name, b"\x00" * nbytes, "audio/mp4")}


def _spool_contents():
    return sorted(p.name for p in SPOOL.iterdir())


def main():
    step(1, "names resolve exactly, and only exactly")
    check(match_whisper_model("whisper-1")[0] == WHISPER_DEFAULT,
          "whisper-1 is OpenAI's id and resolves to the default")
    check(match_whisper_model("WHISPER-1")[0] == WHISPER_DEFAULT,
          "matching is case-insensitive")
    check(match_whisper_model("  whisper-1 ")[0] == WHISPER_DEFAULT,
          "surrounding space is stripped")
    check(match_whisper_model("")[0] == WHISPER_DEFAULT,
          "an empty model resolves to the default")
    check(match_whisper_model(WHISPER_DEFAULT)[0] == WHISPER_DEFAULT,
          "the real id resolves to itself")
    check(match_whisper_model("whisper-small")[0] == "whisper-small",
          "the second catalogue entry resolves")
    # These four are the point of exact matching. Each one would resolve under
    # the substring rule the chat backends use, and each would be wrong.
    for wrong in ("turbo", "large-v3", "whisper-large-v3-turbo-q4", "qwen2.5-vl"):
        check(match_whisper_model(wrong) is None,
              f"{wrong!r} does not resolve by substring")
    check(match_whisper_model(WHISPER_DEFAULT)[2] == WHISPER_MODELS[WHISPER_DEFAULT]["memory_mb"],
          "the matcher carries the measured memory cost")

    step(2, "[chat-cannot-reach-whisper] the chat resolver cannot route to a transcriber")
    mgr = _manager()
    server.mgr = mgr
    mgr.models[WHISPER_DEFAULT] = _managed()
    # Exact name, resident, right in mgr.models — and still not a chat model.
    check(asyncio.run(server._resolve_model(WHISPER_DEFAULT)) is None,
          "an exact request for a resident whisper model resolves to nothing")
    # And the substring scan, which is what would have matched it by accident.
    check(asyncio.run(server._resolve_model("whisper")) is None,
          "a substring that matches it resolves to nothing")
    mgr.models["qwen3:1.7b"] = _managed("qwen3:1.7b", Backend.OLLAMA)
    mgr.ensure_running = lambda name: True
    got = asyncio.run(server._resolve_model("qwen3"))
    check(got is not None and got.name == "qwen3:1.7b",
          "a real chat model still resolves by substring (the scan still works)")

    step(3, "a request is refused before anything is loaded")
    mgr = _manager()
    client, calls = _client(mgr)

    r = client.post("/v1/audio/transcriptions",
                    files=_audio(), data={"response_format": "mp3"})
    check(r.status_code == 400, f"unknown response_format -> 400 (got {r.status_code})")
    check(r.json()["detail"]["error"] == "unsupported_response_format",
          "it names the reason")

    r = client.post("/v1/audio/transcriptions",
                    files=_audio(), data={"response_format": "verbose_json",
                                          "timestamp_granularities[]": "phoneme"})
    check(r.status_code == 400, f"unknown granularity -> 400 (got {r.status_code})")

    r = client.post("/v1/audio/transcriptions",
                    files=_audio(), data={"timestamp_granularities[]": "word"})
    check(r.status_code == 400,
          f"granularity without verbose_json -> 400 (got {r.status_code})")
    check(r.json()["detail"]["error"] == "granularity_needs_verbose_json",
          "it says which format it needs")

    r = client.post("/v1/audio/transcriptions",
                    files=_audio(), data={"model": "qwen2.5-vl"})
    check(r.status_code == 400, f"a chat model on the audio route -> 400 (got {r.status_code})")
    check(r.json()["detail"]["error"] == "unknown_transcription_model",
          "it names the reason")
    check(WHISPER_DEFAULT in r.json()["detail"]["available"],
          "and lists what it would accept")

    check(calls == [], f"no load and no transcription happened: {calls}")
    check(_spool_contents() == [], f"and the spool is empty: {_spool_contents()}")

    step(4, "[oversize-chunked] the upload limit is counted, not declared")
    mgr = _manager()
    client, calls = _client(mgr)
    over = config.WHISPER_MAX_UPLOAD_MB * 1024 * 1024 + 1024
    r = client.post("/v1/audio/transcriptions", files=_audio(over))
    check(r.status_code == 413, f"over the limit -> 413 (got {r.status_code})")
    check(r.json()["detail"]["limit_mb"] == config.WHISPER_MAX_UPLOAD_MB,
          "it names the limit")
    check(("transcribe" not in [c[0] for c in calls]),
          "nothing was transcribed")
    check(_spool_contents() == [], f"the part-written file was removed: {_spool_contents()}")
    check(mgr.models[WHISPER_DEFAULT].in_flight == 0,
          "and the model is not left held in-flight by the refusal")

    r = client.post("/v1/audio/transcriptions", files=_audio(0))
    check(r.status_code == 400, f"an empty file -> 400 (got {r.status_code})")
    check(_spool_contents() == [], "no zero-byte file left in the spool")

    step(5, "the five response formats")
    mgr = _manager()
    client, calls = _client(mgr)

    r = client.post("/v1/audio/transcriptions", files=_audio())
    check(r.status_code == 200, f"default -> 200 (got {r.status_code})")
    check(r.json() == {"text": RESULT["text"].strip()},
          f"json is exactly {{'text': ...}}: {r.json()}")

    r = client.post("/v1/audio/transcriptions", files=_audio(),
                    data={"response_format": "text"})
    check(r.text.strip() == RESULT["text"].strip(), f"text is the transcript: {r.text!r}")
    check(r.headers["content-type"].startswith("text/plain"),
          f"served as text/plain ({r.headers['content-type']})")

    r = client.post("/v1/audio/transcriptions", files=_audio(),
                    data={"response_format": "srt"})
    srt = r.text
    check(srt.startswith("1\n00:00:00,000 --> 00:00:02,500\n"),
          f"srt numbers from 1 and uses a comma: {srt.splitlines()[:2]}")
    check("2\n00:00:02,500 --> 00:00:05,500" in srt, "and carries the second cue")

    r = client.post("/v1/audio/transcriptions", files=_audio(),
                    data={"response_format": "vtt"})
    vtt = r.text
    check(vtt.startswith("WEBVTT\n"), f"vtt starts with its header: {vtt[:20]!r}")
    check("00:00:00.000 --> 00:00:02.500" in vtt, "and uses a dot for milliseconds")

    r = client.post("/v1/audio/transcriptions", files=_audio(),
                    data={"response_format": "verbose_json"})
    body = r.json()
    check(body["language"] == "english", f"verbose_json reports the language NAME: {body['language']}")
    check(body["language_code"] == "en", "and the code alongside it")
    check(body["duration"] == RESULT["duration"], "and the audio duration")
    check(len(body["segments"]) == 2, f"segments by default: {len(body.get('segments', []))}")
    check("words" not in body, "and no word list unless asked")

    r = client.post("/v1/audio/transcriptions", files=_audio(),
                    data={"response_format": "verbose_json",
                          "timestamp_granularities[]": "word"})
    body = r.json()
    check("segments" not in body, "granularity=word alone returns no segments")
    check([w["word"] for w in body["words"]] == ["Okay,", "shed", "Three"],
          f"words are flattened across segments: {body.get('words')}")
    asked = [c[1]["word_timestamps"] for c in calls if c[0] == "transcribe"]
    check(asked[-1] is True, "the child was asked for word timings")
    check(asked[0] is False, "and was not, for a request that did not want them")

    step(6, "the spool holds nothing after a success or a failure")
    check(_spool_contents() == [], f"clean after six transcriptions: {_spool_contents()}")
    check(not server._whisper_gate.locked(),
          "and the resolve gate was released by every one of them")

    mgr = _manager()
    client, calls = _client(mgr)

    async def child_undecodable(**kwargs):
        raise WhisperFailed("could not decode audio: Invalid data found", 422)

    mgr.whisper.transcribe = child_undecodable
    r = client.post("/v1/audio/transcriptions", files=_audio())
    check(r.status_code == 400,
          f"the child's 422 is the caller's 400 (got {r.status_code})")
    check(r.json()["detail"]["error"] == "undecodable_audio", "named as their file")
    check(_spool_contents() == [], f"and the file is gone: {_spool_contents()}")

    async def child_gone(**kwargs):
        raise WhisperUnavailable("connection refused")

    mgr.whisper.transcribe = child_gone
    r = client.post("/v1/audio/transcriptions", files=_audio())
    check(r.status_code == 503, f"an unreachable child is a 503 (got {r.status_code})")
    check(_spool_contents() == [], "and the file is gone")

    async def child_broke(**kwargs):
        raise WhisperFailed("ffmpeg not found at /opt/homebrew/bin/ffmpeg", 500)

    mgr.whisper.transcribe = child_broke
    r = client.post("/v1/audio/transcriptions", files=_audio())
    check(r.status_code == 502,
          f"the child's own 500 is a 502, not a 400 (got {r.status_code})")
    check(mgr.models[WHISPER_DEFAULT].in_flight == 0,
          "and three failures left nothing held in-flight")

    step(7, "a load that does not fit is a 503 with the numbers, not a spawn")
    mgr = _manager()
    server.mgr = mgr
    # Refuse everything: the gate asks capacity.collect() and nothing else.
    mgr._original_collect = capacity.collect
    capacity.collect = lambda: {"memory_available_mb": 100}
    spawned = []
    import subprocess as sp
    original_popen = sp.Popen

    def no_spawn(*a, **k):
        spawned.append(a)
        raise AssertionError("spawned a child after refusing on capacity")

    sp.Popen = no_spawn
    try:
        client = TestClient(server.app, raise_server_exceptions=False)
        r = client.post("/v1/audio/transcriptions", files=_audio())
        check(r.status_code == 503, f"refused with 503 (got {r.status_code})")
        detail = r.json().get("detail", {})
        check(detail.get("error") == "insufficient_capacity",
              f"through the same funnel as every other backend: {detail}")
        check(detail.get("need_mb") == WHISPER_MODELS[WHISPER_DEFAULT]["memory_mb"],
              "carrying the model's measured cost")
        check(spawned == [], "and no child process was started")
        check(_spool_contents() == [], "and nothing was spooled")
    finally:
        capacity.collect = mgr._original_collect
        sp.Popen = original_popen

    step(8, "the cooldown loop leaves a transcription that is still running alone")
    mgr = _manager()
    mm = _managed(in_flight=1)
    mm.last_used = 0.0                    # idle since 1970
    mgr.models[WHISPER_DEFAULT] = mm
    check(mgr.check_cooldowns() == [],
          "a model with a request in flight is not unloaded")
    check(WHISPER_DEFAULT in mgr.models, "and is still resident")
    mm.in_flight = 0
    results = mgr.check_cooldowns()
    check([r["status"] for r in results] == ["unloaded"],
          f"and is unloaded once the request finishes: {results}")
    check(WHISPER_DEFAULT not in mgr.models, "and is gone from the registry")

    step(9, "discovery lists the transcriber, by id and only when installed")
    mgr = _manager()
    server.mgr = mgr
    original_installed = server.hf_model_installed
    server.hf_model_installed = lambda repo: "whisper-large-v3-turbo" in repo
    try:
        client = TestClient(server.app, raise_server_exceptions=False)
        ids = {m["id"]: m["owned_by"] for m in client.get("/v1/models").json()["data"]}
        check(ids.get(WHISPER_DEFAULT) == Backend.WHISPER.value,
              f"/v1/models lists the installed model under its backend: {ids.get(WHISPER_DEFAULT)}")
        check("whisper-small" not in ids,
              "and not the one whose weights are absent")
        check("whisper-1" not in ids,
              "and not the alias, which resolves without being advertised")
        body = client.get("/models").json()
        rows = {row["model"]: row for row in body["available_whisper"]}
        check(rows[WHISPER_DEFAULT]["installed"] is True, "/models says which are on disk")
        check(rows["whisper-small"]["installed"] is False,
              "so an absent one is visible rather than looking unconfigured")
        check(rows[WHISPER_DEFAULT]["default"] is True, "and which one a bare request gets")
        check(body["whisper_aliases"]["whisper-1"] == WHISPER_DEFAULT,
              "and what whisper-1 means here")
    finally:
        server.hf_model_installed = original_installed

    step(10, "[swap-while-busy] a swap cannot kill a transcription that is running")
    mgr = _manager()
    client, calls = _client(mgr)
    # turbo resident and mid-request; the caller now asks for the other model.
    # One child on one port, so honouring that ask means killing turbo's child
    # under a transcription that has already started.
    busy = _managed(WHISPER_DEFAULT, in_flight=1)
    mgr.models[WHISPER_DEFAULT] = busy
    mgr.load_whisper_model = ModelManager.load_whisper_model.__get__(mgr)
    import subprocess as sp2
    original_popen2 = sp2.Popen

    def no_spawn2(*a, **k):
        raise AssertionError("swapped the child while a transcription was running")

    sp2.Popen = no_spawn2
    try:
        r = client.post("/v1/audio/transcriptions", files=_audio(),
                        data={"model": "whisper-small"})
        check(r.status_code == 503, f"refused with 503 (got {r.status_code})")
        detail = r.json().get("detail", {})
        check(detail.get("error") == "transcriber_busy",
              f"named as busy, not as a capacity or startup failure: {detail}")
        check(detail.get("resident") == WHISPER_DEFAULT,
              f"and names what IS resident: {detail.get('resident')}")
        check(r.headers.get("retry-after") == "30",
              f"with a Retry-After header ({r.headers.get('retry-after')})")
        check(WHISPER_DEFAULT in mgr.models and mgr.models[WHISPER_DEFAULT] is busy,
              "the running model is untouched")
        check(busy.in_flight == 1,
              f"and its in-flight count is unchanged ({busy.in_flight})")
        check("whisper-small" not in mgr.models, "the asked-for model was not loaded")
    except BaseException:
        sp2.Popen = original_popen2
        raise

    # The same ask, once the transcription has finished, reaches the spawn. The
    # spawn is blocked and the attempt is what gets asserted. An earlier version
    # of this check let the load run and treated any exception as proof the
    # refusal had lifted — which passed while the spawn happened to fail, and on a
    # box with room started a real 2.6GB child out of a test whose first line
    # says it loads nothing.
    busy.in_flight = 0
    attempted = []

    def tripwire(*a, **k):
        attempted.append(a)
        raise RuntimeError("spawn reached")

    sp2.Popen = tripwire
    try:
        ModelManager.load_whisper_model(mgr, "whisper-small")
        check(False, "an idle swap reaches the spawn (nothing raised at all)")
    except TranscriberBusy as e:
        check(False, f"an idle swap is still refused as busy: {e}")
    except capacity.InsufficientCapacity as e:
        print(f"  SKIP — the box has no room for whisper-small right now: {e}")
    except RuntimeError as e:
        check("spawn reached" in str(e),
              f"an idle swap reaches the spawn (raised {e})")
        check(attempted != [], "and a child was actually about to be started")
        check(WHISPER_DEFAULT not in mgr.models,
              "and the idle model was unloaded to make room")
    finally:
        sp2.Popen = original_popen2

    step(11, "[dead-child] a child that stopped answering is replaced, not served")
    mgr = _manager()
    mm = _managed()
    mm.lease_id = "lease_fake"
    mgr.models[WHISPER_DEFAULT] = mm
    released = []
    mgr.sc.release_lease = lambda lease_id: released.append(lease_id)

    # Alive: the resident path hands back the same entry and spawns nothing.
    mgr.whisper.health = lambda timeout=2.0: {"status": "ok", "model": "x"}
    got = ModelManager.load_whisper_model(mgr, WHISPER_DEFAULT)
    check(got is mm, "a live child is reused")
    check(got.request_count == 1, f"and the request is counted once ({got.request_count})")

    # Not answering: the entry must be dropped rather than handed back. Nothing
    # else would ever replace it — the cooldown loop is the only thing that
    # removes an entry, and the audio route stamps last_used even on a failure,
    # so serving the dead entry is a 503 on every request until a restart.
    mgr.whisper.health = lambda timeout=2.0: None
    spawned2 = []

    def record_spawn(*a, **k):
        spawned2.append(a)
        raise RuntimeError("spawn blocked by the test")

    import subprocess as sp3
    original_popen3 = sp3.Popen
    sp3.Popen = record_spawn
    try:
        try:
            ModelManager.load_whisper_model(mgr, WHISPER_DEFAULT)
            check(False, "a dead child leads to a fresh load")
        except RuntimeError as e:
            check("spawn blocked by the test" in str(e),
                  f"a dead child leads to a fresh load attempt ({e})")
    finally:
        sp3.Popen = original_popen3
    check(spawned2 != [], "a replacement child was actually spawned")
    check(released == ["lease_fake"],
          f"and the dead entry's lease was released, not leaked: {released}")

    print(f"\n{'='*60}")
    for leftover in SPOOL.iterdir():
        print(f"  leftover in spool: {leftover}")
    if not any(SPOOL.iterdir()):
        SPOOL.rmdir()
    if failures:
        print(f"  {len(failures)} FAILED:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
