#!/usr/bin/env python
"""mlx-whisper child process — one model, one box, one transcription at a time.

Run by the gateway, not by hand:

    $WHISPER_PYTHON ollama/whisper_server.py \
        --model mlx-community/whisper-large-v3-turbo \
        --host 127.0.0.1 --port 8803 \
        --spool ~/var/samcloud-services/spool/whisper \
        --ffmpeg /opt/homebrew/bin/ffmpeg

`$WHISPER_PYTHON` is a DIFFERENT interpreter from the one running the gateway —
the one with mlx-whisper in it (requirements-whisper.txt). That is why this file
imports nothing from its own package: it is executed as a script by an
interpreter that has never heard of `ollama/`, so a relative import here is an
ImportError at startup rather than at review time.

Two endpoints:

    GET  /health      {"status": "ok", "model": ..., "busy": bool}
    POST /transcribe  {"path": ...}  ->  mlx-whisper's own result dict

The model is loaded before uvicorn binds the port, so a 200 from /health means
"ready to transcribe", not "process started". The gateway polls exactly that.
"""

import argparse
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [whisper-child] %(levelname)s: %(message)s",
)
log = logging.getLogger("whisper-child")

SAMPLE_RATE = 16000

# Set by main() before the app serves.
MODEL_REPO: str = ""
SPOOL_DIR: Path = Path()
FFMPEG: str = "ffmpeg"

# One model on one GPU: two concurrent transcriptions would double the 2.5GB
# peak the gateway leased for one. Requests queue on this rather than racing.
_lock = threading.Lock()


class AudioDecodeError(Exception):
    """ffmpeg could not read the file. The caller sent something we can't play."""


def decode(path: str) -> np.ndarray:
    """Audio file -> mono float32 at 16kHz, the only thing whisper listens to.

    Does what `mlx_whisper.audio.load_audio` does, with one difference that is
    the reason it is written out here: it execs the ffmpeg the gateway named
    rather than whichever one is on PATH. This process is spawned by a process
    spawned by launchd, whose PATH is /usr/bin:/bin:/usr/sbin:/sbin — no
    /opt/homebrew/bin — so the library's own lookup fails on this box.
    """
    cmd = [
        FFMPEG, "-nostdin", "-threads", "0",
        "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError as e:
        # Not the caller's fault: the box is misconfigured. Distinct from a
        # file we cannot decode, and the gateway maps the two differently.
        raise RuntimeError(f"ffmpeg not found at {FFMPEG}") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", "replace").strip()
        raise AudioDecodeError(stderr.splitlines()[-1] if stderr else "ffmpeg failed")
    if not out:
        raise AudioDecodeError("no audio stream in the file")
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def resolve_in_spool(raw: str) -> Path:
    """The path the caller named, or a refusal.

    The gateway writes an upload into the spool and hands over its path, so
    this process reads a filename it was given rather than a body it parsed.
    That is only safe while "given" means "inside the spool": resolve the path
    (following any symlink) and require the result to sit under SPOOL_DIR.
    Asserted by test_whisper_child.py cases [escape] and [symlink].
    """
    p = Path(raw).resolve()
    root = SPOOL_DIR.resolve()
    if not p.is_relative_to(root):
        raise HTTPException(status_code=400, detail=f"path is not inside {root}")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="no such file in the spool")
    return p


def jsonable(obj):
    """Replace every numpy scalar in a result with the Python number it holds.

    What was measured (slice, mlx-whisper 0.4.3, numpy 2.5.3, a real
    transcription with `word_timestamps=True`): the result holds `str`, `int`,
    `float` and `np.float64`, and `json.dumps` accepts it unchanged. So this
    coercion is NOT fixing an observed failure — removing it leaves every
    response working, which was confirmed by removing it and re-running.

    What it is for, kept separate from that: `np.float64` survives json only
    because it subclasses Python `float`. `np.float32` and `np.int32` do not
    subclass anything json knows, and json refuses both. Which of the three a
    decode yields follows from the model's dtype and from numpy's version,
    neither of which this file chooses, and the failure would be a 500 on one
    response format only. One walk over the result is cheap enough not to
    depend on that.

    Asserted by test_whisper_child.py case [numpy], which checks both halves:
    that the coercion fixes the types json refuses, and that today's real
    output did not need it.
    """
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class TranscribeRequest(BaseModel):
    path: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    word_timestamps: bool = False


app = FastAPI(title="mlx-whisper child")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_REPO,
        "busy": _lock.locked(),
        "sample_rate": SAMPLE_RATE,
    }


def _transcribe(req: TranscribeRequest, path: Path) -> dict:
    import mlx_whisper

    audio = decode(str(path))
    opts: dict = {
        "path_or_hf_repo": MODEL_REPO,
        "word_timestamps": req.word_timestamps,
    }
    if req.language:
        opts["language"] = req.language
    if req.prompt:
        opts["initial_prompt"] = req.prompt
    if req.temperature is not None:
        # mlx-whisper's default is a fallback LADDER (0.0, 0.2, ... 1.0): it
        # re-decodes hotter when a segment looks degenerate. A caller naming one
        # temperature is asking for that one, so pass the scalar and lose the
        # ladder — which is what OpenAI's endpoint documents too.
        opts["temperature"] = req.temperature

    t0 = time.time()
    with _lock:
        result = mlx_whisper.transcribe(audio, **opts)
    result = jsonable(result)
    # mlx-whisper reports the language as an ISO code ("en"); OpenAI's
    # verbose_json reports the name ("english"). Both are returned rather than
    # one converted into the other, so neither client has to guess.
    from mlx_whisper.tokenizer import LANGUAGES
    code = result.get("language")
    result["language_name"] = LANGUAGES.get(code, code)
    result["duration"] = round(len(audio) / SAMPLE_RATE, 3)
    result["transcribe_seconds"] = round(time.time() - t0, 3)
    return result


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    path = resolve_in_spool(req.path)
    try:
        # Off the event loop: a transcription is minutes of GPU work, and
        # /health has to keep answering while it runs — the gateway reads it to
        # decide whether this process is alive.
        return await run_in_threadpool(_transcribe, req, path)
    except AudioDecodeError as e:
        raise HTTPException(status_code=422, detail=f"could not decode audio: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


def preload(repo: str):
    """Load the weights and run one transcription before the port opens.

    A warmup on a second of silence rather than a bare `load_model`, because
    the first real request pays for more than the weights: numba compiles the
    DTW kernel, the tokenizer is fetched, MLX builds its kernels. Measured on
    slice with a cold process and warm HF cache, that is the difference between
    a first request of ~20s and one of ~1s. /health must not go green before
    it is paid.
    """
    import mlx_whisper

    t0 = time.time()
    mlx_whisper.transcribe(
        np.zeros(SAMPLE_RATE, dtype=np.float32),
        path_or_hf_repo=repo,
        word_timestamps=True,
    )
    log.info(f"{repo} ready in {time.time() - t0:.1f}s")


def main():
    global MODEL_REPO, SPOOL_DIR, FFMPEG

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="mlx-community whisper repo id")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8803)
    ap.add_argument("--spool", required=True, help="the only directory paths may name")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    args = ap.parse_args()

    MODEL_REPO = args.model
    SPOOL_DIR = Path(args.spool).expanduser()
    FFMPEG = args.ffmpeg
    SPOOL_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"loading {MODEL_REPO} (pid {os.getpid()})")
    preload(MODEL_REPO)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
