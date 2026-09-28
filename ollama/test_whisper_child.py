#!/usr/bin/env python3
"""The whisper child's own logic: path handling, audio decoding, JSON coercion.

Run with WHISPER_PYTHON, not the gateway's interpreter — this imports
`whisper_server`, which imports numpy:

    ~/code/mlx-whisper-server/.venv/bin/python ollama/test_whisper_child.py

Needs ffmpeg for the decode cases; pass its path in FFMPEG_BIN if it is not on
PATH. Set WHISPER_TEST_AUDIO to a real audio file to add a live transcription
(it loads a model and takes GPU time, so it is opt-in and skipped otherwise).
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

spec = importlib.util.spec_from_file_location("whisper_server", HERE / "whisper_server.py")
ws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ws)

import numpy as np                        # noqa: E402  (after the module import, same venv)
from fastapi import HTTPException         # noqa: E402

passed = failed = 0


def check(label, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  \033[32m✓\033[0m {label}")
    else:
        failed += 1
        print(f"  \033[31m✗\033[0m {label}  {detail}")


def refuses(fn, *a, status=None):
    """Call fn and report the HTTPException status, or None if it returned."""
    try:
        fn(*a)
        return None
    except HTTPException as e:
        return e.status_code


def main():
    spool = Path(tempfile.mkdtemp(prefix="whisper-child-test-"))
    outside = Path(tempfile.mkdtemp(prefix="whisper-child-outside-"))
    ws.SPOOL_DIR = spool
    ws.FFMPEG = FFMPEG

    print("whisper child (#858)")
    print(f"  spool   {spool}")
    print(f"  ffmpeg  {FFMPEG}")
    print()

    print("  [numpy] which numpy scalars json refuses, and which it does not")
    # Two halves, kept apart, because the obvious story here is wrong: it is
    # easy to assume every numpy scalar breaks json.dumps and to write a comment
    # saying so. np.float64 subclasses Python float and sails through.
    check("np.float64 subclasses float, so json already accepts it",
          issubclass(np.float64, float))
    check("np.float32 does not, so json refuses it",
          not issubclass(np.float32, float))
    check("np.int32 does not either", not issubclass(np.int32, int))

    # Half one: what today's decoder actually returns needs no coercion. Built
    # from the types measured in a real result (str, int, float, np.float64) so
    # the assertion holds without a GPU.
    as_measured = {
        "text": " Okay, shed shelf two.",
        "language": "en",
        "segments": [{
            "id": 0,
            "seek": 0,
            "start": np.float64(0.0),
            "end": np.float64(2.5),
            "temperature": 0.0,
            "words": [{"word": " Okay,", "start": np.float64(0.0),
                       "end": np.float64(0.38), "probability": 0.58}],
        }],
    }
    try:
        json.dumps(as_measured)
        check("a result of the measured shape serialises WITHOUT jsonable()", True)
    except TypeError as e:
        check("a result of the measured shape serialises WITHOUT jsonable()", False,
              f"then the docstring is out of date: {e}")

    # Half two: the types that would break it, which is what jsonable() is for.
    # A dtype change in the model, or a numpy release, is all it takes.
    would_break = {
        "segments": [{
            "start": np.float32(0.0),
            "avg_logprob": np.float32(-0.21),
            "tokens": [np.int32(50364), np.int32(1033)],
        }],
    }
    try:
        json.dumps(would_break)
        check("np.float32/np.int32 are refused by json (else nothing to fix)", False,
              "they serialised, so this guard has no failure to prevent")
    except TypeError:
        check("np.float32/np.int32 are refused by json", True)
    clean = ws.jsonable(would_break)
    try:
        json.dumps(clean)
        check("jsonable() makes them serialisable", True)
    except TypeError as e:
        check("jsonable() makes them serialisable", False, str(e))
    seg = clean["segments"][0]
    check("and the values are unchanged", round(seg["avg_logprob"], 2) == -0.21
          and type(seg["start"]) is float,
          f"avg_logprob={seg['avg_logprob']!r} start={type(seg['start']).__name__}")
    check("ints stay ints", seg["tokens"] == [50364, 1033] and
          all(type(t) is int for t in seg["tokens"]),
          f"tokens={seg['tokens']!r}")

    print("  [spool] a path inside the spool resolves")
    good = spool / "note.m4a"
    good.write_bytes(b"\x00" * 16)
    check("a real file in the spool is accepted",
          ws.resolve_in_spool(str(good)) == good.resolve())

    print("  [escape] a path outside the spool is refused")
    secret = outside / "secret.txt"
    secret.write_text("not audio")
    check("an absolute path elsewhere -> 400",
          refuses(ws.resolve_in_spool, str(secret)) == 400)
    check("a traversal out of the spool -> 400",
          refuses(ws.resolve_in_spool, str(spool / ".." / secret.name)) == 400)
    check("a sibling directory with the same prefix -> 400",
          refuses(ws.resolve_in_spool, f"{spool}-evil/note.m4a") == 400)

    print("  [symlink] a symlink inside the spool pointing out is refused")
    link = spool / "link.m4a"
    link.symlink_to(secret)
    check("the symlink is resolved before it is checked -> 400",
          refuses(ws.resolve_in_spool, str(link)) == 400)

    print("  [missing] a name with no file behind it is a 404, not a 400")
    check("nothing there -> 404",
          refuses(ws.resolve_in_spool, str(spool / "gone.m4a")) == 404)

    print("  [decode] real audio decodes to mono float32 at 16kHz")
    wav = spool / "tone.wav"
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2", str(wav)],
        check=True,
    )
    audio = ws.decode(str(wav))
    check("dtype is float32", audio.dtype == np.float32, str(audio.dtype))
    check("length is duration * 16000", abs(len(audio) - 2 * ws.SAMPLE_RATE) < 1600,
          f"{len(audio)} samples")
    check("samples are normalised to -1..1", float(np.abs(audio).max()) <= 1.0,
          f"max={float(np.abs(audio).max())}")

    print("  [undecodable] a file that is not audio is the caller's error")
    notaudio = spool / "notes.txt"
    notaudio.write_text("this is not a sound")
    try:
        ws.decode(str(notaudio))
        check("a text file raises AudioDecodeError", False, "it decoded")
    except ws.AudioDecodeError as e:
        check("a text file raises AudioDecodeError", True)
        check("and carries ffmpeg's own last line", str(e).strip() != "", repr(str(e)))
    except Exception as e:
        check("a text file raises AudioDecodeError", False,
              f"raised {type(e).__name__} instead: {e}")

    print("  [misconfigured] a missing ffmpeg is OUR error, not the caller's")
    # The two must stay distinguishable: the gateway maps AudioDecodeError to a
    # 400 and everything else to a 502, so collapsing them would blame a caller
    # for a box with no ffmpeg on it.
    saved = ws.FFMPEG
    ws.FFMPEG = str(spool / "no-such-ffmpeg")
    try:
        ws.decode(str(wav))
        check("a missing ffmpeg raises RuntimeError", False, "it decoded")
    except ws.AudioDecodeError:
        check("a missing ffmpeg raises RuntimeError", False,
              "it raised AudioDecodeError, which the gateway turns into a 400")
    except RuntimeError as e:
        check("a missing ffmpeg raises RuntimeError", True)
        check("and names the path it tried", ws.FFMPEG in str(e), str(e))
    finally:
        ws.FFMPEG = saved

    live = os.environ.get("WHISPER_TEST_AUDIO")
    if live:
        print(f"  [live] a real transcription of {live}")
        repo = os.environ.get("WHISPER_TEST_MODEL",
                              "mlx-community/whisper-large-v3-turbo")
        ws.MODEL_REPO = repo
        target = spool / Path(live).name
        shutil.copy(live, target)
        req = ws.TranscribeRequest(path=str(target), word_timestamps=True)
        result = ws._transcribe(req, target)
        check("it returns text", bool(result.get("text", "").strip()),
              repr(result.get("text"))[:120])
        check("it reports the audio duration", result.get("duration", 0) > 0,
              str(result.get("duration")))
        check("it reports the language and its name",
              result.get("language") and result.get("language_name"),
              f"{result.get('language')} / {result.get('language_name')}")
        try:
            json.dumps(result)
            check("the whole result serialises, word timestamps included", True)
        except TypeError as e:
            check("the whole result serialises, word timestamps included", False, str(e))
    else:
        print("  [live] SKIP — set WHISPER_TEST_AUDIO to a file to run it")

    shutil.rmtree(spool, ignore_errors=True)
    shutil.rmtree(outside, ignore_errors=True)

    print()
    if failed:
        print(f"  FAILED: {failed} failed, {passed} passed")
        sys.exit(1)
    print(f"  ALL PASSED: whisper child {passed} passed, 0 failed")


if __name__ == "__main__":
    main()
