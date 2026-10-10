"""Every offered entry says what it can be asked for, and whether it is a mutex (#931).

The facts were already in the code -- `Backend`'s own docstring says whisper
"serves ONE route and cannot answer a chat", and that exo is held under an
exclusive lease -- but they were published nowhere, so each consumer had to
hardcode them from the backend string. Two did not:

  * a container seat's model picker read `/warm` as chat candidates
  * `wake-install`'s runtime preflight did the same at seat BIRTH, and would
    have taken the pool's exclusive lease before the seat had done any work

and on a `think` decline the gateway offered **both whisper models** as
alternatives to a chat completion -- the one surface that is a recommendation
rather than a list.

Measured on models-cs 2026-10-10: `whisper-small` is 1,600 MB and
`whisper-large-v3-turbo` 2,600 MB, so a smallest-first chooser picks
`qwen3:1.7b` at 1,296 MB and dodges them **on today's inventory**. That safety
is an accident of what is installed this week, not a property of the list: it
ends the moment anyone installs a chat model under 1,600 MB or an ASR model
under 1,296. And `think`'s own size is ABSENT here and 0 on `/warm`, so it sorts
first under any `memory_mb or 0` fallback -- which is the near-miss that
actually happened.

Run from the repo ROOT: python -m ollama.test_offer_capability
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama.manager import (Backend, BACKEND_SERVES, backend_serves,   # noqa: E402
                            backend_is_exclusive)

PASS = FAIL = 0


def check(desc, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \033[0;32mok\033[0m   {desc}")
    else:
        FAIL += 1
        print(f"  \033[0;31mFAIL\033[0m {desc}")
        if detail:
            print(f"       {detail}")


def main():
    print("\n  offered entries carry capability and exclusivity\n")

    # --- the table covers every backend, or a new one serves nothing ----
    missing = [b.value for b in Backend if b not in BACKEND_SERVES]
    check("every Backend has a row in the table",
          not missing,
          f"a backend with no row serves nothing and is silently filtered out "
          f"of every recommendation: {missing}")

    # --- the fact the enum states in prose, now as data -----------------
    check("whisper does NOT serve chat — the thing both consumers assumed",
          "chat" not in backend_serves("mlx-whisper"),
          str(backend_serves("mlx-whisper")))
    check("...and it does serve transcription",
          backend_serves("mlx-whisper") == ["transcription"])
    for b in ("ollama", "llama-server", "exo"):
        check(f"{b} serves chat", "chat" in backend_serves(b))
    check("mlx-vlm serves chat AND vision",
          set(backend_serves("mlx-vlm")) == {"chat", "vision"},
          str(backend_serves("mlx-vlm")))

    # --- exclusivity, the half a chooser cannot see ---------------------
    check("exo is marked exclusive", backend_is_exclusive("exo") is True)
    for b in ("ollama", "llama-server", "mlx-vlm", "mlx-whisper"):
        check(f"{b} is not exclusive", backend_is_exclusive(b) is False)

    # --- an UNKNOWN backend serves nothing, which is the safe direction -
    check("an unknown backend serves nothing, rather than defaulting to chat",
          backend_serves("some-new-thing") == [],
          "defaulting to chat is how an unservable entry gets recommended — "
          "the exact bug this closes")
    check("...and is not exclusive",
          backend_is_exclusive("some-new-thing") is False)
    check("None serves nothing and does not raise", backend_serves(None) == [])

    # --- the filter the decline now applies -----------------------------
    # Mirrors _pool_unavailable_503: not the pool, and chat-capable.
    offered = [
        {"name": "qwen3:1.7b", "backend": "ollama", "serves": ["chat"]},
        {"name": "whisper-small", "backend": "mlx-whisper", "serves": ["transcription"]},
        {"name": "whisper-large-v3-turbo", "backend": "mlx-whisper",
         "serves": ["transcription"]},
        {"name": "gemma-4-31b", "backend": "mlx-vlm", "serves": ["chat", "vision"]},
        {"name": "think", "backend": "exo", "serves": ["chat"]},
        {"name": "mystery", "backend": "future-thing", "serves": []},
    ]
    alts = [m["name"] for m in offered
            if m.get("backend") != Backend.EXO.value
            and "chat" in (m.get("serves") or [])]
    check("a chat decline offers only chat-capable models",
          alts == ["qwen3:1.7b", "gemma-4-31b"], str(alts))
    check("...so neither whisper model is recommended for a chat",
          not [a for a in alts if "whisper" in a], str(alts))
    check("...and the pool is still excluded, as before",
          "think" not in alts)
    check("...and an unrecognised backend is excluded rather than guessed in",
          "mystery" not in alts)

    print(f"\n  {'all checks passed' if not FAIL else 'FAILED'}: "
          f"{PASS} passed, {FAIL} failed\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
