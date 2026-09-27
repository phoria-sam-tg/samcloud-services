#!/usr/bin/env python3
"""SC_TOKEN comes from the environment if it is there, and from a 0600 file if not.

#845: the launcher wrapped its env file in `set -a`, so SC_TOKEN was exported and
every child of the gateway inherited it — visible in `ps eww <pid>` to any reader
of that uid. Nothing on disk held a literal (slice's env file used
`SC_TOKEN=$(cat ~/.samcloud/token)`, evaluated at source time), so the fix is to
read that same file at the point of use and stop exporting the value.

The environment still wins when set, deliberately: that makes deploying this
change a no-op until the launcher stops exporting, so the two steps are safe in
either order and neither can cause an authentication outage on its own.

Run: python3 ollama/test_token_source.py
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

passed = failed = 0


def check(label, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  \033[32m✓\033[0m {label}")
    else:
        failed += 1
        print(f"  \033[31m✗\033[0m {label}  {detail}")


def load(token_env=None, token_file=None):
    """Re-import config with a given environment. Returns the module."""
    for k in ("SC_TOKEN", "SC_TOKEN_FILE"):
        os.environ.pop(k, None)
    if token_env is not None:
        os.environ["SC_TOKEN"] = token_env
    if token_file is not None:
        os.environ["SC_TOKEN_FILE"] = token_file
    from ollama import config
    return importlib.reload(config)


def write(path: Path, body: str, mode: int) -> str:
    path.write_text(body)
    path.chmod(mode)
    return str(path)


def main():
    print("SC_TOKEN source (#845)")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        raw = write(tmp / "token", "sc_agent_rawtoken\n", 0o600)
        hdr = write(tmp / "token.hdr", "Authorization: Bearer sc_agent_hdrtoken\n", 0o600)
        loose = write(tmp / "loose", "sc_agent_loosetoken\n", 0o644)

        print("  [1] the environment wins when it is set")
        # Deliberate: it makes deploying this change a no-op until the launcher
        # stops exporting, so neither step alone can cause an auth outage.
        c = load(token_env="sc_agent_fromenv", token_file=raw)
        check("token came from the environment", c.SC_TOKEN == "sc_agent_fromenv", c.SC_TOKEN[:12])
        check("and the source says so", c.SC_TOKEN_SOURCE == "environment", c.SC_TOKEN_SOURCE)

        print("  [2] a raw 0600 file is read when the environment is empty")
        c = load(token_env=None, token_file=raw)
        check("token came from the file", c.SC_TOKEN == "sc_agent_rawtoken", c.SC_TOKEN[:12])
        check("trailing newline stripped", "\n" not in c.SC_TOKEN, repr(c.SC_TOKEN[-3:]))
        check("the source names the file", c.SC_TOKEN_SOURCE.startswith("file "), c.SC_TOKEN_SOURCE)

        print("  [3] a curl header file is accepted too")
        # Both shapes exist side by side in these accounts; picking one would make
        # the other fail silently with an empty token.
        c = load(token_env=None, token_file=hdr)
        check("the bearer is parsed out of the header",
              c.SC_TOKEN == "sc_agent_hdrtoken", c.SC_TOKEN[:12])

        print("  [4] an empty SC_TOKEN falls through to the file")
        # `set -a` with an unset variable exports an EMPTY string, which is the
        # commonest way this would arrive rather than absent.
        c = load(token_env="", token_file=raw)
        check("empty env does not win", c.SC_TOKEN == "sc_agent_rawtoken", c.SC_TOKEN[:12])
        check("and the source is the file", c.SC_TOKEN_SOURCE.startswith("file "), c.SC_TOKEN_SOURCE)

        print("  [5] a permissive file WARNS and still works")
        # Refusing would turn a mode bit into a total authentication outage.
        import logging
        seen = []
        c = load(token_env=None, token_file=loose)
        real = c.log.warning
        c.log.warning = lambda m, *a, **k: seen.append(str(m))
        try:
            got = c._read_token_file(loose)
        finally:
            c.log.warning = real
        check("the token is still returned", got == "sc_agent_loosetoken", got[:12])
        check("and a warning names the mode",
              seen and "0o644" in seen[0] and "not 0600" in seen[0], str(seen))

        print("  [6] a missing file is empty and says NOT FOUND, not a crash")
        c = load(token_env=None, token_file=str(tmp / "nope"))
        check("token is empty", c.SC_TOKEN == "", repr(c.SC_TOKEN))
        check("the source is NOT FOUND, so the log can say why",
              c.SC_TOKEN_SOURCE == "NOT FOUND", c.SC_TOKEN_SOURCE)

        print("  [7] ~ is expanded, since that is how it will be configured")
        c = load(token_env=None, token_file="~/definitely-not-a-real-credential-file")
        check("an unexpanded ~ would raise rather than return empty",
              c.SC_TOKEN == "", repr(c.SC_TOKEN))

    print()
    if failed:
        print(f"  FAILED: {failed} failed, {passed} passed")
        sys.exit(1)
    print(f"  ALL PASSED: SC_TOKEN source {passed} passed, 0 failed")


if __name__ == "__main__":
    main()
