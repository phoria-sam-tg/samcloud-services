"""Ollama API client for model management."""

import re
import httpx
import json
from dataclasses import dataclass, field
from typing import Optional, Iterator

from . import config

OLLAMA_BASE = config.OLLAMA_BASE

# Approximate VRAM requirements (MB) for common model sizes.
# Ollama reports actual size once pulled - these are planning estimates.
MODEL_MEMORY_ESTIMATES = {
    "1b": 1200,
    "3b": 2500,
    "7b": 5000,
    "8b": 5500,
    "13b": 8500,
    "14b": 9500,
    "32b": 20000,
    "70b": 42000,
    "72b": 44000,
}


# Match a parameter-count tag only on a digit boundary. Plain substring
# matching read "qwen3.8:27b-mlx" as 7b and leased 5000MB for a model that
# actually resides at ~17500MB — the registry under-counted GPU memory by 3.5x,
# so a second service could be told there was room that did not exist.
# "13b" hit "3b" the same way. Anchor it.
_SIZE_TAG_RE = re.compile(r"(?<![0-9.])(\d+(?:\.\d+)?)b(?![a-z0-9])")


def estimate_memory_mb(model_name: str) -> int:
    """Estimate memory needed based on model name/size tag.

    Only a fallback — prefer OllamaClient.memory_estimate_mb(), which reads the
    real weight size from Ollama instead of guessing from the name.
    """
    name = model_name.lower()
    m = _SIZE_TAG_RE.search(name)
    if m:
        raw = m.group(1)
        # "1.0" -> "1", but never turn "70" into "7".
        tag = f"{raw[:-2] if raw.endswith('.0') else raw}b"
        if tag in MODEL_MEMORY_ESTIMATES:
            return MODEL_MEMORY_ESTIMATES[tag]
        # Unlisted size: interpolate at ~0.65 GB per billion params (Q4-ish).
        try:
            return max(1024, round(float(raw) * 650))
        except ValueError:
            pass
    # Default conservative estimate for unknown models
    return 4000


@dataclass
class OllamaClient:
    base_url: str = OLLAMA_BASE
    _http: httpx.Client = field(default=None, repr=False)

    def __post_init__(self):
        # Used for non-streaming requests only (health, list, show, etc.)
        self._http = httpx.Client(base_url=self.base_url, timeout=600)

    def version(self) -> str:
        r = self._http.get("/api/version")
        r.raise_for_status()
        return r.json().get("version", "unknown")

    def memory_estimate_mb(self, model_name: str) -> int:
        """Memory to lease for a model, from its real weight size where known."""
        try:
            for m in self.list_models():
                name = m.get("name", "")
                if name == model_name or name == f"{model_name}:latest":
                    size = int(m.get("size", 0))
                    if size > 0:
                        return max(1024, round(size / 1024 / 1024))
        except Exception:
            pass
        return estimate_memory_mb(model_name)

    def list_models(self) -> list[dict]:
        r = self._http.get("/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])

    def list_running(self) -> list[dict]:
        r = self._http.get("/api/ps")
        r.raise_for_status()
        return r.json().get("models", [])

    def pull_model(self, model: str, stream: bool = True) -> Iterator[dict]:
        """Pull a model. Yields progress dicts if stream=True."""
        with self._http.stream(
            "POST", "/api/pull", json={"model": model, "stream": stream}, timeout=None
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)

    def load_model(self, model: str, keep_alive: str | int = "1h") -> dict:
        """Load a model into memory without generating (warm-up).
        keep_alive: duration string ("1h", "30m") or -1 for indefinite."""
        r = self._http.post(
            "/api/generate",
            json={"model": model, "prompt": "", "keep_alive": keep_alive},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()

    def unload_model(self, model: str) -> dict:
        """Unload a model from memory by setting keep_alive to 0."""
        r = self._http.post(
            "/api/generate",
            json={"model": model, "prompt": "", "keep_alive": "0"},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    _stream_timeout = httpx.Timeout(connect=10, read=300, write=10, pool=10)

    def generate(self, model: str, prompt: str, **kwargs) -> Iterator[dict]:
        """Generate completion, streaming."""
        payload = {"model": model, "prompt": prompt, "stream": True, **kwargs}
        with self._http.stream(
            "POST", "/api/generate", json=payload, timeout=self._stream_timeout
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)

    def chat(self, model: str, messages: list[dict], **kwargs) -> Iterator[dict]:
        """Chat completion, streaming (sync, for non-streaming collection)."""
        payload = {"model": model, "messages": messages, "stream": True, **kwargs}
        with self._http.stream(
            "POST", "/api/chat", json=payload, timeout=self._stream_timeout
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)

    async def chat_stream(self, model: str, messages: list[dict], **kwargs):
        """Chat completion, async streaming. Kills connection on cancel."""
        import aiohttp
        payload = {"model": model, "messages": messages, "stream": True, **kwargs}
        session = aiohttp.ClientSession()
        try:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.content:
                    line = line.decode().strip()
                    if line:
                        yield json.loads(line)
        finally:
            await session.close()

    async def generate_stream(self, model: str, prompt: str, **kwargs):
        """Generate completion, async streaming. Kills connection on cancel."""
        import aiohttp
        payload = {"model": model, "prompt": prompt, "stream": True, **kwargs}
        session = aiohttp.ClientSession()
        try:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.content:
                    line = line.decode().strip()
                    if line:
                        yield json.loads(line)
        finally:
            await session.close()

    def show_model(self, model: str) -> dict:
        """Get model metadata."""
        r = self._http.post("/api/show", json={"model": model})
        r.raise_for_status()
        return r.json()

    def delete_model(self, model: str) -> dict:
        r = self._http.request("DELETE", "/api/delete", json={"model": model})
        r.raise_for_status()
        return r.json()
