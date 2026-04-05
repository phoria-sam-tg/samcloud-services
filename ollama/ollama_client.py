"""Ollama API client for model management."""

import httpx
import json
from dataclasses import dataclass, field
from typing import Optional, Iterator

OLLAMA_BASE = "http://localhost:11434"

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


def estimate_memory_mb(model_name: str) -> int:
    """Estimate memory needed based on model name/size tag."""
    name = model_name.lower()
    for size_tag, mb in MODEL_MEMORY_ESTIMATES.items():
        if size_tag in name:
            return mb
    # Default conservative estimate for unknown models
    return 4000


@dataclass
class OllamaClient:
    base_url: str = OLLAMA_BASE
    _http: httpx.Client = field(default=None, repr=False)

    def __post_init__(self):
        # Pool limits prevent connection accumulation from abandoned streams
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=600,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )

    def version(self) -> str:
        r = self._http.get("/api/version")
        r.raise_for_status()
        return r.json().get("version", "unknown")

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

    # 5 min read timeout for streaming — long enough for slow inference,
    # short enough to not leak connections from abandoned requests.
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
        """Chat completion, streaming."""
        payload = {"model": model, "messages": messages, "stream": True, **kwargs}
        with self._http.stream(
            "POST", "/api/chat", json=payload, timeout=self._stream_timeout
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)

    def show_model(self, model: str) -> dict:
        """Get model metadata."""
        r = self._http.post("/api/show", json={"model": model})
        r.raise_for_status()
        return r.json()

    def delete_model(self, model: str) -> dict:
        r = self._http.request("DELETE", "/api/delete", json={"model": model})
        r.raise_for_status()
        return r.json()
