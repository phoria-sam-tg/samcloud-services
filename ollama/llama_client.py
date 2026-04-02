"""llama-server (llama.cpp) process management and API client."""

import subprocess
import signal
import time
import httpx
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LLAMA_SERVER_BIN = "/opt/homebrew/bin/llama-server"
MODELS_DIR = Path("/Users/sam/models")

# Known models and their approximate memory usage (MB)
KNOWN_MODELS = {
    "Qwen3-32B-Q6_K.gguf": {
        "memory_mb": 25000,
        "params": "32B",
        "quant": "Q6_K",
        "family": "qwen3",
        "ctx_default": 12288,
    },
    "Qwen3.5-35B-A3B-Q4_K_M.gguf": {
        "memory_mb": 21000,
        "params": "35B-MoE",
        "quant": "Q4_K_M",
        "family": "qwen3.5",
        "ctx_default": 12288,
    },
}


@dataclass
class LlamaInstance:
    """A running llama-server instance."""
    model_file: str
    port: int
    pid: Optional[int] = None
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    ctx_size: int = 12288
    gpu_layers: int = 99
    threads: int = 4
    flash_attn: bool = True
    cache_type_k: str = "q4_0"
    cache_type_v: str = "q4_0"

    @property
    def model_name(self) -> str:
        return Path(self.model_file).stem

    @property
    def memory_mb(self) -> int:
        basename = Path(self.model_file).name
        if basename in KNOWN_MODELS:
            return KNOWN_MODELS[basename]["memory_mb"]
        # Rough estimate from file size (model + KV cache overhead)
        try:
            size_gb = Path(self.model_file).stat().st_size / (1024**3)
            return int(size_gb * 1100)  # ~10% overhead for KV cache
        except FileNotFoundError:
            return 8000


@dataclass
class LlamaServerClient:
    """Manages llama-server instances."""
    models_dir: Path = MODELS_DIR
    bin_path: str = LLAMA_SERVER_BIN
    instances: dict[int, LlamaInstance] = field(default_factory=dict)  # port -> instance

    def discover_running(self) -> list[LlamaInstance]:
        """Find already-running llama-server processes."""
        import re
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True
        )
        instances = []
        for line in result.stdout.splitlines():
            if "llama-server" not in line or "grep" in line:
                continue
            # Extract model and port from command line
            model_match = re.search(r"--model\s+(\S+)", line)
            port_match = re.search(r"--port\s+(\d+)", line)
            pid_match = re.search(r"^\S+\s+(\d+)", line)

            if model_match and port_match and pid_match:
                inst = LlamaInstance(
                    model_file=model_match.group(1),
                    port=int(port_match.group(1)),
                    pid=int(pid_match.group(1)),
                    flash_attn="--flash-attn" in line,
                )
                # Parse other flags
                ctx_match = re.search(r"--ctx-size\s+(\d+)", line)
                if ctx_match:
                    inst.ctx_size = int(ctx_match.group(1))
                gpu_match = re.search(r"--n-gpu-layers\s+(\d+)", line)
                if gpu_match:
                    inst.gpu_layers = int(gpu_match.group(1))

                self.instances[inst.port] = inst
                instances.append(inst)

        return instances

    def health(self, port: int) -> dict:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5)
            return r.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def models(self, port: int) -> list[dict]:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            return r.json().get("data", [])
        except Exception:
            return []

    def available_models(self) -> list[dict]:
        """List GGUF files available to load."""
        models = []
        for f in self.models_dir.glob("*.gguf"):
            info = KNOWN_MODELS.get(f.name, {})
            models.append({
                "file": str(f),
                "name": f.stem,
                "size_gb": round(f.stat().st_size / (1024**3), 1),
                "memory_mb": info.get("memory_mb", int(f.stat().st_size / (1024**3) * 1100)),
                "params": info.get("params", "unknown"),
                "quant": info.get("quant", "unknown"),
                "family": info.get("family", "unknown"),
            })
        return models

    def start(
        self,
        model_file: str,
        port: int = 8000,
        ctx_size: int = 12288,
        gpu_layers: int = 99,
        threads: int = 4,
        flash_attn: bool = True,
        extra_args: Optional[list[str]] = None,
    ) -> LlamaInstance:
        """Start a new llama-server process."""
        cmd = [
            self.bin_path,
            "--model", model_file,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--ctx-size", str(ctx_size),
            "--n-gpu-layers", str(gpu_layers),
            "-np", "1",
            "-t", str(threads),
            "--jinja",
            "--reasoning-format", "none",
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
        ]
        if flash_attn:
            cmd.extend(["--flash-attn", "on"])
        if extra_args:
            cmd.extend(extra_args)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        inst = LlamaInstance(
            model_file=model_file,
            port=port,
            pid=proc.pid,
            process=proc,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers,
            threads=threads,
            flash_attn=flash_attn,
        )
        self.instances[port] = inst

        # Wait for server to be ready
        for _ in range(60):
            h = self.health(port)
            if h.get("status") == "ok":
                return inst
            time.sleep(1)

        raise RuntimeError(f"llama-server on port {port} failed to start within 60s")

    def stop(self, port: int) -> dict:
        """Stop a llama-server instance."""
        inst = self.instances.get(port)
        if not inst:
            return {"status": "not_found"}

        pid = inst.pid
        try:
            if inst.process:
                inst.process.terminate()
                inst.process.wait(timeout=10)
            elif pid:
                import os
                os.kill(pid, signal.SIGTERM)
                # Wait for process to exit
                for _ in range(20):
                    try:
                        os.kill(pid, 0)
                        time.sleep(0.5)
                    except ProcessLookupError:
                        break
        except Exception as e:
            return {"status": "error", "error": str(e)}

        del self.instances[port]
        return {"status": "stopped", "port": port, "pid": pid, "model": inst.model_name}
