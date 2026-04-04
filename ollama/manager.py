"""
Unified Model Service Manager

Manages ALL model backends on slice-test with SAMcloud resource leasing:
  - Ollama (pull/load/unload any HuggingFace or Ollama-hub model)
  - llama-server (llama.cpp - run GGUF models with full Metal acceleration)

Lifecycle:
  1. Request comes in for a model
  2. Check if already loaded -> serve directly
  3. Estimate memory, request a lease from SAMcloud
  4. Start/load the model on the appropriate backend
  5. Serve requests, track usage
  6. On cooldown (no requests for COOLDOWN_SECONDS), stop/unload and release lease
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .ollama_client import OllamaClient, estimate_memory_mb
from .llama_client import LlamaServerClient, LlamaInstance
from .samcloud import SamcloudClient

log = logging.getLogger("model-manager")

RESOURCE_ID = "slice-test/gpu-0"
SERVICE_ID = "slice-test/model-service"
COOLDOWN_SECONDS = 300  # 5 min idle before unload
LEASE_TTL = 3600  # 1 hour default lease
LEASE_RENEW_AT = 0.5  # renew when 50% of TTL has elapsed
OLLAMA_KEEP_ALIVE = -1  # indefinite — our lease system manages memory, not Ollama's timer


class Backend(str, Enum):
    OLLAMA = "ollama"
    LLAMA = "llama-server"


@dataclass
class ManagedModel:
    """Tracks a model that's loaded with an active lease."""
    name: str
    backend: Backend
    memory_mb: int
    lease_id: Optional[str]
    port: int  # where to reach this model
    loaded_at: float
    last_used: float
    request_count: int = 0
    managed: bool = True  # False = pre-existing process we adopted
    llama_instance: Optional[LlamaInstance] = field(default=None, repr=False)


@dataclass
class ModelManager:
    sc: SamcloudClient
    ollama: OllamaClient = field(default_factory=OllamaClient)
    llama: LlamaServerClient = field(default_factory=LlamaServerClient)
    models: dict[str, ManagedModel] = field(default_factory=dict)
    _cooldown_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _health_task: Optional[asyncio.Task] = field(default=None, repr=False)

    def discover(self) -> list[ManagedModel]:
        """Discover and adopt already-running model processes."""
        adopted = []

        # Discover llama-server instances
        for inst in self.llama.discover_running():
            h = self.llama.health(inst.port)
            if h.get("status") != "ok":
                continue
            name = inst.model_name
            mm = ManagedModel(
                name=name,
                backend=Backend.LLAMA,
                memory_mb=inst.memory_mb,
                lease_id=None,
                port=inst.port,
                loaded_at=time.time(),
                last_used=time.time(),
                managed=False,
                llama_instance=inst,
            )
            self.models[name] = mm
            adopted.append(mm)
            log.info(f"Adopted llama-server: {name} on port {inst.port} (~{inst.memory_mb}MB)")

        # Discover running Ollama models
        for m in self.ollama.list_running():
            name = m.get("name", "unknown")
            size_mb = int(m.get("size", 0) / 1024 / 1024)
            if name not in self.models:
                # Pin the model so Ollama doesn't unload it on its own timer
                try:
                    self.ollama.load_model(name, keep_alive=OLLAMA_KEEP_ALIVE)
                    log.info(f"Pinned Ollama model {name} (keep_alive={OLLAMA_KEEP_ALIVE})")
                except Exception as e:
                    log.warning(f"Failed to pin {name}: {e}")
                mm = ManagedModel(
                    name=name,
                    backend=Backend.OLLAMA,
                    memory_mb=size_mb or estimate_memory_mb(name),
                    lease_id=None,
                    port=11434,
                    loaded_at=time.time(),
                    last_used=time.time(),
                    managed=False,
                )
                self.models[name] = mm
                adopted.append(mm)
                log.info(f"Adopted Ollama model: {name} (~{mm.memory_mb}MB)")

        return adopted

    def claim_leases(self) -> list[dict]:
        """Request leases for all models that don't have one."""
        results = []
        for name, mm in self.models.items():
            if mm.lease_id:
                continue
            try:
                lease_resp = self.sc.request_lease(
                    resource_id=RESOURCE_ID,
                    service_id=SERVICE_ID,
                    memory_mb=mm.memory_mb,
                    purpose=f"model:{name}",
                    ttl_seconds=LEASE_TTL,
                )
                lease_id = lease_resp.get("id") or lease_resp.get("lease_id")
                mm.lease_id = str(lease_id) if lease_id else None
                results.append({"model": name, "lease_id": mm.lease_id, "memory_mb": mm.memory_mb})
                log.info(f"Lease claimed for {name}: {mm.lease_id} ({mm.memory_mb}MB)")
            except Exception as e:
                log.warning(f"Failed to claim lease for {name}: {e}")
                results.append({"model": name, "error": str(e)})
        return results

    def status(self) -> dict:
        """Full manager status."""
        return {
            "backends": {
                "ollama": self.ollama.version(),
                "llama_server": {
                    port: {
                        "model": inst.model_name,
                        "pid": inst.pid,
                        "health": self.llama.health(port),
                    }
                    for port, inst in self.llama.instances.items()
                },
            },
            "models": {
                name: {
                    "backend": mm.backend.value,
                    "port": mm.port,
                    "memory_mb": mm.memory_mb,
                    "lease_id": mm.lease_id,
                    "loaded_at": mm.loaded_at,
                    "last_used": mm.last_used,
                    "idle_seconds": int(time.time() - mm.last_used),
                    "request_count": mm.request_count,
                    "managed": mm.managed,
                }
                for name, mm in self.models.items()
            },
            "available": {
                "ollama": [m["name"] for m in self.ollama.list_models()],
                "gguf": [m["name"] for m in self.llama.available_models()],
            },
            "resource": self._get_resource_summary(),
            "leases": self._get_active_leases(),
        }

    def _get_resource_summary(self) -> dict:
        try:
            dash = self.sc.resource_dashboard()
            for r in dash:
                if r["id"] == RESOURCE_ID:
                    return {
                        "memory_pct": r["memory_pct"],
                        "compute_pct": r["compute_pct"],
                        "health": r["health"],
                        "available_memory_mb": r["available_memory_mb"],
                    }
        except Exception as e:
            log.warning(f"Resource dashboard error: {e}")
        return {}

    def _get_active_leases(self) -> list[dict]:
        try:
            return self.sc.list_leases(resource=RESOURCE_ID, status="active")
        except Exception:
            return []

    # -- Ollama model operations --

    def load_ollama_model(self, model_name: str) -> ManagedModel:
        """Load an Ollama model with lease management."""
        if model_name in self.models:
            mm = self.models[model_name]
            mm.last_used = time.time()
            mm.request_count += 1
            return mm

        memory_mb = estimate_memory_mb(model_name)

        # Unload other Ollama models first to prevent Ollama from
        # silently evicting them (Ollama evicts to fit new models,
        # ignoring keep_alive). We explicitly unload + release leases.
        ollama_models = [
            name for name, mm in self.models.items()
            if mm.backend == Backend.OLLAMA
        ]
        for existing in ollama_models:
            log.info(f"Unloading {existing} to make room for {model_name}")
            self.unload(existing, force=True)

        # Pull if needed
        local = [m["name"] for m in self.ollama.list_models()]
        if model_name not in local and f"{model_name}:latest" not in local:
            log.info(f"Pulling {model_name}...")
            for progress in self.ollama.pull_model(model_name):
                s = progress.get("status", "")
                if s == "success":
                    log.info(f"Pull complete: {model_name}")

        # Request lease
        lease_id = self._request_lease(model_name, memory_mb)

        # Load with indefinite keep_alive — our lease system manages memory
        log.info(f"Loading {model_name} into Ollama (keep_alive={OLLAMA_KEEP_ALIVE})...")
        self.ollama.load_model(model_name, keep_alive=OLLAMA_KEEP_ALIVE)

        # Get actual VRAM from Ollama ps
        actual_mb = memory_mb
        for m in self.ollama.list_running():
            if model_name in m.get("name", ""):
                reported = int(m.get("size", 0) / 1024 / 1024)
                if reported > 0:
                    actual_mb = reported
                    log.info(f"Actual VRAM for {model_name}: {actual_mb}MB (estimated {memory_mb}MB)")
                break

        now = time.time()
        mm = ManagedModel(
            name=model_name,
            backend=Backend.OLLAMA,
            memory_mb=actual_mb,
            lease_id=lease_id,
            port=11434,
            loaded_at=now,
            last_used=now,
            request_count=1,
            managed=True,
        )
        self.models[model_name] = mm
        return mm

    # -- llama-server model operations --

    def load_llama_model(
        self, model_file: str, port: int = 8000, **kwargs
    ) -> ManagedModel:
        """Start a llama-server instance with lease management."""
        from pathlib import Path
        name = Path(model_file).stem

        if name in self.models:
            mm = self.models[name]
            mm.last_used = time.time()
            mm.request_count += 1
            return mm

        # Check if port is already in use by an adopted instance
        if port in self.llama.instances:
            inst = self.llama.instances[port]
            if inst.model_name == name:
                mm = self.models.get(name)
                if mm:
                    mm.last_used = time.time()
                    return mm

        # Estimate memory from known models or file size
        from .llama_client import KNOWN_MODELS
        basename = Path(model_file).name
        memory_mb = KNOWN_MODELS.get(basename, {}).get("memory_mb", 8000)

        # Request lease
        lease_id = self._request_lease(name, memory_mb)

        # Start server
        log.info(f"Starting llama-server for {name} on port {port}...")
        inst = self.llama.start(model_file, port=port, **kwargs)

        now = time.time()
        mm = ManagedModel(
            name=name,
            backend=Backend.LLAMA,
            memory_mb=memory_mb,
            lease_id=lease_id,
            port=port,
            loaded_at=now,
            last_used=now,
            request_count=1,
            managed=True,
            llama_instance=inst,
        )
        self.models[name] = mm
        return mm

    # -- Common operations --

    def _request_lease(self, model_name: str, memory_mb: int) -> Optional[str]:
        try:
            resp = self.sc.request_lease(
                resource_id=RESOURCE_ID,
                service_id=SERVICE_ID,
                memory_mb=memory_mb,
                purpose=f"model:{model_name}",
                ttl_seconds=LEASE_TTL,
            )
            lease_id = resp.get("id") or resp.get("lease_id")
            # Validate that the lease has an expiry — reject indefinite leases
            if resp.get("expires_at") is None:
                log.warning(f"Lease {lease_id} granted without expiry — will rely on renewal loop to keep it bounded")
            log.info(f"Lease for {model_name}: {lease_id} ({memory_mb}MB, TTL={LEASE_TTL}s)")
            return str(lease_id) if lease_id else None
        except Exception as e:
            log.warning(f"Lease request failed for {model_name}: {e}")
            return None

    def touch(self, model_name: str):
        """Mark a model as recently used (resets cooldown timer)."""
        if model_name in self.models:
            self.models[model_name].last_used = time.time()
            self.models[model_name].request_count += 1

    def ensure_running(self, model_name: str) -> bool:
        """Check if an Ollama model is actually running. Reload if dropped."""
        if model_name not in self.models:
            return False
        mm = self.models[model_name]
        if mm.backend != Backend.OLLAMA:
            return True  # llama-server managed separately
        # Check if Ollama still has it loaded
        running = [m.get("name", "") for m in self.ollama.list_running()]
        if mm.name in running or any(mm.name in r for r in running):
            return True
        # Model was dropped by Ollama — reload it
        log.warning(f"Model {mm.name} dropped by Ollama — reloading (keep_alive={OLLAMA_KEEP_ALIVE})")
        try:
            self.ollama.load_model(mm.name, keep_alive=OLLAMA_KEEP_ALIVE)
            log.info(f"Reloaded {mm.name}")
            return True
        except Exception as e:
            log.error(f"Failed to reload {mm.name}: {e}")
            return False

    def unload(self, model_name: str, force: bool = False) -> dict:
        """Unload/stop a model and release its lease."""
        if model_name not in self.models:
            return {"status": "not_found"}

        mm = self.models[model_name]

        # Don't auto-unload adopted processes unless forced
        if not mm.managed and not force:
            return {"status": "skipped", "reason": "adopted process - use force=True"}

        # Stop the backend
        if mm.backend == Backend.OLLAMA:
            try:
                self.ollama.unload_model(mm.name)
            except Exception as e:
                log.warning(f"Ollama unload error: {e}")
        elif mm.backend == Backend.LLAMA:
            result = self.llama.stop(mm.port)
            log.info(f"Stopped llama-server: {result}")

        # Release lease
        if mm.lease_id:
            try:
                self.sc.release_lease(mm.lease_id)
                log.info(f"Released lease {mm.lease_id}")
            except Exception as e:
                log.warning(f"Lease release error: {e}")

        del self.models[model_name]
        return {
            "status": "unloaded",
            "model": model_name,
            "backend": mm.backend.value,
            "memory_freed_mb": mm.memory_mb,
            "served_requests": mm.request_count,
            "uptime_seconds": int(time.time() - mm.loaded_at),
        }

    def check_cooldowns(self) -> list[dict]:
        """Unload idle models past cooldown."""
        now = time.time()
        results = []
        to_unload = [
            name
            for name, mm in self.models.items()
            if mm.managed and (now - mm.last_used) > COOLDOWN_SECONDS
        ]
        for name in to_unload:
            idle = int(now - self.models[name].last_used)
            log.info(f"{name} idle {idle}s (>{COOLDOWN_SECONDS}s), unloading...")
            results.append(self.unload(name))
        return results

    async def cooldown_loop(self):
        while True:
            await asyncio.sleep(60)
            try:
                self.check_cooldowns()
            except Exception as e:
                log.warning(f"Cooldown error: {e}")

    async def health_loop(self):
        """Report health to SAMcloud every 60s."""
        while True:
            try:
                self.sc.report_health(SERVICE_ID)
            except Exception as e:
                log.warning(f"Health report error: {e}")
            await asyncio.sleep(60)

    async def lease_renewal_loop(self):
        """Renew leases before they expire. Runs every LEASE_TTL * LEASE_RENEW_AT seconds."""
        interval = int(LEASE_TTL * LEASE_RENEW_AT)
        while True:
            await asyncio.sleep(interval)
            try:
                self._renew_leases()
            except Exception as e:
                log.warning(f"Lease renewal error: {e}")

    def _renew_leases(self):
        """Release and re-request leases to prevent expiry."""
        for name, mm in list(self.models.items()):
            if not mm.lease_id:
                continue
            try:
                self.sc.release_lease(mm.lease_id)
                new_id = self._request_lease(name, mm.memory_mb)
                mm.lease_id = new_id
                log.info(f"Renewed lease for {name}: {new_id}")
            except Exception as e:
                log.warning(f"Failed to renew lease for {name}: {e}")

    def start_background_tasks(self):
        if self._cooldown_task is None or self._cooldown_task.done():
            self._cooldown_task = asyncio.create_task(self.cooldown_loop())
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self.health_loop())
        self._renewal_task = asyncio.create_task(self.lease_renewal_loop())
        log.info("Background tasks started (cooldown, health, lease renewal)")

    def shutdown(self) -> list[dict]:
        """Release all leases. Only stop processes we started (managed=True)."""
        results = []
        for name in list(self.models.keys()):
            mm = self.models[name]
            if mm.managed:
                results.append(self.unload(name, force=True))
            elif mm.lease_id:
                # Release lease but don't kill adopted processes
                try:
                    self.sc.release_lease(mm.lease_id)
                    results.append({"model": name, "status": "lease_released", "process": "kept"})
                except Exception:
                    pass
        for task in [self._cooldown_task, self._health_task, getattr(self, '_renewal_task', None)]:
            if task and not task.done():
                task.cancel()
        return results
