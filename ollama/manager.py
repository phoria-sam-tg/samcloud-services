"""
Unified Model Service Manager

Manages ALL model backends on slice-test with SAMcloud resource leasing:
  - Ollama (pull/load/unload any HuggingFace or Ollama-hub model)
  - llama-server (llama.cpp - run GGUF models with full Metal acceleration)
  - mlx-vlm (vision-language models on MLX - Gemma 4, Qwen2.5-VL, etc.)

Lifecycle:
  1. Request comes in for a model
  2. Check if already loaded -> serve directly
  3. Estimate memory, request a lease from SAMcloud
  4. Start/load the model on the appropriate backend
  5. Serve requests, track usage
  6. On cooldown (no requests for COOLDOWN_SECONDS), stop/unload and release lease
"""

import asyncio
import os
import signal
import subprocess
import time
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from . import capacity
from . import config
from .ollama_client import OllamaClient, estimate_memory_mb
from .exo_client import ExoClient, ExoUnavailable
from .llama_client import LlamaServerClient, LlamaInstance
from .samcloud import SamcloudClient

log = logging.getLogger("model-manager")

RESOURCE_ID = config.SC_RESOURCE_ID
SERVICE_ID = config.SC_SERVICE_ID
COOLDOWN_SECONDS = config.COOLDOWN_SECONDS
LEASE_TTL = config.LEASE_TTL
LEASE_RENEW_AT = config.LEASE_RENEW_AT
OLLAMA_KEEP_ALIVE = config.OLLAMA_KEEP_ALIVE


class Backend(str, Enum):
    OLLAMA = "ollama"
    LLAMA = "llama-server"
    VLM = "mlx-vlm"
    # The pool. Unlike the three above, the gateway does not own the process
    # and does not place the model — it holds an exclusive lease around a
    # generation and proxies. See exo_client.py.
    EXO = "exo"

VLM_PORT = config.VLM_PORT
VLM_HOST = config.VLM_HOST
VLM_PYTHON = config.VLM_PYTHON
VLM_STARTUP_TIMEOUT = config.VLM_STARTUP_TIMEOUT
VLM_MODELS = {
    "gemma-4": {"default": "mlx-community/gemma-4-31b-it-nvfp4", "memory_mb": 18700},
    "qwen2.5-vl": {"default": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit", "memory_mb": 5700},
}
VLM_DEFAULT_MEMORY_MB = 18700

EXO_RESOURCE_ID = config.EXO_RESOURCE_ID
EXO_TIERS = config.EXO_TIERS
EXO_LEASE_TTL = config.EXO_LEASE_TTL


def match_exo_tier(model_name: str) -> bool:
    """Is this request asking for the pool?

    Deliberately an exact, case-insensitive match on a tier name and nothing
    else. Every other matcher in this file does substring matching so callers
    can say "agi" or "gemma-4" — that is right when a name identifies a file we
    can load on demand, and wrong here. The pool holds one model at a time and
    a swap costs 30s-10min, so a request naming a model is a request we cannot
    honour; only "give me the think tier, whatever is in it" is honest. Fuzzy
    matching would also let any request containing the substring "think" fall
    through to a 64 GB pool by accident.
    """
    return model_name.strip().lower() in EXO_TIERS


def match_vlm_model(model_name: str):
    """Map a requested name to a known VLM. Returns (resolved_id, memory_mb) or None.

    Used by both the load path and the gateway's auto-load resolver to decide
    whether a request should be served by an mlx-vlm process.
    """
    lower = model_name.lower()
    for prefix, info in VLM_MODELS.items():
        default = info["default"].lower()
        if prefix in lower or lower in default or default in lower:
            return info["default"], info["memory_mb"]
    # Heuristic fallback for vision-language model names we don't have pinned.
    if "-vl" in lower or "vlm" in lower or "vision" in lower:
        return model_name, VLM_DEFAULT_MEMORY_MB
    return None


def match_gguf_model(model_name: str, available: list[dict]):
    """Map a requested short name to a known local GGUF file.

    `available` is LlamaServerClient.available_models() — the caller passes
    it in so this stays a pure function like match_vlm_model. Lets
    /models/load and the auto-load resolver accept "agi" instead of the
    exact qwen2.5-32b-agi-q4_k_m.gguf filename.
    """
    lower = model_name.lower()
    for m in available:
        name = m["name"].lower()
        if lower in name or name in lower:
            return m
    return None


@dataclass
class LeaseOutcome:
    """The verdict on a lease request: was it granted, and if not, why not.

    Exists so the answer cannot collapse back to a truthy lease id. A queued
    lease has an id too, which is exactly how "202 reads as granted" survived
    — the old code returned `str(lease_id)` and every caller tested it for
    truthiness. Callers now have to look at `granted`.
    """
    granted: bool
    state: str                                  # active | queued | conflict | error
    lease_id: Optional[str] = None
    held_by: Optional[str] = None
    expires_at: Optional[str] = None
    queue_position: Optional[int] = None
    message: Optional[str] = None


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
    # Set only for Backend.EXO, where the dict key is the *tier* a caller asks
    # for ("think") while `name` is whatever model the pool currently holds.
    # Every other backend keys on the model name itself.
    tier: Optional[str] = None
    llama_instance: Optional[LlamaInstance] = field(default=None, repr=False)
    vlm_process: Optional[subprocess.Popen] = field(default=None, repr=False)


@dataclass
class ModelManager:
    sc: SamcloudClient
    ollama: OllamaClient = field(default_factory=OllamaClient)
    llama: LlamaServerClient = field(default_factory=LlamaServerClient)
    exo: ExoClient = field(default_factory=ExoClient)
    models: dict[str, ManagedModel] = field(default_factory=dict)
    # Pool leases currently held by this process, so shutdown can give back
    # what a killed request did not. A set rather than a single slot: the
    # registry is the thing that enforces one-at-a-time, and this bookkeeping
    # should not be the component that quietly assumes it.
    _pool_leases: set = field(default_factory=set, repr=False)
    _pool_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cooldown_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _health_task: Optional[asyncio.Task] = field(default=None, repr=False)

    def discover(self) -> list[ManagedModel]:
        """Discover and adopt already-running model processes.

        Each backend is probed independently — if ollama isn't running we
        still want to pick up llama-server and mlx-vlm. No backend being up
        is a valid state for a fresh service start.
        """
        adopted = []

        # Discover llama-server instances
        try:
            instances = self.llama.discover_running()
        except Exception as e:
            log.info(f"llama-server discovery skipped: {e}")
            instances = []
        for inst in instances:
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
        try:
            running = self.ollama.list_running()
        except Exception as e:
            log.info(f"ollama not reachable, skipping discovery: {e}")
            running = []
        for m in running:
            name = m.get("name", "unknown")
            size_mb = int(m.get("size", 0) / 1024 / 1024)
            if name not in self.models:
                # Re-apply the configured keep_alive so the model honours the
                # idle timer (OLLAMA_KEEP_ALIVE). Previously this pinned -1 and
                # marked the model managed=False, which exempted it from cooldown
                # (check_cooldowns / unload both skip non-managed) — so an
                # on-demand load became stuck in memory forever and was re-pinned
                # on every restart. Adopt it as a managed Ollama model instead so
                # the cooldown loop can spin it back down when idle (ticket #97).
                try:
                    self.ollama.load_model(name, keep_alive=OLLAMA_KEEP_ALIVE)
                    log.info(f"Re-applied keep_alive={OLLAMA_KEEP_ALIVE} to adopted Ollama model {name}")
                except Exception as e:
                    log.warning(f"Failed to set keep_alive on {name}: {e}")
                mm = ManagedModel(
                    name=name,
                    backend=Backend.OLLAMA,
                    memory_mb=size_mb or estimate_memory_mb(name),
                    lease_id=None,
                    port=11434,
                    loaded_at=time.time(),
                    last_used=time.time(),
                    managed=True,
                )
                self.models[name] = mm
                adopted.append(mm)
                log.info(f"Adopted Ollama model: {name} (~{mm.memory_mb}MB)")

        # mlx-vlm is NOT adopted: the gateway owns its lifecycle and spins it
        # up on demand (see load_vlm_model). Any mlx-vlm left running from a
        # previous gateway is a stray — kill it so we always start from a clean,
        # owned state and never route to a process we can't tear down.
        self._kill_stray_vlm()

        return adopted

    def claim_leases(self) -> list[dict]:
        """Request leases for all models that don't have one."""
        results = []
        for name, mm in self.models.items():
            if mm.lease_id:
                continue
            if mm.backend == Backend.EXO:
                # The pool is never leased for residency — its lease is taken
                # per generation and released after, and it is exclusive, so
                # claiming one here would close the pool for as long as the
                # gateway runs.
                continue
            try:
                lease_resp = self.sc.request_lease(
                    resource_id=RESOURCE_ID,
                    service_id=SERVICE_ID,
                    memory_mb=mm.memory_mb,
                    ttl_seconds=LEASE_TTL,
                )
            except Exception as e:
                log.warning(f"Failed to claim lease for {name}: {e}")
                results.append({"model": name, "error": str(e)})
                continue

            # Same rule as _request_lease: queued is not granted. This path had
            # the identical defect — it read an id out of any response that did
            # not raise — and it runs over every adopted model at startup.
            outcome = self._lease_outcome(lease_resp)
            if outcome.granted:
                mm.lease_id = outcome.lease_id
                results.append({"model": name, "lease_id": mm.lease_id,
                                "memory_mb": mm.memory_mb})
                log.info(f"Lease claimed for {name}: {mm.lease_id} ({mm.memory_mb}MB)")
            else:
                if outcome.state == "queued":
                    self._release_lease_quietly(outcome.lease_id)
                mm.lease_id = None
                log.warning(
                    f"Lease NOT claimed for {name} ({outcome.state}): {outcome.message}"
                )
                results.append({"model": name, "lease": outcome.state,
                                "detail": outcome.message})
        return results

    def status(self) -> dict:
        """Full manager status."""
        try:
            ollama_ver = self.ollama.version()
        except Exception as e:
            ollama_ver = f"unreachable: {e}"
        return {
            "backends": {
                "ollama": ollama_ver,
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
                "ollama": self._safe_ollama_list(),
                "gguf": [m["name"] for m in self.llama.available_models()],
            },
            "resource": self._get_resource_summary(),
            "leases": self._get_active_leases(),
        }

    def _safe_ollama_list(self) -> list[str]:
        try:
            return [m["name"] for m in self.ollama.list_models()]
        except Exception:
            return []

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
            # resource_id=, not resource= — the plane silently ignores unknown
            # query filters, so `resource=` returned every active lease on the
            # fleet and /status named another box's exclusive lease as this
            # GPU's. An unknown field in a POST body is a 422; an unknown filter
            # in a GET is the whole table. See #774.
            return self.sc.list_leases(resource_id=RESOURCE_ID, status="active")
        except Exception:
            return []

    # -- Ollama model operations --

    def catalogue_mb(self) -> dict:
        """Every local Ollama model mapped to its weight size in MB.

        The fit decision and the advertised offering both derive from this, so
        `offering:full` keeps meaning "the biggest model we hold fits" even as
        the catalogue changes — fixed MB bands go stale the moment it does.
        """
        out = {}
        try:
            for m in self.ollama.list_models():
                name = m.get("name", "")
                size = int(m.get("size", 0))
                if name and size:
                    out[name] = round(size / 1024 / 1024)
        except Exception as e:
            log.warning(f"Could not read model catalogue: {e}")
        return out

    def load_ollama_model(self, model_name: str) -> ManagedModel:
        """Load an Ollama model with lease management."""
        if model_name in self.models:
            mm = self.models[model_name]
            mm.last_used = time.time()
            mm.request_count += 1
            return mm


        # Fit against what this box can hand over WITHOUT swapping, instead of
        # evicting whatever is resident to force-fit the ask. The contract is
        # "20GB used by us, 5GB left, so I can offer the small model" — the
        # caller picks from what fits rather than us making room by force.
        # Ollama would otherwise silently evict to fit, ignoring keep_alive.
        need_mb = self.ollama.memory_estimate_mb(model_name)
        avail_mb = capacity.collect().get("memory_available_mb", 0)

        if not capacity.fits(need_mb, avail_mb):
            resident = [
                name for name, mm in self.models.items()
                if mm.backend == Backend.OLLAMA
            ]
            reclaimable = sum(self.models[n].memory_mb or 0 for n in resident)
            if config.AUTO_EVICT and capacity.fits(need_mb, avail_mb + reclaimable):
                for existing in resident:
                    log.info(f"Unloading {existing} to make room for {model_name}")
                    self.unload(existing, force=True)
            else:
                fitting = capacity.servable(self.catalogue_mb(), avail_mb)
                detail = (
                    f"{model_name} needs ~{need_mb}MB, but only "
                    f"{capacity.usable_mb(avail_mb)}MB is on offer "
                    f"({int(capacity.USABLE_FRACTION * 100)}% of the {avail_mb}MB "
                    f"free right now)"
                )
                if resident:
                    detail += (
                        f" ({reclaimable}MB held by {', '.join(resident)}, "
                        f"released on idle)"
                    )
                detail += (
                    f". Fits right now: {', '.join(fitting)}" if fitting
                    else ". Nothing in the catalogue fits right now."
                )
                log.info(f"Refusing load: {detail}")
                raise capacity.InsufficientCapacity(
                    detail,
                    need_mb=need_mb,
                    usable_mb=capacity.usable_mb(avail_mb),
                    available_mb=avail_mb,
                    fits_now=fitting,
                )

        # Pull if needed
        local = [m["name"] for m in self.ollama.list_models()]
        if model_name not in local and f"{model_name}:latest" not in local:
            log.info(f"Pulling {model_name}...")
            for progress in self.ollama.pull_model(model_name):
                s = progress.get("status", "")
                if s == "success":
                    log.info(f"Pull complete: {model_name}")

        # Re-read the size now the weights are local — the gate above may
        # have run against a name-guess for a model we had not pulled yet.
        memory_mb = self.ollama.memory_estimate_mb(model_name)

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

    # -- mlx-vlm model operations --

    def _kill_stray_vlm(self):
        """Terminate any mlx-vlm server process the gateway doesn't own.

        Frees VLM_PORT and guarantees we never route to an un-owned process.
        Processes we currently track (managed VLMs) are left alone.
        """
        owned = {
            mm.vlm_process.pid
            for mm in self.models.values()
            if mm.backend == Backend.VLM and mm.vlm_process
        }
        try:
            out = subprocess.run(
                ["pgrep", "-f", "mlx_vlm.server"],
                capture_output=True, text=True,
            )
        except Exception as e:
            log.warning(f"Could not scan for stray mlx-vlm processes: {e}")
            return
        for line in out.stdout.split():
            try:
                pid = int(line)
            except ValueError:
                continue
            if pid in owned or pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                log.info(f"Killed stray mlx-vlm process {pid}")
            except ProcessLookupError:
                pass
            except Exception as e:
                log.warning(f"Failed to kill stray mlx-vlm process {pid}: {e}")

    def load_vlm_model(self, model_name: str) -> ManagedModel:
        """Start an mlx-vlm server for a vision-language model on demand.

        The gateway owns the process: it spawns mlx_vlm.server, leases memory,
        and tears both down on cooldown/unload. Only one VLM runs at a time on
        VLM_PORT — a request for a different VLM swaps the current one out.
        """
        import httpx

        match = match_vlm_model(model_name)
        if match is None:
            raise RuntimeError(f"{model_name} is not a recognised mlx-vlm model")
        resolved, memory_mb = match

        if resolved in self.models:
            mm = self.models[resolved]
            mm.last_used = time.time()
            mm.request_count += 1
            return mm

        # Single VLM per port — evict any other VLM first.
        for existing, mm in list(self.models.items()):
            if mm.backend == Backend.VLM:
                log.info(f"Unloading VLM {existing} to make room for {resolved}")
                self.unload(existing, force=True)

        # Ensure the port is clear of any un-owned process.
        self._kill_stray_vlm()

        lease_id = self._request_lease(resolved, memory_mb)

        log.info(f"Starting mlx-vlm server for {resolved} on {VLM_HOST}:{VLM_PORT}...")
        proc = subprocess.Popen(
            [
                VLM_PYTHON, "-m", "mlx_vlm.server",
                "--model", resolved,
                "--port", str(VLM_PORT),
                "--host", VLM_HOST,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        base = f"http://{VLM_HOST}:{VLM_PORT}"
        ready = False
        for _ in range(VLM_STARTUP_TIMEOUT):
            if proc.poll() is not None:
                self._release_lease_quietly(lease_id)
                raise RuntimeError(
                    f"mlx-vlm server for {resolved} exited early (code {proc.returncode})"
                )
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if not ready:
            proc.terminate()
            self._release_lease_quietly(lease_id)
            raise RuntimeError(
                f"mlx-vlm server for {resolved} did not become healthy in {VLM_STARTUP_TIMEOUT}s"
            )

        now = time.time()
        mm = ManagedModel(
            name=resolved,
            backend=Backend.VLM,
            memory_mb=memory_mb,
            lease_id=lease_id,
            port=VLM_PORT,
            loaded_at=now,
            last_used=now,
            request_count=1,
            managed=True,
            vlm_process=proc,
        )
        self.models[resolved] = mm
        return mm

    # -- the pool (Backend.EXO) --

    def resolve_exo_tier(self, tier: str) -> ManagedModel:
        """Register or refresh the pool tier. Takes no lease and loads nothing.

        The asymmetry with every other `load_*` method here is the point. For
        Ollama, llama-server and mlx-vlm, "resolve" means *make it resident* and
        the lease covers that residency. The pool is already resident, placed
        out of band, with its pages already wired and already counted by each
        box's own capacity gate — so there is nothing to load and nothing to
        lease for. The lease that matters is taken later, around the generation,
        by `pool_lease()`.

        The resident model is re-read from the pool on every resolve rather
        than cached, so an operator swapping the model is picked up without a
        deploy here. That is also why `memory_mb` is 0: claiming a figure would
        double-count pages the box already sees, and a lease written from a
        pre-load estimate was measured 4.7x wrong.
        """
        if not config.EXO_ENABLED:
            raise ExoUnavailable("exo backend is disabled (EXO_ENABLED=0)")

        status = self.exo.pool_status()        # raises ExoUnavailable if down
        resident = status["resident_model"]
        if not resident:
            # Quote the runner states. Mid-swap these carry layer progress, so
            # a caller learns "23/47 layers" instead of an opaque refusal on a
            # wait that can run to ten minutes.
            detail = "; ".join(
                f"{i['model']}: {', '.join(i['runners'].values())}"
                for i in status["instances"]
            ) or "no instance placed"
            raise ExoUnavailable(
                f"the exo pool at {config.EXO_BASE} is reachable but has no "
                f"model resident and ready — it is most likely mid-swap, or was "
                f"never placed after a reboot (it cannot restart itself). "
                f"Runners: {detail}"
            )

        now = time.time()
        existing = self.models.get(tier)
        if existing is not None and existing.backend == Backend.EXO:
            if existing.name != resident:
                log.info(
                    f"pool resident model changed: {existing.name} -> {resident} "
                    f"(tier '{tier}' now serves {resident})"
                )
                existing.name = resident
            existing.last_used = now
            existing.request_count += 1
            return existing

        mm = ManagedModel(
            name=resident,
            backend=Backend.EXO,
            memory_mb=0,          # not ours to account for; see docstring
            lease_id=None,        # per-generation, not per-residency
            port=0,               # not a local port; reached over the network
            loaded_at=now,
            last_used=now,
            request_count=1,
            # We did not start this process and must never stop it. managed=False
            # also keeps the cooldown loop from trying to unload the pool after
            # five idle minutes, which it has no right to do.
            managed=False,
            tier=tier,
        )
        self.models[tier] = mm
        log.info(f"Pool tier '{tier}' -> {resident} (exclusive lease per request)")
        return mm

    def exo_request_defaults(self, model_id: str) -> dict:
        """Per-model request defaults for the pool, matched on the model id.

        Same shape as the Ollama `think:false` workaround: a small table of
        things a given model needs that the caller should not have to know. On
        an exclusive resource the `max_tokens` cap is not a nicety — an
        unbounded generation owns the pool until it stops talking.
        """
        lower = (model_id or "").lower()
        defaults = {"max_tokens": config.EXO_MAX_TOKENS}
        for key, overrides in config.EXO_MODEL_DEFAULTS.items():
            if key in lower:
                defaults.update(overrides)
        return defaults

    def _release_lease_quietly(self, lease_id: Optional[str]):
        if not lease_id:
            return
        try:
            self.sc.release_lease(lease_id)
        except Exception as e:
            log.warning(f"Lease release error: {e}")

    # -- Common operations --

    def _lease_outcome(self, resp: dict) -> "LeaseOutcome":
        """Read a lease response into a verdict. The load-bearing half.

        This used to return `str(lease_id)` for anything that did not raise,
        which meant a **queued** lease read as a granted one. On a shared,
        byte-metered resource that is close to harmless — the load proceeds and
        the registry's accounting is a little optimistic. On an *exclusive*
        resource it is the whole bug: "you are second in line" and "the pool is
        yours" are opposite answers, and acting on the wrong one puts two
        consumers inside one inference instance.

        The verdict cannot come from the status code alone. The API index
        documents `201 granted / 202 queued`, but the registry's handler sets
        no status code on either path, so **both** return a plain `200` — as
        the measurement on #748 recorded (`grant 200`). Code-only logic would
        therefore either accept everything (today's bug) or, if written to the
        documentation, reject every grant — the same bug pointing the other
        way. So: treat any non-2xx as a refusal, and among 2xx let the body
        decide. `status: "queued"` or a `queue_position` means not ours.
        """
        code = resp.get("status_code", 0)
        lease_id = resp.get("id") or resp.get("lease_id")
        lease_id = str(lease_id) if lease_id else None
        status = (resp.get("status") or "").lower()
        queue_position = resp.get("queue_position")

        # A 409's body is FastAPI-wrapped: {"detail": {"detail", "held_by",
        # "expires_at"}}. Flatten so the holder's expiry is reachable either way.
        detail = resp.get("detail")
        if isinstance(detail, dict):
            held_by = detail.get("held_by")
            expires_at = detail.get("expires_at") or resp.get("expires_at")
            message = detail.get("detail") or str(detail)
        else:
            held_by = resp.get("service_id")
            expires_at = resp.get("expires_at")
            message = detail if isinstance(detail, str) else None

        if code == 409:
            return LeaseOutcome(
                granted=False, state="conflict", lease_id=None,
                held_by=held_by, expires_at=expires_at,
                queue_position=queue_position,
                message=message or "resource is held exclusively",
            )
        if code not in (200, 201, 202):
            return LeaseOutcome(
                granted=False, state="error", lease_id=None,
                message=message or f"lease request returned {code}",
            )
        if status == "queued" or queue_position is not None:
            return LeaseOutcome(
                granted=False, state="queued", lease_id=lease_id,
                queue_position=queue_position,
                message=message or "lease queued, not granted",
            )
        if not lease_id:
            # 2xx with no identifier is not something to proceed on.
            return LeaseOutcome(
                granted=False, state="error", lease_id=None,
                message="lease response carried no id",
            )
        return LeaseOutcome(
            granted=True, state="active", lease_id=lease_id,
            expires_at=resp.get("expires_at"), held_by=resp.get("service_id"),
        )

    def _request_lease(self, model_name: str, memory_mb: int) -> Optional[str]:
        """Lease memory for a model on the shared GPU resource.

        Returns a lease id only when one was actually granted. A queued lease
        is released rather than kept: we are not going to wait for it, and a
        row left sitting in `queued` is renewed by the renewal loop and
        released by nobody. The load itself still proceeds — the capacity gate
        upstream has already decided the model fits this box, and refusing on
        a registry byte-count disagreement would be a regression — but it
        proceeds honestly unleased instead of recording a lease it does not
        hold.
        """
        try:
            resp = self.sc.request_lease(
                resource_id=RESOURCE_ID,
                service_id=SERVICE_ID,
                memory_mb=memory_mb,
                ttl_seconds=LEASE_TTL,
            )
        except Exception as e:
            log.warning(f"Lease request failed for {model_name}: {e}")
            return None

        outcome = self._lease_outcome(resp)
        if outcome.granted:
            if outcome.expires_at is None:
                log.warning(
                    f"Lease {outcome.lease_id} granted without expiry — relying "
                    f"on the renewal loop to keep it bounded"
                )
            log.info(
                f"Lease for {model_name}: {outcome.lease_id} "
                f"({memory_mb}MB, TTL={LEASE_TTL}s)"
            )
            return outcome.lease_id

        if outcome.state == "queued":
            log.warning(
                f"Lease for {model_name} was QUEUED, not granted "
                f"(position {outcome.queue_position}) — releasing the queued row "
                f"and loading unleased; the capacity gate already cleared this fit"
            )
            self._release_lease_quietly(outcome.lease_id)
        else:
            log.warning(
                f"Lease for {model_name} not granted ({outcome.state}): "
                f"{outcome.message}"
            )
        return None

    # -- The pool (Backend.EXO): an exclusive lease around one generation --

    def _retry_after_s(self, expires_at: Optional[str]) -> Optional[int]:
        """Seconds until `expires_at`, floored at 1. None if unparseable.

        The holder's expiry is the only honest retry hint available on an
        exclusive conflict: there is no queue to report a position in, so
        "come back when the current holder's lease lapses" is the real answer.
        """
        if not expires_at:
            return None
        try:
            from datetime import datetime, timezone
            ts = expires_at.replace("Z", "+00:00")
            when = datetime.fromisoformat(ts)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            delta = (when - datetime.now(timezone.utc)).total_seconds()
            return max(1, int(delta))
        except Exception:
            return None

    def acquire_pool(self, purpose: str = "generation") -> str:
        """Take the exclusive lease on the pool, or raise capacity.PoolBusy.

        There is no wait-and-retry here on purpose. The pool serves one request
        at a time and a generation is minutes, so blocking a caller until it is
        free would be the hang the design rules out; a structured decline that
        says when to come back is the contract instead.
        """
        try:
            resp = self.sc.request_lease(
                resource_id=EXO_RESOURCE_ID,
                service_id=SERVICE_ID,
                memory_mb=None,      # the pool is TAKEN, not metered in bytes
                ttl_seconds=EXO_LEASE_TTL,
                exclusive=True,
            )
        except Exception as e:
            # A registry we cannot reach is not a free pool. Refusing to start
            # work we cannot announce is the safe direction on an exclusive
            # resource: the alternative collides with whoever does hold it.
            raise capacity.PoolBusy(
                f"cannot confirm the pool is free — registry unreachable: {e}",
                resource_id=EXO_RESOURCE_ID,
                retry_after_s=30,
            )

        outcome = self._lease_outcome(resp)
        if outcome.granted:
            # Register here rather than in pool_lease(), so that a caller which
            # acquires directly is swept by shutdown() too. The streaming path
            # has to acquire before it returns a response — otherwise a busy
            # pool would be discovered after the 200 headers were already on
            # the wire and a decline could only be delivered as stream content.
            with self._pool_lock:
                self._pool_leases.add(outcome.lease_id)
            log.info(
                f"Pool lease {outcome.lease_id} acquired for {purpose} "
                f"(exclusive, TTL={EXO_LEASE_TTL}s)"
            )
            return outcome.lease_id

        if outcome.state == "queued":
            # Should be unreachable while we send no memory_mb, but if the
            # registry ever queues an exclusive request, queued is not granted.
            self._release_lease_quietly(outcome.lease_id)
            raise capacity.PoolBusy(
                "the pool queued this request rather than granting it",
                resource_id=EXO_RESOURCE_ID,
                queue_position=outcome.queue_position,
                retry_after_s=60,
            )

        retry_after = self._retry_after_s(outcome.expires_at)
        holder = f" (held by {outcome.held_by})" if outcome.held_by else ""
        raise capacity.PoolBusy(
            f"the exo pool is serving another request{holder}; it holds one "
            f"request at a time" + (
                f", and the current lease lapses in {retry_after}s"
                if retry_after else ""
            ),
            resource_id=EXO_RESOURCE_ID,
            queue_position=outcome.queue_position,
            retry_after_s=retry_after,
            expires_at=outcome.expires_at,
        )

    def release_pool(self, lease_id: Optional[str]):
        """Give the pool back. Safe to call twice; never raises."""
        if not lease_id:
            return
        with self._pool_lock:
            self._pool_leases.discard(lease_id)
        try:
            self.sc.release_lease(lease_id)
            log.info(f"Pool lease {lease_id} released")
        except Exception as e:
            # Worth a louder log than a shared lease: an exclusive lease left
            # behind does not make the pool slower, it makes it closed until
            # the TTL lapses.
            log.error(
                f"FAILED to release pool lease {lease_id}: {e} — the pool stays "
                f"closed to other consumers until its {EXO_LEASE_TTL}s TTL expires"
            )

    @contextmanager
    def pool_lease(self, purpose: str = "generation"):
        """Hold the pool for the duration of a block, and give it back after.

        The equivalent of the exo repo's `scripts/with-pool.sh` trapping
        EXIT/INT/TERM: the `finally` covers a normal return, an exception, and
        a cancelled request (asyncio raises CancelledError *into* the frame, so
        a client that disconnects mid-generation still unwinds through here).
        What it cannot cover is SIGKILL or a power cut, which is what the
        lease TTL is for, and why `shutdown()` sweeps any still-held lease.
        """
        lease_id = self.acquire_pool(purpose)
        try:
            yield lease_id
        finally:
            self.release_pool(lease_id)

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
        if mm.backend == Backend.EXO:
            # Nothing local to restart. "Running" means the pool still answers
            # and still holds a ready model — and if it has been swapped under
            # us, pick up the new one rather than sending a request for a model
            # that is no longer there.
            try:
                resident = self.exo.resident_model()
            except ExoUnavailable as e:
                log.warning(f"pool unreachable for tier '{model_name}': {e}")
                return False
            if not resident:
                log.warning(f"pool has no ready model for tier '{model_name}'")
                return False
            if resident != mm.name:
                log.info(f"pool resident model changed: {mm.name} -> {resident}")
                mm.name = resident
            return True
        if mm.backend == Backend.VLM:
            # Restart our mlx-vlm process if it died.
            if mm.managed and mm.vlm_process and mm.vlm_process.poll() is not None:
                log.warning(f"mlx-vlm process for {mm.name} died — restarting")
                name = mm.name
                del self.models[name]
                try:
                    self.load_vlm_model(name)
                    return True
                except Exception as e:
                    log.error(f"Failed to restart mlx-vlm for {name}: {e}")
                    return False
            return True
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

        if mm.backend == Backend.EXO:
            # Deregister the tier, never touch the pool. We did not start the
            # exo instance and stopping it would strand whoever else is using
            # it — and it cannot be restarted unattended, so a stop here is
            # effectively permanent until a human opens a Terminal. `force`
            # deliberately does not override this.
            del self.models[model_name]
            log.info(f"Deregistered pool tier '{model_name}' (pool left running)")
            return {
                "status": "deregistered",
                "model": model_name,
                "backend": Backend.EXO.value,
                "note": "the exo pool is not owned by this gateway; nothing was stopped",
            }

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
        elif mm.backend == Backend.VLM:
            if mm.vlm_process:
                try:
                    mm.vlm_process.terminate()
                    try:
                        mm.vlm_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        mm.vlm_process.kill()
                    log.info(f"Stopped mlx-vlm server for {mm.name}")
                except Exception as e:
                    log.warning(f"mlx-vlm stop error: {e}")
            else:
                self._kill_stray_vlm()

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
        # Give the pool back first. uvicorn runs lifespan shutdown on SIGTERM
        # and SIGINT, so this is the trap that `scripts/with-pool.sh` installs
        # on EXIT/INT/TERM: a request killed mid-generation does not strand an
        # exclusive lease and close the pool until its TTL lapses. SIGKILL is
        # still beyond reach — that is what the TTL is for.
        with self._pool_lock:
            held = list(self._pool_leases)
        for lease_id in held:
            self.release_pool(lease_id)
            results.append({"pool_lease": lease_id, "status": "released"})
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
