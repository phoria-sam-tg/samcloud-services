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

import httpx
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
EXO_LEASE_MAX_TOTAL_S = config.EXO_LEASE_MAX_TOTAL_S
EXO_WEDGE_RECHECK_S = config.EXO_WEDGE_RECHECK_S


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


def vlm_cache_dir():
    """Where mlx-vlm will look for a repo's weights.

    Reads the same env huggingface_hub reads, in its order, because the VLM is
    spawned as a child of this process (`load_vlm_model`) and so inherits it.
    Do not shortcut to `~/.cache/huggingface/hub` -- that is only the default.
    """
    from pathlib import Path
    v = os.environ.get("HF_HUB_CACHE")
    if v:
        return Path(v)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def vlm_installed(repo_id: str) -> bool:
    """Are this VLM's weights already on local disk?

    The other two local backends only advertise what is installed -- Ollama
    lists what has been pulled, llama-server lists GGUF files that exist -- and
    this is the VLM equivalent, so `/v1/models` can hold to the same meaning
    across all three.

    It matters more here than the symmetry suggests, because the fit gate in
    `capacity.py` measures RAM and nothing measures disk. Measured 2026-09-24
    on slice: gemma-4-31b needs 18700MB of RAM, which fit, against 15GB of free
    disk and ~18.7GB of weights not yet fetched. Nothing in the load path would
    have refused that -- `load_vlm_model` would have spawned mlx-vlm, which
    downloads on demand, and the box would have run itself out of disk before
    `VLM_STARTUP_TIMEOUT` expired. Advertising a model is an invitation to
    request it, so the listing is the honest place to draw the line.

    False is the safe answer on any error: it hides a model from discovery,
    which `/models` still shows with `installed: false`, and costs nothing else
    -- `match_vlm_model` is not gated on this, so a caller who asks for the
    model by name anyway still reaches the existing load path.
    """
    from pathlib import Path
    try:
        p = Path(repo_id).expanduser()
        if p.is_dir():          # a pinned local directory, not a hub repo
            return True
        snapshots = vlm_cache_dir() / ("models--" + repo_id.replace("/", "--")) / "snapshots"
        return any(any(s.iterdir()) for s in snapshots.iterdir())
    except OSError:
        return False


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
    # When each pool lease was taken, so a renewal can say how long the
    # lease has been held. That age is the gate for #827 stage (b) — the
    # default TTL drops only once a lease over 120s is seen renewing —
    # so it is load-bearing, not decoration.
    _pool_lease_acquired: dict = field(default_factory=dict, repr=False)
    # Last renewal reason per lease, so the at_ceiling line is logged once
    # rather than every interval (samclaude-admin, reviewing PR #35). A
    # CHANGE of reason always logs, so silence only ever means 'the same
    # thing as last time' and never 'something new I decided to hide'.
    _pool_lease_last_reason: dict = field(default_factory=dict, repr=False)
    # Guards pool bookkeeping that is mutated OFF the event loop — the lease set
    # above, and the tier's request counter in resolve_exo_tier. Both reach this
    # object from `asyncio.to_thread`, so the single-threaded loop no longer
    # serialises them for free. That is the general cost of moving work off the
    # loop: implicit serialisation leaves with it, and anything read-modify-write
    # has to say so explicitly.
    _pool_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cooldown_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _health_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _stats_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _pool_renewal_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _offering_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _offering_tier: Optional[str] = field(default=None, repr=False)

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
                # idle timer (OLLAMA_KEEP_ALIVE), and adopt it as MANAGED so the
                # cooldown loop can spin it back down when idle (ticket #97).
                #
                # Previously this marked the model managed=False, and
                # check_cooldowns / unload both skip non-managed entries. With
                # OLLAMA_KEEP_ALIVE=-1 (the config default) that pinned the model
                # in memory forever. Both boxes actually run 300, so ollama's own
                # timer frees the memory and what leaks is the BOOKKEEPING: the
                # entry never leaves self.models, because cooldown is the only
                # thing that removes it. Three consequences, all from that flag:
                #   1. /status reports the model resident after ollama dropped it
                #   2. load_ollama_model short-circuits on `name in self.models`
                #      and returns the stale entry without reloading
                #   3. a capacity refusal lies to the caller. `resident` and
                #      `reclaimable` build the 503 detail ungated, so the body
                #      names a model ollama dropped long ago, quotes memory that
                #      is already free, and promises it is "released on idle"
                #      when check_cooldowns skips that entry permanently. Every
                #      clause false, in the one message a caller has to trust.
                #      (The same sum also over-counts evictable memory in the
                #      fit path itself, but that read is behind AUTO_EVICT,
                #      which defaults False and is set on neither box.)
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
        """Active leases on this box's GPU, matched here rather than on the wire.

        `GET /leases` documents `?resource=` and ignores it, honours an
        undocumented `?resource_id=`, and #774 may reconcile the two in either
        direction — so neither spelling is safe to depend on. `?status=` is
        documented and works, and the fleet-wide active set is small, so this
        asks for the one filter that is in the contract and matches the
        resource itself. The `except` is deliberately narrow: a filter that
        stops working must not become a silent empty list again.
        """
        try:
            leases = self.sc.list_leases(status="active")
        except Exception:
            return []
        return [lease for lease in leases if lease.get("resource_id") == RESOURCE_ID]

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
            # `pgrep -f` is a SUBSTRING match over the whole command line, so it
            # also returns any process that merely *mentions* "mlx_vlm.server" —
            # a grep, an editor, a shell script quoting it — and this loop sends
            # SIGTERM to everything it returns. Verify each candidate is really a
            # `python -m mlx_vlm.server` of ours before signalling it.
            # (#806: the seventh instance of a pattern search matching something
            # that was not the thing; the only one wired to a kill with no uid
            # filter, in a long-running daemon.)
            if not self._is_vlm_server(pid):
                log.debug(f"pid {pid} matched the vlm pattern but is not one; leaving it")
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                log.info(f"Killed stray mlx-vlm process {pid}")
            except ProcessLookupError:
                pass
            except Exception as e:
                log.warning(f"Failed to kill stray mlx-vlm process {pid}: {e}")

    @staticmethod
    def _is_vlm_server(pid: int) -> bool:
        """True only if `pid` is genuinely a `python -m mlx_vlm.server` we own.

        Checks three things, all of which the real process satisfies and a
        coincidental `pgrep -f` match does not: our uid, `argv[0]` equal to
        VLM_PYTHON, and `-m mlx_vlm.server` as adjacent whitespace-delimited
        argv tokens. That is exactly the shape load_vlm_model spawns.
        """
        try:
            out = subprocess.run(
                ["ps", "-o", "uid=,command=", "-p", str(pid)],
                capture_output=True, text=True, timeout=5,
            )
        except Exception:
            return False
        line = out.stdout.strip()
        if not line:
            return False
        uid_s, _, cmd = line.partition(" ")
        try:
            if int(uid_s.strip()) != os.getuid():
                return False
        except ValueError:
            return False
        toks = cmd.split()
        if not toks:
            return False
        # Adjacent `-m mlx_vlm.server` tokens are NOT sufficient on their own,
        # and the reason is that `ps` output carries no quoting. A single quoted
        # argument reappears in `ps -o command=` as separate whitespace tokens,
        # so a long-lived
        #     grep -r -- "-m mlx_vlm.server" /some/tree
        # prints as `grep -r -- -m mlx_vlm.server /some/tree`, and `cmd.split()`
        # sees `-m` immediately followed by `mlx_vlm.server`. That is a real
        # process, under our own uid, that the caller would SIGTERM. The
        # vulnerable shape comes from quoting loss in `ps`, not from anyone
        # passing two separate arguments.
        #
        # This comment previously gave `grep -- -m mlx_vlm.server .` as the
        # example. That has the right argv shape but cannot actually be caught:
        # `--` ends option parsing, so `-m` is the pattern and `mlx_vlm.server`
        # is a filename that does not exist, and grep exits at once. Established
        # by trying to reproduce it against a live process and failing.
        #
        # Only argv[0] says what a process was EXECUTED AS, and load_vlm_model
        # execs VLM_PYTHON by exactly this string (see its Popen list below).
        # Asserted by test_vlm_kill_guard.py: without this check, cases [2],
        # [2b], [5] and [8] fail.
        if toks[0] != VLM_PYTHON:
            return False
        return any(
            t == "-m" and toks[i + 1] == "mlx_vlm.server"
            for i, t in enumerate(toks[:-1])
        )

    def load_vlm_model(self, model_name: str) -> ManagedModel:
        """Start an mlx-vlm server for a vision-language model on demand.

        The gateway owns the process: it spawns mlx_vlm.server, leases memory,
        and tears both down on cooldown/unload. Only one VLM runs at a time on
        VLM_PORT — a request for a different VLM swaps the current one out.
        """
        # httpx is imported at module level now — kept here harmlessly so this
        # function reads the same as before.
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
            # Guarded because this method now runs in a thread pool, not on the
            # event loop. `+= 1` is LOAD/ADD/STORE and was serialised only
            # implicitly, by there being one thread; two concurrent `think`
            # resolves can interleave it and lose a count. Concurrent resolves
            # are not hypothetical — resolve happens *before* the lease, so they
            # are precisely the case where one caller wins the pool and the
            # other is declined. The assignments above are left unguarded on
            # purpose: they are idempotent, every racer writing the same
            # resident value read from the same pool.
            with self._pool_lock:
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
                # Explicit, not the registry default: liveness and exposure are
                # different numbers now (#827 b1). 600s says "assume I died after
                # this"; 1800s says "this is the longest I could legitimately need
                # the pool". Renewals extend inside the second until granted_at+1800.
                max_total_s=EXO_LEASE_MAX_TOTAL_S,
            )
        except httpx.HTTPStatusError as e:
            # A 404 here is a misconfiguration, not a transient. It means this
            # gateway is asking for a resource that does not exist on the
            # registry — the default `EXO_RESOURCE_ID` is `<device>/exo-pool`,
            # and the pool is ONE physical node fronted by ONE plane resource
            # (`claude-services-slice/exo-pool`), so any other box enabling EXO
            # without overriding it lands here forever. Retrying cannot fix it,
            # so it gets no retry hint and its own code.
            if e.response.status_code == 404:
                raise capacity.PoolBusy(
                    f"{EXO_RESOURCE_ID} does not exist on the registry. The exo "
                    f"pool is one node fronted by one resource; a second box "
                    f"must point EXO_RESOURCE_ID at the existing one rather than "
                    f"register its own, or both would hold 'the exclusive lease' "
                    f"at once and generate into the same slot.",
                    resource_id=EXO_RESOURCE_ID,
                    error="pool_resource_missing",
                )
            raise capacity.PoolBusy(
                f"cannot determine whether the pool is free — the registry "
                f"answered {e.response.status_code}: {e}",
                resource_id=EXO_RESOURCE_ID,
                retry_after_s=30,
                error="lease_unavailable",
            )
        except Exception as e:
            # A registry we cannot reach is not a free pool. Refusing to start
            # work we cannot announce is the safe direction on an exclusive
            # resource: the alternative collides with whoever does hold it.
            #
            # But it is NOT `resource_busy`, which is what this used to report.
            # That conflated "someone else holds the lease" with "I cannot
            # reach or resolve my own resource" — indistinguishable from the
            # 503, so a caller honouring the retry hint loops forever through a
            # plane outage or a misconfiguration, and whoever debugs it goes
            # looking for the holder of a lease that was never taken.
            raise capacity.PoolBusy(
                f"cannot determine whether the pool is free — registry "
                f"unreachable: {e}",
                resource_id=EXO_RESOURCE_ID,
                retry_after_s=30,
                error="lease_unavailable",
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
                self._pool_lease_acquired[outcome.lease_id] = time.monotonic()
            log.info(
                f"Pool lease {outcome.lease_id} acquired for {purpose} "
                f"(exclusive, TTL={EXO_LEASE_TTL}s)"
            )
            self._refuse_if_wedged(outcome.lease_id)
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

    def _refuse_if_wedged(self, lease_id: str):
        """Having just won the exclusive lease, refuse if the pool says it is busy.

        We hold the only lease, so a runner still reporting `RunnerRunning` is
        not another lease-holder. It is one of two things and neither is safe to
        dispatch into:

        - **Wedged.** A previous client died mid-generation and exo was never
          told, so its single slot is occupied by work nobody is reading. exo
          *does* have cancellation (`POST /v1/cancel/{command_id}`, and a
          disconnect handler), but neither fired for the two clients that
          wedged this pool on 2026-09-22, so in practice it does not clear
          itself and someone has to cancel it explicitly. Why the disconnect
          handler did not fire on a non-streaming response is unmeasured
          (#806) — do not assume it will.
        - **Driven directly.** Someone is generating against `:52415` without
          taking a lease, which the lease cannot prevent.

        Without this check the gateway grants itself the lease and dispatches
        anyway, and the caller hangs for EXO_GENERATE_TIMEOUT holding the pool
        shut. That is not hypothetical: it is what this gateway did on
        2026-09-20 after a SIGKILL stranded a generation, and the `busy` signal
        that detects it already existed and was consumed by nothing.

        exo dispatches `TextGeneration` only under `RunnerReady`, so refusing
        here also matches the engine rather than merely being cautious.
        """
        # Two samples, not one. `busy` is an observation of something that
        # moves: a generation finishing in the window between acquiring the
        # lease and reading the pool makes a single sample say "busy" about a
        # pool that is about to be free, and we would decline a working pool.
        # That is the shape of the RunnerReady-only bug — correct logic applied
        # to one reading of a moving thing — and the second read costs ~45ms on
        # a path that routinely spends fifteen seconds.
        runners = {}
        for attempt in range(2):
            if attempt:
                time.sleep(EXO_WEDGE_RECHECK_S)
            try:
                status = self.exo.pool_status()
            except Exception as e:
                # A guard, not the readiness gate: if we cannot read the pool,
                # let the generation attempt surface the real error rather than
                # inventing a refusal from a failed probe.
                log.warning(f"wedge guard could not read the pool: {e}")
                return
            if not status.get("busy"):
                return
            for inst in status.get("instances") or []:
                if inst.get("busy"):
                    runners = inst.get("runners") or {}
                    break

        log.error(
            f"Pool busy across two reads while we hold the only lease — "
            f"refusing to dispatch. Runners: {runners}"
        )
        self.release_pool(lease_id)
        # Deliberately does not assert a wedge. Until every consumer goes
        # through this gateway, a legitimate caller driving :52415 directly
        # trips this too, and that is not a fault — the pool genuinely is busy.
        # Naming the observation rather than the diagnosis keeps the next reader
        # from being sent after a phantom.
        raise capacity.PoolBusy(
            "the pool is serving a request that holds no lease, so this gateway "
            "cannot serialise against it: either something is driving exo "
            "directly, or a client died mid-generation and left the slot "
            "occupied. Declining rather than dispatching into it, which would "
            "hang instead of serving. To clear it: read the stuck task's "
            "commandId from GET /state on the pool and POST /v1/cancel/"
            "{commandId} — that frees the runner and keeps the model loaded. "
            f"Re-placing the model on {EXO_RESOURCE_ID} also works but costs a "
            "30s-10min reload, so it is the fallback, not the first move. This "
            "gateway does neither; ask whoever operates the pool (#806).",
            resource_id=EXO_RESOURCE_ID,
            retry_after_s=60,
            error="pool_busy_unleased",
        )

    def release_pool(self, lease_id: Optional[str]):
        """Give the pool back. Safe to call twice; never raises."""
        if not lease_id:
            return
        with self._pool_lock:
            self._pool_leases.discard(lease_id)
            self._pool_lease_acquired.pop(lease_id, None)
            self._pool_lease_last_reason.pop(lease_id, None)
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
            # it. The pool is restartable unattended now (#806), so a stop here
            # is no longer permanent — but it would still drop the resident
            # model, which only an operator can place back. `force` deliberately
            # does not override this.
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

    def _collect_stats(self) -> dict:
        """Canonical capacity reading — see ollama/capacity.py.

        Was a local vm_stat parse reporting total and used only. Two things
        were wrong with it: it assumed 4KiB pages on a machine that uses 16KiB,
        under-reporting memory by 4x; and "used" counts pages the compressor is
        merely sitting on, so it read 22.7GiB used on a 36GiB box whose entire
        process RSS was ~3.9GiB. The number that answers "will this load swap"
        is memory_available_mb (free + inactive + speculative), which only the
        shared collector reports.
        """
        try:
            return capacity.collect()
        except Exception as e:
            log.warning(f"Capacity read failed: {e}")
            return {}

    async def stats_loop(self):
        """Push unified-memory stats to this box's SAMcloud resource every 15s.

        Without it a resource shows `last_stats: never` and the dashboard reads
        NaN. The reading is the full `capacity.collect()`; `registry_payload()`
        narrows it to the keys the registry's strict ResourceStats model will
        accept, because memory_available_mb — the whole point of the collector —
        is not yet on the wire and an unknown key 422s the push.
        """
        while True:
            try:
                stats = self._collect_stats()
                if stats:
                    self.sc.push_stats(RESOURCE_ID, capacity.registry_payload(stats))
            except Exception as e:
                log.warning(f"Stats push error: {e}")
            await asyncio.sleep(15)

    def _compute_offering(self) -> Optional[str]:
        """Offering tier derived from what actually fits (doc #8 Stage 1).

        Good-citizen flex on a box shared with other work: the tier drops as
        unified memory fills and restores when it frees, so a caller reading
        the registry is told what this box can really serve right now.

        Was fixed MB bands over `total - used`. Both halves were wrong: `used`
        counts reclaimable compressed pages, and the bands were calibrated when
        the largest model was ~6GB — so wafer advertised `offering:full` on
        ~14GiB available while a 17.5GB model could not load without swapping.
        Deriving the tier from the catalogue keeps `full` meaning "the big one
        fits" as models come and go, and needs no per-box threshold tuning.
        Returns None if memory cannot be read, leaving the tier unchanged.
        """
        try:
            avail = capacity.collect().get("memory_available_mb")
            if avail is None:
                return None
            return capacity.offering_tier(self.catalogue_mb(), avail)
        except Exception as e:
            log.warning(f"Offering computation failed: {e}")
            return None

    def _apply_offering(self, tier: str):
        """Publish the tier as an `offering:<tier>` capability (replacing any prior
        offering:* entry, preserving all other capabilities). PATCH emits a
        service.updated event, satisfying doc #8's 'emit on tier change'."""
        try:
            svc = self.sc.get_service(SERVICE_ID)
            caps = [
                c for c in (svc.get("capabilities") or [])
                if not str(c).startswith("offering:")
            ]
            caps.append(f"offering:{tier}")
            self.sc.update_service(SERVICE_ID, capabilities=caps)
            log.info(f"Offering tier {self._offering_tier} -> {tier} (capabilities now {caps})")
            self._offering_tier = tier
        except Exception as e:
            log.warning(f"Failed to apply offering:{tier}: {e}")

    async def offering_loop(self):
        """Stage-1 capacity offering (doc #8): recompute the offering tier from
        live resource pressure and advertise it via the service's capabilities.
        The first reading is published immediately; subsequent changes require the
        new tier to hold for OFFERING_HYSTERESIS consecutive polls to avoid
        flapping on transient spikes."""
        pending: Optional[str] = None
        pending_count = 0
        while True:
            try:
                tier = self._compute_offering()
                if tier is not None:
                    if self._offering_tier is None:
                        # First advertisement — no hysteresis wait.
                        self._apply_offering(tier)
                        pending, pending_count = None, 0
                    elif tier == self._offering_tier:
                        pending, pending_count = None, 0
                    elif tier == pending:
                        pending_count += 1
                        if pending_count >= config.OFFERING_HYSTERESIS:
                            self._apply_offering(tier)
                            pending, pending_count = None, 0
                    else:
                        pending, pending_count = tier, 1
            except Exception as e:
                log.warning(f"Offering loop error: {e}")
            await asyncio.sleep(config.OFFERING_POLL_SECONDS)

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
        """Release and re-request MODEL leases to prevent expiry.

        NOT THE PATTERN FOR AN EXCLUSIVE LEASE, and deliberately left alone.
        Releasing and then re-requesting opens a window in which the resource is
        unheld; for the memory leases here that is survivable, because they are
        shares of a pool rather than the pool itself. On the exclusive pool lease
        the same two lines would let another caller in mid-generation — the exact
        class of fault #827 exists to remove. `_renew_pool_leases` below renews in
        place against POST /leases/<id>/renew and must stay that way.
        """
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

    # Leases younger than this are skipped by a renewal tick. See the comment at
    # the skip for why it is 2s rather than one interval.
    _RENEW_MIN_AGE_S: float = 2.0

    async def pool_renewal_loop(self):
        """Keep held pool leases alive while their generation runs (#827 P3).

        OFF THE EVENT LOOP, via to_thread. `_renew_pool_leases` makes a
        synchronous httpx POST carrying the plane's 10s timeout, and it runs
        every interval for as long as a generation is streaming — on the same
        loop as that stream. A slow plane answer would stall every token of
        every concurrent response for its duration, and today's release 502s
        took seconds. Worse, the streaming path is where the 60s inter-token
        rule lives, so a renewal blocking the loop could trip this gateway's
        own stall detector against a generation that never faltered.

        The rest of the pool path already reaches this object from
        `asyncio.to_thread` (see `_pool_lock`); the renewal was the one call
        still on the loop.
        """
        while True:
            await asyncio.sleep(config.EXO_LEASE_RENEW_INTERVAL_S)
            try:
                await asyncio.to_thread(self._renew_pool_leases)
            except Exception as e:
                log.warning(f"Pool lease renewal error: {e}")

    def _renew_pool_leases(self):
        """Extend every held pool lease IN PLACE. Never releases anything.

        A failure here does NOT release the lease. The generation is still
        running and the pool is still ours until the registry says otherwise;
        dropping the claim because we failed to restate it would hand a busy
        single-slot pool to the next caller, which is worse than a lease that
        outlives us by one TTL.
        """
        with self._pool_lock:
            held = [(lid, self._pool_lease_acquired.get(lid))
                    for lid in self._pool_leases]
        for lease_id, since in held:
            age = time.monotonic() - since if since else None
            shown = f"{age:.0f}s" if age is not None else "unknown"

            # A lease this young cannot need renewing, and renewing it anyway is
            # how a benign truncation became an ERROR (#827, 21:15:19Z). The
            # heartbeat's offset from acquisition is 0 to one interval, and at the
            # 0 end `now + ttl` is only microseconds ahead of `granted + ttl` — the
            # registry's `int()` of that difference was 0, which it then reported
            # as `ttl_below_granted`, a caller misconfiguration that had not
            # happened. samclaude-admin is fixing the classification in 0.12.56;
            # this skips the request that provokes it, which is worth doing anyway
            # because it cannot achieve anything.
            #
            # DELIBERATELY SMALL, and not "younger than one interval". Skipping a
            # whole interval would push the worst-case renewal gap from 60s to
            # ~120s, and clause 2's invariant (three attempts inside 80% of the
            # TTL) is stated over 60/120/180 against 480. It would still hold at
            # 120/180/240, with a third of the margin. Two seconds removes the
            # sub-second case and perturbs nothing.
            if age is not None and age < self._RENEW_MIN_AGE_S:
                log.debug(
                    f"Pool lease {lease_id} is {age:.2f}s old; skipping this "
                    f"renewal tick — nothing to extend yet."
                )
                continue
            try:
                resp = self.sc.renew_lease(lease_id, EXO_LEASE_TTL) or {}

                # ABSENT IS A THIRD OUTCOME, not a falsy one
                # (claude-wafer-services, #827). `resp.get("extended_by_s") or 0`
                # reads a RENAMED field as a real zero and takes the at_ceiling
                # branch, logging "NO extension" forever while every renewal is in
                # fact unaccounted for. No fixture catches that — a live plane that
                # renamed the field produces exactly the same silence — so the check
                # has to be here, where the read happens.
                #
                # Only the keys we BRANCH on are required. expires_at and
                # max_total_s appear solely inside log messages, where a missing
                # value degrades a line rather than choosing a path, and `note` is
                # conditional by design (the registry returns it only for an
                # indefinite lease). Requiring everything read would log a
                # violation on every ordinary renewal, and a check that fires on
                # the normal path carries no more information than one that never
                # fires.
                missing = [k for k in ("extended_by_s", "capped", "reason")
                           if k not in resp]
                if missing:
                    log.error(
                        f"contract violation: renewal answered without "
                        f"{', '.join(missing)} for {lease_id} after {shown} held. "
                        f"Keys present: {sorted(resp)}. The gateway branches on "
                        f"those fields, so it cannot tell an extension from a "
                        f"ceiling — treating this renewal as unknown rather than "
                        f"guessing. This is a RENAMED field, not an old plane: "
                        f"`reason` shipped with /renew in 0.12.51, and a registry "
                        f"without /renew answers 404 rather than a 200. #827."
                    )
                    with self._pool_lock:
                        self._pool_lease_last_reason[lease_id] = "contract_violation"
                    continue

                extended = resp.get("extended_by_s") or 0
                # `reason` (#33) rather than deriving one from two booleans. The
                # registry knows which of four things happened; inferring it here
                # is how "no extension" got mislabelled as "the ceiling" once
                # already. When it is absent we say so instead of guessing —
                # a quiet fallback to the old inference is the failure this
                # whole ticket keeps turning up.
                # No inference fallback any more. `reason` is one of the three
                # keys the totality check above requires, so an absent one is a
                # contract violation and never reaches here. The old code guessed
                # it from extended_by_s/capped for a registry predating #827 P3 —
                # but that guess cannot distinguish an OLD plane from a RENAMED
                # field, which is the case that matters, and guessing is the exact
                # behaviour being removed. A plane older than 0.12.51 now fails
                # renewals loudly instead of silently, which is the right
                # direction for a mechanism exclusivity will depend on.
                reason = resp["reason"]
                every = f"renewing every {config.EXO_LEASE_RENEW_INTERVAL_S}s"

                if reason == "extended":
                    # capped and extended arrive together on the LAST renewal
                    # before the ceiling: target lands on the ceiling, still later
                    # than the current expiry. Worth saying, because every renewal
                    # after it reports at_ceiling and an operator should see that
                    # transition coming rather than meet it.
                    tail = (f" — this is the LAST extension, the "
                            f"{resp.get('max_total_s')}s ceiling is now reached"
                            if resp.get("capped") else "")
                    log.info(
                        f"Pool lease {lease_id} renewed in place after {shown} "
                        f"held: +{extended}s, expires {resp.get('expires_at')} "
                        f"({every}){tail}"
                    )
                elif reason == "at_ceiling" and (
                        self._pool_lease_last_reason.get(lease_id) == "at_ceiling"):
                    # Same as last interval. At stage (a) EVERY renewal is capped,
                    # so a 1500s generation would otherwise print 25 identical
                    # lines. Suppressed only while the reason is UNCHANGED — the
                    # first one logs, and any change logs — because a log that
                    # goes quiet for a new reason is the failure this whole ticket
                    # is about.
                    pass
                elif reason == "at_ceiling":
                    # Accepted, extended nothing, the ceiling bound it. Expected
                    # for the whole of stage (a) — but it must not print as an
                    # extension, because a renewal that silently does nothing
                    # reads exactly like one that silently failed.
                    log.info(
                        f"Pool lease {lease_id} renewal accepted after {shown} "
                        f"held, NO extension (at the {resp.get('max_total_s')}s "
                        f"ceiling; granted TTL is {EXO_LEASE_TTL}s). Expected "
                        f"until #827 stage (b) lowers the granted TTL."
                    )
                elif reason == "ttl_below_granted":
                    # #34 splits already_later in two, and THIS half has exactly
                    # one cause, so it can be named rather than listed. Same
                    # severity: every renewal extends nothing, the lease lapses on
                    # its original expiry, and the loop reports success throughout.
                    log.error(
                        f"Pool lease renewal is NOT EXTENDING: {lease_id} after "
                        f"{shown} held returned ttl_below_granted — we are asking "
                        f"for {EXO_LEASE_TTL}s but the lease was granted for "
                        f"longer, so the expiry is already further out. Every "
                        f"renewal will do nothing and the lease lapses on its "
                        f"original expiry ({resp.get('expires_at')}). #827."
                    )
                elif reason == "already_later":
                    # The pre-#34 registry (0.12.51) does not split this, so here
                    # it still carries both causes and neither can be asserted.
                    # Kept rather than replaced: this gateway may meet either
                    # version depending on which release the plane is running, and
                    # mapping an old value onto the new message would claim a
                    # precision the response does not have.
                    log.error(
                        f"Pool lease renewal is NOT EXTENDING: {lease_id} after "
                        f"{shown} held returned already_later — the expiry is "
                        f"already beyond now+{EXO_LEASE_TTL}s. Either the renewal "
                        f"TTL is below the granted one, or the grant exceeded the "
                        f"registry ceiling ({resp.get('max_total_s')}s). Every "
                        f"renewal will do nothing and the lease lapses on its "
                        f"original expiry ({resp.get('expires_at')}). #827."
                    )
                elif reason == "indefinite":
                    log.warning(
                        f"Pool lease {lease_id} has no expiry to renew after "
                        f"{shown} held ({resp.get('note') or 'indefinite'}). "
                        f"Unexpected — this gateway always grants with a TTL."
                    )
                else:
                    log.warning(
                        f"Pool lease {lease_id} renewal after {shown} held "
                        f"returned an unrecognised reason {reason!r}: {resp}"
                    )
                # Under the lock: `release_pool` pops this from another thread.
                with self._pool_lock:
                    self._pool_lease_last_reason[lease_id] = reason
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # A 404 means two different things and they need different
                    # volumes (claude-wafer-services, #827).
                    #
                    # The loop reads the held-lease map under `_pool_lock` and then
                    # POSTs OUTSIDE it — deliberately, because holding a lock
                    # across a 10s HTTP call is the wrong trade. So a generation
                    # that finishes in that window releases its lease and the
                    # renewal 404s on a lease we no longer own. That is BENIGN and
                    # entirely normal: the median lease is 9.4s against a 60s tick.
                    #
                    # Reporting it as a reap would be a false alarm on the normal
                    # path, which is as bad as silence — an error that fires for
                    # benign reasons trains its reader to skip it, and the once it
                    # means what it says is the once it gets skipped.
                    #
                    # So re-check ownership rather than locking longer. Still held
                    # -> the reap this error exists for. No longer held -> we raced
                    # our own release.
                    with self._pool_lock:
                        still_ours = lease_id in self._pool_leases
                    if not still_ours:
                        log.debug(
                            f"Pool lease {lease_id} was released while its renewal "
                            f"was in flight ({shown} held); the 404 is our own race "
                            f"with release_pool, not a reap."
                        )
                    else:
                        log.error(
                            f"Pool lease {lease_id} is GONE from the registry after "
                            f"{shown} held and we STILL HOLD it — reaped while the "
                            f"generation was running, so the pool is no longer "
                            f"exclusively ours. Check the lease TTL against #827 "
                            f"staging."
                        )
                elif e.response.status_code in (403, 422):
                    # Not transient. 403 means our identity is not the lease's
                    # holder; 422 means we are sending a body the model rejects.
                    # Either way renewal will fail identically every 60s forever,
                    # so liveness is not in force at all while looking like it is
                    # — the precise failure #843 was. As loud as a reap.
                    log.error(
                        f"Pool lease renewal is NOT WORKING: {lease_id} after "
                        f"{shown} held answered {e.response.status_code} "
                        f"({'not the holder' if e.response.status_code == 403 else 'bad request body'}). "
                        f"This will repeat every "
                        f"{config.EXO_LEASE_RENEW_INTERVAL_S}s and no lease is "
                        f"being kept alive. #827."
                    )
                else:
                    log.warning(
                        f"Could not renew pool lease {lease_id} after {shown} "
                        f"held ({e.response.status_code}); keeping it — the "
                        f"generation is still running"
                    )
            except Exception as e:
                log.warning(
                    f"Could not renew pool lease {lease_id} after {shown} held "
                    f"({type(e).__name__}: {e}); keeping it"
                )

    def start_background_tasks(self):
        if self._cooldown_task is None or self._cooldown_task.done():
            self._cooldown_task = asyncio.create_task(self.cooldown_loop())
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self.health_loop())
        self._renewal_task = asyncio.create_task(self.lease_renewal_loop())
        if self._pool_renewal_task is None or self._pool_renewal_task.done():
            self._pool_renewal_task = asyncio.create_task(self.pool_renewal_loop())
        if self._stats_task is None or self._stats_task.done():
            self._stats_task = asyncio.create_task(self.stats_loop())
        if config.OFFERING_ENABLED and (self._offering_task is None or self._offering_task.done()):
            self._offering_task = asyncio.create_task(self.offering_loop())
        log.info("Background tasks started (cooldown, health, lease renewal, POOL lease renewal, stats, offering)")

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
        for task in [self._cooldown_task, self._health_task, getattr(self, '_renewal_task', None),
                     getattr(self, '_pool_renewal_task', None), self._stats_task, self._offering_task]:
            if task and not task.done():
                task.cancel()
        return results
