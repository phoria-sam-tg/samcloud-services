"""SAMcloud API client for service registration and resource/lease management."""

import httpx
import time
from dataclasses import dataclass, field
from typing import Optional

from . import config


@dataclass
class SamcloudClient:
    token: str
    base: str = field(default_factory=lambda: config.SC_BASE)
    device: str = field(default_factory=lambda: config.SC_DEVICE)
    _http: httpx.Client = field(default=None, repr=False)

    def __post_init__(self):
        self._http = httpx.Client(
            base_url=self.base,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30,
        )

    # -- Resources --

    def list_resources(self, **filters) -> list[dict]:
        r = self._http.get("/resources", params=filters)
        r.raise_for_status()
        return r.json()

    def get_resource(self, resource_id: str) -> dict:
        r = self._http.get(f"/resources/{resource_id}")
        r.raise_for_status()
        return r.json()

    def resource_dashboard(self) -> list[dict]:
        r = self._http.get("/resources/dashboard")
        r.raise_for_status()
        return r.json()

    def push_stats(self, resource_id: str, stats: dict) -> dict:
        r = self._http.post(f"/resources/{resource_id}/stats", json=stats)
        r.raise_for_status()
        return r.json()

    # -- Leases --

    def request_lease(
        self,
        resource_id: str,
        service_id: Optional[str] = None,
        memory_mb: Optional[int] = None,
        ttl_seconds: int = 3600,
        exclusive: bool = False,
    ) -> dict:
        """Request a lease. Returns `{"status_code": int, ...body}`.

        Does NOT raise on 409. A conflict is an *answer* this caller has to
        read — the body carries the current holder and its `expires_at`, which
        is the only way to tell a caller when to come back — so raising here
        would throw away the useful half of the response. Genuine faults (5xx,
        network, 404 on an unregistered resource) still raise.

        `memory_mb=None` omits the field entirely rather than sending a zero,
        and that distinction is load-bearing on a resource that leases no
        bytes. The registry queues a request when `memory_mb > available`,
        where `available` is `total - leased` and `total` is read from
        `vram_mb`/`gpu_memory_mb`/`unified_memory_mb`/`ram_mb` in the
        resource's specs. An instance-style resource like `exo-pool` carries
        none of those, so `total` is 0 and *any* positive `memory_mb` is
        instantly oversubscribed: the request would be queued forever instead
        of granted, no matter how idle the resource is. Send no byte count and
        the queue branch is never entered.

        `exclusive=True` asks for the resource itself rather than a slice of
        its memory. Passing `service_id` is worth doing even though it is
        optional: the registry renders a 409's `held_by` from the holder's
        `service_id`, so a lease taken without one reports as "held by None"
        to whoever collides with it.
        """
        payload: dict = {"ttl_seconds": ttl_seconds}
        if service_id is not None:
            payload["service_id"] = service_id
        if memory_mb is not None:
            payload["memory_mb"] = memory_mb
        if exclusive:
            payload["exclusive"] = True

        r = self._http.post(f"/resources/{resource_id}/leases", json=payload)
        if r.status_code not in (200, 201, 202, 409):
            r.raise_for_status()
        try:
            body = r.json()
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {"detail": body}
        return {"status_code": r.status_code, **body}

    def release_lease(self, lease_id: str) -> dict:
        r = self._http.delete(f"/leases/{lease_id}")
        r.raise_for_status()
        return r.json()

    def list_leases(self, **filters) -> list[dict]:
        r = self._http.get("/leases", params=filters)
        r.raise_for_status()
        return r.json()

    # -- Services --

    def create_service(
        self,
        name: str,
        port: int,
        description: str,
        health_endpoint: str = "/health",
        subdomain: Optional[str] = None,
        capabilities: Optional[list[str]] = None,
        version: Optional[str] = None,
    ) -> dict:
        # `description` is required by the API and was missing here — POST /services
        # answers 422 {"loc": ["body", "description"], "msg": "Field required"} without
        # it. No caller on either box, so it had drifted unnoticed; measured against the
        # live plane on #760.
        payload = {
            "name": name,
            "device_id": self.device,
            "port": port,
            "description": description,
            "health_endpoint": health_endpoint,
        }
        if subdomain:
            payload["subdomain"] = subdomain
        if capabilities:
            payload["capabilities"] = capabilities
        if version:
            payload["version"] = version
        r = self._http.post("/services", json=payload)
        r.raise_for_status()
        return r.json()

    def update_service(self, service_id: str, **fields) -> dict:
        r = self._http.patch(f"/services/{service_id}", json=fields)
        r.raise_for_status()
        return r.json()

    def report_health(self, service_id: str) -> dict:
        r = self._http.post(f"/services/{service_id}/health")
        r.raise_for_status()
        return r.json()

    def get_service(self, service_id: str) -> dict:
        r = self._http.get(f"/services/{service_id}")
        r.raise_for_status()
        return r.json()

    def list_services(self) -> list[dict]:
        r = self._http.get("/services")
        r.raise_for_status()
        return r.json()

    def delete_service(self, service_id: str) -> dict:
        r = self._http.delete(f"/services/{service_id}")
        r.raise_for_status()
        return r.json()
