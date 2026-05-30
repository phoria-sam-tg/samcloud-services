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
        service_id: str,
        memory_mb: int,
        purpose: str = "model-serving",
        ttl_seconds: int = 3600,
    ) -> dict:
        """Request a memory lease. Returns 201 granted, 202 queued, 409 conflict."""
        r = self._http.post(
            f"/resources/{resource_id}/leases",
            json={
                "service_id": service_id,
                "memory_mb": memory_mb,
                "purpose": purpose,
                "ttl_seconds": ttl_seconds,
            },
        )
        r.raise_for_status()
        return {"status_code": r.status_code, **r.json()}

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
        health_endpoint: str = "/health",
        subdomain: Optional[str] = None,
        capabilities: Optional[list[str]] = None,
        version: Optional[str] = None,
    ) -> dict:
        payload = {
            "name": name,
            "device_id": self.device,
            "port": port,
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
