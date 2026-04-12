"""Vast.ai cloud GPU provider service."""

import asyncio
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

VASTAI_API_BASE = "https://console.vast.ai/api/v0"


class VastAIProvider:
    """Interact with the Vast.ai API to search, rent, and manage GPU instances."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> dict | list:
        """Make an authenticated request to the Vast.ai API."""
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.request(
                method,
                f"{VASTAI_API_BASE}{path}",
                headers=self._headers,
                **kwargs,
            )
            resp.raise_for_status()
            return resp.json()

    async def search_gpus(
        self,
        min_gpu_ram_gb: float = 16.0,
        gpu_type: Optional[str] = None,
        num_gpus: int = 1,
        max_dph: Optional[float] = None,
        order: str = "dph_total",
        limit: int = 20,
    ) -> list[dict]:
        """Search the Vast.ai marketplace for available GPU offers.

        Returns a list of offers sorted by price.
        """
        # Build the search query using Vast.ai's query format
        query: dict[str, Any] = {
            "gpu_ram": {"gte": min_gpu_ram_gb},
            "num_gpus": {"eq": num_gpus},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "type": "on-demand",
            "order": [[order, "asc"]],
            "limit": limit,
        }

        if gpu_type:
            query["gpu_name"] = {"eq": gpu_type}
        if max_dph:
            query["dph_total"] = {"lte": max_dph}

        try:
            import json as _json
            data = await self._request(
                "GET",
                "/bundles/",
                params={"q": _json.dumps(query)},
            )
            offers = data.get("offers", []) if isinstance(data, dict) else data
            return [
                {
                    "id": o.get("id"),
                    "gpu_name": o.get("gpu_name", "Unknown"),
                    "num_gpus": o.get("num_gpus", 1),
                    "gpu_ram_gb": round(o.get("gpu_ram", 0) / 1024, 1)
                    if o.get("gpu_ram", 0) > 100
                    else o.get("gpu_ram", 0),
                    "cpu_cores": o.get("cpu_cores_effective", 0),
                    "ram_gb": round(o.get("cpu_ram", 0) / 1024, 1),
                    "disk_gb": round(o.get("disk_space", 0), 0),
                    "dph_total": round(o.get("dph_total", 0), 4),
                    "reliability": round(o.get("reliability2", 0), 3),
                    "inet_down": round(o.get("inet_down", 0), 1),
                    "inet_up": round(o.get("inet_up", 0), 1),
                    "cuda_max_good": o.get("cuda_max_good"),
                    "machine_id": o.get("machine_id", o.get("id")),
                    "verified": o.get("verification", "") == "verified",
                }
                for o in offers[:limit]
            ]
        except Exception as e:
            logger.error(f"Failed to search Vast.ai GPUs: {e}")
            raise

    async def create_instance(
        self,
        offer_id: int,
        docker_image: str,
        disk_gb: int = 50,
        env: Optional[dict[str, str]] = None,
        onstart_cmd: Optional[str] = None,
    ) -> dict:
        """Rent a GPU instance from Vast.ai.

        Returns instance details including the instance ID.
        """
        payload: dict[str, Any] = {
            "client_id": "me",
            "image": docker_image,
            "disk": disk_gb,
            "runtype": "ssh",
        }

        if env:
            payload["env"] = env
        if onstart_cmd:
            payload["onstart"] = onstart_cmd

        try:
            data = await self._request(
                "PUT",
                f"/asks/{offer_id}/",
                json=payload,
            )
            instance_id = data.get("new_contract")
            if not instance_id:
                raise RuntimeError(f"No instance ID in response: {data}")

            return {
                "instance_id": str(instance_id),
                "success": data.get("success", True),
            }
        except Exception as e:
            logger.error(f"Failed to create Vast.ai instance: {e}")
            raise

    async def get_instance(self, instance_id: str) -> dict:
        """Get instance details."""
        try:
            data = await self._request("GET", f"/instances/{instance_id}/")
            instances = data.get("instances", [data]) if isinstance(data, dict) else data
            if not instances:
                raise RuntimeError(f"Instance {instance_id} not found")
            inst = instances[0] if isinstance(instances, list) else instances
            return {
                "id": str(inst.get("id")),
                "status": inst.get("actual_status", inst.get("status_msg", "unknown")),
                "ssh_host": inst.get("ssh_host"),
                "ssh_port": inst.get("ssh_port"),
                "gpu_name": inst.get("gpu_name"),
                "num_gpus": inst.get("num_gpus"),
                "dph_total": inst.get("dph_total"),
            }
        except Exception as e:
            logger.error(f"Failed to get Vast.ai instance {instance_id}: {e}")
            raise

    async def destroy_instance(self, instance_id: str) -> bool:
        """Destroy (terminate) a cloud GPU instance."""
        try:
            await self._request("DELETE", f"/instances/{instance_id}/")
            logger.info(f"Destroyed Vast.ai instance {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to destroy Vast.ai instance {instance_id}: {e}")
            return False

    async def execute_command(self, instance_id: str, command: str) -> dict:
        """Execute a command on a running Vast.ai instance via their API."""
        try:
            # Vast.ai allows updating the onstart script on a running instance
            # which gets executed immediately if the instance is already running
            data = await self._request(
                "PUT",
                f"/instances/{instance_id}/",
                json={"onstart": command},
            )
            return {"success": True, "data": data}
        except Exception as e:
            logger.error(f"Failed to execute command on instance {instance_id}: {e}")
            raise

    async def get_ssh_details(self, instance_id: str) -> dict:
        """Get SSH connection details for an instance."""
        info = await self.get_instance(instance_id)
        return {
            "ssh_host": info.get("ssh_host"),
            "ssh_port": info.get("ssh_port"),
            "instance_id": instance_id,
        }

    async def wait_for_ready(
        self, instance_id: str, timeout: int = 300, poll_interval: int = 10
    ) -> dict:
        """Poll until the instance is ready (running status)."""
        elapsed = 0
        while elapsed < timeout:
            info = await self.get_instance(instance_id)
            status = info.get("status", "")
            if status == "running":
                return info
            if status in ("exited", "error"):
                raise RuntimeError(
                    f"Instance {instance_id} entered terminal state: {status}"
                )
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(
            f"Instance {instance_id} not ready after {timeout}s (status: {info.get('status')})"
        )
