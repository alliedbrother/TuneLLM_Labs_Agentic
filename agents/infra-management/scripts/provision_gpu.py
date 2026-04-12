#!/usr/bin/env python3
"""Provision a remote GPU instance via Vast.ai.

Searches for available GPUs matching requirements, rents an instance,
waits for it to be ready, and outputs SSH connection details.

Usage:
    python provision_gpu.py --api-key <VASTAI_KEY> --min-gpu-ram 16 --max-dph 0.50
    python provision_gpu.py --api-key <VASTAI_KEY> --gpu-type "RTX 4090"
"""

import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))

from infra.vastai_provider import VastAIProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def provision(args):
    api_key = args.api_key or os.environ.get("VASTAI_API_KEY")
    if not api_key:
        logger.error("No Vast.ai API key provided. Use --api-key or set VASTAI_API_KEY env var.")
        sys.exit(1)

    provider = VastAIProvider(api_key)

    # Search for GPUs
    logger.info(f"Searching Vast.ai for GPUs (min RAM: {args.min_gpu_ram}GB, max $/hr: {args.max_dph})...")
    offers = await provider.search_gpus(
        min_gpu_ram_gb=args.min_gpu_ram,
        gpu_type=args.gpu_type,
        num_gpus=args.num_gpus,
        max_dph=args.max_dph,
        limit=10,
    )

    if not offers:
        logger.error("No GPU offers found matching criteria.")
        sys.exit(1)

    # Pick the cheapest reliable offer
    best = None
    for offer in offers:
        if offer.get("reliability", 0) >= 0.9:
            best = offer
            break
    if not best:
        best = offers[0]

    logger.info(f"Selected: {best['gpu_name']} x{best['num_gpus']} | "
                f"{best['gpu_ram_gb']}GB VRAM | ${best['dph_total']}/hr | "
                f"reliability: {best.get('reliability', 'N/A')}")

    # Rent the instance
    docker_image = args.docker_image or "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel"
    logger.info(f"Renting instance (offer {best['id']}, image: {docker_image})...")

    result = await provider.create_instance(
        offer_id=best["id"],
        docker_image=docker_image,
        disk_gb=args.disk_gb,
        onstart_cmd="apt-get update && apt-get install -y openssh-server && service ssh start",
    )

    instance_id = result["instance_id"]
    logger.info(f"Instance created: {instance_id}. Waiting for ready...")

    # Wait for the instance to be running
    info = await provider.wait_for_ready(instance_id, timeout=args.timeout)

    ssh_host = info.get("ssh_host")
    ssh_port = info.get("ssh_port")

    output = {
        "instance_id": instance_id,
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
        "gpu_name": best["gpu_name"],
        "num_gpus": best["num_gpus"],
        "gpu_ram_gb": best["gpu_ram_gb"],
        "dph_total": best["dph_total"],
        "docker_image": docker_image,
        "provider": "vastai",
    }

    logger.info(f"GPU ready! SSH: ssh root@{ssh_host} -p {ssh_port}")
    print(f"__GPU_PROVISIONED__:{json.dumps(output)}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Provision a remote GPU via Vast.ai")
    parser.add_argument("--api-key", type=str, help="Vast.ai API key")
    parser.add_argument("--min-gpu-ram", type=float, default=16.0, help="Minimum GPU RAM in GB")
    parser.add_argument("--gpu-type", type=str, default=None, help="Specific GPU type (e.g., 'RTX 4090')")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--max-dph", type=float, default=1.0, help="Maximum $/hr")
    parser.add_argument("--disk-gb", type=int, default=50, help="Disk space in GB")
    parser.add_argument("--docker-image", type=str, default=None, help="Docker image to use")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout waiting for instance (seconds)")
    args = parser.parse_args()

    asyncio.run(provision(args))


if __name__ == "__main__":
    main()
