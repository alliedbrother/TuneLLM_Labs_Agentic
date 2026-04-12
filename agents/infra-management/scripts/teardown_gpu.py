#!/usr/bin/env python3
"""Teardown (destroy) a remote GPU instance.

Usage:
    python teardown_gpu.py --api-key <VASTAI_KEY> --instance-id <ID>
    python teardown_gpu.py --api-key <VASTAI_KEY> --instance-id <ID> --provider vastai
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


async def teardown(args):
    api_key = args.api_key or os.environ.get("VASTAI_API_KEY")
    if not api_key:
        logger.error("No Vast.ai API key. Use --api-key or set VASTAI_API_KEY.")
        sys.exit(1)

    provider = VastAIProvider(api_key)
    instance_id = args.instance_id

    logger.info(f"Destroying instance {instance_id}...")
    success = await provider.destroy_instance(instance_id)

    result = {
        "instance_id": instance_id,
        "status": "destroyed" if success else "failed",
        "provider": "vastai",
    }

    if success:
        logger.info(f"Instance {instance_id} destroyed successfully")
    else:
        logger.error(f"Failed to destroy instance {instance_id}")

    print(f"__GPU_TEARDOWN__:{json.dumps(result)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Teardown a remote GPU instance")
    parser.add_argument("--api-key", type=str, help="Vast.ai API key")
    parser.add_argument("--instance-id", required=True, help="Instance ID to destroy")
    parser.add_argument("--provider", default="vastai", help="Cloud provider (default: vastai)")
    args = parser.parse_args()

    asyncio.run(teardown(args))


if __name__ == "__main__":
    main()
