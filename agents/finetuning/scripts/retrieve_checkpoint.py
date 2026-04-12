#!/usr/bin/env python3
"""Retrieve trained model checkpoint from remote GPU to local workspace.

Usage:
    python retrieve_checkpoint.py --host <ssh_host> --port <ssh_port> \
        --remote-dir /workspace/models/v0.1.0-math \
        --local-dir workspace/models/v0.1.0-math
"""

import argparse
import asyncio
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_ssh_key():
    for path in [
        os.path.expanduser("~/.ssh/id_rsa"),
        os.path.expanduser("~/.ssh/id_ed25519"),
        os.path.expanduser("~/.ssh/id_ecdsa"),
        os.path.expanduser("~/.tunellm/ssh/tunellm_rsa"),
    ]:
        if os.path.exists(path):
            return path
    return None


async def retrieve(args):
    key = get_ssh_key()
    if not key:
        logger.error("No SSH key found")
        sys.exit(1)

    host = args.host
    port = args.port
    remote_dir = args.remote_dir.rstrip("/")
    local_dir = args.local_dir

    os.makedirs(local_dir, exist_ok=True)

    # First, list what's on the remote
    logger.info(f"Listing remote checkpoint at {remote_dir}...")
    ssh_cmd = (
        f"ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i {key} -p {port} "
        f"root@{host} 'find {remote_dir} -type f | head -50'"
    )
    proc = await asyncio.create_subprocess_shell(
        ssh_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    files = stdout.decode().strip().split("\n")
    logger.info(f"Found {len(files)} files on remote")

    # SCP the entire directory
    logger.info(f"Downloading checkpoint to {local_dir}...")
    scp_cmd = (
        f"scp -r -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
        f"-o BatchMode=yes -i {key} -P {port} "
        f"root@{host}:{remote_dir}/* {local_dir}/"
    )
    proc = await asyncio.create_subprocess_shell(
        scp_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)

    if proc.returncode != 0:
        logger.error(f"Download failed: {stderr.decode().strip()}")
        sys.exit(1)

    # Count local files
    local_files = []
    for root, dirs, fnames in os.walk(local_dir):
        for fname in fnames:
            local_files.append(os.path.join(root, fname))

    total_size = sum(os.path.getsize(f) for f in local_files)

    result = {
        "status": "retrieved",
        "local_dir": os.path.abspath(local_dir),
        "remote_dir": remote_dir,
        "files_count": len(local_files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }

    logger.info(f"Checkpoint retrieved: {len(local_files)} files, {result['total_size_mb']}MB")
    print(f"__CHECKPOINT_RETRIEVED__:{json.dumps(result)}")


def main():
    parser = argparse.ArgumentParser(description="Retrieve model checkpoint from remote GPU")
    parser.add_argument("--host", required=True, help="SSH host")
    parser.add_argument("--port", type=int, required=True, help="SSH port")
    parser.add_argument("--remote-dir", required=True, help="Remote model directory")
    parser.add_argument("--local-dir", required=True, help="Local directory to save to")
    args = parser.parse_args()

    asyncio.run(retrieve(args))


if __name__ == "__main__":
    main()
