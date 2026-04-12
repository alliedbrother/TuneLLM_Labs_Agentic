#!/usr/bin/env python3
"""Test SSH connectivity to a remote GPU instance.

Usage:
    python connect_ssh.py --host <ssh_host> --port <ssh_port>
    python connect_ssh.py --host <ssh_host> --port <ssh_port> --command "nvidia-smi"
"""

import argparse
import asyncio
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_ssh_keys():
    """Find available SSH keys."""
    keys = []
    tunellm_key = os.path.expanduser("~/.tunellm/ssh/tunellm_rsa")
    if os.path.exists(tunellm_key):
        keys.append(tunellm_key)
    for name in ["id_rsa", "id_ed25519", "id_ecdsa"]:
        p = os.path.expanduser(f"~/.ssh/{name}")
        if os.path.exists(p) and p not in keys:
            keys.append(p)
    return keys


async def test_connection(host, port, command=None, timeout=15):
    """Test SSH connection and optionally run a command."""
    keys = get_ssh_keys()
    if not keys:
        logger.error("No SSH keys found in ~/.ssh/ or ~/.tunellm/ssh/")
        return {"status": "failed", "error": "No SSH keys available"}

    cmd_to_run = command or "echo 'SSH connection successful' && hostname && uptime"

    for key_path in keys:
        try:
            ssh_cmd = (
                f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout={timeout} "
                f"-o BatchMode=yes -i {key_path} -p {port} root@{host} "
                f"'{cmd_to_run}'"
            )
            proc = await asyncio.create_subprocess_shell(
                ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 5)

            if proc.returncode == 0:
                output = stdout.decode().strip()
                logger.info(f"SSH connected via {key_path}")
                result = {
                    "status": "connected",
                    "host": host,
                    "port": port,
                    "key": key_path,
                    "output": output,
                }
                print(f"__SSH_CONNECTED__:{json.dumps(result)}")
                return result
            else:
                logger.warning(f"SSH failed with key {key_path}: {stderr.decode().strip()}")
        except asyncio.TimeoutError:
            logger.warning(f"SSH timed out with key {key_path}")
        except Exception as e:
            logger.warning(f"SSH error with key {key_path}: {e}")

    result = {"status": "failed", "host": host, "port": port, "error": "All SSH keys failed"}
    logger.error("All SSH connection attempts failed")
    return result


def main():
    parser = argparse.ArgumentParser(description="Test SSH connectivity to remote GPU")
    parser.add_argument("--host", required=True, help="SSH host")
    parser.add_argument("--port", type=int, required=True, help="SSH port")
    parser.add_argument("--command", type=str, default=None, help="Command to run on remote")
    parser.add_argument("--timeout", type=int, default=15, help="Connection timeout in seconds")
    args = parser.parse_args()

    asyncio.run(test_connection(args.host, args.port, args.command, args.timeout))


if __name__ == "__main__":
    main()
