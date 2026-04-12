#!/usr/bin/env python3
"""Check resources on a remote GPU instance or locally.

Reports GPU status, disk usage, and system info.

Usage:
    python check_resources.py --host <ssh_host> --port <ssh_port>
    python check_resources.py --local
"""

import argparse
import asyncio
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)


def get_ssh_keys():
    keys = []
    for path in [
        os.path.expanduser("~/.tunellm/ssh/tunellm_rsa"),
        os.path.expanduser("~/.ssh/id_rsa"),
        os.path.expanduser("~/.ssh/id_ed25519"),
    ]:
        if os.path.exists(path):
            keys.append(path)
    return keys


async def check_remote(host, port, timeout=15):
    """Check resources on a remote instance via SSH."""
    keys = get_ssh_keys()
    if not keys:
        return {"status": "error", "error": "No SSH keys"}

    key = keys[0]

    check_script = """
import json, subprocess, shutil
result = {}

# GPU info via nvidia-smi
try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
         '--format=csv,noheader,nounits'],
        text=True
    ).strip()
    gpus = []
    for line in out.split('\\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            gpus.append({
                'name': parts[0],
                'memory_total_mb': int(parts[1]),
                'memory_used_mb': int(parts[2]),
                'memory_free_mb': int(parts[3]),
                'utilization_pct': int(parts[4]),
            })
    result['gpus'] = gpus
except Exception as e:
    result['gpus'] = []
    result['gpu_error'] = str(e)

# Disk
total, used, free = shutil.disk_usage('/workspace')
result['disk'] = {
    'total_gb': round(total / 1e9, 1),
    'used_gb': round(used / 1e9, 1),
    'free_gb': round(free / 1e9, 1),
}

# CPU/Memory
import os
result['cpu_count'] = os.cpu_count()
try:
    with open('/proc/meminfo') as f:
        for line in f:
            if 'MemTotal' in line:
                result['ram_total_gb'] = round(int(line.split()[1]) / 1e6, 1)
            elif 'MemAvailable' in line:
                result['ram_available_gb'] = round(int(line.split()[1]) / 1e6, 1)
except Exception:
    pass

print(json.dumps(result))
"""

    ssh_cmd = (
        f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout={timeout} "
        f"-o BatchMode=yes -i {key} -p {port} root@{host} "
        f"python3 -c '{check_script}'"
    )

    proc = await asyncio.create_subprocess_shell(
        ssh_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 10)

    if proc.returncode == 0:
        try:
            data = json.loads(stdout.decode().strip())
            data["status"] = "ok"
            data["host"] = host
            data["port"] = port
            return data
        except json.JSONDecodeError:
            return {"status": "error", "error": "Failed to parse output", "raw": stdout.decode()}
    else:
        return {"status": "error", "error": stderr.decode().strip()}


def check_local():
    """Check local resources (disk only, no GPU expected)."""
    import shutil
    workspace = os.path.join(PROJECT_ROOT, "workspace")
    result = {"status": "ok", "location": "local"}

    if os.path.exists(workspace):
        total, used, free = shutil.disk_usage(workspace)
        result["disk"] = {
            "total_gb": round(total / 1e9, 1),
            "used_gb": round(used / 1e9, 1),
            "free_gb": round(free / 1e9, 1),
        }

    result["cpu_count"] = os.cpu_count()
    return result


async def main_async(args):
    if args.local:
        result = check_local()
    else:
        if not args.host or not args.port:
            logger.error("Must provide --host and --port for remote check, or use --local")
            sys.exit(1)
        result = await check_remote(args.host, args.port)

    print(f"__RESOURCES__:{json.dumps(result)}")

    # Pretty print summary
    if result.get("gpus"):
        for i, gpu in enumerate(result["gpus"]):
            logger.info(f"GPU {i}: {gpu['name']} | "
                       f"{gpu['memory_used_mb']}/{gpu['memory_total_mb']}MB | "
                       f"{gpu['utilization_pct']}% util")
    if result.get("disk"):
        d = result["disk"]
        logger.info(f"Disk: {d['used_gb']}/{d['total_gb']}GB ({d['free_gb']}GB free)")


def main():
    parser = argparse.ArgumentParser(description="Check resources on remote or local machine")
    parser.add_argument("--host", type=str, help="SSH host")
    parser.add_argument("--port", type=int, help="SSH port")
    parser.add_argument("--local", action="store_true", help="Check local resources")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
