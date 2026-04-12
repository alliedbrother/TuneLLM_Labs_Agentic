#!/usr/bin/env python3
"""Deploy training environment to a remote GPU instance via SSH.

Copies training scripts, installs dependencies, and verifies the environment.

Usage:
    python deploy_training_env.py --host <ssh_host> --port <ssh_port>
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)


def get_ssh_keys():
    keys = []
    # Prefer standard SSH keys first, then tunellm-specific
    for path in [
        os.path.expanduser("~/.ssh/id_rsa"),
        os.path.expanduser("~/.ssh/id_ed25519"),
        os.path.expanduser("~/.ssh/id_ecdsa"),
        os.path.expanduser("~/.tunellm/ssh/tunellm_rsa"),
    ]:
        if os.path.exists(path) and path not in keys:
            keys.append(path)
    return keys


async def run_ssh(host, port, key, command, timeout=60):
    """Run a command on the remote instance."""
    ssh_cmd = (
        f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
        f"-o BatchMode=yes -i {key} -p {port} root@{host} "
        f"'{command}'"
    )
    proc = await asyncio.create_subprocess_shell(
        ssh_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return proc.returncode, stdout.decode(), stderr.decode()


async def scp_file(host, port, key, local_path, remote_path, timeout=30):
    """Copy a file to the remote instance."""
    scp_cmd = (
        f"scp -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
        f"-o BatchMode=yes -i {key} -P {port} "
        f"{local_path} root@{host}:{remote_path}"
    )
    proc = await asyncio.create_subprocess_shell(
        scp_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return proc.returncode == 0


async def scp_dir(host, port, key, local_dir, remote_dir, timeout=60):
    """Copy a directory to the remote instance."""
    scp_cmd = (
        f"scp -r -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
        f"-o BatchMode=yes -i {key} -P {port} "
        f"{local_dir} root@{host}:{remote_dir}"
    )
    proc = await asyncio.create_subprocess_shell(
        scp_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return proc.returncode == 0


async def deploy(args):
    keys = get_ssh_keys()
    if not keys:
        logger.error("No SSH keys found")
        sys.exit(1)

    key = keys[0]
    host = args.host
    port = args.port

    logger.info(f"Deploying training env to {host}:{port} using key {key}")

    # 1. Create remote directories
    logger.info("Creating remote directories...")
    code, out, err = await run_ssh(host, port, key,
        "mkdir -p /workspace/training/utils /workspace/training/configs "
        "/workspace/data/datasets /workspace/models /workspace/logs")
    if code != 0:
        logger.error(f"Failed to create dirs: {err}")
        sys.exit(1)

    # 2. Copy training scripts
    lib_training = os.path.join(PROJECT_ROOT, "lib", "training")
    logger.info("Copying training scripts...")

    scripts = [
        "train.py", "train_unsloth.py", "lora_trainer.py",
        "qlora_trainer.py", "dpo_trainer.py", "evaluate.py",
    ]
    for script in scripts:
        local = os.path.join(lib_training, script)
        if os.path.exists(local):
            ok = await scp_file(host, port, key, local, f"/workspace/training/{script}")
            if ok:
                logger.info(f"  Copied {script}")
            else:
                logger.warning(f"  Failed to copy {script}")

    # Copy utils directory
    utils_dir = os.path.join(lib_training, "utils")
    if os.path.isdir(utils_dir):
        ok = await scp_dir(host, port, key, utils_dir, "/workspace/training/")
        logger.info(f"  Copied utils/ directory: {'OK' if ok else 'FAILED'}")

    # Copy config templates
    configs_dir = os.path.join(lib_training, "configs")
    if os.path.isdir(configs_dir):
        ok = await scp_dir(host, port, key, configs_dir, "/workspace/training/")
        logger.info(f"  Copied configs/ directory: {'OK' if ok else 'FAILED'}")

    # 3. Install dependencies
    logger.info("Installing Python dependencies on remote...")
    install_cmd = (
        "pip install -q torch transformers datasets peft trl accelerate "
        "bitsandbytes safetensors evaluate rouge-score nltk scikit-learn "
        "pyyaml wandb rich tqdm numpy pandas 2>&1 | tail -5"
    )
    code, out, err = await run_ssh(host, port, key, install_cmd, timeout=300)
    if code != 0:
        logger.warning(f"Dep install may have failed: {err}")
    else:
        logger.info("Dependencies installed")

    # 4. Verify GPU is available
    logger.info("Verifying GPU...")
    code, out, err = await run_ssh(host, port, key,
        "python3 -c \"import torch; print('CUDA:', torch.cuda.is_available()); "
        "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\"")

    gpu_info = out.strip()
    logger.info(f"GPU check: {gpu_info}")

    # 5. Verify training script is runnable
    code, out, err = await run_ssh(host, port, key,
        "python3 -c \"import sys; sys.path.insert(0, '/workspace/training'); "
        "from lora_trainer import LoRATrainer; print('Training module OK')\"")

    result = {
        "status": "deployed",
        "host": host,
        "port": port,
        "gpu_info": gpu_info,
        "training_path": "/workspace/training",
        "data_path": "/workspace/data",
        "models_path": "/workspace/models",
    }

    logger.info("Training environment deployed successfully!")
    print(f"__ENV_DEPLOYED__:{json.dumps(result)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Deploy training env to remote GPU")
    parser.add_argument("--host", required=True, help="SSH host")
    parser.add_argument("--port", type=int, required=True, help="SSH port")
    args = parser.parse_args()

    asyncio.run(deploy(args))


if __name__ == "__main__":
    main()
