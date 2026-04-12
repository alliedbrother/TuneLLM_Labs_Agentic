#!/usr/bin/env python3
"""Deploy dataset and config to remote GPU, launch training, and stream logs.

This is the main Finetuning Agent execution script. It:
1. SCPs the dataset and training config to the remote GPU
2. Launches the training script via SSH
3. Streams training logs in real-time
4. Reports progress via stdout markers

Usage:
    python deploy_and_train.py --host <ssh_host> --port <ssh_port> \
        --config <local_config.yaml> --dataset-dir <local_dataset_dir> \
        --output-dir <remote_output_dir>
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

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


async def scp(host, port, key, local, remote, is_dir=False, timeout=120):
    """SCP a file or directory to the remote."""
    flag = "-r" if is_dir else ""
    cmd = (
        f"scp {flag} -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
        f"-o BatchMode=yes -i {key} -P {port} "
        f"{local} root@{host}:{remote}"
    )
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    if proc.returncode != 0:
        logger.error(f"SCP failed: {stderr.decode().strip()}")
        return False
    return True


async def run_ssh(host, port, key, command, timeout=60):
    """Run a command on the remote and return output."""
    ssh_cmd = (
        f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
        f"-o BatchMode=yes -i {key} -p {port} root@{host} "
        f"'{command}'"
    )
    proc = await asyncio.create_subprocess_shell(
        ssh_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return proc.returncode, stdout.decode(), stderr.decode()


async def run_ssh_streaming(host, port, key, command):
    """Run a long-running command on remote and stream stdout in real-time."""
    ssh_cmd = (
        f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
        f"-o BatchMode=yes -o ServerAliveInterval=30 "
        f"-i {key} -p {port} root@{host} "
        f"'{command}'"
    )
    proc = await asyncio.create_subprocess_shell(
        ssh_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )

    training_complete = False
    final_metrics = None
    last_progress = None

    async def read_stream(stream, label):
        nonlocal training_complete, final_metrics, last_progress
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode().strip()
            if not text:
                continue

            # Parse structured markers from training output
            if "__PROGRESS__:" in text:
                try:
                    progress = json.loads(text.split("__PROGRESS__:")[1])
                    last_progress = progress
                    print(f"__TRAINING_PROGRESS__:{json.dumps(progress)}", flush=True)
                except (json.JSONDecodeError, IndexError):
                    pass
            elif "__FINAL_METRICS__:" in text:
                try:
                    final_metrics = json.loads(text.split("__FINAL_METRICS__:")[1])
                    print(f"__TRAINING_FINAL_METRICS__:{json.dumps(final_metrics)}", flush=True)
                except (json.JSONDecodeError, IndexError):
                    pass
            elif "__TRAINING_COMPLETE__" in text:
                training_complete = True
                logger.info("Training complete signal received!")
            elif "__BASELINE_METRICS__:" in text:
                try:
                    baseline = json.loads(text.split("__BASELINE_METRICS__:")[1])
                    print(f"__TRAINING_BASELINE__:{json.dumps(baseline)}", flush=True)
                except (json.JSONDecodeError, IndexError):
                    pass
            else:
                # Log non-marker output
                logger.info(f"[{label}] {text}")

    await asyncio.gather(
        read_stream(proc.stdout, "stdout"),
        read_stream(proc.stderr, "stderr"),
    )
    await proc.wait()

    return proc.returncode, training_complete, final_metrics, last_progress


async def deploy_and_train(args):
    key = get_ssh_key()
    if not key:
        logger.error("No SSH key found")
        sys.exit(1)

    host = args.host
    port = args.port
    start_time = time.time()

    # 1. Create remote directories
    logger.info("Setting up remote directories...")
    await run_ssh(host, port, key,
        "mkdir -p /workspace/data /workspace/models /workspace/configs /workspace/logs")

    # 2. Upload dataset
    logger.info(f"Uploading dataset from {args.dataset_dir}...")
    ok = await scp(host, port, key, args.dataset_dir, "/workspace/data/", is_dir=True, timeout=300)
    if not ok:
        logger.error("Failed to upload dataset")
        sys.exit(1)
    dataset_name = os.path.basename(args.dataset_dir.rstrip("/"))
    remote_dataset = f"/workspace/data/{dataset_name}/train.jsonl"
    logger.info(f"Dataset uploaded to {remote_dataset}")

    # 3. Upload config (modify dataset path to point to remote location)
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["dataset"]["source"] = remote_dataset
    remote_config_path = f"/workspace/configs/{os.path.basename(args.config)}"

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config, tmp, default_flow_style=False)
        tmp_config = tmp.name

    ok = await scp(host, port, key, tmp_config, remote_config_path)
    os.unlink(tmp_config)
    if not ok:
        logger.error("Failed to upload config")
        sys.exit(1)
    logger.info(f"Config uploaded to {remote_config_path}")

    # 4. Launch training
    remote_output_dir = args.remote_output_dir or f"/workspace/models/{config.get('run_name', 'run')}"
    train_cmd = (
        f"cd /workspace/training && "
        f"python3 train.py --config {remote_config_path} "
        f"--output-dir {remote_output_dir} 2>&1"
    )

    logger.info(f"Launching training: {config.get('run_name', 'run')}")
    logger.info(f"  Model: {config.get('base_model')}")
    logger.info(f"  Method: {config.get('method')}")
    logger.info(f"  Output: {remote_output_dir}")

    exit_code, complete, metrics, progress = await run_ssh_streaming(
        host, port, key, train_cmd
    )

    elapsed = time.time() - start_time

    result = {
        "status": "completed" if complete and exit_code == 0 else "failed",
        "exit_code": exit_code,
        "training_complete": complete,
        "elapsed_seconds": round(elapsed),
        "elapsed_hours": round(elapsed / 3600, 2),
        "remote_output_dir": remote_output_dir,
        "final_metrics": metrics,
        "last_progress": progress,
        "config": {
            "run_name": config.get("run_name"),
            "base_model": config.get("base_model"),
            "method": config.get("method"),
        },
    }

    if complete and exit_code == 0:
        logger.info(f"Training completed successfully in {elapsed/3600:.2f} hours!")
    else:
        logger.error(f"Training failed with exit code {exit_code}")

    print(f"__TRAINING_RESULT__:{json.dumps(result)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Deploy and train on remote GPU")
    parser.add_argument("--host", required=True, help="SSH host")
    parser.add_argument("--port", type=int, required=True, help="SSH port")
    parser.add_argument("--config", required=True, help="Local training config YAML")
    parser.add_argument("--dataset-dir", required=True, help="Local dataset directory to upload")
    parser.add_argument("--remote-output-dir", default=None, help="Remote output directory for model")
    args = parser.parse_args()

    asyncio.run(deploy_and_train(args))


if __name__ == "__main__":
    main()
