#!/usr/bin/env python3
"""Run model evaluation on a remote GPU or locally.

Wraps TuneLLM's evaluate.py to benchmark a model checkpoint.

Usage:
    # Remote evaluation
    python run_eval.py --host <ssh_host> --port <ssh_port> \
        --model "Qwen/Qwen2.5-1.5B-Instruct" --adapter /workspace/models/v0.1.0-math \
        --test-dataset /workspace/data/ds-v0.1.0-math/eval.jsonl \
        --output-dir workspace/eval_results/v0.1.0-math/

    # Local evaluation (if model is small enough)
    python run_eval.py --local \
        --model "Qwen/Qwen2.5-1.5B-Instruct" --adapter workspace/models/v0.1.0-math \
        --test-dataset workspace/datasets/ds-v0.1.0-math/eval.jsonl \
        --output-dir workspace/eval_results/v0.1.0-math/
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


def get_ssh_key():
    for path in [
        os.path.expanduser("~/.tunellm/ssh/tunellm_rsa"),
        os.path.expanduser("~/.ssh/id_rsa"),
        os.path.expanduser("~/.ssh/id_ed25519"),
    ]:
        if os.path.exists(path):
            return path
    return None


async def run_remote_eval(args):
    """Run evaluation on a remote GPU instance."""
    key = get_ssh_key()
    if not key:
        logger.error("No SSH key found")
        sys.exit(1)

    host = args.host
    port = args.port

    # Build the remote evaluation command
    eval_cmd = (
        f"cd /workspace/training && python3 evaluate.py "
        f"--model {args.model} "
        f"--test-dataset {args.test_dataset} "
        f"--output-dir {args.remote_output_dir or '/workspace/eval_results'} "
        f"--max-samples {args.max_samples}"
    )
    if args.adapter:
        eval_cmd += f" --adapter {args.adapter}"

    logger.info(f"Running remote evaluation on {host}:{port}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Adapter: {args.adapter or 'none'}")

    ssh_cmd = (
        f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "
        f"-o BatchMode=yes -o ServerAliveInterval=30 "
        f"-i {key} -p {port} root@{host} "
        f"'{eval_cmd}'"
    )

    proc = await asyncio.create_subprocess_shell(
        ssh_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )

    eval_metrics = None
    async def read_output(stream):
        nonlocal eval_metrics
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode().strip()
            if "__EVAL_METRICS__:" in text:
                try:
                    eval_metrics = json.loads(text.split("__EVAL_METRICS__:")[1])
                except (json.JSONDecodeError, IndexError):
                    pass
            elif text:
                logger.info(f"[eval] {text}")

    await asyncio.gather(
        read_output(proc.stdout),
        read_output(proc.stderr),
    )
    await proc.wait()

    # If we got metrics, download the results
    if eval_metrics and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info(f"Eval metrics saved to {metrics_path}")

    return eval_metrics, proc.returncode


def run_local_eval(args):
    """Run evaluation locally."""
    import subprocess

    eval_script = os.path.join(PROJECT_ROOT, "lib", "training", "evaluate.py")
    cmd = [
        sys.executable, eval_script,
        "--model", args.model,
        "--test-dataset", args.test_dataset,
        "--output-dir", args.output_dir,
        "--max-samples", str(args.max_samples),
    ]
    if args.adapter:
        cmd.extend(["--adapter", args.adapter])

    logger.info(f"Running local evaluation")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Adapter: {args.adapter or 'none'}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    eval_metrics = None
    for line in result.stdout.split("\n"):
        if "__EVAL_METRICS__:" in line:
            try:
                eval_metrics = json.loads(line.split("__EVAL_METRICS__:")[1])
            except (json.JSONDecodeError, IndexError):
                pass
        elif line.strip():
            logger.info(f"[eval] {line}")

    if result.stderr:
        for line in result.stderr.split("\n"):
            if line.strip():
                logger.info(f"[eval:err] {line}")

    return eval_metrics, result.returncode


async def main_async(args):
    if args.local:
        metrics, code = run_local_eval(args)
    else:
        if not args.host or not args.port:
            logger.error("Must provide --host and --port for remote eval, or use --local")
            sys.exit(1)
        metrics, code = await run_remote_eval(args)

    if metrics:
        logger.info(f"Evaluation complete. Metrics: {json.dumps(metrics, indent=2)}")
        print(f"__EVAL_COMPLETE__:{json.dumps(metrics)}")
    else:
        logger.error("No evaluation metrics captured")
        sys.exit(1 if code != 0 else 0)


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--test-dataset", required=True, help="Test dataset path")
    parser.add_argument("--output-dir", required=True, help="Local output directory for results")
    parser.add_argument("--remote-output-dir", default=None, help="Remote output dir (if remote eval)")
    parser.add_argument("--max-samples", type=int, default=50, help="Max eval samples")
    parser.add_argument("--host", type=str, help="SSH host (for remote eval)")
    parser.add_argument("--port", type=int, help="SSH port (for remote eval)")
    parser.add_argument("--local", action="store_true", help="Run evaluation locally")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
