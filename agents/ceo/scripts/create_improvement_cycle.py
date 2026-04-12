#!/usr/bin/env python3
"""Create a full improvement cycle as a chain of Paperclip subtasks.

Creates the task chain:
  1. Data Selection → curate dataset
  2. Infra Management → provision GPU
  3. Finetuning → train model
  4. Model Registry → register checkpoint
  5. Evaluation → benchmark model
  6. Infra Management → teardown GPU
  7. Model Registry → promote (if approved)

Usage:
    python create_improvement_cycle.py \
        --weakness "math_reasoning" \
        --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
        --dataset-source "hf://gsm8k" \
        --parent-task-id <paperclip_task_id>
"""

import argparse
import json
import logging
import os
import sys

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_URL = os.environ.get("PAPERCLIP_API_URL", "http://127.0.0.1:3200")
API_KEY = os.environ.get("PAPERCLIP_API_KEY", "")
COMPANY_ID = os.environ.get("PAPERCLIP_COMPANY_ID", "")
RUN_ID = os.environ.get("PAPERCLIP_RUN_ID", "")

# Agent IDs (from our Paperclip company)
AGENT_IDS = {
    "ceo": "826cd065-4b44-4b72-bd48-e61f211257a1",
    "data-selection": "d3893b68-0bb5-4e2e-8817-79cc0d5e81c6",
    "data-creation": "2304b058-d02e-4631-8ee5-1164c398c5e0",
    "finetuning": "57bc5441-4e38-4272-9ee1-4ed30a9072e5",
    "evaluation": "628a9be4-09f0-4135-b8a7-f9e423ddf3f3",
    "inference": "84a43e25-fb3b-4f1f-a501-6915af75d278",
    "infra-management": "9a545453-7cdd-4f15-9405-e69f013e4e3b",
    "model-registry": "ad4b42b2-bc9b-4023-b058-ccef1dbab4b6",
}


def api_headers():
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    if RUN_ID:
        headers["X-Paperclip-Run-Id"] = RUN_ID
    return headers


def create_issue(title, description, assignee_agent_id, parent_id=None, goal_id=None):
    """Create a Paperclip issue/task."""
    payload = {
        "title": title,
        "description": description,
        "assigneeAgentId": assignee_agent_id,
        "status": "todo",
    }
    if parent_id:
        payload["parentId"] = parent_id
    if goal_id:
        payload["goalId"] = goal_id

    resp = httpx.post(
        f"{API_URL}/api/companies/{COMPANY_ID}/issues",
        headers=api_headers(),
        json=payload,
    )
    resp.raise_for_status()
    issue = resp.json()
    logger.info(f"Created task: {title} → {issue.get('identifier', issue.get('id'))}")
    return issue


def main():
    parser = argparse.ArgumentParser(description="Create improvement cycle task chain")
    parser.add_argument("--weakness", required=True, help="Target weakness (e.g., math_reasoning)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model")
    parser.add_argument("--dataset-source", default="hf://gsm8k", help="Dataset source")
    parser.add_argument("--parent-task-id", default=None, help="Parent task ID")
    parser.add_argument("--method", default="lora", help="Training method")
    parser.add_argument("--version-tag", default=None, help="Version tag (auto-generated if omitted)")
    args = parser.parse_args()

    if not COMPANY_ID:
        logger.error("PAPERCLIP_COMPANY_ID not set")
        sys.exit(1)

    version_tag = args.version_tag or f"v0.1.0-{args.weakness.replace('_', '-')}"
    dataset_version = f"ds-{version_tag}"
    parent = args.parent_task_id

    workspace = os.environ.get("WORKSPACE", "workspace")

    logger.info(f"Creating improvement cycle for weakness: {args.weakness}")
    logger.info(f"  Base model: {args.base_model}")
    logger.info(f"  Version tag: {version_tag}")

    # Task 1: Curate dataset
    task1 = create_issue(
        title=f"Curate dataset for {args.weakness}",
        description=f"""## Objective
Curate a training dataset to improve {args.weakness} capability.

## Parameters
- Source: `{args.dataset_source}`
- Format: Alpaca (instruction/input/output)
- Max samples: 5000
- Eval split: 10%
- Output: `{workspace}/datasets/{dataset_version}/`

## Scripts
```bash
source env.sh
python agents/data-selection/scripts/select_data.py \\
    --source "{args.dataset_source}" \\
    --max-samples 5000 \\
    --output-dir {workspace}/datasets/{dataset_version}/

python agents/data-selection/scripts/validate_dataset.py \\
    --dataset-dir {workspace}/datasets/{dataset_version}/

python agents/data-selection/scripts/create_manifest.py \\
    --dataset-dir {workspace}/datasets/{dataset_version}/ \\
    --source "{args.dataset_source}" --weakness "{args.weakness}"
```

Post the dataset path and manifest summary as a comment when done.""",
        assignee_agent_id=AGENT_IDS["data-selection"],
        parent_id=parent,
    )

    # Task 2: Provision GPU
    task2 = create_issue(
        title=f"Provision GPU for {version_tag} training",
        description=f"""## Objective
Provision a remote GPU instance for fine-tuning.

## Requirements
- Minimum 16GB VRAM (24GB preferred for comfort)
- Max $0.50/hr
- Provider: Vast.ai

## Scripts
```bash
source env.sh
python agents/infra-management/scripts/provision_gpu.py \\
    --min-gpu-ram 16 --max-dph 0.50

# After provisioning, deploy training environment:
python agents/infra-management/scripts/deploy_training_env.py \\
    --host <SSH_HOST> --port <SSH_PORT>
```

Post the SSH connection details (host, port, instance_id) as a comment when done.""",
        assignee_agent_id=AGENT_IDS["infra-management"],
        parent_id=parent,
    )

    # Task 3: Fine-tune
    task3 = create_issue(
        title=f"Fine-tune model {version_tag}",
        description=f"""## Objective
Fine-tune {args.base_model} using {args.method} to improve {args.weakness}.

## Inputs
- Base model: `{args.base_model}`
- Dataset: `{workspace}/datasets/{dataset_version}/` (from data selection task)
- SSH details: (from GPU provisioning task)
- Method: `{args.method}`

## Scripts
```bash
source env.sh

# 1. Generate config
python agents/finetuning/scripts/generate_config.py \\
    --base-model "{args.base_model}" \\
    --dataset-path {workspace}/datasets/{dataset_version}/train.jsonl \\
    --method {args.method} --run-name "{version_tag}" \\
    --output agents/finetuning/configs/{version_tag}.yaml

# 2. Deploy and train on remote GPU
python agents/finetuning/scripts/deploy_and_train.py \\
    --host <SSH_HOST> --port <SSH_PORT> \\
    --config agents/finetuning/configs/{version_tag}.yaml \\
    --dataset-dir {workspace}/datasets/{dataset_version}/

# 3. Retrieve checkpoint
python agents/finetuning/scripts/retrieve_checkpoint.py \\
    --host <SSH_HOST> --port <SSH_PORT> \\
    --remote-dir /workspace/models/{version_tag} \\
    --local-dir {workspace}/models/{version_tag}/

# 4. Package model
python agents/finetuning/scripts/package_model.py \\
    --model-dir {workspace}/models/{version_tag}/ \\
    --base-model "{args.base_model}" \\
    --dataset-version "{dataset_version}" \\
    --method {args.method} --weakness "{args.weakness}"
```

Post training metrics and checkpoint path as a comment when done.""",
        assignee_agent_id=AGENT_IDS["finetuning"],
        parent_id=parent,
    )

    # Task 4: Register model
    task4 = create_issue(
        title=f"Register model {version_tag}",
        description=f"""## Objective
Register the trained model checkpoint in the model registry.

## Scripts
```bash
source env.sh
python agents/model-registry/scripts/register_model.py \\
    --version "{version_tag}" \\
    --base-model "{args.base_model}" \\
    --dataset "{dataset_version}" \\
    --method {args.method} \\
    --weakness "{args.weakness}" \\
    --artifacts-path {workspace}/models/{version_tag}/
```""",
        assignee_agent_id=AGENT_IDS["model-registry"],
        parent_id=parent,
    )

    # Task 5: Evaluate
    task5 = create_issue(
        title=f"Evaluate model {version_tag}",
        description=f"""## Objective
Run full evaluation suite on {version_tag} and compare against baseline.

## Scripts
```bash
source env.sh

# Run evaluation (on remote GPU if still available, or locally)
python agents/evaluation/scripts/run_eval.py \\
    --model "{args.base_model}" \\
    --adapter {workspace}/models/{version_tag}/ \\
    --test-dataset {workspace}/datasets/{dataset_version}/eval.jsonl \\
    --output-dir {workspace}/eval_results/{version_tag}/ \\
    --local

# Compare against baseline
python agents/evaluation/scripts/compare_models.py \\
    --baseline {workspace}/eval_results/v0.0.1/eval_metrics.json \\
    --candidate {workspace}/eval_results/{version_tag}/eval_metrics.json \\
    --output {workspace}/eval_results/{version_tag}/report.yaml \\
    --target-weakness "{args.weakness}"
```

Post the recommendation (APPROVE/REJECT/NEEDS_REVIEW) and key metrics as a comment.""",
        assignee_agent_id=AGENT_IDS["evaluation"],
        parent_id=parent,
    )

    # Task 6: Teardown GPU
    task6 = create_issue(
        title=f"Teardown GPU after {version_tag} training",
        description=f"""## Objective
Destroy the cloud GPU instance to stop billing.

## Scripts
```bash
source env.sh
python agents/infra-management/scripts/teardown_gpu.py \\
    --instance-id <INSTANCE_ID>
```

Use the instance_id from the GPU provisioning task.""",
        assignee_agent_id=AGENT_IDS["infra-management"],
        parent_id=parent,
    )

    created = [task1, task2, task3, task4, task5, task6]
    ids = [t.get("identifier", t.get("id")) for t in created]

    logger.info(f"\nImprovement cycle created! {len(created)} tasks:")
    for t in created:
        logger.info(f"  {t.get('identifier', '?')}: {t.get('title', '?')}")

    print(f"__CYCLE_CREATED__:{json.dumps({'task_ids': ids, 'version_tag': version_tag})}")


if __name__ == "__main__":
    main()
