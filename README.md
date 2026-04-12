# TuneLLM Labs Agentic

An autonomous multi-agent system for fine-tuning LLMs. Create one ticket, and a team of AI agents handles the entire pipeline — from dataset curation to GPU provisioning, training, evaluation, and model registration — with zero human intervention.

Built on [Paperclip](https://github.com/paperclipai/paperclip), an open-source orchestration framework for AI agent companies.

## How It Works

```
You create a ticket: "Fine-tune Llama 3.2 1B on Python coding"
                              │
                              ▼
                    CEO Agent reads the ticket
                    Creates the first pipeline task
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                     ▼
   Data Selection      Infra Agent          Finetuning Agent
   Searches HF Hub     Provisions GPU       Trains with Unsloth
   Downloads data      on Vast.ai           Pre + Post eval
   Formats to JSONL    Writes SSH info      Retrieves adapter
         │                    │                     │
         └────────┬───────────┘                     │
                  ▼                                  ▼
           Each agent hands off              Evaluation Agent
           to the next one                   APPROVE / REJECT
           automatically                            │
                                                    ▼
                                             Model Registry
                                             Registers version
                                                    │
                                                    ▼
                                             Infra Teardown
                                             Destroys GPU
```

**Key idea:** Each agent finishes its work, then creates a task for the next agent in the chain. Paperclip auto-wakes the assigned agent. The pipeline is self-propelling — no human needed after the initial ticket.

## The Agents

| Agent | Role | What It Does |
|-------|------|-------------|
| **CEO** | Orchestrator | Reads the ticket, decides model/dataset/method, kicks off the pipeline |
| **Data Selection** | Data Curator | Searches HuggingFace Hub, web, local files. Downloads and formats datasets. Never generates data. |
| **Data Creation** | Synthetic Data | Creates training data from PDFs and documents using LLM-powered Q&A generation |
| **Infra Management** | GPU Ops | Provisions NVIDIA GPUs on Vast.ai/Lambda/AWS. Detects CUDA version. Writes `active_gpu.yaml` |
| **Finetuning** | Training | Runs Unsloth LoRA/QLoRA training on remote GPU with pre and post evaluation |
| **Evaluation** | Quality Gate | Benchmarks models (F1, ROUGE, BLEU), detects regressions, produces APPROVE/REJECT |
| **Model Registry** | Versioning | Registers model versions, tracks lineage, manages promotion stages |
| **Inference** | Serving | Deploys models with vLLM + KV caching for high-performance serving |

## Pipeline Modes

**End-to-end pipeline:** Add a `## Pipeline Context` section to the task description. Each agent hands off to the next automatically.

**Standalone tasks:** No pipeline context = agent does its work and stops. Useful for one-off data curation, GPU provisioning, or evaluation.

---

## Setup Guide

### Prerequisites

- **macOS or Linux** (tested on macOS with Apple Silicon)
- **Python 3.10+**
- **Node.js 18+** and **pnpm** (`npm install -g pnpm`)
- **Git**
- **SSH key** in `~/.ssh/` (for remote GPU access)
- **Vast.ai account** with API key (for GPU provisioning) — [vast.ai](https://vast.ai)
- **HuggingFace account** with API token — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **Claude Code CLI** — [claude.ai/code](https://claude.ai/code)

### Step 1: Clone the repo

```bash
git clone https://github.com/alliedbrother/TuneLLM_Labs_Agentic.git
cd TuneLLM_Labs_Agentic
```

### Step 2: Clone Paperclip (the orchestration framework)

```bash
git clone https://github.com/paperclipai/paperclip.git
cd paperclip
pnpm install
pnpm build
cd ..
```

### Step 3: Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Create a Paperclip instance

```bash
cd paperclip
pnpm paperclipai onboard -y
```

This creates a local Paperclip instance with an embedded PostgreSQL database. Follow the prompts if it asks for configuration.

### Step 5: Start Paperclip

```bash
pnpm paperclipai run
```

Note the port it starts on (default: 3100). Keep this terminal running.

### Step 6: Import the agent company

In a new terminal:

```bash
# Copy company package to a clean dir (avoid including the paperclip/ subdirectory)
mkdir -p /tmp/tunellm-import
cp COMPANY.md .paperclip.yaml /tmp/tunellm-import/
cp -r agents teams skills projects tasks /tmp/tunellm-import/

# Import into Paperclip
cd paperclip
pnpm paperclipai company import \
  --from /tmp/tunellm-import \
  --include company,agents,skills,projects,tasks \
  --target new
```

### Step 7: Configure API keys

Add your API keys to all agents via the Paperclip UI (http://127.0.0.1:3100) or via API:

```bash
# Get the company ID and agent IDs
pnpm paperclipai company list
pnpm paperclipai agent list -C <company-id>

# For each agent, add env vars:
curl -X PATCH "http://127.0.0.1:3100/api/agents/<agent-id>" \
  -H "Content-Type: application/json" \
  -d '{
    "adapterConfig": {
      "model": "claude-sonnet-4-6",
      "dangerouslySkipPermissions": true,
      "env": {
        "HF_TOKEN": "hf_your_token_here",
        "VASTAI_API_KEY": "your_vastai_key_here"
      }
    }
  }'
```

Or use the Paperclip UI → Agent → Settings → Environment Variables.

### Step 8: Run your first fine-tuning pipeline

Create a ticket for the CEO:

```bash
curl -X POST "http://127.0.0.1:3100/api/companies/<company-id>/issues" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "End-to-end fine-tune: Llama 3.2 1B on Python code instructions",
    "description": "Run the full fine-tuning pipeline.\n\nModel: meta-llama/Llama-3.2-1B-Instruct\nDataset: iamtarun/python_code_instructions_18k_alpaca\nMethod: LoRA via Unsloth\nVersion tag: v0.1.0-python-code",
    "assigneeAgentId": "<ceo-agent-id>",
    "status": "todo"
  }'
```

The CEO will wake up automatically and start the pipeline. Monitor progress in the Paperclip UI or via:

```bash
pnpm paperclipai issue list -C <company-id> --status todo,in_progress
```

---

## Project Structure

```
TuneLLM_Labs_Agentic/
│
├── agents/                        # Agent definitions (AGENTS.md) and helper scripts
│   ├── ceo/                       # Orchestrator — delegates, never executes
│   ├── data-selection/            # Searches and downloads datasets
│   ├── data-creation/             # Creates data from PDFs/documents
│   ├── finetuning/                # Trains models on remote GPUs
│   ├── evaluation/                # Benchmarks and compares models
│   ├── infra-management/          # Provisions and destroys cloud GPUs
│   ├── model-registry/            # Tracks model versions
│   └── inference/                 # Deploys models with vLLM
│
├── lib/                           # Shared libraries (adapted from TuneLLM)
│   ├── training/                  # Training scripts, trainers, evaluation, metrics
│   └── infra/                     # Vast.ai, AWS, SSH provider clients
│
├── workspace/                     # Shared workspace for agent outputs
│   ├── datasets/                  # Prepared datasets (train.jsonl + eval.jsonl)
│   ├── models/                    # Trained model checkpoints
│   ├── eval_results/              # Evaluation metrics and reports
│   ├── infra/                     # active_gpu.yaml — current GPU info
│   └── registry/                  # registry.yaml — model version tracking
│
├── datasets/                      # Raw data lake (downloaded by Data Selection Agent)
│
├── skills/                        # Paperclip skill definitions
├── teams/                         # Team groupings
├── projects/                      # Project definitions
├── tasks/                         # Recurring task templates
│
├── COMPANY.md                     # Paperclip company definition
├── .paperclip.yaml                # Agent adapter configuration
├── env.sh                         # Environment variables for agents
└── requirements.txt               # Python dependencies
```

## Key Conventions

### Data Flow

```
datasets/                          → Raw downloads (organized by topic)
workspace/datasets/ds-v{X}-{tag}/  → Prepared datasets (Alpaca JSONL + manifest)
workspace/models/v{X}-{tag}/       → Trained adapters + metadata
workspace/eval_results/v{X}-{tag}/ → Evaluation metrics + reports
workspace/registry/registry.yaml   → Model version registry
workspace/infra/active_gpu.yaml    → Current GPU connection info
```

### Pipeline Context

The `## Pipeline Context` section in task descriptions is the signal for end-to-end pipeline mode. It carries forward accumulated information from each agent:

```
## Pipeline Context
pipeline: e2e-finetune
topic: python_code_instructions
base_model: meta-llama/Llama-3.2-1B-Instruct
method: lora
version_tag: v0.2.0-python-code
dataset_path: /path/to/prepared/dataset
```

Without this section, agents operate in standalone mode — no handoffs.

### GPU Info Contract

The Infra Agent writes `workspace/infra/active_gpu.yaml` with SSH details and CUDA version. All downstream agents read this file to connect to the GPU and install CUDA-matched dependencies.

---

## Adapting for Your Use Case

### Use a different model
Change `base_model` in the ticket description. The Infra Agent auto-sizes the GPU (1-3B → 16GB, 7B → 24GB, 13B → 48GB, 70B → 80GB+).

### Use a different dataset
Point to any HuggingFace dataset or provide a local path. The Data Selection Agent handles format detection and conversion.

### Use PDFs as training data
Assign the task to the Data Creation Agent instead. It extracts text, chunks it, and generates Q&A pairs using the Anthropic API.

### Use a different GPU provider
The Infra Agent supports Vast.ai (default), Lambda Labs, AWS EC2, and generic SSH. Set the appropriate API key and specify the provider in the task.

---

## Credits

- **[Paperclip](https://github.com/paperclipai/paperclip)** — Agent orchestration framework
- **[TuneLLM](https://github.com/alliedbrother/TuneLLM)** — Training and evaluation code adapted from this project
- **[Unsloth](https://github.com/unslothai/unsloth)** — 2x faster LoRA training with 70% less VRAM
- **[vLLM](https://github.com/vllm-project/vllm)** — High-performance inference with KV caching

## License

MIT
