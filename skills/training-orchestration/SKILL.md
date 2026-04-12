---
name: training-orchestration
description: Configure, launch, monitor, and manage fine-tuning training runs with experiment tracking and checkpoint selection
---

# Training Orchestration Skill

Manage the full lifecycle of a fine-tuning training run — from config design through execution, monitoring, and checkpoint selection.

## Training Methods

| Method | When to Use | Memory | Quality |
|--------|------------|--------|---------|
| Full fine-tune | Large budget, major capability shift | Very high | Highest |
| LoRA | Default choice, targeted improvements | Medium | High |
| QLoRA | Budget constrained, large models | Low | Good |
| DoRA | When LoRA underfits | Medium | High |

Default: **LoRA** with rank 16-64 depending on model size and dataset complexity.

## Configuration Template

```yaml
run_name: improvement-cycle-{N}-{weakness_tag}
base_model: model-registry-version-id
dataset: dataset-version-id
method: lora

hyperparameters:
  learning_rate: 2e-5
  lr_scheduler: cosine
  warmup_ratio: 0.1
  batch_size: 4
  gradient_accumulation_steps: 8
  epochs: 3
  max_steps: -1  # -1 = use epochs
  weight_decay: 0.01
  max_grad_norm: 1.0

lora_config:
  rank: 32
  alpha: 64
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

evaluation:
  eval_steps: 100
  eval_strategy: steps
  metric_for_best_model: eval_loss
  greater_is_better: false

checkpointing:
  save_steps: 200
  save_total_limit: 3
  load_best_model_at_end: true

compute:
  precision: bf16
  seed: 42
  dataloader_num_workers: 4
```

## Monitoring Checklist

During training, monitor:
- [ ] Loss curve trending downward (no divergence)
- [ ] Validation loss not increasing (no overfitting)
- [ ] Gradient norms stable (no explosion)
- [ ] GPU utilization > 80% (efficient resource use)
- [ ] Learning rate following expected schedule
- [ ] No NaN/Inf values in loss

## Checkpoint Selection

1. Track validation loss at every eval step
2. Select checkpoint with lowest validation loss
3. If multiple checkpoints within 0.5% of best, prefer earlier checkpoint (less overfitting risk)
4. Record selection rationale in experiment log

## Experiment Log Format

```yaml
experiment_id: exp-{uuid}
run_name: improvement-cycle-3-math-reasoning
status: completed | failed | stopped
started_at: ISO-8601
finished_at: ISO-8601
duration_hours: 4.2
base_model: v1.1.0
dataset: ds-v1.2.0-math
config: {full config snapshot}
results:
  final_train_loss: 0.42
  best_eval_loss: 0.51
  best_checkpoint_step: 800
  total_steps: 1200
  tokens_processed: 12500000
compute:
  gpu_type: A100-80GB
  gpu_count: 4
  gpu_hours: 16.8
  estimated_cost: $45.00
```
