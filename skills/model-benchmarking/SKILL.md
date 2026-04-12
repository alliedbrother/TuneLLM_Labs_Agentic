---
name: model-benchmarking
description: Run comprehensive model evaluation benchmarks across capability dimensions, safety checks, and regression detection
---

# Model Benchmarking Skill

Evaluate LLM checkpoints across a standardized benchmark suite. This skill defines the evaluation protocol, benchmark categories, and reporting format.

## Benchmark Categories

### Core Capabilities
- **Reasoning**: Multi-step logical reasoning, common sense, causal inference
- **Coding**: Code generation, debugging, code explanation across languages
- **Math**: Arithmetic, algebra, calculus, word problems, proof construction
- **Language understanding**: Reading comprehension, summarization, translation
- **Instruction following**: Format compliance, constraint satisfaction, multi-step instructions

### Safety & Alignment
- **Toxicity**: Harmful content generation resistance
- **Refusal accuracy**: Correct refusal of harmful requests without over-refusing benign ones
- **Bias**: Demographic bias detection across protected categories
- **Hallucination**: Factual accuracy and calibration (knowing when it doesn't know)

### Performance
- **Latency**: Time-to-first-token and tokens-per-second
- **Consistency**: Variance across multiple runs of the same prompt

## Evaluation Protocol

1. Load model checkpoint from Model Registry
2. Run each benchmark category with fixed random seeds
3. Compute per-benchmark scores with confidence intervals
4. Compare against production baseline
5. Generate structured report with pass/fail determination

## Report Format

```yaml
model_version: v1.2.0-math
base_model: v1.1.0
evaluation_date: YYYY-MM-DD
overall_recommendation: APPROVE | REJECT | NEEDS_REVIEW

benchmarks:
  - name: benchmark_name
    category: core | safety | performance
    score: 0.85
    baseline_score: 0.82
    delta: +0.03
    confidence_interval: [0.83, 0.87]
    status: improved | regressed | neutral

safety_gate: PASS | FAIL
regression_gate: PASS | FAIL
target_improvement_gate: PASS | FAIL
```
