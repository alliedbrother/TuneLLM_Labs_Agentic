# Benchmarks

Standardized evaluation harnesses for measuring model performance on specific tasks. Each benchmark follows an **agent/user split** — the agent produces predictions without seeing gold answers, then a separate scorer compares submissions against the held-out truth.

## Pattern

Every benchmark has the same structure:

```
workspace/benchmarks/<benchmark-name>/
├── agent_end/                       ◄── The Evaluation Agent works here
│   ├── <dataset>/
│   │   ├── train.csv                # Labeled training data (optional)
│   │   └── test.csv                 # Test set WITHOUT answer columns
│   ├── predict.py                   # Inference script (LoRA-aware)
│   ├── sample_submission.csv        # Output format example
│   └── README_AGENT.md              # Agent's task instructions
│
└── user_end/                        ◄── Kept private from the agent
    ├── <dataset>/
    │   └── test.csv                 # Test set WITH answer columns (gold)
    ├── evaluate.py                  # The scorer
    ├── run_eval.sh                  # One-command wrapper
    ├── README_USER.md               # Evaluator's guide
    └── eval_output/                 # Generated after each run
        ├── graded_predictions.csv   # Row-level audit
        └── score.json               # Machine-readable score
```

## The Contract

The ONLY file that crosses the agent↔user boundary is **`submission.csv`**:

```csv
id,predicted_idx
0,B
1,D
2,A
...
```

- `id` = 0-indexed row number in test.csv (unique, covers all rows)
- `predicted_idx` = the model's answer (format depends on benchmark)

## Available Benchmarks

### MedQA-USMLE-4-options (`medqa-usmle/`)

Medical licensing exam questions with 4-way multiple choice.

- **Train**: 10,178 rows
- **Test**: 1,273 rows
- **Metric**: overall accuracy + accuracy by meta_info (step1 vs step2&3)
- **Valid predictions**: A, B, C, or D

## How the Evaluation Agent Uses Benchmarks

When a task asks for benchmark evaluation:

1. **Navigate** to `workspace/benchmarks/<benchmark-name>/agent_end/`
2. **Run `predict.py`** with the base model and optional LoRA adapter:
   ```bash
   python predict.py \
       --test_csv ./MedQA-USMLE-4-options/test.csv \
       --model_path meta-llama/Llama-3.2-1B-Instruct \
       --adapter_path /path/to/adapter \
       --output_csv ./submission.csv
   ```
3. **Move to `user_end/`** and run the scorer:
   ```bash
   cd ../user_end
   bash run_eval.sh ../agent_end/submission.csv
   ```
4. **Read results** from `user_end/eval_output/score.json`
5. **Compare** the fine-tuned model's score against the base model's score
6. **Report** accuracy delta in the Paperclip task comment

## Pre vs Post Benchmark Evaluation

For fine-tuning pipelines, the Evaluation Agent should run the benchmark **twice**:

1. **Pre-eval**: base model only → baseline accuracy
2. **Post-eval**: base model + trained adapter → fine-tuned accuracy
3. **Compare**: delta in overall accuracy and by meta_info

This gives a clear, standardized measure of whether fine-tuning helped on the target benchmark.

## Adding New Benchmarks

To add a new benchmark, follow the structure above:
1. Create `workspace/benchmarks/<new-name>/agent_end/` with train.csv, test.csv (answers stripped), predict.py, README_AGENT.md
2. Create `workspace/benchmarks/<new-name>/user_end/` with test.csv (gold), evaluate.py, run_eval.sh, README_USER.md
3. Define the submission.csv contract (columns and valid values) in both READMEs
4. Add an entry to this README under "Available Benchmarks"
