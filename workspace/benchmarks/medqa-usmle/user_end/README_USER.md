# README_USER.md — Evaluation System Overview

This document describes the structure of the evaluation system end-to-end.

All paths below are relative to the repository root (`evaluation/`).

---

## 1. Roles

The system has two roles:

- **Agent** (works entirely inside `agent_end/`): produces predictions.
  Trains on `train.csv`, fills in `predict.py`, runs inference, and ends
  up with one file: `submission.csv`.
- **User** (you, works inside `user_end/`): runs the evaluation. Takes
  the agent's `submission.csv`, compares it against the gold `test.csv`,
  and gets a score.

The only thing that crosses between the two sides is `submission.csv`.

---

## 2. Directory layout

```
evaluation/
│
├── agent_end/                       ◄── shipped to the agent
│   ├── MedQA-USMLE-4-options/
│   │   ├── train.csv                # labeled training data
│   │   └── test.csv                 # test set WITHOUT answer columns
│   ├── predict.py                   # inference template to fill in
│   ├── sample_submission.csv        # output format example
│   └── README_AGENT.md              # agent's task instructions
│
└── user_end/                        ◄── kept private by you
    ├── MedQA-USMLE-4-options/
    │   └── test.csv                 # test set WITH answer columns (gold)
    ├── evaluate.py                  # the scorer
    ├── run_eval.sh                  # one-command wrapper
    ├── README_USER.md               # you are here
    └── eval_output/                 # populated after each run
        ├── graded_predictions.csv   # row-level audit
        └── score.json               # machine-readable final score
```

### Who sees what

| resource                                       | agent | user |
|------------------------------------------------|:-----:|:----:|
| `agent_end/MedQA-USMLE-4-options/train.csv`    |   ✅   |  ✅   |
| `agent_end/MedQA-USMLE-4-options/test.csv` (no answers) |   ✅   |  ✅   |
| `user_end/MedQA-USMLE-4-options/test.csv` (gold)|   ❌   |  ✅   |
| `agent_end/predict.py`                         |   ✅   |  ✅   |
| `agent_end/sample_submission.csv`              |   ✅   |  ✅   |
| `agent_end/README_AGENT.md`                    |   ✅   |  ✅   |
| `user_end/evaluate.py`                         |   ❌   |  ✅   |
| `user_end/run_eval.sh`                         |   ❌   |  ✅   |
| `user_end/README_USER.md`                      |   ❌   |  ✅   |

> The test set is split into two physically separate files: `agent_end/`'s
> copy has `answer` and `answer_idx` stripped, `user_end/`'s copy keeps
> them. Both have the same row order, so row index = `id` stays consistent
> across both sides.

---

## 3. End-to-end flow

```
  ┌────────────────────── AGENT SIDE (agent_end/) ─────────────────────┐
  │                                                                    │
  │   train.csv ──► [ train freely ] ──► ./my_model                    │
  │                                            │                       │
  │   test.csv ──────────────┐                 │                       │
  │    (no answers)          ▼                 ▼                       │
  │                      ┌─────────────────────────┐                   │
  │                      │       predict.py        │                   │
  │                      │  (load_model +          │                   │
  │                      │   predict_one filled)   │                   │
  │                      └─────────────────────────┘                   │
  │                                  │                                 │
  │                                  ▼                                 │
  │                           submission.csv ─────────────────────┐    │
  │                                                               │    │
  └───────────────────────────────────────────────────────────────┼────┘
                                                                  │
                              (hand-off: submission.csv)          │
                                                                  │
  ┌────────────────────── USER SIDE (user_end/) ──────────────────┼────┐
  │                                                               │    │
  │   test.csv (with gold answers) ──┐                            │    │
  │                                  ▼                            ▼    │
  │                           run_eval.sh  ──►  evaluate.py            │
  │                                                    │               │
  │                                                    ▼               │
  │                                      eval_output/score.json        │
  │                                      eval_output/graded_predictions.csv
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
```

---

## 4. The interface between agent and user

The only contract is `submission.csv`:

```
id,predicted_idx
0,B
1,D
2,A
...
```

- `id` = 0-indexed row number in `test.csv` (unique, covers all rows — same
  index space on both sides, since the two `test.csv` files share row order)
- `predicted_idx` ∈ `{A, B, C, D}`
- Row order doesn't matter — evaluator joins on `id`

As long as this format stays stable, the agent side (model, prompts,
training recipe) and the user side (metrics, audit format, serving
infrastructure) can each evolve independently.

---

## 5. Running an evaluation

From inside `user_end/`:

```bash
bash run_eval.sh  <submission.csv>  [test.csv]  [output_dir]
```

Arguments:

- `submission.csv` — the agent's file (**required**)
- `test.csv` — the gold test set (default: `./MedQA-USMLE-4-options/test.csv`
  inside `user_end/`)
- `output_dir` — where audit files go (default: `./eval_output`)

Example (submission has been placed at `user_end/submission.csv`):

```bash
cd user_end
bash run_eval.sh ./submission.csv
```

What `evaluate.py` does internally:

1. Loads `test.csv` and extracts `(id, answer_idx, meta_info)` as ground truth.
2. Loads `submission.csv`, normalizes types, strips whitespace, uppercases letters.
3. Validates: required columns, warns on duplicates / missing ids / invalid letters / extra ids.
4. Left-joins submission onto ground truth — missing / invalid predictions count as wrong.
5. Computes overall accuracy and per-`meta_info` accuracy.
6. Writes audit files and prints a summary.

---

## 6. Output artifacts

After each run, `user_end/eval_output/` contains:

### `score.json`

Machine-readable summary, easy to aggregate into a leaderboard:

```json
{
  "overall_accuracy": 0.5387,
  "n_total": 1273,
  "n_correct": 686,
  "by_meta_info": {
    "step1":   { "accuracy": 0.5052, "n": 679 },
    "step2&3": { "accuracy": 0.5769, "n": 594 }
  }
}
```

### `graded_predictions.csv`

Row-level audit trail:

```
id,answer_idx,predicted_idx,meta_info
0,B,B,step1
1,D,A,step1
...
```

### Terminal summary

```
============================================================
                    EVALUATION RESULT
============================================================
Overall Accuracy : 0.5387  (686 / 1273)
------------------------------------------------------------
By meta_info:
  step1       acc = 0.5052   (n = 679)
  step2&3     acc = 0.5769   (n = 594)
============================================================
```

---

## 7. Metrics

- **Primary**: overall accuracy on the 1,273-row test set.
- **Secondary**: accuracy broken down by `meta_info` (`step1` vs `step2&3`).

Accuracy is appropriate here: single-label 4-way classification, near-balanced
classes, one correct answer per row. Additional metrics (macro-F1 by letter,
Brier score, etc.) are a drop-in change inside `score()` in `evaluate.py` —
the agent-side format doesn't need to change.

---

## 8. Failure modes — by design

The scorer is lenient about *content* errors and strict about *structural*
errors. The agent should never see a crash when the real problem is "you
predicted 1,270 rows instead of 1,273".

| agent mistake                              | scorer behaviour                         |
|--------------------------------------------|------------------------------------------|
| missing rows                               | warn, count missing as wrong             |
| duplicate `id`                             | warn, keep the first occurrence          |
| `predicted_idx` outside `{A,B,C,D}`        | warn, count as wrong                     |
| extra `id` not in test.csv                 | warn, ignore                             |
| wrong CSV header (missing required column) | **hard fail with a clear error message** |
| file not found / not a CSV                 | **hard fail with a clear error message** |

---

## 9. Recommended workflow

1. **Distribute** — ship the entire `agent_end/` directory to the agent.
   Keep `user_end/` private.

2. **Collect** — the agent returns one file: `submission.csv`. Drop it
   anywhere under `user_end/` (e.g. `user_end/submission.csv`).

3. **Score** —
   ```bash
   cd user_end
   bash run_eval.sh ./submission.csv
   ```

4. **Inspect** —
   - `eval_output/score.json` — aggregate metric
   - `eval_output/graded_predictions.csv` — row-level diff for error analysis

---

## 10. Extending the system

Because the agent ↔ user interface is just `submission.csv`, changes on
either side don't force changes on the other:

- **New metric** → edit `score()` in `user_end/evaluate.py`; agent is unaffected.
- **Extra option letter (e.g. 'E')** → update `VALID_CHOICES` in both
  `user_end/evaluate.py` and `agent_end/predict.py`.
- **Remote scoring service** → wrap `run_eval.sh` in an HTTP endpoint; the
  agent still just POSTs a `submission.csv`.
- **Additional dev/validation split** → any CSV with `answer_idx` +
  `meta_info` columns works with `evaluate.py` as-is.

---

## One-line summary

> The agent works inside `agent_end/`, produces `submission.csv` from
> `test.csv` using `predict.py`. You drop that file into `user_end/`, run
> `bash run_eval.sh submission.csv`, and get a score.
