# README_AGENT.md

This document tells you, the agent, exactly what to do.

All paths below are relative to this directory (`agent_end/`).

---

## Goal

Produce a file called `submission.csv` (at the root of `agent_end/`) that
contains your predicted answer (A / B / C / D) for every question in
`MedQA-USMLE-4-options/test.csv`.

---

## What you have

```
agent_end/
├── MedQA-USMLE-4-options/
│   ├── train.csv            # 10,178 rows, labeled — train on this
│   └── test.csv             # 1,273 rows, no answer columns — predict on this
├── predict.py               # inference script — fill in 2 functions
├── sample_submission.csv    # example of the output format
└── README_AGENT.md          # you are here
```

Both CSVs have these columns (the test set has `answer` / `answer_idx` stripped):

| column            | description                                                           |
|-------------------|-----------------------------------------------------------------------|
| `question`        | clinical vignette                                                     |
| `options`         | Python-dict-like string, e.g. `"{'A': '...', 'B': '...', ...}"`       |
| `meta_info`       | `step1` or `step2&3`                                                  |
| `metamap_phrases` | pre-extracted medical phrases (may be empty)                          |
| `answer`          | **train only** — full-text gold answer                                |
| `answer_idx`      | **train only** — gold letter ∈ {A, B, C, D}                           |

---

## What to do

### 1. Train a model on `train.csv`

Any approach: fine-tune an LLM, prompt a frozen LLM, build a RAG system,
train a classifier on `metamap_phrases`, use rules — whatever works. Save
your trained artifact somewhere inside `agent_end/`, e.g. `./my_model/`.

### 2. Fill in `predict.py`

Open `predict.py` and edit **only these two functions**:

```python
def load_model(model_path: str):
    # load your model and return it
    ...

def predict_one(model, question, options, meta_info, metamap_phrases) -> str:
    # return one of 'A', 'B', 'C', 'D'
    ...
```

- `options` is already a dict when it reaches `predict_one` — don't parse it again.
- You may add helper functions, extra imports, and extra CLI flags.
- Do NOT change the output CSV format or the existing CLI flags.

### 3. Run inference

The shipped `predict.py` is **LoRA-aware** — it can load a HuggingFace base model plus an optional LoRA adapter out of the box. You do NOT need to re-write `load_model` or `predict_one` unless you want to change the prompting strategy.

**Base model only:**
```bash
python predict.py \
    --test_csv    ./MedQA-USMLE-4-options/test.csv \
    --model_path  meta-llama/Llama-3.2-1B-Instruct \
    --output_csv  ./submission.csv
```

**Base model + fine-tuned LoRA adapter:**
```bash
python predict.py \
    --test_csv    ./MedQA-USMLE-4-options/test.csv \
    --model_path  meta-llama/Llama-3.2-1B-Instruct \
    --adapter_path /path/to/lora/adapter \
    --output_csv  ./submission.csv
```

This writes `submission.csv` at the root of `agent_end/`.

---

## Required output format

`submission.csv` must be a UTF-8 CSV with a header row and exactly two
columns, in this order:

```
id,predicted_idx
0,B
1,D
2,A
3,C
...
```

Rules:

- `id` = 0-indexed row number in `test.csv`. Row 0 → `id=0`, row 1 → `id=1`, etc.
- `id` must be **unique** and cover **every** row of `test.csv` (1,273 predictions total).
- `predicted_idx` ∈ `{A, B, C, D}` — uppercase, no quotes, no whitespace.
- No extra columns, no index column, no trailing metadata.
- Row order doesn't matter; the evaluator joins on `id`.

What happens if you get it wrong:

| mistake                              | consequence                   |
|--------------------------------------|-------------------------------|
| missing id                           | counted wrong (not an error)  |
| duplicate id                         | only the first is kept        |
| value outside `{A,B,C,D}`            | counted wrong                 |
| missing required column / bad header | **hard fail**, no score       |

If you use `predict.py` as provided, all of this is handled automatically.

See `sample_submission.csv` for a minimal valid example.

---

## Constraints

- Do NOT use `test.csv` for training or tuning.
- Do NOT look up gold answers for `test.csv` from external sources.
- Do NOT change the output schema of `submission.csv`.

---

## Hand-off

When done, the only file that leaves `agent_end/` is `submission.csv`. Hand it
to the user.

---

## Checklist before you finish

- [ ] `submission.csv` exists at the root of `agent_end/`.
- [ ] Header is exactly `id,predicted_idx`.
- [ ] Row count (excluding header) equals the number of rows in `test.csv` (1,273).
- [ ] Every `id` is a unique integer in `[0, 1272]`.
- [ ] Every `predicted_idx` is one of `A`, `B`, `C`, `D`.
