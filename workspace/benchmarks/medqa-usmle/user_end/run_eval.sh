#!/usr/bin/env bash
# ==============================================================================
# run_eval.sh — score a submission.csv against the held-out test set.
#
# Run from inside user_end/:
#     bash run_eval.sh  <submission.csv>  [test.csv]  [output_dir]
#
# Defaults:
#     test.csv   = ./MedQA-USMLE-4-options/test.csv
#     output_dir = ./eval_output
# ==============================================================================
set -euo pipefail

SUBMISSION="${1:?missing arg 1: path to submission.csv}"
TEST_CSV="${2:-./MedQA-USMLE-4-options/test.csv}"
OUTPUT_DIR="${3:-./eval_output}"

if [[ ! -f "$SUBMISSION" ]]; then
    echo "[ERROR] submission not found: $SUBMISSION" >&2
    exit 1
fi
if [[ ! -f "$TEST_CSV" ]]; then
    echo "[ERROR] test.csv not found: $TEST_CSV" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATOR="$SCRIPT_DIR/evaluate.py"
if [[ ! -f "$EVALUATOR" ]]; then
    echo "[ERROR] evaluate.py not found next to run_eval.sh at $EVALUATOR" >&2
    exit 1
fi

echo "[INFO] submission : $SUBMISSION"
echo "[INFO] test_csv   : $TEST_CSV"
echo "[INFO] output_dir : $OUTPUT_DIR"
echo

python3 "$EVALUATOR" \
    --test_csv    "$TEST_CSV" \
    --submission  "$SUBMISSION" \
    --output_dir  "$OUTPUT_DIR"
