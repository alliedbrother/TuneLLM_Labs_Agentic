#!/bin/bash
# Check system status: current model, registry state, workspace summary
# Usage: bash agents/ceo/scripts/check_status.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
WORKSPACE="$PROJECT_ROOT/workspace"

echo "=== Self-Improving LLM — System Status ==="
echo ""

# Current production model
echo "--- Model Registry ---"
if [ -f "$WORKSPACE/registry/registry.yaml" ]; then
    python3 "$PROJECT_ROOT/agents/model-registry/scripts/get_model_info.py" --list 2>/dev/null
else
    echo "  No registry found"
fi
echo ""

# Datasets
echo "--- Datasets ---"
if [ -d "$WORKSPACE/datasets" ]; then
    for dir in "$WORKSPACE/datasets"/*/; do
        if [ -d "$dir" ]; then
            name=$(basename "$dir")
            train_count=$(wc -l < "$dir/train.jsonl" 2>/dev/null || echo "0")
            echo "  $name: $train_count training examples"
        fi
    done
    [ "$(ls -A "$WORKSPACE/datasets" 2>/dev/null)" ] || echo "  No datasets"
else
    echo "  No datasets directory"
fi
echo ""

# Models
echo "--- Model Artifacts ---"
if [ -d "$WORKSPACE/models" ]; then
    for dir in "$WORKSPACE/models"/*/; do
        if [ -d "$dir" ]; then
            name=$(basename "$dir")
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "  $name: $size"
        fi
    done
    [ "$(ls -A "$WORKSPACE/models" 2>/dev/null)" ] || echo "  No model artifacts"
else
    echo "  No models directory"
fi
echo ""

# Eval results
echo "--- Evaluation Results ---"
if [ -d "$WORKSPACE/eval_results" ]; then
    for dir in "$WORKSPACE/eval_results"/*/; do
        if [ -d "$dir" ]; then
            name=$(basename "$dir")
            if [ -f "$dir/report.yaml" ]; then
                rec=$(grep "recommendation:" "$dir/report.yaml" | head -1 | awk '{print $2}')
                echo "  $name: $rec"
            elif [ -f "$dir/eval_metrics.json" ]; then
                echo "  $name: metrics available (no report)"
            fi
        fi
    done
    [ "$(ls -A "$WORKSPACE/eval_results" 2>/dev/null)" ] || echo "  No eval results"
else
    echo "  No eval results directory"
fi
echo ""

# Disk usage
echo "--- Disk Usage ---"
du -sh "$WORKSPACE" 2>/dev/null | awk '{print "  Workspace total: " $1}'
