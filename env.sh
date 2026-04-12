#!/bin/bash
# Environment setup for Self-Improving LLM agents
# Source this before running any agent script: source env.sh

export PROJECT_ROOT="/Users/saiakhil/Documents/Personal_Projects_Git_Sync/fine_tune_framework"
export WORKSPACE="$PROJECT_ROOT/workspace"
export LIB="$PROJECT_ROOT/lib"
export VENV="$PROJECT_ROOT/venv"

# Activate venv if it exists
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi

# Add lib to PYTHONPATH so scripts can import from lib/
export PYTHONPATH="$LIB:$PYTHONPATH"
