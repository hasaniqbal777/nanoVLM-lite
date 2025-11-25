#!/bin/bash

# Baseline Evaluation Script
# Runs baseline evaluation on nanoVLM with A-OKVQA

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

set -e

# Show usage
show_usage() {
    c_echo $YELLOW "Usage: $0 [MAX_SAMPLES] [MODE] [SPLIT]"
    echo ""
    echo "Arguments:"
    echo "  MAX_SAMPLES  Number of samples to evaluate (default: 100)"
    echo "  MODE         Evaluation mode: mcq, oa, both (default: both)"
    echo "  SPLIT        Dataset split: train, validation, test (default: validation)"
    echo ""
    echo "Examples:"
    echo "  $0 100              # 100 samples, both modes, validation split"
    echo "  $0 100 mcq          # 100 samples, MCQ only"
    echo "  $0 100 oa           # 100 samples, open-answer only"
    echo "  $0 100 both test    # 100 samples, both modes, test split"
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

c_echo $GREEN "============================================================"
c_echo $GREEN "nanoVLM Baseline Evaluation"
c_echo $GREEN "============================================================"

# Check if nanoVLM is set up
if [ ! -d "$NANOVLM_DIR" ]; then
    c_echo $RED "nanoVLM not found at: $NANOVLM_DIR"
    c_echo $YELLOW "Run ./scripts/1-setup.sh first"
    exit 1
fi

# Ensure results directory exists
ensure_dir "$RESULTS_DIR"

# Set Python path
export PYTHONPATH="$NANOVLM_DIR:$PROJECT_ROOT:$PYTHONPATH"

# Parse arguments
MAX_SAMPLES="${1:-100}"
MODE="${2:-both}"
SPLIT="${3:-validation}"

echo ""
c_echo $YELLOW "Configuration:"
echo "  Max samples: $MAX_SAMPLES"
echo "  Mode: $MODE"
echo "  Split: $SPLIT"
echo "  Results dir: $RESULTS_DIR"
echo ""

# Run evaluation
c_echo $YELLOW "Running baseline evaluation..."
echo ""
uv run python src/evaluation/baseline.py \
    --max-samples "$MAX_SAMPLES" \
    --mode "$MODE" \
    --split "$SPLIT" \
    --output "$RESULTS_DIR/baseline_results.json"

echo ""
c_echo $GREEN "✓ Baseline evaluation complete!"
c_echo $YELLOW "Results saved to: $RESULTS_DIR/baseline_results.json"
