#!/bin/bash

# Baseline Evaluation Script
# Runs baseline evaluation on nanoVLM with A-OKVQA

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

set -e

# Show usage
show_usage() {
    c_echo $YELLOW "Usage: $0 [MAX_SAMPLES] [MODE] [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  MAX_SAMPLES  Number of samples to evaluate (default: 100, 0 = all)"
    echo "  MODE         Evaluation mode: mcq, oa, both (default: both)"
    echo ""
    echo "Options:"
    echo "  --split SPLIT      Dataset split: train, validation, test (default: validation)"
    echo "  --output FILE      Output JSON file (default: results/baseline_results.json)"
    echo ""
    echo "Examples:"
    echo "  $0                                        # Full dataset, both modes"
    echo "  $0 100                                    # 100 samples, both modes"
    echo "  $0 100 mcq                                # 100 samples, MCQ only"
    echo "  $0 1 both --output results/baseline_1.json  # 1 sample, custom output"
    echo "  $0 --output results/baseline_full.json    # Full dataset, custom output"
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
# If first arg starts with --, treat as flag (no MAX_SAMPLES provided)
if [[ "$1" == --* ]]; then
    MAX_SAMPLES="0"
    MODE="both"
else
    MAX_SAMPLES="${1:-100}"
    MODE="${2:-both}"
    shift 2 2>/dev/null || shift $# 2>/dev/null
fi

# Parse optional flags
SPLIT="validation"
OUTPUT="$RESULTS_DIR/baseline_results.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            c_echo $RED "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

echo ""
c_echo $YELLOW "Configuration:"
echo "  Max samples: $MAX_SAMPLES"
echo "  Mode: $MODE"
echo "  Split: $SPLIT"
echo "  Output: $OUTPUT"
echo ""

# Run evaluation
c_echo $YELLOW "Running baseline evaluation..."
echo ""
uv run python src/evaluation/baseline.py \
    --max-samples "$MAX_SAMPLES" \
    --mode "$MODE" \
    --split "$SPLIT" \
    --output "$OUTPUT"

echo ""
c_echo $GREEN "✓ Baseline evaluation complete!"
c_echo $YELLOW "Results saved to: $OUTPUT"
