#!/bin/bash

# Token Pooling Evaluation Script
# Evaluates nanoVLM with post-hoc token pooling on A-OKVQA

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

set -e

# Show usage
show_usage() {
    c_echo $YELLOW "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model PATH         Model path or HF ID (default: lusxvr/nanoVLM)"
    echo "  --max-samples NUM    Number of samples to evaluate (default: 100)"
    echo "  --mode MODE          Evaluation mode: mcq, oa, both (default: mcq)"
    echo "  --split SPLIT        Dataset split: train, validation, test (default: validation)"
    echo "  --pooling METHOD     Token pooling method:"
    echo "                       - none: No token pooling (baseline)"
    echo "                       - avg: Average pooling (default)"
    echo "                       - max: Max pooling"
    echo "                       - adaptive: Adaptive pooling with learned weights"
    echo "  --kernel NUM         Kernel size for avg/max pooling (default: 2)"
    echo "  --stride NUM         Stride for avg/max pooling (default: 2)"
    echo "  --grid NUM           Target grid size for adaptive pooling (default: 8)"
    echo "  --resolution RES     Vision encoder resolution (default: model default)"
    echo "  --output FILE        Output JSON file (default: auto-generated)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Default: avg pooling, 2×2 kernel"
    echo "  $0 --pooling none                     # Baseline (no pooling)"
    echo "  $0 --pooling avg --kernel 2 --stride 2   # 2×2 average pooling"
    echo "  $0 --pooling max --kernel 4 --stride 4   # 4×4 max pooling"
    echo "  $0 --pooling adaptive --grid 8        # Adaptive pooling to 8×8 grid"
    echo "  $0 --max-samples 500 --pooling avg    # 500 samples with avg pooling"
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

c_echo $GREEN "============================================================"
c_echo $GREEN "nanoVLM Token Pooling Evaluation"
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

# Default values
MODEL="lusxvr/nanoVLM"
MAX_SAMPLES=100
MODE="mcq"
SPLIT="validation"
POOLING="avg"
KERNEL=2
STRIDE=2
GRID=8
RESOLUTION=""
OUTPUT=""
OUTPUT_OVERRIDE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --pooling)
            POOLING="$2"
            shift 2
            ;;
        --kernel)
            KERNEL="$2"
            shift 2
            ;;
        --stride)
            STRIDE="$2"
            shift 2
            ;;
        --grid)
            GRID="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            OUTPUT_OVERRIDE=true
            shift 2
            ;;
        *)
            c_echo $RED "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Setup logging
LOG_DIR="$PROJECT_ROOT/logs"
ensure_dir "$LOG_DIR"

if [ "$OUTPUT_OVERRIDE" = false ]; then
    if [ "$POOLING" = "none" ]; then
        OUTPUT="$RESULTS_DIR/baseline_results.json"
        LOG_FILE="$LOG_DIR/baseline_evaluation.log"
    elif [ "$POOLING" = "adaptive" ]; then
        OUTPUT="$RESULTS_DIR/token_pooling_${POOLING}_grid${GRID}_results.json"
        LOG_FILE="$LOG_DIR/token_pooling_${POOLING}_grid${GRID}.log"
    else
        OUTPUT="$RESULTS_DIR/token_pooling_${POOLING}_k${KERNEL}_s${STRIDE}_results.json"
        LOG_FILE="$LOG_DIR/token_pooling_${POOLING}_k${KERNEL}_s${STRIDE}.log"
    fi
else
    OUTPUT_BASENAME=$(basename "$OUTPUT" .json)
    LOG_FILE="$LOG_DIR/${OUTPUT_BASENAME}.log"
fi

# Function to log both to console and file
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
c_echo $YELLOW "Configuration:"
echo "  Model: $MODEL"
echo "  Max samples: $MAX_SAMPLES"
echo "  Mode: $MODE"
echo "  Split: $SPLIT"
echo "  Pooling method: $POOLING"
if [ "$POOLING" = "adaptive" ]; then
    echo "  Target grid: ${GRID}×${GRID} (${GRID}*${GRID}=$((GRID*GRID)) tokens)"
elif [ "$POOLING" != "none" ]; then
    echo "  Kernel size: ${KERNEL}×${KERNEL}"
    echo "  Stride: $STRIDE"
fi
if [ -n "$RESOLUTION" ]; then
    echo "  Resolution: ${RESOLUTION}×${RESOLUTION}"
fi
echo "  Output: $OUTPUT"
echo ""

# Run evaluation
c_echo $YELLOW "Running token pooling evaluation..."
echo ""

# Build command
EVAL_CMD="uv run python src/evaluation/evaluate_token_pooling.py \
    --model-path '$MODEL' \
    --max-samples $MAX_SAMPLES \
    --mode $MODE \
    --split $SPLIT \
    --pooling $POOLING \
    --kernel $KERNEL \
    --stride $STRIDE \
    --grid $GRID \
    --output $OUTPUT"

if [ -n "$RESOLUTION" ]; then
    EVAL_CMD="$EVAL_CMD --resolution $RESOLUTION"
fi

eval $EVAL_CMD

echo ""
c_echo $GREEN "✓ Token pooling evaluation complete!"
c_echo $YELLOW "Results saved to: $OUTPUT"
c_echo $YELLOW "Log saved to: $LOG_FILE"
