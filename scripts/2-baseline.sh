#!/bin/bash

# Baseline Evaluation Script
# Runs baseline evaluation with enhanced prompts on nanoVLM with A-OKVQA

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
    echo "  --max-samples NUM    Number of samples to evaluate (default: 100, 0 = all)"
    echo "  --mode MODE          Evaluation mode: mcq, oa, both (default: both)"
    echo "  --split SPLIT        Dataset split: train, validation, test (default: validation)"
    echo "  --resolution RES     Target resolution: 384, 256, 192 (default: model default 512)"
    echo "  --output FILE        Output JSON file (default: results/baseline_results_res{RES}.json)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Default: 100 samples, both modes, 512 resolution"
    echo "  $0 --max-samples 50                          # 50 samples"
    echo "  $0 --max-samples 100 --mode mcq              # 100 samples, MCQ only"
    echo "  $0 --model checkpoints/mcq_finetuning/mcq_finetuned  # Evaluate finetuned model"
    echo "  $0 --resolution 384                          # 384x384 resolution"
    echo "  $0 --max-samples 100 --resolution 256        # 100 samples, 256 resolution"
    echo "  $0 --output results/custom.json              # Custom output file"
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

# Default values
MODEL="lusxvr/nanoVLM"
MAX_SAMPLES=100
MODE="both"
SPLIT="validation"
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

# Set default output with resolution in filename if not overridden
if [ "$OUTPUT_OVERRIDE" = false ]; then
    if [ -n "$RESOLUTION" ]; then
        OUTPUT="$RESULTS_DIR/baseline_results_res${RESOLUTION}.json"
    else
        OUTPUT="$RESULTS_DIR/baseline_results_res512.json"
    fi
fi

# Setup logging based on output filename
LOG_DIR="$PROJECT_ROOT/logs"
ensure_dir "$LOG_DIR"
OUTPUT_BASENAME=$(basename "$OUTPUT" .json)
LOG_FILE="$LOG_DIR/${OUTPUT_BASENAME}.log"

# Function to log both to console and file
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
c_echo $YELLOW "Configuration:"
echo "  Model: $MODEL"
echo "  Max samples: $MAX_SAMPLES"
echo "  Mode: $MODE"
echo "  Split: $SPLIT"
if [ -n "$RESOLUTION" ]; then
    echo "  Resolution: ${RESOLUTION}×${RESOLUTION}"
else
    echo "  Resolution: Model default (512×512)"
fi
echo "  Output: $OUTPUT"
echo ""

# Run evaluation
c_echo $YELLOW "Running baseline evaluation..."
echo ""

# Build command with optional resolution
EVAL_CMD="uv run python src/evaluation/baseline.py \
    --model-path '$MODEL' \
    --max-samples $MAX_SAMPLES \
    --mode $MODE \
    --split $SPLIT \
    --output $OUTPUT"

if [ -n "$RESOLUTION" ]; then
    EVAL_CMD="$EVAL_CMD --resolution $RESOLUTION"
fi

eval $EVAL_CMD

echo ""
c_echo $GREEN "✓ Baseline evaluation complete!"
c_echo $YELLOW "Results saved to: $OUTPUT"
c_echo $YELLOW "Log saved to: $LOG_FILE"
