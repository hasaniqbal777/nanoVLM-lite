#!/bin/bash

# Fine-tune nanoVLM on A-OKVQA MCQ task
# Trains only the language model while freezing vision encoder

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

set -e

# Setup logging
LOG_DIR="$PROJECT_ROOT/logs"
ensure_dir "$LOG_DIR"
LOG_FILE="$LOG_DIR/finetune_mcq_$(date +%Y%m%d_%H%M%S).log"

# Function to log both to console and file
exec > >(tee -a "$LOG_FILE") 2>&1

# Show usage
show_usage() {
    c_echo $YELLOW "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --epochs N          Number of training epochs (default: 2)"
    echo "  --lr FLOAT          Learning rate (default: 2e-4)"
    echo "  --grad-accum N      Gradient accumulation steps (default: 32)"
    echo "  --eval-every N      Evaluate every N steps (default: 500)"
    echo "  --eval-samples N    Max samples for eval (-1 for all, default: -1)"
    echo "  --output DIR        Output directory (default: checkpoints/mcq_finetuning)"
    echo "  --no-freeze-vision  Don't freeze vision encoder (train end-to-end)"
    echo "  --freeze-proj       Freeze modality projector"
    echo ""
    echo "Examples:"
    echo "  $0                                      # Default settings"
    echo "  $0 --epochs 3 --lr 1e-4                # Custom epochs and LR"
    echo "  $0 --eval-samples 100                  # Quick eval on 100 samples"
    echo "  $0 --no-freeze-vision --freeze-proj    # Train vision, freeze projector"
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

c_echo $GREEN "============================================================"
c_echo $GREEN "nanoVLM MCQ Fine-tuning"
c_echo $GREEN "============================================================"

# Check if nanoVLM is set up
if [ ! -d "$NANOVLM_DIR" ]; then
    c_echo $RED "nanoVLM not found at: $NANOVLM_DIR"
    c_echo $YELLOW "Run ./scripts/1-setup.sh first"
    exit 1
fi

# Set Python path
export PYTHONPATH="$NANOVLM_DIR:$PROJECT_ROOT:$PYTHONPATH"

# Default arguments
EPOCHS=2
LR="2e-4"
GRAD_ACCUM=32
EVAL_EVERY=500
EVAL_SAMPLES=-1
OUTPUT="checkpoints/mcq_finetuning"
FREEZE_VISION="--freeze_vision"
FREEZE_PROJ=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --eval-every)
            EVAL_EVERY="$2"
            shift 2
            ;;
        --eval-samples)
            EVAL_SAMPLES="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --no-freeze-vision)
            FREEZE_VISION=""
            shift
            ;;
        --freeze-proj)
            FREEZE_PROJ="--freeze_proj"
            shift
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
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Eval every: $EVAL_EVERY steps"
echo "  Eval samples: $EVAL_SAMPLES"
echo "  Output: $OUTPUT"
echo "  Freeze vision: $([ -n "$FREEZE_VISION" ] && echo "Yes" || echo "No")"
echo "  Freeze projector: $([ -n "$FREEZE_PROJ" ] && echo "Yes" || echo "No")"
echo ""

# Ensure checkpoints directory exists
ensure_dir "checkpoints"

# Run fine-tuning
c_echo $YELLOW "Starting fine-tuning..."
echo ""
uv run python src/finetuning/finetune_mcq.py \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --grad_accum "$GRAD_ACCUM" \
    --eval_every "$EVAL_EVERY" \
    --eval_samples "$EVAL_SAMPLES" \
    --output_dir "$OUTPUT" \
    $FREEZE_VISION \
    $FREEZE_PROJ

echo ""
c_echo $GREEN "✓ Fine-tuning complete!"
c_echo $YELLOW "Model saved to: $OUTPUT"
c_echo $YELLOW "Log saved to: $LOG_FILE"
