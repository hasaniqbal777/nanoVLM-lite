#!/bin/bash

# FLOP Counter Script
# Measures FLOPs for nanoVLM vision encoder and generation

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

set -e

# Setup logging
LOG_DIR="$PROJECT_ROOT/logs"
ensure_dir "$LOG_DIR"
LOG_FILE="$LOG_DIR/flop_counter_$(date +%Y%m%d_%H%M%S).log"

# Function to log both to console and file
exec > >(tee -a "$LOG_FILE") 2>&1

# Show usage
show_usage() {
    c_echo $YELLOW "Usage: $0 [IMAGE] [QUERY_FILE] [OUTPUT]"
    echo ""
    echo "Arguments:"
    echo "  IMAGE      Path to test image (default: test/test_image.webp)"
    echo "  QUERY_FILE Path to query file (default: test/test_query.txt)"
    echo "  OUTPUT     Path to save JSON results (default: results/flops.json)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 test/test_image.webp test/test_query.txt"
    echo "  $0 test/test_image.webp test/test_query.txt results/my_flops.json"
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

c_echo $GREEN "============================================================"
c_echo $GREEN "nanoVLM FLOP Counter"
c_echo $GREEN "============================================================"

# Check if nanoVLM is set up
if [ ! -d "$NANOVLM_DIR" ]; then
    c_echo $RED "nanoVLM not found at: $NANOVLM_DIR"
    c_echo $YELLOW "Run ./scripts/1-setup.sh first"
    exit 1
fi

# Set Python path
export PYTHONPATH="$NANOVLM_DIR:$PROJECT_ROOT:$PYTHONPATH"

# Parse arguments
IMAGE="${1:-test/test_image.webp}"
QUERY_FILE="${2:-test/test_query.txt}"
OUTPUT="${3:-results/flops.json}"

# Read question from file
if [ ! -f "$QUERY_FILE" ]; then
    c_echo $RED "Query file not found: $QUERY_FILE"
    exit 1
fi
QUESTION=$(cat "$QUERY_FILE")

echo ""
c_echo $YELLOW "Configuration:"
echo "  Image: $IMAGE"
echo "  Query file: $QUERY_FILE"
echo "  Question: $QUESTION"
echo "  Output: $OUTPUT"
echo ""

# Check if image exists
if [ ! -f "$IMAGE" ]; then
    c_echo $RED "Image not found: $IMAGE"
    exit 1
fi

# Run FLOP counter
c_echo $YELLOW "Measuring FLOPs..."
echo ""
uv run python src/evaluation/flop_counter.py \
    --image "$IMAGE" \
    --question "$QUESTION" \
    --gen-tokens 3 \
    --warmup 5 \
    --output "$OUTPUT"

echo ""
c_echo $GREEN "✓ FLOP measurement complete!"
c_echo $YELLOW "Log saved to: $LOG_FILE"
