#!/bin/bash

# nanoVLM Setup Script
# Clones the nanoVLM repository and prepares it for use

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

set -e

# Setup logging
LOG_DIR="$PROJECT_ROOT/logs"
ensure_dir "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup_$(date +%Y%m%d_%H%M%S).log"

# Function to log both to console and file
exec > >(tee -a "$LOG_FILE") 2>&1

c_echo $GREEN "============================================================"
c_echo $GREEN "nanoVLM Setup"
c_echo $GREEN "============================================================"

# Create models directory
ensure_dir "$MODELS_DIR"

# Check if already cloned
if [ -d "$NANOVLM_DIR" ]; then
    c_echo $GREEN "✓ nanoVLM already cloned at: $NANOVLM_DIR"
    echo ""
    c_echo $YELLOW "To update, run:"
    echo "  cd $NANOVLM_DIR && git pull"
else
    c_echo $YELLOW "Cloning nanoVLM to: $NANOVLM_DIR"
    c_echo $YELLOW "This may take a few minutes..."
    echo ""
    
    if git clone https://github.com/huggingface/nanoVLM.git "$NANOVLM_DIR"; then
        c_echo $GREEN "✓ Successfully cloned nanoVLM"
    else
        c_echo $RED "✗ Failed to clone repository"
        echo ""
        echo "Manual setup:"
        echo "  1. Go to: https://github.com/huggingface/nanoVLM"
        echo "  2. Clone or download the repository"
        echo "  3. Place it in: $NANOVLM_DIR"
        exit 1
    fi
fi

# Check for required files
echo ""
c_echo $YELLOW "Checking required files:"

all_files_present=true

check_file_status "$NANOVLM_DIR/models/vision_language_model.py" || all_files_present=false
check_file_status "$NANOVLM_DIR/models/config.py" || all_files_present=false
check_file_status "$NANOVLM_DIR/train.py" || all_files_present=false

if [ "$all_files_present" = true ]; then
    echo ""
    c_echo $GREEN "✓ All required files present!"
else
    echo ""
    c_echo $YELLOW "⚠ Some required files are missing"
fi

# Add to Python path instructions
echo ""
c_echo $YELLOW "To use nanoVLM, add this to your environment:"
echo "  export PYTHONPATH=\"$NANOVLM_DIR:\$PYTHONPATH\""
echo ""
c_echo $YELLOW "Or run:"
echo "  export PYTHONPATH=\"$NANOVLM_DIR:\$PYTHONPATH\""
echo "  uv run python src/evaluation/evaluate.py"

echo ""
c_echo $GREEN "============================================================"
c_echo $GREEN "Setup complete!"
c_echo $GREEN "============================================================"
echo ""
c_echo $YELLOW "Log saved to: $LOG_FILE"
