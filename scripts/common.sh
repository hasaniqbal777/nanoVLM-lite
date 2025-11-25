#!/bin/bash

################################################################################
# Shell dependencies                                                           #
################################################################################
# https://stackoverflow.com/questions/592620/how-can-i-check-if-a-program-exists-from-a-bash-script

if ! [ -x "$(command -v git)" ]; then
  echo 'Error: git is not installed.' >&2
  exit 1
fi

################################################################################
# Common Functions                                                             #
################################################################################

# Echo's the text to the screen with the designated color
c_echo () {
    local color=$1
    local txt=$2

    echo -e "${color}${txt}${NC}"
}

# Enforces you are running from project root
force_project_root () {
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    PARENT_DIR=$(dirname $DIR)

    if [ "$(pwd)" != $PARENT_DIR ]
    then
        c_echo $RED "You must be in $PARENT_DIR to run"
        exit 1
    fi
}

# Ensure directory exists
ensure_dir () {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        c_echo $YELLOW "Created directory: $1"
    fi
}

# Check if file exists and print status
check_file_status () {
    local file_path="$1"
    local file_name="$(basename "$file_path")"
    
    if [ -f "$file_path" ]; then
        c_echo $GREEN "  ✓ $file_name"
        return 0
    else
        c_echo $RED "  ✗ $file_name - NOT FOUND"
        return 1
    fi
}

# Setup Python path with nanoVLM
setup_python_path () {
    if [ -d "$NANOVLM_DIR" ]; then
        export PYTHONPATH="$NANOVLM_DIR:$PYTHONPATH"
        c_echo $YELLOW "Added nanoVLM to PYTHONPATH"
    else
        c_echo $RED "nanoVLM directory not found at: $NANOVLM_DIR"
        c_echo $RED "Run ./scripts/1-setup.sh first"
        exit 1
    fi
}

# Ensure common directories exist
ensure_common_dirs () {
    ensure_dir "$MODELS_DIR"
    ensure_dir "$RESULTS_DIR"
    ensure_dir "$CHECKPOINTS_DIR"
    ensure_dir "$EXPERIMENTS_DIR"
}

################################################################################
# Color Constants                                                              #
################################################################################
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

################################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ENFORCE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
################################################################################
force_project_root

################################################################################
# Application Versioning Information                                           #
################################################################################
APP_NAME=nanoVLMOptimization
APP_VERSION=$(git describe --tags --dirty) || exit 1
COMMIT=$(git rev-parse HEAD)

################################################################################
# Common Paths                                                                 #
################################################################################
PROJECT_ROOT="$(pwd)"
MODELS_DIR="$PROJECT_ROOT/models"
NANOVLM_DIR="$MODELS_DIR/nanoVLM"
RESULTS_DIR="$PROJECT_ROOT/results"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
SRC_DIR="$PROJECT_ROOT/src"
BUILDBIN_PATH=$(pwd)/tmp/build/bin