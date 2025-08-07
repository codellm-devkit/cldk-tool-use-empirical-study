#!/bin/bash
set -e

echo "[structural_searcher] Setting up isolated virtual environment for codeanalyzer..."

# Define virtual environment path
VENV_DIR="/tmp/codeanalyzer_venv"

# Check for python3 availability
if ! command -v python3 &>/dev/null; then
  echo "[ERROR] python3 is not installed or not in PATH."
  exit 1
fi

# Create the virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install codeanalyzer only if not already installed
if ! "$VENV_DIR/bin/codeanalyzer" --version &>/dev/null; then
  echo "[structural_searcher] Installing codeanalyzer in isolated environment..."
  pip install -U codeanalyzer-python
fi

# Define paths
PROJECT="/testbed"
CACHE_DIR="/analysis"
OUTPUT="${CACHE_DIR}/.codeanalyzer_output"

# Ensure /analysis directory exists
mkdir -p "${CACHE_DIR}"

# Run codeanalyzer from the virtual environment
echo "[structural_searcher] Running codeanalyzer..."
if ! "$VENV_DIR/bin/codeanalyzer" \
  --input "${PROJECT}" \
  --analysis-level 1 \
  --output "${OUTPUT}" \
  --format json \
  --cache-dir "${CACHE_DIR}" \
  -v; then
  echo "[ERROR] codeanalyzer execution failed."
  exit 2
fi

echo "[structural_searcher] Symbol table precomputed at ${OUTPUT}."
