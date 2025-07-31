#!/bin/bash
set -e

echo "[structural_searcher] Installing dependencies..."

# Support for Python 3.9+
pip install -U codeanalyzer-python

echo "[structural_searcher] Done."

# Precompute symbol table:
PROJECT="/testbed"
CACHE_DIR="/analysis"
OUTPUT="${CACHE_DIR}/.codeanalyzer_output"

# Ensure /analysis directory exists
mkdir -p "${CACHE_DIR}"

# Run codeanalyzer and save output outside /testbed
codeanalyzer \
  --input "${PROJECT}" \
  --analysis-level 1 \
  --output "${OUTPUT}" \
  --format json \
  --cache-dir "${CACHE_DIR}" \
  -v

echo "[structural_searcher] Symbol table precomputed at ${OUTPUT}."
