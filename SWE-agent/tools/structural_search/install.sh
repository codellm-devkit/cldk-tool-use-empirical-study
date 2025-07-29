#!/bin/bash
set -e

echo "[structural_searcher] Installing dependencies..."

# Support for Python 3.9+
pip install -U codeanalyzer-python

echo "[structural_searcher] Done."

# Precompute symbol table:
PROJECT="/testbed"
OUTPUT="${PROJECT}/.codeanalyzer_output"
codeanalyzer --input "${PROJECT}" --analysis-level 1 --output "${OUTPUT}" --format json -v
echo "[structural_searcher] Symbol table precomputed at ${OUTPUT}."
