#!/bin/bash
set -e

echo "[structural_searcher] Installing dependencies..."

# Only works if using Python 3.10
pip install codeanalyzer-python==0.1.9

echo "[structural_searcher] Done."
