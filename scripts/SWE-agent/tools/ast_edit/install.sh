#!/bin/bash
set -e

echo "[ast_editor] Installing dependencies..."

pip install 'black'
pip install 'zss'
pip install 'tree-sitter'
pip install 'tree-sitter-python'
