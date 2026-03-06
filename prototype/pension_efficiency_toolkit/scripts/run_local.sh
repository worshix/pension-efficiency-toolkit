#!/usr/bin/env bash
# Run the full analysis pipeline locally with sample data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Pension Efficiency Toolkit — Local Run ==="
echo ""

echo ">>> Installing dependencies..."
uv sync

echo ""
echo ">>> Running analysis pipeline..."
uv run python -m pension_toolkit.cli analyze \
  --input tests/sample_data.csv \
  --out out/ \
  --bootstrap-B 200 \
  --seed 42

echo ""
echo ">>> Running tests..."
uv run pytest -q

echo ""
echo "=== Done. Results are in the out/ directory. ==="
echo ""
echo "To launch the Streamlit dashboard:"
echo "  uv run streamlit run pension_toolkit/ui_streamlit.py"
