#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r scripts/requirements.txt

echo ""
echo "Setup complete. Activate with:"
echo "  source \"$PROJECT_DIR/.venv/bin/activate\""
echo ""
echo "Then run experiments in order:"
echo "  python scripts/01_train_nsl_kdd.py    # ~5 min"
echo "  python scripts/02_train_guide.py      # ~40 min"
echo "  python scripts/03_train_hdfs.py       # ~1h"
