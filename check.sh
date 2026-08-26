#!/usr/bin/env bash
set -euo pipefail

echo "=== ruff ==="
.venv/bin/ruff check src/ tests/
echo "=== mypy ==="
.venv/bin/mypy src/
echo "=== pytest ==="
.venv/bin/python -m pytest tests/ -q
echo "=== ALL CLEAN ==="
