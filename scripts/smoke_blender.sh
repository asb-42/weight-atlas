#!/usr/bin/env bash
# Smoke test for Blender renderer (local only, not CI).
# Verifies: scan → render blender → second render byte-identical PNGs.
#
# Prerequisites:
#   - Blender installed and on PATH (or WEIGHT_ATLAS_BLENDER set)
#   - weight-atlas installed in dev mode
#
# Usage:
#   bash scripts/smoke_blender.sh
#
# Exit codes:
#   0 = success (PNG byte-identical)
#   1 = failure

set -euo pipefail

# Create temp directory
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== Weight Atlas Blender Smoke Test ==="
echo "Temp dir: $TMPDIR"

# Generate fixture model
echo ""
echo "1. Generating fixture model..."
MODEL_DIR="$TMPDIR/model"
mkdir -p "$MODEL_DIR"
.venv/bin/python -c "
from tests.fixtures import make_fake_model
from pathlib import Path
make_fake_model(Path('$MODEL_DIR/model.safetensors'), n_layers=4, hidden=32, seed=42)
print('   Fixture model created.')
"

# Scan the model
echo ""
echo "2. Scanning model..."
OUT_DIR="$TMPDIR/scan"
mkdir -p "$OUT_DIR"
.venv/bin/python -m weight-atlas scan "$MODEL_DIR/model.safetensors" --out "$OUT_DIR"
echo "   Scan complete. Artefacts:"
ls -la "$OUT_DIR"

# First render
echo ""
echo "3. First Blender render..."
RENDER1="$TMPDIR/render1"
mkdir -p "$RENDER1"
.venv/bin/python -m weight-atlas render "$OUT_DIR" --renderer blender
# Copy outputs to render1
cp "$OUT_DIR/render/"* "$RENDER1/" 2>/dev/null || true
echo "   First render complete."

# Second render
echo ""
echo "4. Second Blender render..."
RENDER2="$TMPDIR/render2"
mkdir -p "$RENDER2"
.venv/bin/python -m weight-atlas render "$OUT_DIR" --renderer blender
# Copy outputs to render2
cp "$OUT_DIR/render/"* "$RENDER2/" 2>/dev/null || true
echo "   Second render complete."

# Compare PNGs
echo ""
echo "5. Comparing PNGs..."
PNG1="$RENDER1/terrain_smooth.png"
PNG2="$RENDER2/terrain_smooth.png"

if [ ! -f "$PNG1" ]; then
    echo "ERROR: First render PNG not found: $PNG1"
    exit 1
fi
if [ ! -f "$PNG2" ]; then
    echo "ERROR: Second render PNG not found: $PNG2"
    exit 1
fi

SHA1=$(sha256sum "$PNG1" | cut -d' ' -f1)
SHA2=$(sha256sum "$PNG2" | cut -d' ' -f1)

echo "   First render SHA-256:  $SHA1"
echo "   Second render SHA-256: $SHA2"

if [ "$SHA1" = "$SHA2" ]; then
    echo ""
    echo "=== SUCCESS: PNGs are byte-identical ==="
    exit 0
else
    echo ""
    echo "=== FAILURE: PNGs differ ==="
    echo "This indicates non-deterministic rendering."
    echo "Check Workbench lighting rotation and PNG encoder metadata."
    exit 1
fi
