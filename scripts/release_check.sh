#!/bin/bash
# Release check script for weight-atlas v0.1.0
# Tests core functionality without optional dependencies.
# Exit codes: 0 = success, 1 = failure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== weight-atlas Release Check ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}PASS${NC}: $1"; }
fail() { echo -e "${RED}FAIL${NC}: $1"; }
skip() { echo -e "${YELLOW}SKIP${NC}: $1"; }

FAILED=0

# 1. Version check
echo "--- Version Check ---"
VERSION=$(python -c "import weight_atlas; print(weight_atlas.__version__)" 2>/dev/null || echo "unknown")
if [ "$VERSION" = "0.1.0" ]; then
    pass "Version is $VERSION"
else
    fail "Version mismatch: expected 0.1.0, got $VERSION"
    FAILED=1
fi
echo ""

# 2. Core imports
echo "--- Core Imports ---"
if python -c "from weight_atlas.scan import scan; from weight_atlas.cli import main; print('OK')" 2>/dev/null; then
    pass "Core imports successful"
else
    fail "Core imports failed"
    FAILED=1
fi
echo ""

# 3. CLI help
echo "--- CLI Help ---"
if weight-atlas --help >/dev/null 2>&1; then
    pass "CLI help works"
else
    fail "CLI help failed"
    FAILED=1
fi
echo ""

# 4. Fixture scan → sheet → compare
echo "--- Fixture Scan → Sheet → Compare ---"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

cd "$PROJECT_ROOT"

python -c "
import numpy as np
from safetensors.numpy import save_file
from pathlib import Path
from weight_atlas.scan import scan
from weight_atlas.compare import compute_compare_summary
from weight_atlas.core.types import AtlasSpec
from weight_atlas.fields.tif_io import read_tif

# Create test model
model_path = Path('$TMPDIR/test.safetensors')
rng = np.random.default_rng(42)
tensors = {}
for layer in range(4):
    for slot in ['self_attn.q_proj', 'self_attn.k_proj', 'mlp.gate_proj']:
        tensors[f'model.layers.{layer}.{slot}.weight'] = rng.normal(0, 0.1, (32, 32)).astype(np.float32)
tensors['model.embed_tokens.weight'] = rng.normal(0, 0.1, (100, 32)).astype(np.float32)
save_file(tensors, str(model_path))

# Load spec
spec = AtlasSpec.from_json(Path('$PROJECT_ROOT/specs/atlas_spec.v1.json'))

# Scan
scan(model_path, Path('$TMPDIR/out1'), spec)
scan(model_path, Path('$TMPDIR/out2'), spec)

# Compare
field_a = read_tif(Path('$TMPDIR/out1/field_height_raw.tif'))
field_b = read_tif(Path('$TMPDIR/out2/field_height_raw.tif'))
summary = compute_compare_summary(field_a, field_b, spec, mode='strict')
print('Scan/Sheet/Compare: OK')
"

if [ $? -eq 0 ]; then
    pass "Fixture scan → sheet → compare successful"
else
    fail "Fixture scan → sheet → compare failed"
    FAILED=1
fi
echo ""

# 5. Determinism check (second run)
echo "--- Determinism Check ---"
python -c "
import numpy as np
from pathlib import Path
from weight_atlas.fields.tif_io import read_tif

# Compare two runs
f1 = read_tif(Path('$TMPDIR/out1/field_height_raw.tif'))
f2 = read_tif(Path('$TMPDIR/out2/field_height_raw.tif'))
if np.allclose(f1, f2, equal_nan=True):
    print('Determinism: OK')
else:
    print('Determinism: FAILED')
    exit(1)
"

if [ $? -eq 0 ]; then
    pass "Determinism verified (byte-identical second run)"
else
    fail "Determinism check failed"
    FAILED=1
fi
echo ""

# 6. Embedding generation
echo "--- Embedding Generation ---"
python -c "
import numpy as np
from safetensors.numpy import save_file
from pathlib import Path
from weight_atlas.scan import scan
from weight_atlas.core.types import AtlasSpec

model_path = Path('$TMPDIR/embed_test.safetensors')
rng = np.random.default_rng(42)
tensors = {
    'model.embed_tokens.weight': rng.normal(0, 0.1, (100, 32)).astype(np.float32),
    'model.layers.0.self_attn.q_proj.weight': rng.normal(0, 0.1, (32, 32)).astype(np.float32),
}
save_file(tensors, str(model_path))

spec = AtlasSpec.from_json(Path('$PROJECT_ROOT/specs/atlas_spec.v1.json'))
spec_dict = {'embedding': {'method': 'pca', 'grid': 256, 'components': 3, 'seeds': {'pca': 0}}}
spec = AtlasSpec(**{**spec.__dict__, **spec_dict})

scan(model_path, Path('$TMPDIR/embed_out'), spec)

# Check artefacts exist
pca_path = Path('$TMPDIR/embed_out/embedding_pca.npy')
density_path = Path('$TMPDIR/embed_out/field_embed_density_raw.tif')
if not pca_path.exists():
    print('PCA file missing')
    exit(1)
if not density_path.exists():
    print('Density field missing')
    exit(1)
print('Embedding: OK')
"

if [ $? -eq 0 ]; then
    pass "Embedding generation successful"
else
    fail "Embedding generation failed"
    FAILED=1
fi
echo ""

# 7. Blender smoke (only if binary available)
echo "--- Blender Smoke Test ---"
if command -v blender &>/dev/null || [ -n "${WEIGHT_ATLAS_BLENDER:-}" ]; then
    skip "Blender smoke test (binary found but not automated in CI)"
else
    skip "Blender smoke test (no blender binary found)"
fi
echo ""

# 8. Activity smoke (only with extra)
echo "--- Activity Smoke Test ---"
if python -c "import torch; import transformers" 2>/dev/null; then
    skip "Activity smoke test (torch/transformers available but not automated in CI)"
else
    skip "Activity smoke test (install with: pip install -e '.[activity]')"
fi
echo ""

# Summary
echo "=== Summary ==="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed!${NC}"
    exit 1
fi
