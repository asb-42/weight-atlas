"""End-to-end: scan + render via CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path

from tests.fixtures import make_fake_model
from weight_atlas.cli import main


def test_render_huge_panel_is_bounded(tmp_path):
    """Rendering a huge expert panel must stay memory-bounded.

    Regression: the sheet renderer sized the figure in inches proportional to
    the field, so a 736x7168 expert panel (Kimi K3) became a ~537,600 px wide
    figure whose ~95 GB RGBA buffer OOM-killed the worker. The figure's long
    edge is now capped, so the output PNG must be bounded too.
    """
    import numpy as np
    from PIL import Image

    import weight_atlas.render  # noqa: F401 — registers renderers
    from weight_atlas.core.registry import get_renderer
    from weight_atlas.core.types import AtlasSpec, Field2D

    spec = AtlasSpec.from_json(Path("specs/atlas_spec.v2.4.json"))
    data = np.random.default_rng(0).normal(0, 1, (120, 2500)).astype(np.float64)
    field = Field2D(
        channel="expert_mlp_down_height",
        data=data,
        row_labels=[str(i) for i in range(120)],
        col_labels=[str(i) for i in range(2500)],
    )
    out = tmp_path / "render"
    paths = get_renderer("sheet")().render(field, spec, out)
    assert len(paths) == 1
    assert paths[0].exists()

    from weight_atlas.render.matplotlib_sheet import _MAX_RENDER_PIXELS

    w, h = Image.open(paths[0]).size
    assert w * h <= _MAX_RENDER_PIXELS


def test_cli_scan_render():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "model.safetensors"
        make_fake_model(model)
        out = tmp / "artefacts"

        assert main(["scan", str(model), "--out", str(out)]) == 0
        assert (out / "fingerprint.json").exists()
        assert (out / "manifest.json").exists()
        tifs = list(out.glob("field_*.tif"))
        assert len(tifs) >= 3  # at least one channel

        assert main(["render", str(out)]) == 0
