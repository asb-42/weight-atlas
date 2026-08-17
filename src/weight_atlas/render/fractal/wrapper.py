"""Fractal terrain renderer plugin.

Generates a genuinely fractal terrain from the model's real tensor
statistics: per-slot fBm parameters (octaves/persistence/lacunarity/base
frequency) are derived from per-slot aggregate stats (effective_rank →
octaves, kurtosis → persistence, sparsity → lacunarity, spectral_norm →
base frequency) rather than painting a fractal texture on a heightmap. The
resulting height field is rendered through the same Blender Workbench
pipeline as the plain terrain (same smoothing, lights, subsurf, metadata
stripping), so the two renders stay directly comparable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from weight_atlas.core.registry import register_renderer
from weight_atlas.core.types import AtlasSpec, Field2D
from weight_atlas.render.blender.blender_wrapper import (
    _get_spec_value,
    build_blender_command,
    build_blender_env,
    resolve_blender_path,
    run_blender_command,
)
from weight_atlas.render.blender.render_terrain import normalise_height, resample_bilinear
from weight_atlas.render.fractal.fbm import slot_fractal_field
from weight_atlas.render.fractal.params import slot_fractal_params

# Spec defaults (same fallbacks as the blender block).
_DEFAULT_GRID = 1024
_DEFAULT_RESOLUTION = 2048
_DEFAULT_Z_SCALE = 0.3
_DEFAULT_PITCH = 18.0
_DEFAULT_CLIP = 0.01
_DEFAULT_SUBSURF_LEVELS = 1
_DEFAULT_FILL_LIGHT_ENERGY = 0.35
_DEFAULT_CELL_H = 8
_DEFAULT_CELL_W = 8
_DEFAULT_MAPPING = {
    "octaves": {"stat": "effective_rank", "lo": 4, "hi": 8},
    "persistence": {"stat": "kurtosis", "lo": 0.4, "hi": 0.7},
    "lacunarity": {"stat": "sparsity", "lo": 1.8, "hi": 2.4},
    "base_freq": {"stat": "spectral_norm", "lo": 1.0, "hi": 2.5},
}

_OBJ_DOWNSAMPLE = 256


def _write_obj(height: np.ndarray, tint: np.ndarray, out: Path, clip: float = _DEFAULT_CLIP) -> None:
    """Diffable OBJ mesh of the fractal terrain (256² bilinear downsample).

    Uses the same robust percentile normalisation as the PNG renderer so the
    mesh surface matches the rendered geometry.
    """
    target = _OBJ_DOWNSAMPLE
    h_small = normalise_height(resample_bilinear(height, target), clip)

    x = np.linspace(-1.0, 1.0, target)
    y = np.linspace(-1.0, 1.0, target)
    xx, yy = np.meshgrid(x, y)

    lines = ["# weight-atlas fractal terrain OBJ", f"# vertices: {target * target}"]
    for i in range(target):
        for j in range(target):
            lines.append(f"v {xx[i, j]:.6f} {yy[i, j]:.6f} {h_small[i, j]:.6f}")
    for i in range(target - 1):
        for j in range(target - 1):
            v0 = i * target + j + 1
            v1 = i * target + (j + 1) + 1
            v2 = (i + 1) * target + (j + 1) + 1
            v3 = (i + 1) * target + j + 1
            lines.append(f"f {v0} {v1} {v2} {v3}")
    out.write_text("\n".join(lines) + "\n")


def _make_tint(
    height: np.ndarray,
    n_rows: int,
    n_cols: int,
    slots: list[str],
    tint_mapping: dict[str, Any],
    out_dir: Path,
    seed: int,
) -> np.ndarray:
    """A second per-slot fractal strip used as the tint channel.

    The tint field is generated from the same slot stats through an
    independent mapping (defaults to the height mapping), so tint bands still
    encode per-slot structure rather than being a flat colour wash.
    """
    params = slot_fractal_params(out_dir, slots, {"mapping": tint_mapping}, seed + 17)
    cell_h = max(1, height.shape[0] // n_rows)
    cell_w = max(1, height.shape[1] // n_cols)
    return slot_fractal_field(n_rows, n_cols, params, slots, cell_h=cell_h, cell_w=cell_w)


@register_renderer("fractal")
class FractalRenderer:
    """Fractal terrain renderer – registry ID ``"fractal"``.

    Per-slot tensor statistics drive the fBm parameters of a self-similar
    terrain. The geometry is genuinely fractal (not a heightmap with a
    fractal texture), rendered via the same Blender Workbench pipeline as the
    ``"blender"`` renderer. Produces a PNG and a diffable OBJ mesh.
    """

    renderer_id = "fractal"

    def __init__(self) -> None:
        # The fractal output depends only on the fingerprint + seed, not the
        # channel. The API/CLI invoke render() once per channel (height, tint,
        # rough, vision_*); dedupe within a single renderer instance so Blender
        # runs once per model and every channel reuses the identical artefacts.
        self._last_key: tuple | None = None
        self._last_produced: list[Path] = []

    def render(self, field: Field2D, spec: AtlasSpec, out: Path, *, field_name: str = "height", scatter_path: Path | None = None) -> list[Path]:
        out.mkdir(parents=True, exist_ok=True)

        fractal_cfg = getattr(spec, "fractal", None) or {}
        seeds = getattr(spec, "seeds", None) or {}
        seed = int(fractal_cfg.get("seed", seeds.get("fractal", 0)))

        n_rows = len(field.row_labels) if field.row_labels else int(field.data.shape[0])
        n_cols = len(field.col_labels) if field.col_labels else int(field.data.shape[1])
        if not field.col_labels:
            raise ValueError("fractal renderer requires slot column labels (field.col_labels)")
        slots = list(field.col_labels)

        # The fractal depends on the model's fingerprint + seed, not the
        # channel. The API/CLI invoke render() once per channel (height, tint,
        # rough, vision_*); dedupe within a single renderer instance so Blender
        # runs once per model and every channel reuses the identical artefacts
        # (the primary language raster's layout, never overwritten by the
        # smaller vision layout).
        key = (str(out.resolve()), seed)
        if self._last_key == key:
            return list(self._last_produced)
        self._last_key = key

        cell_h = int(fractal_cfg.get("cell_h", _DEFAULT_CELL_H))
        cell_w = int(fractal_cfg.get("cell_w", _DEFAULT_CELL_W))
        mapping = fractal_cfg.get("mapping", _DEFAULT_MAPPING)

        params = slot_fractal_params(out.parent, slots, fractal_cfg, seed)
        height = slot_fractal_field(n_rows, n_cols, params, slots, cell_h=cell_h, cell_w=cell_w)
        tint = _make_tint(height, n_rows, n_cols, slots, mapping, out.parent, seed)

        blender_path = resolve_blender_path()
        script_path = Path(__file__).resolve().parent.parent / "blender" / "render_terrain.py"
        blender_env = build_blender_env()

        grid = int(_get_spec_value(spec, "grid", _DEFAULT_GRID))
        resolution = int(_get_spec_value(spec, "resolution", _DEFAULT_RESOLUTION))
        z_scale = float(_get_spec_value(spec, "z_scale", _DEFAULT_Z_SCALE))
        pitch = float(_get_spec_value(spec, "pitch", _DEFAULT_PITCH))
        clip = float(_get_spec_value(spec, "clip", _DEFAULT_CLIP))
        subsurf = int(_get_spec_value(spec, "subsurf_levels", _DEFAULT_SUBSURF_LEVELS))
        fill_energy = float(_get_spec_value(spec, "fill_light_energy", _DEFAULT_FILL_LIGHT_ENERGY))

        produced: list[Path] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            height_npy = tmpdir_path / "height.npy"
            tint_npy = tmpdir_path / "tint.npy"
            np.save(height_npy, height.astype(np.float64))
            np.save(tint_npy, tint.astype(np.float64))

            out_png = out / "terrain_fractal.png"
            cmd = build_blender_command(
                blender_path=blender_path,
                script_path=script_path,
                height_npy=height_npy,
                tint_npy=tint_npy,
                out_png=out_png,
                grid=grid,
                z_scale=z_scale,
                resolution=resolution,
                pitch=pitch,
                clip=clip,
                adaptive_z_scale=False,
                subsurf_levels=subsurf,
                fill_light_energy=fill_energy,
            )
            run_blender_command(cmd, blender_env)

            if out_png.exists():
                produced.append(out_png)

        obj_path = out / "terrain_fractal.obj"
        _write_obj(height, tint, obj_path, clip=clip)
        produced.append(obj_path)
        self._last_produced = list(produced)
        return produced
