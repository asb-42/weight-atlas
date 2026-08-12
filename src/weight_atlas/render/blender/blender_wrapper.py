"""Blender headless renderer plugin.

Reads TIFF height/tint artefacts from an M1 scan, exports them to ``.npy``,
invokes Blender in headless mode to render an orthographic top-view PNG,
and additionally writes a diffable OBJ mesh (256² downsample).

The Blender binary is resolved from ``WEIGHT_ATLAS_BLENDER`` env var,
falling back to ``shutil.which("blender")``. Neither is installed as a
pip dependency – Blender is always an external tool.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from weight_atlas.core.registry import register_renderer
from weight_atlas.core.types import AtlasSpec, Field2D
from weight_atlas.fields.tif_io import read_tif

# Spec defaults for Blender-specific settings
_DEFAULT_GRID = 1024
_DEFAULT_RESOLUTION = 2048
_DEFAULT_Z_SCALE = 0.3
_OBJ_DOWNSAMPLE = 256


def resolve_blender_path() -> Path:
    """Return the path to the Blender binary.

    Priority:
    1. ``WEIGHT_ATLAS_BLENDER`` env var
    2. ``shutil.which("blender")``

    Raises:
        FileNotFoundError: neither source yields a valid path.
    """
    env_path = os.environ.get("WEIGHT_ATLAS_BLENDER")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"WEIGHT_ATLAS_BLENDER points to non-existent path: {env_path}"
        )

    which_path = shutil.which("blender")
    if which_path:
        return Path(which_path)

    raise FileNotFoundError(
        "Blender binary not found. Install Blender and ensure it is on PATH, "
        "or set WEIGHT_ATLAS_BLENDER=/path/to/blender"
    )


def build_blender_command(
    blender_path: Path,
    script_path: Path,
    height_npy: Path,
    tint_npy: Path,
    out_png: Path,
    grid: int,
    z_scale: float,
    resolution: int,
) -> list[str]:
    """Construct the command-line for a headless Blender render."""
    return [
        str(blender_path),
        "-b",  # headless (no GUI)
        "-P", str(script_path),  # execute Python script
        "--",  # script args follow
        "--height", str(height_npy),
        "--tint", str(tint_npy),
        "--out", str(out_png),
        "--grid", str(grid),
        "--z-scale", str(z_scale),
        "--resolution", str(resolution),
    ]


def write_obj(height: np.ndarray, tint: np.ndarray, out: Path) -> None:
    """Write a Wavefront OBJ mesh (plain text, diffable).

    The mesh is a downsample of the smooth height field to 256² vertices.
    """
    # Downsample
    src_h, src_w = height.shape
    target = _OBJ_DOWNSAMPLE
    row_idx = (np.arange(target) * (src_h - 1) / (target - 1)).astype(int)
    col_idx = (np.arange(target) * (src_w - 1) / (target - 1)).astype(int)
    h_small = height[np.ix_(row_idx, col_idx)]

    # Normalise
    h_min = float(h_small.min())
    h_max = float(h_small.max())
    h_range = h_max - h_min if h_max > h_min else 1.0
    h_norm = (h_small - h_min) / h_range

    x = np.linspace(-1.0, 1.0, target)
    y = np.linspace(-1.0, 1.0, target)
    xx, yy = np.meshgrid(x, y)

    lines = ["# weight-atlas terrain OBJ", f"# vertices: {target * target}"]

    # Vertices
    for i in range(target):
        for j in range(target):
            lines.append(f"v {xx[i, j]:.6f} {yy[i, j]:.6f} {h_norm[i, j]:.6f}")

    # Faces (quads, 1-indexed)
    for i in range(target - 1):
        for j in range(target - 1):
            v0 = i * target + j + 1
            v1 = i * target + (j + 1) + 1
            v2 = (i + 1) * target + (j + 1) + 1
            v3 = (i + 1) * target + j + 1
            lines.append(f"f {v0} {v1} {v2} {v3}")

    out.write_text("\n".join(lines) + "\n")


def _get_spec_value(spec: AtlasSpec, key: str, default: Any) -> Any:
    """Get a value from spec.blender block, falling back to default."""
    blender_spec = getattr(spec, "blender", None)
    if blender_spec and isinstance(blender_spec, dict):
        return blender_spec.get(key, default)
    return default


@register_renderer("blender")
class BlenderRenderer:
    """Blender headless renderer – registry ID ``"blender"``.

    Renders a topographic view of the height field using an external
    Blender binary. Produces PNGs (raw + smooth) and a diffable OBJ mesh.
    """

    renderer_id = "blender"

    def render(self, field: Field2D, spec: AtlasSpec, out: Path, *, field_name: str = "height") -> list[Path]:
        out.mkdir(parents=True, exist_ok=True)

        # Check for TIFF files first (before resolving Blender path)
        height_tif = out.parent / f"field_{field_name}_smooth.tif"
        height_raw_tif = out.parent / f"field_{field_name}_raw.tif"
        tint_tif = out.parent / "field_tint_smooth.tif"

        if not height_tif.exists():
            raise FileNotFoundError(f"height TIFF not found: {height_tif}")
        if not tint_tif.exists():
            raise FileNotFoundError(f"tint TIFF not found: {tint_tif}")

        blender_path = resolve_blender_path()
        script_path = Path(__file__).parent / "render_terrain.py"

        height = read_tif(height_tif)
        tint = read_tif(tint_tif)

        # Get spec values (from spec.blender block or defaults)
        grid = int(_get_spec_value(spec, "grid", _DEFAULT_GRID))
        resolution = int(_get_spec_value(spec, "resolution", _DEFAULT_RESOLUTION))
        z_scale = float(_get_spec_value(spec, "z_scale", _DEFAULT_Z_SCALE))

        produced: list[Path] = []

        # Render smooth version
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            height_npy = tmpdir_path / "height.npy"
            tint_npy = tmpdir_path / "tint.npy"
            np.save(height_npy, height.astype(np.float64))
            np.save(tint_npy, tint.astype(np.float64))

            out_png_smooth = out / "terrain_smooth.png"
            cmd = build_blender_command(
                blender_path=blender_path,
                script_path=script_path,
                height_npy=height_npy,
                tint_npy=tint_npy,
                out_png=out_png_smooth,
                grid=grid,
                z_scale=z_scale,
                resolution=resolution,
            )

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Blender render failed (exit {result.returncode}):\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}"
                )

            if out_png_smooth.exists():
                produced.append(out_png_smooth)

        # Render raw version if raw TIFF exists
        if height_raw_tif.exists():
            height_raw = read_tif(height_raw_tif)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                height_npy = tmpdir_path / "height.npy"
                tint_npy = tmpdir_path / "tint.npy"
                np.save(height_npy, height_raw.astype(np.float64))
                np.save(tint_npy, tint.astype(np.float64))

                out_png_raw = out / "terrain_raw.png"
                cmd = build_blender_command(
                    blender_path=blender_path,
                    script_path=script_path,
                    height_npy=height_npy,
                    tint_npy=tint_npy,
                    out_png=out_png_raw,
                    grid=grid,
                    z_scale=z_scale,
                    resolution=resolution,
                )

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Blender render failed (exit {result.returncode}):\n"
                        f"STDOUT:\n{result.stdout}\n"
                        f"STDERR:\n{result.stderr}"
                    )

                if out_png_raw.exists():
                    produced.append(out_png_raw)

        # Also write the OBJ mesh (deterministic, no Blender needed)
        obj_path = out / "terrain.obj"
        write_obj(height, tint, obj_path)
        produced.append(obj_path)

        return produced
