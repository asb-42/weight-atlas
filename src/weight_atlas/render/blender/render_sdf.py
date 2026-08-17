"""bpy script: build a mesh from verts/faces arrays, render to PNG.

Called by the wrapper via ``blender -b -P render_sdf.py -- <args>``. The
wrapper writes the SDF mosaic mesh to ``.npy`` files in a tempdir; this
script reads them, builds the mesh using foreach_set (fast), colours vertices
from a per-vertex tint array, and renders an orthographic view to PNG — the
same deterministic Workbench pipeline as ``render_terrain.py``.

Command-line arguments (passed after ``--``):
    --verts    path to verts.npy (N, 3) float64
    --faces    path to faces.npy (M, 3) int64
    --tint     path to tint.npy (N,) float64 per-vertex colour channel
    --out      output PNG path
    --grid     unused (kept for CLI symmetry); the mesh carries its own res
    --z-scale  base vertical exaggeration factor (default 0.3)
    --pitch    camera pitch angle in degrees (default 18.0)
    --resolution  render resolution in pixels (default 2048)
    --fill-light-energy  fill sun energy, lifts the shadow side (default 0.35)

Determinism guarantees (same as the terrain script):
    - Workbench engine (no GPU sampling noise)
    - Fixed world colour, fixed light rotation, no timestamps in PNG output
      (Blender's Date/RenderTime tEXt chunks are stripped after rendering)
    - Same inputs → byte-identical PNG (verified by SHA-256 in smoke test)

Geometry: the mesh is smooth-shaded with a 30° auto-smooth cutoff so the
fractal facets read crisply while shared normals keep shading continuous.
No subdivision is applied — the SDF mesh is already the surface.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# The terrain script ships the shared bpy helpers; make sure its directory is
# importable regardless of how Blender invoked this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_terrain import (  # type: ignore[import-not-found]
    clear_scene,
    render_to_png,
    setup_camera,
    setup_lighting,
    setup_render_engine,
    setup_world,
)


def parse_args() -> argparse.Namespace:
    """Parse args passed after ``--`` in the Blender command line."""
    argv = sys.argv
    argv = [] if "--" not in argv else argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description="Blender SDF mesh renderer")
    parser.add_argument("--verts", required=True, help="Path to verts.npy")
    parser.add_argument("--faces", required=True, help="Path to faces.npy")
    parser.add_argument("--tint", required=True, help="Path to tint.npy")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--grid", type=int, default=1024, help="Unused (CLI symmetry)")
    parser.add_argument("--z-scale", type=float, default=0.3, help="Base Z scale factor")
    parser.add_argument("--pitch", type=float, default=18.0,
                        help="Camera pitch angle in degrees (0 = top-down)")
    parser.add_argument("--resolution", type=int, default=2048, help="Render resolution")
    parser.add_argument("--fill-light-energy", type=float, default=0.35,
                        help="Fill sun energy (soft opposite-side light)")
    return parser.parse_args(argv)


def make_mesh(verts: np.ndarray, faces: np.ndarray) -> object:
    """Create a mesh object from vertex and triangle arrays (smooth-shaded)."""
    import bpy  # type: ignore[import-not-found]
    mesh = bpy.data.meshes.new("SdfMesh")
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])

    obj = bpy.data.objects.new("SdfMosaic", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    for poly in mesh.polygons:
        poly.use_smooth = True
    try:
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = np.radians(30.0)
    except AttributeError:
        pass
    return obj


def add_vertex_colors(obj: object, tint: np.ndarray) -> None:
    """Add per-vertex colours from the tint channel (normalised to [0,1])."""
    mesh = obj.data  # type: ignore[attr-defined]
    t_min = float(tint.min())
    t_max = float(tint.max())
    t_range = t_max - t_min if t_max > t_min else 1.0
    t_norm = (tint - t_min) / t_range

    colors = np.zeros((len(tint), 4), dtype=np.float32)
    colors[:, 0] = t_norm
    colors[:, 1] = t_norm * 0.5  # slight green tint
    colors[:, 2] = 1.0 - t_norm  # inverse for blue
    colors[:, 3] = 1.0

    col_attr = mesh.color_attributes.new(name="Tint", type="FLOAT_COLOR", domain="POINT")
    col_attr.data.foreach_set("color", colors.ravel())


def main() -> None:
    args = parse_args()

    verts = np.load(args.verts).astype(np.float64)
    faces = np.load(args.faces).astype(np.int64)
    tint = np.load(args.tint).astype(np.float64)
    if len(faces) == 0:
        raise ValueError("empty SDF mesh: no faces")

    # The wrapper already normalises the mosaic footprint into the [-1, 1]²
    # camera frame; here we only apply the z exaggeration so relief reads.
    verts[:, 2] *= args.z_scale
    z_scale = float(np.abs(verts[:, 2]).max()) if len(verts) else args.z_scale

    clear_scene()
    obj = make_mesh(verts, faces)
    add_vertex_colors(obj, tint)
    setup_world()
    setup_lighting(args.fill_light_energy)
    setup_camera(1024, args.resolution, pitch=args.pitch, z_scale=max(z_scale, args.z_scale))
    setup_render_engine()
    render_to_png(args.out)


if __name__ == "__main__":
    main()
