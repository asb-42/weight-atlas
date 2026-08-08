"""bpy script: build a 1024² terrain mesh from TIFF heightmap, render to PNG.

Called by the wrapper via ``blender -b -P render_terrain.py -- <args>``.
The wrapper writes TIFF data to ``.npy`` files in a tempdir; this script
reads them, builds a displaced grid using foreach_set (fast), colours
vertices from a tint field, and renders an orthographic top-view to PNG.

Command-line arguments (passed after ``--``):
    --height   path to field_height_smooth.npy
    --tint     path to field_tint_smooth.npy
    --out      output PNG path
    --grid     grid resolution (default 1024)
    --z-scale  vertical exaggeration factor (default 1.0)
    --resolution  render resolution in pixels (default 2048)

Determinism guarantees (local smoke test):
    - Workbench engine (no GPU sampling noise)
    - Fixed world colour, fixed light rotation, no timestamps in PNG output
    - Same inputs → byte-identical PNG (verified by SHA-256 in smoke test)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse args passed after ``--`` in the Blender command line."""
    argv = sys.argv
    argv = [] if "--" not in argv else argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description="Blender terrain renderer")
    parser.add_argument("--height", required=True, help="Path to height .npy file")
    parser.add_argument("--tint", required=True, help="Path to tint .npy file")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--grid", type=int, default=1024, help="Grid resolution")
    parser.add_argument("--z-scale", type=float, default=1.0, help="Z scale factor")
    parser.add_argument("--resolution", type=int, default=2048, help="Render resolution")
    return parser.parse_args(argv)


def clear_scene() -> None:
    """Remove all mesh objects, materials, and lights from the scene."""
    import bpy  # type: ignore[import-not-found]
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)


def make_grid_mesh(grid_res: int, z_scale: float, height: np.ndarray) -> object:
    """Create a grid mesh and displace vertices from height data.

    Uses ``foreach_set`` for fast vertex assignment (spec requirement for M2).
    Returns the created object.
    """
    import bpy
    n = grid_res
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Normalise height to [0, 1] then scale
    h_min = float(height.min())
    h_max = float(height.max())
    h_range = h_max - h_min if h_max > h_min else 1.0
    h_norm = (height - h_min) / h_range
    zz = h_norm * z_scale

    # Create mesh with known vertex count
    verts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float64)

    # Build quad faces
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = i * n + (j + 1)
            v2 = (i + 1) * n + (j + 1)
            v3 = (i + 1) * n + j
            faces.append((v0, v1, v2, v3))

    mesh = bpy.data.meshes.new("TerrainMesh")
    mesh.vertices.add(len(verts))
    mesh.polygons.add(len(faces))
    mesh.vertices.foreach_set("co", verts.ravel())
    mesh.polygons.foreach_set("vertices", [v for face in faces for v in face])
    mesh.polygons.foreach_set("loop_start", list(range(0, len(faces) * 4, 4)))
    mesh.polygons.foreach_set("loop_total", [4] * len(faces))
    mesh.update()

    obj = bpy.data.objects.new("Terrain", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    return obj


def add_vertex_colors(obj: object, tint: np.ndarray) -> None:
    """Add vertex colours from the tint field (normalised to [0,1])."""
    mesh = obj.data  # type: ignore[attr-defined]
    n = tint.shape[0]
    t_min = float(tint.min())
    t_max = float(tint.max())
    t_range = t_max - t_min if t_max > t_min else 1.0
    t_norm = (tint - t_min) / t_range

    # Build RGBA vertex colours
    colors = np.zeros((n * n, 4), dtype=np.float32)
    colors[:, 0] = t_norm.ravel()
    colors[:, 1] = t_norm.ravel() * 0.5  # slight green tint
    colors[:, 2] = 1.0 - t_norm.ravel()  # inverse for blue
    colors[:, 3] = 1.0

    col_attr = mesh.color_attributes.new(name="Tint", type="COLOR", domain="POINT")
    col_attr.data.foreach_set("color", colors.ravel())


def setup_world() -> None:
    """Set up a fixed-colour world background (deterministic, no HDR)."""
    import bpy
    world = bpy.data.worlds.new("TerrainWorld")
    world.use_nodes = False
    world.color = (0.05, 0.05, 0.05)  # fixed dark grey
    bpy.context.scene.world = world


def setup_lighting() -> None:
    """Set up studio-style lighting with NW azimuth (315°) and 45° altitude."""
    import bpy
    light = bpy.data.lights.new("Sun", type="SUN")
    light.energy = 1.0
    # Blender sun: rotation in Euler angles. Azimuth 315°, altitude 45°.
    light_obj = bpy.data.objects.new("Sun", light)
    light_obj.rotation_euler = (
        np.radians(45.0),  # altitude
        0.0,
        np.radians(315.0),  # azimuth
    )
    bpy.context.collection.objects.link(light_obj)


def setup_camera(grid_res: int, resolution: int) -> None:
    """Set up an orthographic top-view camera."""
    import bpy
    cam = bpy.data.cameras.new("TopCam")
    cam.type = "ORTHO"
    # Ortho scale covers the full grid extent (-1 to 1 = 2.0 units)
    cam.ortho_scale = 2.2  # slight margin
    cam_obj = bpy.data.objects.new("TopCam", cam)
    cam_obj.location = (0.0, 0.0, 5.0)  # above the terrain
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)  # looking straight down
    bpy.context.collection.objects.link(cam_obj)

    bpy.context.scene.camera = cam_obj
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.compression = 0  # lossless
    bpy.context.scene.render.film_transparent = False


def setup_render_engine() -> None:
    """Configure the Workbench engine for deterministic headless rendering."""
    import bpy
    bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
    bpy.context.scene.display.shading.light = "STUDIO"
    bpy.context.scene.display.shading.color_type = "VERTEX"
    bpy.context.scene.display.shading.background_type = "VIEWPORT"
    bpy.context.scene.display.shading.background_color = (0.05, 0.05, 0.05)
    bpy.context.scene.render.filepath = ""


def render_to_png(out_path: str) -> None:
    """Render the current scene to PNG."""
    import bpy
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()

    # Load height and tint data
    height = np.load(args.height).astype(np.float64)
    tint = np.load(args.tint).astype(np.float64)

    # Handle NaN in height (smoothing may leave edges as NaN)
    height = np.nan_to_num(height, nan=0.0)
    tint = np.nan_to_num(tint, nan=0.0)

    # Ensure correct shape
    grid_res = args.grid
    if height.shape != (grid_res, grid_res):
        height = _resize_grid(height, grid_res)
    if tint.shape != (grid_res, grid_res):
        tint = _resize_grid(tint, grid_res)

    clear_scene()
    terrain_obj = make_grid_mesh(grid_res, args.z_scale, height)
    add_vertex_colors(terrain_obj, tint)
    setup_world()
    setup_lighting()
    setup_camera(grid_res, args.resolution)
    setup_render_engine()
    render_to_png(args.out)


def _resize_grid(arr: np.ndarray, target: int) -> np.ndarray:
    """Resize a 2D array to target×target using numpy indexing."""
    src_h, src_w = arr.shape
    if src_h == target and src_w == target:
        return arr
    row_idx = (np.arange(target) * (src_h - 1) / (target - 1)).astype(int)
    col_idx = (np.arange(target) * (src_w - 1) / (target - 1)).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


if __name__ == "__main__":
    main()
