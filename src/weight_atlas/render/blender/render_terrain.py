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
    --z-scale  base vertical exaggeration factor (default 1.0)
    --clip     percentile clip for robust height normalisation (default 0.01)
    --adaptive-z-scale  rescale z so relief std is constant (flag)
    --pitch    camera pitch angle in degrees (default 18.0)
    --resolution  render resolution in pixels (default 2048)
    --subsurf-levels  Catmull-Clark subdivision levels (default 1, 0=off)
    --fill-light-energy  fill sun energy, lifts the shadow side (default 0.35)

Determinism guarantees (local smoke test):
    - Workbench engine (no GPU sampling noise)
    - Fixed world colour, fixed light rotation, no timestamps in PNG output
      (Blender's Date/RenderTime tEXt chunks are stripped after rendering)
    - Same inputs → byte-identical PNG (verified by SHA-256 in smoke test)

Geometry smoothing (terrain, not raw values):
    The height field is bilinearly resampled to the grid (no nearest-
    neighbour blockiness), the mesh is smooth-shaded, and a Catmull-Clark
    subdivision surface interpolates it — the classic terrain-renderer
    smoothing step that keeps the geometry continuous. ``--subsurf-levels 0``
    restores a raw flat-shaded mesh. All steps are deterministic (fixed
    topology operations, no sampling).

Robust height normalisation:
    Raw heights are clipped to the [clip, 1-clip] percentile band and
    rescaled to [0,1] (not plain min/max). A single outlier hotspot can no
    longer squash the bulk of the field into a flat slab. With
    ``--adaptive-z-scale`` the effective z_scale becomes
    ``base_z_scale / std(normalised)`` (capped), giving weak-relief fields a
    constant visible amplitude — at the cost of losing absolute-amplitude
    comparability between models.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

_Z_SCALE_CAP = 5.0
_EPS = 1e-9


def parse_args() -> argparse.Namespace:
    """Parse args passed after ``--`` in the Blender command line."""
    argv = sys.argv
    argv = [] if "--" not in argv else argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description="Blender terrain renderer")
    parser.add_argument("--height", required=True, help="Path to height .npy file")
    parser.add_argument("--tint", required=True, help="Path to tint .npy file")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--grid", type=int, default=1024, help="Grid resolution")
    parser.add_argument("--z-scale", type=float, default=1.0, help="Base Z scale factor")
    parser.add_argument("--clip", type=float, default=0.01,
                        help="Percentile clip (0-1) for robust height normalisation")
    parser.add_argument("--adaptive-z-scale", action="store_true",
                        help="Rescale Z so relief std is constant across fields")
    parser.add_argument("--pitch", type=float, default=18.0,
                        help="Camera pitch angle in degrees (0 = top-down)")
    parser.add_argument("--resolution", type=int, default=2048, help="Render resolution")
    parser.add_argument("--subsurf-levels", type=int, default=1,
                        help="Catmull-Clark subdivision levels (0 = raw flat mesh)")
    parser.add_argument("--fill-light-energy", type=float, default=0.35,
                        help="Fill sun energy (soft opposite-side light)")
    return parser.parse_args(argv)


def normalise_height(height: np.ndarray, clip: float) -> np.ndarray:
    """Robustly normalise a height field to [0,1].

    Clips to the ``[clip, 1-clip]`` percentile band before rescaling, so a few
    outlier hotspots cannot flatten the bulk of the field. Constant fields
    (or NaN-only) yield a zero array. Handles clip=0 as plain min/max.
    """
    height = np.asarray(height, dtype=np.float64)
    finite = height[np.isfinite(height)]
    if finite.size == 0:
        return np.zeros_like(height)

    if clip > 0:
        lo = float(np.percentile(finite, clip * 100.0))
        hi = float(np.percentile(finite, 100.0 - clip * 100.0))
    else:
        lo = float(finite.min())
        hi = float(finite.max())

    span = hi - lo
    if span <= _EPS:
        return np.zeros_like(height)

    # Clip through finite values only, so NaN inputs map to NaN output.
    clipped = np.where(np.isfinite(height), np.clip(height, lo, hi), np.nan)
    norm = (clipped - lo) / span
    return np.nan_to_num(norm, nan=0.0)


def compute_effective_z_scale(
    base_z_scale: float, h_norm: np.ndarray, adaptive: bool
) -> float:
    """Return the effective Z scale.

    ``adaptive=True``: base / std(normalised height), capped at ``_Z_SCALE_CAP``.
    ``adaptive=False``: the base value unchanged.
    """
    if not adaptive:
        return float(base_z_scale)
    std = float(np.std(h_norm))
    if std <= _EPS:
        return float(base_z_scale)
    return min(base_z_scale / std, _Z_SCALE_CAP)


def compute_ortho_scale(pitch_deg: float, z_scale: float, base: float = 2.2) -> float:
    """Compute the ortho scale that fits the tilted grid + relief.

    The projected vertical extent of the ±1 unit grid under a ``pitch_deg``
    tilt is ``2*cos(p) + z_scale*sin(p)``. Returns that (times a 10% margin)
    when it exceeds the top-down ``base``, else ``base``.
    """
    p = np.radians(pitch_deg)
    projected = 2.0 * np.cos(p) + z_scale * np.sin(p)
    return max(float(base), float(projected) * 1.1)


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


def make_grid_mesh(
    grid_res: int,
    z_scale: float,
    h_norm: np.ndarray,
    subsurf_levels: int = 1,
) -> object:
    """Create a grid mesh and displace vertices from a normalised height field.

    ``h_norm`` must already be in [0,1] (see ``normalise_height``); it is
    scaled by ``z_scale`` for vertical exaggeration. Uses ``from_pydata`` which
    handles loop allocation internally and is stable for large meshes on
    Blender 4.x. Smooth-shaded and optionally Catmull-Clark subdivided so the
    geometry renders as continuous terrain rather than flat facets. Returns the
    created object.
    """
    import bpy
    n = grid_res
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    zz = h_norm * z_scale

    # Build vertex list
    verts = [(xx[i, j], yy[i, j], zz[i, j]) for i in range(n) for j in range(n)]

    # Build quad faces (empty edges for from_pydata)
    edges: list = []
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = i * n + (j + 1)
            v2 = (i + 1) * n + (j + 1)
            v3 = (i + 1) * n + j
            faces.append((v0, v1, v2, v3))

    mesh = bpy.data.meshes.new("TerrainMesh")
    mesh.from_pydata(verts, edges, faces)

    obj = bpy.data.objects.new("Terrain", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Smooth shading: one vertex normal shared across faces, so Workbench
    # interpolates shading continuously instead of showing every quad facet.
    # ``obj.shade_smooth()`` is a bpy.ops operator on 4.0 (Object has no such
    # method), so use the mesh data-polygon flag which works on every version.
    for poly in mesh.polygons:
        poly.use_smooth = True
    try:
        # Auto-smooth adds a 30° cutoff so the grid's true edges (rim, seams)
        # stay crisp instead of being fully rounded. Blender 4.1+ moved this
        # property; guard for compatibility.
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = np.radians(30.0)
    except AttributeError:
        pass

    # Catmull-Clark subdivision interpolates the displacement into smooth
    # continuous terrain — the classic terrain-renderer geometry step.
    if subsurf_levels > 0:
        mod = obj.modifiers.new(name="Subsurf", type="SUBSURF")
        mod.levels = min(int(subsurf_levels), 2)
        mod.render_levels = mod.levels
        mod.subdivision_type = "CATMULL_CLARK"

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

    col_attr = mesh.color_attributes.new(name="Tint", type="FLOAT_COLOR", domain="POINT")
    col_attr.data.foreach_set("color", colors.ravel())


def setup_world() -> None:
    """Set up a fixed-colour world background (deterministic, no HDR)."""
    import bpy
    world = bpy.data.worlds.new("TerrainWorld")
    world.use_nodes = False
    world.color = (0.05, 0.05, 0.05)  # fixed dark grey
    bpy.context.scene.world = world


def setup_lighting(fill_energy: float = 0.35) -> None:
    """Set up studio-style lighting: NW key sun + soft opposite-side fill.

    Azimuth 315° / altitude 45° for the key (matches the sheet hillshade), a
    low-energy fill from the opposite azimuth so the shadow side of the terrain
    reads instead of blacking out. Both rotations are fixed → deterministic.
    """
    import bpy
    # Key sun (NW)
    key = bpy.data.lights.new("KeySun", type="SUN")
    key.energy = 1.0
    key_obj = bpy.data.objects.new("KeySun", key)
    key_obj.rotation_euler = (
        np.radians(45.0),  # altitude
        0.0,
        np.radians(315.0),  # azimuth
    )
    bpy.context.collection.objects.link(key_obj)

    # Soft fill sun (SE, opposite azimuth, low altitude + low energy)
    if fill_energy > 0:
        fill = bpy.data.lights.new("FillSun", type="SUN")
        fill.energy = float(fill_energy)
        fill_obj = bpy.data.objects.new("FillSun", fill)
        fill_obj.rotation_euler = (
            np.radians(25.0),  # altitude
            0.0,
            np.radians(135.0),  # azimuth
        )
        bpy.context.collection.objects.link(fill_obj)


def setup_camera(grid_res: int, resolution: int, pitch: float = 18.0, z_scale: float = 1.0) -> None:
    """Set up an orthographic camera tilted by ``pitch`` degrees."""
    import bpy
    cam = bpy.data.cameras.new("TerrainCam")
    cam.type = "ORTHO"
    # Ortho scale covers the full tilted grid + relief extent (computed so the
    # 10% margin keeps edges in frame; never smaller than the top-down size).
    cam.ortho_scale = compute_ortho_scale(pitch, z_scale)
    cam_obj = bpy.data.objects.new("TerrainCam", cam)
    cam_obj.location = (0.0, 0.0, 5.0)  # above the terrain
    cam_obj.rotation_euler = (np.radians(pitch), 0.0, 0.0)  # tilt forward
    bpy.context.collection.objects.link(cam_obj)

    bpy.context.scene.camera = cam_obj
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.compression = 0  # lossless
    bpy.context.scene.render.film_transparent = False


def setup_render_engine() -> None:
    """Configure the Workbench engine for deterministic headless rendering.

    ``use_scene_lights`` + ``use_scene_world`` must be enabled or Workbench's
    STUDIO light mode ignores the scene SUN objects entirely (the key/fill
    lights would be a no-op for the render). All fixed → deterministic.
    """
    import bpy
    bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
    bpy.context.scene.display.shading.light = "STUDIO"
    bpy.context.scene.display.shading.color_type = "VERTEX"
    bpy.context.scene.display.shading.background_type = "VIEWPORT"
    bpy.context.scene.display.shading.background_color = (0.05, 0.05, 0.05)
    bpy.context.scene.display.shading.use_scene_lights = True
    bpy.context.scene.display.shading.use_scene_world = True
    bpy.context.scene.render.filepath = ""


def _strip_png_metadata(path: str) -> None:
    """Remove non-deterministic tEXt chunks (Date/RenderTime) from a PNG.

    Blender stamps every rendered PNG with the wall-clock time and render
    duration, which breaks the byte-identical determinism contract. Rewrites
    the file in place with those chunks dropped (pure stdlib).
    """
    import struct
    import zlib

    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return

    out = bytearray(data[:8])
    off = 8
    while off < len(data):
        ln = struct.unpack(">I", data[off : off + 4])[0]
        typ = data[off + 4 : off + 8]
        if typ == b"tEXt":
            payload = data[off + 8 : off + 8 + ln]
            key = payload.split(b"\x00", 1)[0] if b"\x00" in payload else b""
            if key in (b"Date", b"RenderTime"):
                off += 12 + ln
                continue
        chunk = data[off : off + 12 + ln]
        if typ == b"IEND":
            # Recompute CRC for the rewritten IEND chunk.
            chunk = chunk[:4] + chunk[4:8] + struct.pack(">I", zlib.crc32(chunk[4:-4]))
        out += chunk
        off += 12 + ln

    with open(path, "wb") as fh:
        fh.write(bytes(out))


def render_to_png(out_path: str) -> None:
    """Render the current scene to PNG (with non-deterministic tEXt stripped)."""
    import bpy
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    _strip_png_metadata(out_path)


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

    # Effective Z scale (adaptive = constant relief amplitude)
    h_norm = normalise_height(height, args.clip)
    z_scale = compute_effective_z_scale(args.z_scale, h_norm, args.adaptive_z_scale)

    clear_scene()
    terrain_obj = make_grid_mesh(grid_res, z_scale, h_norm, args.subsurf_levels)
    add_vertex_colors(terrain_obj, tint)
    setup_world()
    setup_lighting(args.fill_light_energy)
    setup_camera(grid_res, args.resolution, pitch=args.pitch, z_scale=z_scale)
    setup_render_engine()
    render_to_png(args.out)


def resample_bilinear(arr: np.ndarray, target: int) -> np.ndarray:
    """Bilinear resample of a 2D array to ``target``×``target``.

    Pure NumPy (no scipy — Blender's bundled Python may lack it). NaN inputs
    are masked out of the interpolation and restored as NaN in the output so
    the downstream NaN handling stays unchanged. Deterministic.
    """
    src_h, src_w = arr.shape
    if src_h == target and src_w == target:
        return arr

    src = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(src)
    src_clean = np.where(valid, src, 0.0)

    # Source coordinates for each output pixel (edge-aligned, like the old
    # nearest-neighbour sampler so geometry extents stay unchanged).
    ys = np.linspace(0.0, src_h - 1.0, target)
    xs = np.linspace(0.0, src_w - 1.0, target)
    y0 = np.floor(ys).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x0 = np.floor(xs).astype(np.int64)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]

    # Gather the four corners (broadcast to target×target).
    v00 = src_clean[y0][:, x0]
    v10 = src_clean[y1][:, x0]
    v01 = src_clean[y0][:, x1]
    v11 = src_clean[y1][:, x1]

    top = v00 * (1 - wy) + v10 * wy
    bottom = v01 * (1 - wy) + v11 * wy
    out = top * (1 - wx) + bottom * wx

    # Restore NaN holes: any output pixel whose 2×2 footprint contains a NaN
    # source sample is NaN (mirrors the old sampler's NaN propagation).
    w_valid = (
        valid[y0][:, x0].astype(np.float64) * (1 - wy) + valid[y1][:, x0].astype(np.float64) * wy
    )
    w_valid = w_valid * (1 - wx) + (
        valid[y0][:, x1].astype(np.float64) * (1 - wy) + valid[y1][:, x1].astype(np.float64) * wy
    ) * wx
    return np.where(w_valid >= 1.0, out, np.nan)


def _resize_grid(arr: np.ndarray, target: int) -> np.ndarray:
    """Resize a 2D array to target×target (bilinear, NaN-safe)."""
    return resample_bilinear(arr, target)


if __name__ == "__main__":
    main()
