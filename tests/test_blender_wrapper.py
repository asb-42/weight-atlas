"""Tests for Blender wrapper – dry-run only, no actual Blender render."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from weight_atlas.core.registry import get_renderer
from weight_atlas.core.registry import reset as registry_reset
from weight_atlas.loaders import (
    gguf_loader,  # noqa: F401 — ensure loader registered
    safetensors_loader,  # noqa: F401 — ensure loader registered
)
from weight_atlas.render import matplotlib_sheet  # noqa: F401 — ensure renderer registered
from weight_atlas.render.blender.blender_wrapper import (
    BlenderRenderer,
    build_blender_command,
    resolve_blender_path,
    write_obj,
)
from weight_atlas.render.blender.render_terrain import (
    _strip_png_metadata,
    compute_effective_z_scale,
    compute_ortho_scale,
    normalise_height,
    resample_bilinear,
)


@pytest.fixture
def _isolated_registry():
    """Provide a clean registry for a single test, then restore."""
    registry_reset()
    yield
    # Re-register everything needed by other tests
    registry_reset()
    importlib.reload(safetensors_loader)
    importlib.reload(matplotlib_sheet)
    importlib.reload(gguf_loader)
    import weight_atlas.render.fractal.wrapper as fractal_wrapper
    importlib.reload(fractal_wrapper)


@pytest.fixture
def tmp_artefacts(tmp_path: Path) -> Path:
    """Create a minimal artefacts directory with height/tint TIFFs."""
    height = np.random.default_rng(42).random((64, 64), dtype=np.float32)
    tint = np.random.default_rng(43).random((64, 64), dtype=np.float32)
    tmp_path.joinpath("field_height_smooth.tif").write_bytes(b"")
    tmp_path.joinpath("field_tint_smooth.tif").write_bytes(b"")
    # Use tifffile to write real data; import here to avoid hard dep in test discovery
    import tifffile
    tifffile.imwrite(tmp_path / "field_height_smooth.tif", height, metadata=None)
    tifffile.imwrite(tmp_path / "field_tint_smooth.tif", tint, metadata=None)
    return tmp_path


@pytest.fixture
def tmp_artefacts_with_raw(tmp_path: Path) -> Path:
    """Create artefacts directory including raw height TIFF."""
    height = np.random.default_rng(42).random((64, 64), dtype=np.float32)
    height_raw = np.random.default_rng(44).random((64, 64), dtype=np.float32)
    tint = np.random.default_rng(43).random((64, 64), dtype=np.float32)
    tmp_path.joinpath("field_height_smooth.tif").write_bytes(b"")
    tmp_path.joinpath("field_height_raw.tif").write_bytes(b"")
    tmp_path.joinpath("field_tint_smooth.tif").write_bytes(b"")
    import tifffile
    tifffile.imwrite(tmp_path / "field_height_smooth.tif", height, metadata=None)
    tifffile.imwrite(tmp_path / "field_height_raw.tif", height_raw, metadata=None)
    tifffile.imwrite(tmp_path / "field_tint_smooth.tif", tint, metadata=None)
    return tmp_path


class TestResolveBlenderPath:
    def test_env_var_set(self, tmp_path: Path):
        fake_blender = tmp_path / "blender"
        fake_blender.write_text("#!/bin/sh\n")
        fake_blender.chmod(0o755)
        with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}):
            result = resolve_blender_path()
        assert result == fake_blender

    def test_env_var_nonexistent(self):
        with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": "/nonexistent/path"}), pytest.raises(FileNotFoundError, match="WEIGHT_ATLAS_BLENDER"):
            resolve_blender_path()

    def test_env_var_unset_falls_back_to_which(self):
        env = {k: v for k, v in os.environ.items() if k != "WEIGHT_ATLAS_BLENDER"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("shutil.which", return_value="/usr/bin/blender"):
            result = resolve_blender_path()
        assert result == Path("/usr/bin/blender")

    def test_no_env_no_which_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "WEIGHT_ATLAS_BLENDER"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("shutil.which", return_value=None), pytest.raises(FileNotFoundError, match="Blender binary not found"):
            resolve_blender_path()


class TestBuildBlenderCommand:
    def test_command_structure(self, tmp_path: Path):
        blender_path = tmp_path / "blender"
        script_path = tmp_path / "render_terrain.py"
        height_npy = tmp_path / "height.npy"
        tint_npy = tmp_path / "tint.npy"
        out_png = tmp_path / "terrain.png"

        cmd = build_blender_command(
            blender_path=blender_path,
            script_path=script_path,
            height_npy=height_npy,
            tint_npy=tint_npy,
            out_png=out_png,
            grid=1024,
            z_scale=0.3,
            resolution=2048,
        )

        assert cmd[0] == str(blender_path)
        assert "-b" in cmd
        assert "-P" in cmd
        assert cmd[cmd.index("-P") + 1] == str(script_path)
        assert "--" in cmd
        args_after_sep = cmd[cmd.index("--") + 1:]
        assert "--height" in args_after_sep
        assert args_after_sep[args_after_sep.index("--height") + 1] == str(height_npy)
        assert "--grid" in args_after_sep
        assert args_after_sep[args_after_sep.index("--grid") + 1] == "1024"
        assert "--z-scale" in args_after_sep
        assert args_after_sep[args_after_sep.index("--z-scale") + 1] == "0.3"
        assert "--resolution" in args_after_sep
        assert args_after_sep[args_after_sep.index("--resolution") + 1] == "2048"

    def test_subsurf_and_fill_args_present(self, tmp_path: Path):
        cmd = build_blender_command(
            blender_path=tmp_path / "blender",
            script_path=tmp_path / "s.py",
            height_npy=tmp_path / "h.npy",
            tint_npy=tmp_path / "t.npy",
            out_png=tmp_path / "o.png",
            grid=1024,
            z_scale=0.3,
            resolution=2048,
            subsurf_levels=2,
            fill_light_energy=0.4,
        )
        args_after_sep = cmd[cmd.index("--") + 1:]
        assert args_after_sep[args_after_sep.index("--subsurf-levels") + 1] == "2"
        assert args_after_sep[args_after_sep.index("--fill-light-energy") + 1] == "0.4"

    def test_subsurf_fill_defaults(self, tmp_path: Path):
        cmd = build_blender_command(
            blender_path=tmp_path / "blender",
            script_path=tmp_path / "s.py",
            height_npy=tmp_path / "h.npy",
            tint_npy=tmp_path / "t.npy",
            out_png=tmp_path / "o.png",
            grid=1024,
            z_scale=0.3,
            resolution=2048,
        )
        args_after_sep = cmd[cmd.index("--") + 1:]
        assert args_after_sep[args_after_sep.index("--subsurf-levels") + 1] == "1"
        assert args_after_sep[args_after_sep.index("--fill-light-energy") + 1] == "0.35"

    def test_new_args_present(self, tmp_path: Path):
        cmd = build_blender_command(
            blender_path=tmp_path / "blender",
            script_path=tmp_path / "render_terrain.py",
            height_npy=tmp_path / "height.npy",
            tint_npy=tmp_path / "tint.npy",
            out_png=tmp_path / "out.png",
            grid=1024,
            z_scale=0.3,
            resolution=2048,
            pitch=18.0,
            clip=0.01,
            adaptive_z_scale=True,
        )
        args_after_sep = cmd[cmd.index("--") + 1:]
        assert "--pitch" in args_after_sep
        assert args_after_sep[args_after_sep.index("--pitch") + 1] == "18.0"
        assert "--clip" in args_after_sep
        assert args_after_sep[args_after_sep.index("--clip") + 1] == "0.01"
        assert "--adaptive-z-scale" in args_after_sep

    def test_default_pitch_and_clip(self, tmp_path: Path):
        cmd = build_blender_command(
            blender_path=tmp_path / "blender",
            script_path=tmp_path / "s.py",
            height_npy=tmp_path / "h.npy",
            tint_npy=tmp_path / "t.npy",
            out_png=tmp_path / "o.png",
            grid=1024,
            z_scale=0.3,
            resolution=2048,
        )
        args_after_sep = cmd[cmd.index("--") + 1:]
        assert args_after_sep[args_after_sep.index("--pitch") + 1] == "18.0"
        assert args_after_sep[args_after_sep.index("--clip") + 1] == "0.01"
        assert "--adaptive-z-scale" not in args_after_sep

    def test_default_values(self, tmp_path: Path):
        cmd = build_blender_command(
            blender_path=Path("/fake/blender"),
            script_path=Path("/fake/script.py"),
            height_npy=Path("/fake/height.npy"),
            tint_npy=Path("/fake/tint.npy"),
            out_png=Path("/fake/out.png"),
            grid=1024,
            z_scale=0.3,
            resolution=2048,
        )
        # Verify all flags present
        assert cmd[:3] == ["/fake/blender", "-b", "-P"]


class TestBlenderRenderer:
    def test_registered_with_id_blender(self, _isolated_registry):
        """Verify the BlenderRenderer is registered with ID 'blender'."""
        # Re-import to trigger registration after fixture reset
        import weight_atlas.render.blender.blender_wrapper as bw_mod
        importlib.reload(bw_mod)
        renderer_cls = get_renderer("blender")
        assert renderer_cls.renderer_id == "blender"

    def test_render_raises_if_height_missing(self, tmp_artefacts: Path):
        height_tif = tmp_artefacts / "field_height_smooth.tif"
        height_tif.unlink()
        spec = mock.MagicMock()
        renderer = BlenderRenderer()
        out = tmp_artefacts / "render"
        out.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="height TIFF not found"):
            renderer.render(mock.MagicMock(), spec, out)

    def test_render_raises_if_tint_missing(self, tmp_artefacts: Path):
        tint_tif = tmp_artefacts / "field_tint_smooth.tif"
        tint_tif.unlink()
        spec = mock.MagicMock()
        renderer = BlenderRenderer()
        out = tmp_artefacts / "render"
        out.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="tint TIFF not found"):
            renderer.render(mock.MagicMock(), spec, out)

    def test_render_dry_run_subprocess_called(self, tmp_artefacts: Path):
        """Dry-run: verify subprocess.run is called with correct command."""
        spec = mock.MagicMock()
        renderer = BlenderRenderer()
        out = tmp_artefacts / "render"
        out.mkdir(parents=True, exist_ok=True)

        # Create a fake blender binary
        fake_blender = tmp_artefacts / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        # Mock the subprocess.run to avoid actually running Blender
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="", stderr="")
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}):
                renderer.render(mock.MagicMock(), spec, out)

        # Verify subprocess.run was called
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert str(fake_blender) in cmd[0]
        assert "-b" in cmd
        assert "-P" in cmd

        # Verify OBJ was written
        assert (out / "terrain.obj").exists()

    def test_render_dry_run_with_raw_variant(self, tmp_artefacts_with_raw: Path):
        """Dry-run: verify raw variant is rendered when raw TIFF exists."""
        spec = mock.MagicMock()
        renderer = BlenderRenderer()
        out = tmp_artefacts_with_raw / "render"
        out.mkdir(parents=True, exist_ok=True)

        fake_blender = tmp_artefacts_with_raw / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="", stderr="")
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}):
                renderer.render(mock.MagicMock(), spec, out)

        # Should be called twice (smooth + raw)
        assert mock_run.call_count == 2

        # Verify first call is for smooth PNG (check --out argument)
        first_cmd = mock_run.call_args_list[0][0][0]
        first_out_idx = first_cmd.index("--out")
        assert "terrain_smooth.png" in first_cmd[first_out_idx + 1]

        # Verify second call is for raw PNG (check --out argument)
        second_cmd = mock_run.call_args_list[1][0][0]
        second_out_idx = second_cmd.index("--out")
        assert "terrain_raw.png" in second_cmd[second_out_idx + 1]

    def test_render_without_raw_variant(self, tmp_artefacts: Path):
        """Verify only smooth variant is rendered when raw TIFF missing."""
        spec = mock.MagicMock()
        renderer = BlenderRenderer()
        out = tmp_artefacts / "render"
        out.mkdir(parents=True, exist_ok=True)

        fake_blender = tmp_artefacts / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="", stderr="")
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}):
                renderer.render(mock.MagicMock(), spec, out)

        # Should be called only once (smooth only)
        mock_run.assert_called_once()

    def test_render_captures_blender_failure(self, tmp_artefacts: Path):
        """Verify RuntimeError on non-zero exit code."""
        spec = mock.MagicMock()
        renderer = BlenderRenderer()
        out = tmp_artefacts / "render"
        out.mkdir(parents=True, exist_ok=True)

        fake_blender = tmp_artefacts / "blender"
        fake_blender.write_text("#!/bin/sh\nexit 1\n")
        fake_blender.chmod(0o755)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=1, stdout="error", stderr="fail"
            )
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}), pytest.raises(RuntimeError, match="Blender render failed"):
                renderer.render(mock.MagicMock(), spec, out)

    def test_render_catches_blender_script_traceback(self, tmp_artefacts: Path):
        """Blender exits 0 even when the -P script crashes; a Python traceback
        in stderr must fail the render (else stale PNGs are silently served)."""
        spec = mock.MagicMock()
        renderer = BlenderRenderer()
        out = tmp_artefacts / "render"
        out.mkdir(parents=True, exist_ok=True)

        fake_blender = tmp_artefacts / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        stderr = (
            "Blender 4.0.2\n"
            "Traceback (most recent call last):\n"
            '  File "/x/render_terrain.py", line 199, in make_grid_mesh\n'
            "AttributeError: 'Object' object has no attribute 'shade_smooth'\n"
        )
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=0, stdout="", stderr=stderr
            )
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}), pytest.raises(RuntimeError, match="Blender script crashed"):
                renderer.render(mock.MagicMock(), spec, out)


class TestWriteObj:
    def test_writes_valid_obj(self, tmp_path: Path):
        """Verify write_obj produces a valid OBJ file with expected structure."""
        height = np.random.default_rng(42).random((16, 16)).astype(np.float32)
        tint = np.random.default_rng(43).random((16, 16)).astype(np.float32)
        out = tmp_path / "terrain.obj"
        write_obj(height, tint, out)
        content = out.read_text()
        lines = content.strip().split("\n")
        # Check header
        assert lines[0].startswith("# weight-atlas")
        # Should have vertices and faces
        vertex_lines = [ln for ln in lines if ln.startswith("v ")]
        face_lines = [ln for ln in lines if ln.startswith("f ")]
        # Downsample to 256x256 from 16x16 (target is always 256)
        assert len(vertex_lines) == 256 * 256
        assert len(face_lines) == 255 * 255

    def test_deterministic(self, tmp_path: Path):
        """Same inputs → byte-identical OBJ."""
        height = np.random.default_rng(42).random((16, 16)).astype(np.float32)
        tint = np.random.default_rng(43).random((16, 16)).astype(np.float32)
        out1 = tmp_path / "terrain1.obj"
        out2 = tmp_path / "terrain2.obj"
        write_obj(height, tint, out1)
        write_obj(height, tint, out2)
        assert out1.read_bytes() == out2.read_bytes()

    def test_handles_nan(self, tmp_path: Path):
        """NaN in input should not crash write_obj."""
        height = np.random.default_rng(42).random((16, 16)).astype(np.float32)
        height[0, 0] = np.nan
        tint = np.random.default_rng(43).random((16, 16)).astype(np.float32)
        out = tmp_path / "terrain.obj"
        write_obj(height, tint, out)
        assert out.exists()

    def test_small_grid_hand_computed(self, tmp_path: Path):
        """Hand-computed OBJ for a 2x2 grid (downsample to 256 still applies)."""
        height = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        tint = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        out = tmp_path / "terrain.obj"
        write_obj(height, tint, out)
        content = out.read_text()
        lines = content.strip().split("\n")
        vertex_lines = [ln for ln in lines if ln.startswith("v ")]
        # Always downsampled to 256x256
        assert len(vertex_lines) == 256 * 256


class TestStripPngMetadata:
    def test_removes_date_and_render_time(self, tmp_path: Path):
        """Blender stamps Date/RenderTime tEXt chunks; stripping them is what
        makes two renders byte-identical. Build a minimal PNG and check."""
        import struct
        import zlib

        def chunk(typ: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + typ
                + payload
                + struct.pack(">I", zlib.crc32(typ + payload))
            )

        ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
        rows = b"".join(b"\x00" + b"\x00\x00\x00\x00\x00\x00" for _ in range(2))
        idat = zlib.compress(rows)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"tEXt", b"Date\x002026/08/17 07:49:32")
            + chunk(b"tEXt", b"RenderTime\x0000:02.83")
            + chunk(b"tEXt", b"Software\x00weight-atlas")
            + chunk(b"IDAT", idat)
            + chunk(b"IEND", b"")
        )
        p = tmp_path / "test.png"
        p.write_bytes(png)
        _strip_png_metadata(str(p))
        out = p.read_bytes()
        assert b"Date" not in out
        assert b"RenderTime" not in out
        assert b"Software" in out
        assert out[:8] == b"\x89PNG\r\n\x1a\n"

    def test_still_valid_png(self, tmp_path: Path):
        """The rewritten file must remain a decodable PNG."""
        import struct
        import zlib

        def chunk(typ: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + typ
                + payload
                + struct.pack(">I", zlib.crc32(typ + payload))
            )

        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        idat = zlib.compress(b"\x00\x80\x80\x80")
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"tEXt", b"RenderTime\x0000:02.84")
            + chunk(b"IDAT", idat)
            + chunk(b"IEND", b"")
        )
        p = tmp_path / "test.png"
        p.write_bytes(png)
        _strip_png_metadata(str(p))
        from PIL import Image
        im = Image.open(p)
        im.load()  # decode
        assert im.size == (1, 1)

    def test_non_png_untouched(self, tmp_path: Path):
        p = tmp_path / "not.png"
        p.write_bytes(b"not a png")
        _strip_png_metadata(str(p))
        assert p.read_bytes() == b"not a png"


class TestResampleBilinear:
    def test_identity_when_same_size(self):
        arr = np.random.default_rng(1).random((64, 64))
        out = resample_bilinear(arr, 64)
        np.testing.assert_allclose(out, arr)

    def test_upsample_constant(self):
        arr = np.full((4, 4), 2.0)
        out = resample_bilinear(arr, 16)
        assert out.shape == (16, 16)
        np.testing.assert_allclose(out, 2.0)

    def test_downsample_constant(self):
        arr = np.full((16, 16), 2.0)
        out = resample_bilinear(arr, 4)
        assert out.shape == (4, 4)
        np.testing.assert_allclose(out, 2.0)

    def test_interpolation_is_smooth(self):
        # Bilinear of a ramp must not step like nearest-neighbour.
        arr = np.arange(16, dtype=float).reshape(4, 4)
        out = resample_bilinear(arr, 8)
        # Interior row should be monotonic (a nearest-neighbour resample of a
        # 4->8 upscale produces plateaus).
        row = out[3]
        assert np.all(np.diff(row) > 0)

    def test_deterministic(self):
        arr = np.random.default_rng(2).random((8, 8))
        a = resample_bilinear(arr, 16)
        b = resample_bilinear(arr, 16)
        np.testing.assert_array_equal(a, b)

    def test_nan_hole_preserved(self):
        arr = np.ones((4, 4))
        arr[1, 1] = np.nan
        out = resample_bilinear(arr, 8)
        # The hole should still contain NaN in its neighbourhood.
        assert np.isnan(out).any()
        # Far from the hole, values stay finite.
        assert np.isfinite(out[0, 0])

    def test_preserves_extrema(self):
        # Corner/edge-aligned sampling keeps the corner value exact.
        arr = np.zeros((4, 4))
        arr[0, 0] = 1.0
        out = resample_bilinear(arr, 16)
        assert out[0, 0] == 1.0


class TestNormaliseHeight:
    def test_clips_outliers(self):
        # One extreme hotspot must not squash the bulk.
        rng = np.random.default_rng(3)
        h = rng.uniform(0.3, 0.7, size=(20, 20))  # realistic bulk
        h[0, 0] = 10.0  # outlier hotspot
        h[1, 1] = -10.0
        norm = normalise_height(h, 0.01)
        assert norm.min() >= 0.0 and norm.max() <= 1.0
        assert np.isfinite(norm).all()
        # Bulk values must stay meaningfully spread (not squashed to one row).
        assert norm.std() > 0.1

    def test_outlier_squashes_plain_minmax(self):
        # Without clip, a single hotspot pushes the bulk to ~0 — the bug
        # robust normalisation exists to fix.
        rng = np.random.default_rng(3)
        h = rng.uniform(0.3, 0.7, size=(20, 20))
        h[0, 0] = 100.0
        plain = normalise_height(h, 0.0)
        robust = normalise_height(h, 0.01)
        assert robust.std() > plain.std() * 5

    def test_plain_minmax_when_clip_zero(self):
        h = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=float)
        norm = normalise_height(h, 0.0)
        assert norm[0, 0] == 0.0
        assert norm[1, 1] == 1.0

    def test_constant_field_zero(self):
        h = np.ones((4, 4))
        norm = normalise_height(h, 0.01)
        assert np.all(norm == 0.0)

    def test_nan_only_zeros(self):
        h = np.full((4, 4), np.nan)
        norm = normalise_height(h, 0.01)
        assert np.all(norm == 0.0)

    def test_nan_masked(self):
        h = np.array([[0.0, 1.0], [np.nan, 2.0]], dtype=float)
        norm = normalise_height(h, 0.01)
        assert np.isfinite(norm).all()


class TestComputeEffectiveZScale:
    def test_non_adaptive_returns_base(self):
        assert compute_effective_z_scale(0.3, np.array([0.5, 0.5]), False) == 0.3

    def test_adaptive_amplifies_weak_relief(self):
        # Nearly-flat field -> small std -> large effective z_scale.
        flat = np.full((10, 10), 0.5)
        flat[0, 0] = 0.51
        z = compute_effective_z_scale(0.3, flat, True)
        assert z > 0.3

    def test_adaptive_caps_at_limit(self):
        # Truly flat field (std ~0) must not explode to infinity.
        flat = np.full((10, 10), 0.5)
        z = compute_effective_z_scale(0.3, flat, True)
        assert np.isfinite(z)
        assert z <= 5.0

    def test_adaptive_constant_std(self):
        # Uniform [0,1] has std ~0.29; effective scale should exceed base.
        rng = np.random.default_rng(7)
        h = rng.random((100, 100))
        z = compute_effective_z_scale(0.3, h, True)
        assert z > 0.3


class TestComputeOrthoScale:
    def test_top_down_uses_base(self):
        assert compute_ortho_scale(0.0, 0.3) == 2.2

    def test_tilted_exceeds_base_with_high_z(self):
        # Large z_scale + tilt -> projected extent larger than 2.2.
        s = compute_ortho_scale(18.0, 5.0)
        assert s > 2.2

    def test_default_tilt_stays_at_base(self):
        # 18° pitch, z_scale 0.3: 2*cos(18°)+0.3*sin(18°) ~1.99 -> base 2.2.
        assert compute_ortho_scale(18.0, 0.3) == 2.2
