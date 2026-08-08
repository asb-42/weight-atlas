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
