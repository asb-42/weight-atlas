"""Tests for the fractal terrain renderer (fBm + per-slot stat mapping)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from weight_atlas.core.registry import get_renderer
from weight_atlas.core.types import AtlasSpec, Field2D
from weight_atlas.render.fractal.fbm import fbm, slot_fractal_field, value_noise
from weight_atlas.render.fractal.mosaic import build_sdf_mosaic
from weight_atlas.render.fractal.params import (
    slot_fractal_params,
    slot_sdf_params,
    slot_stat_medians,
    slot_stat_tint,
    stats_to_params,
)
from weight_atlas.render.fractal.sdf import menger_sdf, sdf_volume
from weight_atlas.render.fractal.surface_nets import surface_nets
from weight_atlas.render.fractal.wrapper import FractalRenderer


def _make_fingerprint(slots: list[str], n_layers: int = 3) -> dict:
    """Slot → realistic GGUF tensor name (name_map maps these to slots)."""
    name_by_slot = {"attn_q": "attn_q", "mlp_gate": "ffn_gate"}
    tensors = {}
    rng = np.random.default_rng(42)
    for layer in range(n_layers):
        for s in slots:
            tn = name_by_slot.get(s, s)
            tensors[f"blk.{layer}.{tn}.weight"] = {
                "kurtosis": float(rng.uniform(0.2, 3.0)),
                "sparsity": float(rng.uniform(0.0, 0.5)),
                "effective_rank": float(rng.uniform(2, 30)),
                "spectral_norm": float(rng.uniform(1, 8)),
            }
    return {"model": {}, "tensors": tensors}


def _make_spec(tmp_path: Path, out_dir: Path) -> AtlasSpec:
    fp = _make_fingerprint(["attn_q", "mlp_gate"])
    (out_dir / "fingerprint.json").write_text(json.dumps(fp))
    spec = AtlasSpec(
        spec_version=4,
        slots=["attn_q", "mlp_gate"],
        channels={},
        grid={},
        sheet={},
        seeds={"svd": 0, "fractal": 0},
        fractal={
            "seed": 0,
            "cell_h": 4,
            "cell_w": 4,
            "mapping": {
                "octaves": {"stat": "effective_rank", "lo": 4, "hi": 8},
                "persistence": {"stat": "kurtosis", "lo": 0.4, "hi": 0.7},
                "lacunarity": {"stat": "sparsity", "lo": 1.8, "hi": 2.4},
                "base_freq": {"stat": "spectral_norm", "lo": 1.0, "hi": 2.5},
            },
        },
    )
    return spec


class TestFbm:
    def test_deterministic(self):
        coords = np.random.default_rng(0).random((16, 16, 2))
        a = fbm(coords, 4, 0.5, 2.0, 1.0, 7)
        b = fbm(coords, 4, 0.5, 2.0, 1.0, 7)
        np.testing.assert_array_equal(a, b)

    def test_value_noise_deterministic_and_seed_sensitive(self):
        coords = np.random.default_rng(1).random((8, 8, 2))
        np.testing.assert_array_equal(value_noise(coords, 7), value_noise(coords, 7))
        assert not np.array_equal(value_noise(coords, 7), value_noise(coords, 8))

    def test_finite_and_in_unit_range(self):
        coords = np.random.default_rng(2).random((32, 32, 2))
        out = fbm(coords, 6, 0.5, 2.0, 1.0, 3)
        assert np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() < 1.0

    def test_octave_count_changes_output(self):
        # More octaves stack more detail; the output must differ while staying
        # deterministic (spectral-share metrics are confounded by fBm's
        # normalisation, so assert the contract directly).
        coords = np.random.default_rng(3).random((64, 64, 2))
        low = fbm(coords, 1, 0.5, 2.0, 2.0, 5)
        high = fbm(coords, 6, 0.5, 2.0, 2.0, 5)
        assert not np.array_equal(low, high)
        assert np.isfinite(low).all() and np.isfinite(high).all()


class TestSlotFractalField:
    def test_shape_and_determinism(self):
        params = {
            "a": {"octaves": 4, "persistence": 0.5, "lacunarity": 2.0, "base_freq": 1.0, "seed": 1},
            "b": {"octaves": 6, "persistence": 0.6, "lacunarity": 2.4, "base_freq": 2.0, "seed": 2},
        }
        f = slot_fractal_field(5, 2, params, ["a", "b"], cell_h=4, cell_w=4)
        assert f.shape == (20, 8)
        assert np.isfinite(f).all()
        np.testing.assert_array_equal(
            f, slot_fractal_field(5, 2, params, ["a", "b"], cell_h=4, cell_w=4)
        )

    def test_slots_produce_distinct_columns(self):
        params = {
            "a": {"octaves": 3, "persistence": 0.3, "lacunarity": 1.8, "base_freq": 0.5, "seed": 1},
            "b": {"octaves": 8, "persistence": 0.7, "lacunarity": 2.4, "base_freq": 3.0, "seed": 2},
        }
        f = slot_fractal_field(4, 2, params, ["a", "b"], cell_h=4, cell_w=4)
        assert not np.allclose(f[:, :4], f[:, 4:])

    def test_normalised_range(self):
        params = {
            "a": {"octaves": 4, "persistence": 0.5, "lacunarity": 2.0, "base_freq": 1.0, "seed": 1},
            "b": {"octaves": 4, "persistence": 0.5, "lacunarity": 2.0, "base_freq": 1.0, "seed": 2},
        }
        f = slot_fractal_field(3, 2, params, ["a", "b"], cell_h=4, cell_w=4)
        assert f.min() >= 0.0 and f.max() <= 1.0


class TestParams:
    def test_slot_stat_medians(self):
        slots = ["attn_q", "mlp_gate"]
        fp = _make_fingerprint(slots, n_layers=3)
        med = slot_stat_medians(fp, slots)
        assert set(med) == set(slots)
        for s in slots:
            assert set(med[s]) == {"kurtosis", "sparsity", "effective_rank", "spectral_norm"}
            assert all(np.isfinite(v) for v in med[s].values())

    def test_slot_stat_medians_skips_non_layer(self):
        slots = ["attn_q"]
        fp = {"tensors": {"embed_tokens.weight": {"kurtosis": 1.0}, "blk.0.attn_q.weight": {"kurtosis": 2.0}}}
        med = slot_stat_medians(fp, slots)
        # embed is non-layer → skipped; only layer tensors counted.
        assert med["attn_q"]["kurtosis"] == 2.0

    def test_stats_to_params_maps_ranges(self):
        slot_stats = {
            "a": {"kurtosis": 0.2, "sparsity": 0.0, "effective_rank": 2.0, "spectral_norm": 1.0},
            "b": {"kurtosis": 3.0, "sparsity": 0.5, "effective_rank": 30.0, "spectral_norm": 8.0},
        }
        mapping = {
            "octaves": {"stat": "effective_rank", "lo": 4, "hi": 8},
            "persistence": {"stat": "kurtosis", "lo": 0.4, "hi": 0.7},
            "lacunarity": {"stat": "sparsity", "lo": 1.8, "hi": 2.4},
        }
        out = stats_to_params(slot_stats, mapping)
        # Extremes map to range endpoints.
        assert out["octaves"]["a"] == 4.0
        assert out["octaves"]["b"] == 8.0
        assert out["persistence"]["a"] == 0.4
        assert out["persistence"]["b"] == 0.7
        assert out["lacunarity"]["a"] == 1.8
        assert out["lacunarity"]["b"] == 2.4

    def test_stats_to_params_nan_falls_back_to_mid(self):
        slot_stats = {"a": {"kurtosis": float("nan"), "effective_rank": 5.0}}
        mapping = {"persistence": {"stat": "kurtosis", "lo": 0.4, "hi": 0.7}}
        out = stats_to_params(slot_stats, mapping)
        assert out["persistence"]["a"] == pytest.approx(0.55)

    def test_slot_fractal_params_deterministic(self, tmp_path: Path):
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        p1 = slot_fractal_params(out_dir, spec.slots, spec.fractal, spec.seeds["fractal"])
        p2 = slot_fractal_params(out_dir, spec.slots, spec.fractal, spec.seeds["fractal"])
        assert p1 == p2

    def test_slot_fractal_params_per_slot_seed(self, tmp_path: Path):
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        p = slot_fractal_params(out_dir, spec.slots, spec.fractal, 0)
        assert p["attn_q"]["seed"] != p["mlp_gate"]["seed"]
        assert p["attn_q"]["octaves"] in range(4, 9)


class TestSdf:
    def test_menger_sdf_deterministic(self):
        coords = np.random.default_rng(4).random((16, 16, 3)) * 2 - 1
        a = menger_sdf(coords, 3, 3.0)
        b = menger_sdf(coords, 3, 3.0)
        np.testing.assert_array_equal(a, b)

    def test_menger_sdf_inside_outside(self):
        # The sponge's solid corner columns are inside (negative); the carved
        # centre hole and a far corner are outside (positive).
        assert menger_sdf(np.array([[0.5, 0.5, 0.5]]), 2, 3.0)[0] < 0
        assert menger_sdf(np.array([[0.0, 0.0, 0.0]]), 2, 3.0)[0] > 0
        assert menger_sdf(np.array([[2.0, 2.0, 2.0]]), 2, 3.0)[0] > 0

    def test_surface_nets_watertight_sphere(self):
        # A sphere has a clean closed surface: every edge shared by exactly
        # two faces (no boundary, no T-junction) → watertight.
        n = 16
        axis = np.linspace(-1.5, 1.5, n + 1)
        zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
        vol = np.sqrt(xx * xx + yy * yy + zz * zz) - 1.0
        verts, faces = surface_nets(vol)
        assert len(verts) > 0 and len(faces) > 0

        from collections import Counter
        edge_count = Counter()
        for a, b, c in faces:
            for e in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                edge_count[tuple(sorted(e))] += 1
        assert all(v == 2 for v in edge_count.values())

    def test_surface_nets_outward_normals_sphere(self):
        # Centre the sphere at the origin; all face normals must point away
        # from the centroid (outward orientation).
        n = 16
        axis = np.linspace(-1.5, 1.5, n + 1)
        zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
        vol = np.sqrt(xx * xx + yy * yy + zz * zz) - 1.0
        verts, faces = surface_nets(vol)
        centre = verts.mean(axis=0)
        for a, b, c in faces[:50]:
            nrm = np.cross(verts[b] - verts[a], verts[c] - verts[a])
            assert np.dot(nrm, verts[a] - centre) > 0

    def test_surface_nets_deterministic(self):
        n = 12
        axis = np.linspace(-1.3, 1.3, n + 1)
        zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
        vol = np.sqrt(xx * xx + yy * yy + zz * zz) - 1.0
        v1, f1 = surface_nets(vol)
        v2, f2 = surface_nets(vol)
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(f1, f2)

    def test_sdf_volume_shape(self):
        vol = sdf_volume("menger", {"iterations": 2, "scale": 3.0}, 8)
        assert vol.shape == (9, 9, 9)
        assert np.isfinite(vol).all()

    def test_sdf_volume_unknown_family_raises(self):
        with pytest.raises(ValueError, match="unknown SDF family"):
            sdf_volume("bogus", {"iterations": 2}, 8)

    def test_slot_sdf_params_deterministic_and_clamped(self, tmp_path: Path):
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        fractal = dict(spec.fractal)
        fractal["sdf"] = {
            "family": "menger",
            "mapping": {
                "iterations": {"stat": "effective_rank", "lo": 1, "hi": 10},
                "scale": {"stat": "kurtosis", "lo": 2.5, "hi": 3.5},
            },
        }
        p1 = slot_sdf_params(out_dir, spec.slots, fractal, 16)
        p2 = slot_sdf_params(out_dir, spec.slots, fractal, 16)
        assert p1 == p2
        # grid=16 → max_iter = round(16/6) = 3; high lo→hi range must clamp.
        assert all(1 <= row["iterations"] <= 3 for row in p1.values())
        assert all(2.5 <= row["scale"] <= 3.5 for row in p1.values())

    def test_build_sdf_mosaic_deterministic(self):
        params = {
            "a": {"iterations": 2, "scale": 3.0},
            "b": {"iterations": 3, "scale": 3.0},
        }
        m1 = build_sdf_mosaic(2, 2, ["a", "b"], params, "menger", 12, cell_h=8, cell_w=8)
        m2 = build_sdf_mosaic(2, 2, ["a", "b"], params, "menger", 12, cell_h=8, cell_w=8)
        for a, b in zip(m1, m2, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_build_sdf_mosaic_footprint_and_tint(self):
        params = {"a": {"iterations": 2, "scale": 3.0}}
        verts, faces, tint = build_sdf_mosaic(
            1, 1, ["a"], params, "menger", 12, cell_h=8, cell_w=8
        )
        assert len(verts) > 0 and len(faces) > 0
        assert len(tint) == len(verts)
        # Footprint normalised to the [-1, 1]² render frame.
        assert verts[:, 0].min() >= -1.0 - 1e-9 and verts[:, 0].max() <= 1.0 + 1e-9
        assert verts[:, 1].min() >= -1.0 - 1e-9 and verts[:, 1].max() <= 1.0 + 1e-9
        assert tint.min() >= 0.0 and tint.max() <= 1.0

    def test_build_sdf_mosaic_mandelbulb(self):
        params = {"a": {"iterations": 3, "power": 6.0}}
        verts, faces, tint = build_sdf_mosaic(
            1, 1, ["a"], params, "mandelbulb", 12, cell_h=8, cell_w=8
        )
        assert len(verts) > 0 and len(faces) > 0
        assert len(tint) == len(verts)

    def test_build_sdf_mosaic_relief_scales_z(self):
        """Objects stand up: tallest vertex reaches the requested relief."""
        params = {"a": {"iterations": 2, "scale": 3.0}}
        verts, _, _ = build_sdf_mosaic(
            1, 1, ["a"], params, "menger", 12, cell_h=8, cell_w=8, relief=0.5
        )
        assert verts[:, 2].min() >= -1e-9
        assert verts[:, 2].max() <= 0.5 + 1e-9
        assert verts[:, 2].max() > 0.1  # real height, not flat

    def test_build_sdf_mosaic_variation_rotates_and_scales_cells(self):
        """Per-cell deterministic size/yaw breaks grid symmetry."""
        params = {"a": {"iterations": 2, "scale": 3.0}}
        base, _, _ = build_sdf_mosaic(
            2, 2, ["a", "b"], params, "menger", 12, cell_h=8, cell_w=8,
            seed=0, variation=False,
        )
        varied, _, _ = build_sdf_mosaic(
            2, 2, ["a", "b"], params, "menger", 12, cell_h=8, cell_w=8,
            seed=0, variation=True,
        )
        assert not np.allclose(base, varied)
        # Same seed → identical variation.
        again, _, _ = build_sdf_mosaic(
            2, 2, ["a", "b"], params, "menger", 12, cell_h=8, cell_w=8,
            seed=0, variation=True,
        )
        np.testing.assert_array_equal(varied, again)

    def test_build_sdf_mosaic_slot_tint_maps_per_cell(self):
        """slot_tint drives per-cell colour; missing slots fall back to 0.5."""
        params = {"a": {"iterations": 2, "scale": 3.0}, "b": {"iterations": 2, "scale": 3.0}}
        _, _, tint = build_sdf_mosaic(
            2, 2, ["a", "b"], params, "menger", 12, cell_h=8, cell_w=8,
            slot_tint={"a": 0.1, "b": 0.9},
        )
        assert tint.min() >= 0.0 and tint.max() <= 1.0
        # Both mapped slots contribute distinct tints.
        assert len(np.unique(tint)) >= 2
        _, _, tint_fallback = build_sdf_mosaic(
            2, 2, ["a", "b"], params, "menger", 12, cell_h=8, cell_w=8,
            slot_tint={"a": 0.1},
        )
        # Unmapped slot b falls back to the midpoint, still within range.
        assert tint_fallback.min() >= 0.0 and tint_fallback.max() <= 1.0

    def test_slot_stat_tint_normalizes_across_slots(self, tmp_path: Path):
        """slot_stat_tint maps a per-slot stat to [0, 1]; NaN → 0.5."""
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        slots = spec.slots
        tint = slot_stat_tint(out_dir, slots, "effective_rank")
        assert set(tint) == set(slots)
        assert all(0.0 <= v <= 1.0 for v in tint.values())
        # Deterministic for identical inputs.
        assert slot_stat_tint(out_dir, slots, "effective_rank") == tint

    def test_sdf_render_mode_dry_run(self, tmp_path: Path):
        """Dry-run: mode='sdf' produces PNG + OBJ via the SDF bpy script."""
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        spec.fractal["mode"] = "sdf"
        spec.fractal["sdf"] = {
            "family": "menger",
            "grid": 12,
            "mapping": {
                "iterations": {"stat": "effective_rank", "lo": 1, "hi": 4},
                "scale": {"stat": "kurtosis", "lo": 2.5, "hi": 3.5},
            },
        }
        out = out_dir / "render"
        out.mkdir(parents=True, exist_ok=True)
        field = Field2D(
            channel="height",
            data=np.zeros((3, 2)),
            row_labels=["0", "1", "2"],
            col_labels=["attn_q", "mlp_gate"],
        )
        renderer = FractalRenderer()

        fake_blender = tmp_path / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        with mock.patch("subprocess.run") as mock_run:
            def _write_png(cmd, **kw):
                (out / "terrain_fractal.png").write_bytes(b"PNG-DUMMY")
                return mock.MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = _write_png
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}):
                produced = renderer.render(field, spec, out)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        args = cmd[cmd.index("--") + 1:]
        assert "render_sdf.py" in cmd[cmd.index("-P") + 1]
        assert "--verts" in args and "--faces" in args and "--tint" in args
        names = {p.name for p in produced}
        assert "terrain_fractal.png" in names
        assert "terrain_fractal.obj" in names
        obj = (out / "terrain_fractal.obj").read_text()
        assert obj.startswith("# weight-atlas fractal SDF mosaic OBJ")

    def test_sdf_render_deterministic_obj(self, tmp_path: Path):
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        spec.fractal["mode"] = "sdf"
        spec.fractal["sdf"] = {
            "family": "menger",
            "grid": 10,
            "mapping": {
                "iterations": {"stat": "effective_rank", "lo": 1, "hi": 3},
                "scale": {"stat": "kurtosis", "lo": 2.5, "hi": 3.5},
            },
        }
        field = Field2D(
            channel="height",
            data=np.zeros((3, 2)),
            row_labels=["0", "1", "2"],
            col_labels=["attn_q", "mlp_gate"],
        )
        renderer = FractalRenderer()

        fake_blender = tmp_path / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        objs = []
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="", stderr="")
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}):
                for i in range(2):
                    out = tmp_path / f"render_sdf{i}"
                    out.mkdir(parents=True, exist_ok=True)
                    renderer.render(field, spec, out)
                    objs.append((out / "terrain_fractal.obj").read_bytes())
        assert objs[0] == objs[1]


class TestFractalRenderer:
    def test_registered_with_id_fractal(self):
        renderer_cls = get_renderer("fractal")
        assert renderer_cls.renderer_id == "fractal"

    def test_requires_col_labels(self, tmp_path: Path):
        renderer = FractalRenderer()
        out = tmp_path / "render"
        field = Field2D(channel="height", data=np.zeros((2, 2)), row_labels=["0", "1"], col_labels=[])
        with pytest.raises(ValueError, match="col_labels"):
            renderer.render(field, mock.MagicMock(), out)

    def test_dry_run_subprocess_and_artefacts(self, tmp_path: Path):
        """Dry-run: Blender subprocess mocked; PNG + OBJ must be produced."""
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        out = out_dir / "render"
        out.mkdir(parents=True, exist_ok=True)

        field = Field2D(
            channel="height",
            data=np.zeros((3, 2)),
            row_labels=["0", "1", "2"],
            col_labels=["attn_q", "mlp_gate"],
        )
        renderer = FractalRenderer()

        fake_blender = tmp_path / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        with mock.patch("subprocess.run") as mock_run:
            def _write_png(cmd, **kw):
                (out / "terrain_fractal.png").write_bytes(b"PNG-DUMMY")
                return mock.MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = _write_png
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}):
                produced = renderer.render(field, spec, out)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert str(fake_blender) in cmd[0]
        assert "-b" in cmd and "-P" in cmd
        args = cmd[cmd.index("--") + 1:]
        assert "terrain_fractal.png" in args[args.index("--out") + 1]

        names = {p.name for p in produced}
        assert "terrain_fractal.png" in names
        assert "terrain_fractal.obj" in names
        assert (out / "terrain_fractal.obj").exists()
        obj = (out / "terrain_fractal.obj").read_text()
        assert obj.startswith("# weight-atlas fractal terrain OBJ")

    def test_per_channel_dedupe_renders_once(self, tmp_path: Path):
        """API/CLI call render() once per channel; Blender must run only once."""
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        out = out_dir / "render"
        out.mkdir(parents=True, exist_ok=True)
        field = Field2D(
            channel="height",
            data=np.zeros((3, 2)),
            row_labels=["0", "1", "2"],
            col_labels=["attn_q", "mlp_gate"],
        )
        renderer = FractalRenderer()

        fake_blender = tmp_path / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        with mock.patch("subprocess.run") as mock_run:
            def _write_png(cmd, **kw):
                (out / "terrain_fractal.png").write_bytes(b"PNG-DUMMY")
                return mock.MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = _write_png
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}):
                for _channel in ("height", "tint", "rough"):
                    produced = renderer.render(field, spec, out)
                    assert any(p.name == "terrain_fractal.png" for p in produced)

        mock_run.assert_called_once()

    def test_expert_and_vision_channels_skipped(self, tmp_path: Path):
        """Expert/vision rasters never define the fractal (primary raster only)."""
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        spec.fractal["mode"] = "sdf"
        spec.fractal["sdf"] = {
            "family": "menger",
            "grid": 10,
            "mapping": {
                "iterations": {"stat": "effective_rank", "lo": 1, "hi": 4},
                "scale": {"stat": "kurtosis", "lo": 2.5, "hi": 3.5},
            },
        }
        out = out_dir / "render"
        out.mkdir(parents=True, exist_ok=True)
        renderer = FractalRenderer()

        fake_blender = tmp_path / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        with mock.patch("subprocess.run") as mock_run:
            def _write_png(cmd, **kw):
                (out / "terrain_fractal.png").write_bytes(b"PNG-DUMMY")
                return mock.MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = _write_png
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}):
                # Primary raster renders first (sets the cache).
                primary = Field2D(
                    channel="height",
                    data=np.zeros((3, 2)),
                    row_labels=["0", "1", "2"],
                    col_labels=["attn_q", "mlp_gate"],
                )
                produced = renderer.render(primary, spec, out)
                assert any(p.name == "terrain_fractal.png" for p in produced)

                # Expert panel (one column per expert) must be skipped.
                expert = Field2D(
                    channel="expert_mlp_down_height",
                    data=np.zeros((3, 896)),
                    row_labels=["0", "1", "2"],
                    col_labels=[str(i) for i in range(896)],
                )
                assert renderer.render(expert, spec, out) == []

                # Vision panel must be skipped too.
                vision = Field2D(
                    channel="vision_height",
                    data=np.zeros((3, 18)),
                    row_labels=["0", "1", "2"],
                    col_labels=[f"v{i}" for i in range(18)],
                )
                assert renderer.render(vision, spec, out) == []

        mock_run.assert_called_once()

    def test_mosaic_decimates_large_rasters(self):
        """Rasters exceeding max_cells are decimated deterministically."""
        params = {"a": {"iterations": 2, "scale": 3.0}}
        verts, faces, tint = build_sdf_mosaic(
            92, 896, ["a"] * 896, params, "menger", 10, cell_h=8, cell_w=8
        )
        # 82,432 cells would be ~115M verts uncapped; the cap must bound it.
        assert len(verts) < 2_000_000
        assert len(tint) == len(verts)

        # Deterministic: same input → same mesh.
        v2, f2, t2 = build_sdf_mosaic(
            92, 896, ["a"] * 896, params, "menger", 10, cell_h=8, cell_w=8
        )
        for a, b in zip((verts, faces, tint), (v2, f2, t2), strict=True):
            np.testing.assert_array_equal(a, b)

    def test_mosaic_small_raster_uncapped(self):
        """Small rasters under max_cells keep one object per cell."""
        params = {"a": {"iterations": 2, "scale": 3.0}}
        one, _f, _t = build_sdf_mosaic(1, 1, ["a"], params, "menger", 10, cell_h=8, cell_w=8)
        assert len(one) > 0

    def test_dedupe_keyed_on_layout(self, tmp_path: Path):
        """Channels with a different raster layout must not reuse the cache."""
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        out = out_dir / "render"
        out.mkdir(parents=True, exist_ok=True)
        renderer = FractalRenderer()

        fake_blender = tmp_path / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="", stderr="")
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}):
                small = Field2D(
                    channel="height",
                    data=np.zeros((3, 2)),
                    row_labels=["0", "1", "2"],
                    col_labels=["attn_q", "mlp_gate"],
                )
                renderer.render(small, spec, out)

                # Different layout (different slot labels) → new render.
                other = Field2D(
                    channel="height",
                    data=np.zeros((3, 3)),
                    row_labels=["0", "1", "2"],
                    col_labels=["attn_q", "mlp_gate", "mlp_up"],
                )
                renderer.render(other, spec, out)
        assert mock_run.call_count == 2

    def test_dry_run_traceback_guard(self, tmp_path: Path):
        """Blender exits 0 with a script traceback → render must fail."""
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        out = out_dir / "render"
        out.mkdir(parents=True, exist_ok=True)
        field = Field2D(
            channel="height",
            data=np.zeros((3, 2)),
            row_labels=["0", "1", "2"],
            col_labels=["attn_q", "mlp_gate"],
        )
        renderer = FractalRenderer()

        fake_blender = tmp_path / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        stderr = "Traceback (most recent call last):\nAttributeError: boom\n"
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="", stderr=stderr)
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}), pytest.raises(RuntimeError, match="Blender script crashed"):
                renderer.render(field, spec, out)

    def test_render_is_deterministic(self, tmp_path: Path):
        """Same inputs → byte-identical OBJ artefacts."""
        out_dir = tmp_path / "scan"
        out_dir.mkdir()
        spec = _make_spec(tmp_path, out_dir)
        field = Field2D(
            channel="height",
            data=np.zeros((3, 2)),
            row_labels=["0", "1", "2"],
            col_labels=["attn_q", "mlp_gate"],
        )
        renderer = FractalRenderer()

        fake_blender = tmp_path / "blender"
        fake_blender.write_text("#!/bin/sh\necho ok\n")
        fake_blender.chmod(0o755)

        objs = []
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="", stderr="")
            with mock.patch.dict(os.environ, {"WEIGHT_ATLAS_BLENDER": str(fake_blender)}):
                for i in range(2):
                    out = tmp_path / f"render{i}"
                    out.mkdir(parents=True, exist_ok=True)
                    renderer.render(field, spec, out)
                    objs.append((out / "terrain_fractal.obj").read_bytes())
        assert objs[0] == objs[1]
