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
from weight_atlas.render.fractal.params import (
    slot_fractal_params,
    slot_stat_medians,
    stats_to_params,
)
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
