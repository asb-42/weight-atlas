"""M9 paired-analysis smoke test (Bonsai-8B style fixture).

Covers the ``quant`` preset (analytic SQNR band, injected-outlier hotspot,
cross-run determinism, strict-only hard-reject, noise-floor veil boolean mask)
and the ``edit`` preset (abliteration surrogate → low_rank_localized + band +
u1-coherence, quantization → full_rank_uniform, weight-space hotspot ranking).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import make_fake_model

# HF slot -> GGUF tensor name for the same logical (layer, slot).
_GGUF_NAME = {
    "model.embed_tokens.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
}


def _gguf_name(hf_name: str, n_layers: int) -> str:
    """Map an HF tensor name to the equivalent GGUF name."""
    for slot, gguf in _GGUF_NAME.items():
        if hf_name == slot:
            return gguf
        if slot.startswith(("self_attn.", "mlp.", "input_", "post_")):
            prefix = f"model.layers.{{}}.{slot}.weight"
            for layer in range(n_layers):
                if hf_name == prefix.format(layer):
                    return f"blk.{layer}.{gguf}.weight"
    raise KeyError(f"no GGUF mapping for {hf_name}")


def _write_q40_gguf(
    path: Path,
    tensors: dict[str, np.ndarray],
    n_layers: int,
    outlier: tuple[int, str, float] | None = None,
) -> None:
    """Write a real Q4_0 GGUF via gguf.quants (canonical block layout).

    ``outlier`` = (layer, "mlp.down_proj", scale) — scales that tensor by
    ``scale`` in the copy before quantizing (hotspot injection).
    """
    from gguf import GGMLQuantizationType, GGUFWriter
    from gguf.quants import quantize

    writer = GGUFWriter(str(path), arch="llama")
    writer.add_architecture()
    writer.add_block_count(n_layers)
    for hf_name, data in tensors.items():
        gguf_name = _gguf_name(hf_name, n_layers)
        payload = data
        if outlier is not None:
            layer, slot, scale = outlier
            if hf_name == f"model.layers.{layer}.{slot}.weight":
                payload = data * scale
        q = quantize(payload, GGMLQuantizationType.Q4_0)
        writer.add_tensor(
            gguf_name, q, raw_shape=list(q.shape), raw_dtype=GGMLQuantizationType.Q4_0
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _q40_analytic_band(a: np.ndarray) -> float:
    """Analytic SQNR band for a tensor quantized to Q4_0.

    Per-block uniform-noise model: each 32-element block is scaled by
    ``d = max_abs / 8`` (Q4_0), so noise variance per element is ``d²/12``.
    Band = 10·log10(signal / Σ 32·d²/12).
    """
    flat = np.asarray(a, dtype=np.float64).reshape(-1)
    nb = flat.size // 32
    blocks = flat[: nb * 32].reshape(nb, 32)
    step = np.abs(blocks).max(axis=1) / 8.0
    signal = float((blocks**2).sum())
    noise = float((32.0 * step**2 / 12.0).sum())
    return 10.0 * np.log10(signal / noise) if signal > 0 and noise > 0 else np.nan


def _run(tmp_path: Path, tensors: dict[str, np.ndarray], n_layers: int, **kwargs):
    """Build ref safetensors + Q4_0 GGUF, run paired (quant preset), return summary + out dir."""
    from weight_atlas.core.types import load_default_spec
    from weight_atlas.paired import run_paired

    ref_path = tmp_path / "ref.safetensors"
    make_fake_model(ref_path, n_layers=n_layers, hidden=32, seed=42)

    out = kwargs.pop("out", tmp_path / "out")
    run_paired(
        ref_path,
        kwargs.pop("model_b"),
        out,
        load_default_spec(),
        preset="quant",
        **kwargs,
    )
    return json.loads((out / "impact_summary.json").read_text()), out


def _make_q40(tmp_path: Path, tensors: dict[str, np.ndarray], n_layers: int, **kwargs) -> Path:
    path = tmp_path / "q40.gguf"
    _write_q40_gguf(path, tensors, n_layers, **kwargs)
    return path


class TestSmokeQ40:
    @pytest.fixture(autouse=True)
    def _register(self):
        # conftest registers loaders/renderers; ensure imports happened.
        import weight_atlas.loaders.gguf_loader  # noqa: F401
        import weight_atlas.loaders.safetensors_loader  # noqa: F401
        import weight_atlas.render.matplotlib_sheet  # noqa: F401
        yield

    def test_per_tensor_sqnr_in_analytic_band(self, tmp_path):
        """Q4_0 copy: every dense tensor's sqnr_db lands in the analytic band."""
        n_layers = 2
        tensors = make_fake_model(tmp_path / "ref.safetensors", n_layers=n_layers, hidden=32, seed=42)
        q40 = _make_q40(tmp_path, tensors, n_layers)

        summary, out = _run(
            tmp_path, tensors, n_layers, model_b=q40, jobs=1, ref_side="a"
        )
        assert summary["alignment"]["n_pairs"] == len(tensors)
        assert summary["alignment"]["n_skipped"] == 0

        per_type = summary["per_type"]
        assert "ggml_2" in per_type  # Q4_0 = type 2
        assert per_type["ggml_2"]["n"] == len(tensors)

        # Per-tensor sqnr_db within ±3 dB of the analytic uniform-noise band.
        for hf_name, arr in tensors.items():
            gguf_name = _gguf_name(hf_name, n_layers)
            band = _q40_analytic_band(arr)
            measured = _measured_sqnr(tmp_path, out, hf_name, gguf_name)
            assert measured == pytest.approx(band, abs=3.0), (
                f"{hf_name}: measured {measured:.2f} dB vs band {band:.2f} dB"
            )

    def test_outlier_ranks_top1(self, tmp_path):
        """A 10x outlier on one mlp_down ranks top-1 lowest SQNR."""
        n_layers = 2
        tensors = make_fake_model(tmp_path / "ref.safetensors", n_layers=n_layers, hidden=32, seed=42)
        q40 = _make_q40(tmp_path, tensors, n_layers, outlier=(1, "mlp.down_proj", 10.0))

        summary, _ = _run(tmp_path, tensors, n_layers, model_b=q40, jobs=1)
        top = summary["hotspot_ranking"][0]
        assert top["slot"] == "mlp_down"
        assert top["layer"] == 1
        assert top["name_b"].endswith("ffn_down.weight")
        assert top["sqnr_db"] < -10.0  # ≈10·log10(1/81) ≈ -19 dB

    def test_second_run_byte_identical(self, tmp_path):
        """Two runs produce identical manifest.json (SHA-256 of artefacts)."""
        n_layers = 2
        tensors = make_fake_model(tmp_path / "ref.safetensors", n_layers=n_layers, hidden=32, seed=42)
        q40 = _make_q40(tmp_path, tensors, n_layers)

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        _run(tmp_path, tensors, n_layers, model_b=q40, jobs=1, out=out1)
        _run(tmp_path, tensors, n_layers, model_b=q40, jobs=2, out=out2)

        m1 = json.loads((out1 / "manifest.json").read_text())
        m2 = json.loads((out2 / "manifest.json").read_text())
        assert set(m1) == set(m2)
        for name in m1:
            assert m1[name] == m2[name], f"{name} differs across jobs"

    def test_mode_aligned_hard_reject(self, tmp_path):
        """Paired analysis is strict-only: aligned mode raises ValueError."""
        from weight_atlas.core.types import load_default_spec
        from weight_atlas.paired import run_paired

        n_layers = 1
        ref = tmp_path / "ref.safetensors"
        make_fake_model(ref, n_layers=n_layers, hidden=32, seed=42)
        q40 = _make_q40(tmp_path, make_fake_model(tmp_path / "x.safetensors", n_layers=n_layers, hidden=32, seed=42), n_layers)
        with pytest.raises(ValueError, match="strict-only"):
            run_paired(ref, q40, tmp_path / "o", load_default_spec(), mode="aligned")

    def test_noise_floor_mask_is_boolean(self, tmp_path):
        """Veil mask is a per-cell boolean, never SSIM-derived."""
        from weight_atlas.compare.render.delta_sheet import _noise_floor_mask

        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        from weight_atlas.fields.tif_io import write_tif

        calib = np.array([[1.0, np.nan], [3.0, 0.5]])
        write_tif(calib_dir / "field_delta_height_raw.tif", calib)

        delta = np.array([[0.5, 2.0], [5.0, 0.1]])
        mask = _noise_floor_mask(delta, calib_dir, "height")
        assert mask.dtype == np.bool_
        assert mask.shape == delta.shape
        # |0.5| <= |1.0| -> veiled; NaN calib -> not veiled; 5 > 3 -> not veiled
        assert mask[0, 0]
        assert not mask[0, 1]
        assert not mask[1, 0]
        assert mask[1, 1]

        # Missing calibration file -> all-False, still boolean.
        empty = _noise_floor_mask(delta, tmp_path / "nope", "height")
        assert empty.dtype == np.bool_
        assert not empty.any()


class TestCli:
    def test_cli_paired(self, tmp_path):
        """CLI: paired subcommand (quant preset) writes the full artefact set."""
        from weight_atlas.cli import main

        n_layers = 1
        ref = tmp_path / "ref.safetensors"
        make_fake_model(ref, n_layers=n_layers, hidden=32, seed=42)
        q40 = tmp_path / "q40.gguf"
        _write_q40_gguf(q40, make_fake_model(tmp_path / "x.safetensors", n_layers=n_layers, hidden=32, seed=42), n_layers)
        out = tmp_path / "out"

        rc = main(
            ["paired", str(tmp_path / "sa"), str(tmp_path / "sb"), "--weights-a", str(ref), "--weights-b", str(q40), "--out", str(out), "--jobs", "1", "--preset", "quant"]
        )
        assert rc == 0
        assert (out / "impact_summary.json").exists()
        assert (out / "manifest.json").exists()
        assert (out / "impact_sqnr_db.png").exists()
        assert (out / "field_impact_sqnr_db_raw.tif").exists()

    def test_cli_qimpact_alias(self, tmp_path):
        """CLI: qimpact remains an alias for the paired quant preset."""
        from weight_atlas.cli import main

        n_layers = 1
        ref = tmp_path / "ref.safetensors"
        make_fake_model(ref, n_layers=n_layers, hidden=32, seed=42)
        q40 = tmp_path / "q40.gguf"
        _write_q40_gguf(q40, make_fake_model(tmp_path / "x.safetensors", n_layers=n_layers, hidden=32, seed=42), n_layers)

        rc = main(
            ["qimpact", str(tmp_path / "sa"), str(tmp_path / "sb"), "--weights-a", str(ref), "--weights-b", str(q40), "--out", str(tmp_path / "o"), "--jobs", "1"]
        )
        assert rc == 0
        assert (tmp_path / "o" / "impact_summary.json").exists()

    def test_cli_paired_mode_aligned_rejected(self, tmp_path):
        """CLI: --mode aligned is hard-rejected."""
        from weight_atlas.cli import main

        n_layers = 1
        ref = tmp_path / "ref.safetensors"
        make_fake_model(ref, n_layers=n_layers, hidden=32, seed=42)
        q40 = tmp_path / "q40.gguf"
        _write_q40_gguf(q40, make_fake_model(tmp_path / "x.safetensors", n_layers=n_layers, hidden=32, seed=42), n_layers)

        rc = main(
            ["paired", str(tmp_path / "sa"), str(tmp_path / "sb"), "--weights-a", str(ref), "--weights-b", str(q40), "--out", str(tmp_path / "o"), "--mode", "aligned"]
        )
        assert rc != 0

    def test_compare_noise_floor_emits_delta_tifs(self, tmp_path):
        """compare --noise-floor writes field_delta_* artefacts (veil source)."""
        from tests.fixtures import make_fake_model
        from weight_atlas.cli import main

        a = tmp_path / "a.safetensors"
        b = tmp_path / "b.safetensors"
        make_fake_model(a, n_layers=1, hidden=32, seed=42)
        make_fake_model(b, n_layers=1, hidden=32, seed=43)
        sa, sb = tmp_path / "scan_a", tmp_path / "scan_b"
        assert main(["scan", str(a), "--out", str(sa)]) == 0
        assert main(["scan", str(b), "--out", str(sb)]) == 0

        calib = tmp_path / "calib"
        assert main(["compare", str(sa), str(sb), "--out", str(calib)]) == 0
        assert (calib / "field_delta_height_raw.tif").exists()
        assert (calib / "field_delta_height_smooth.tif").exists()

        out = tmp_path / "out"
        rc = main(
            ["compare", str(sa), str(sb), "--out", str(out), "--mode", "strict", "--noise-floor", str(calib)]
        )
        assert rc == 0
        assert (out / "field_delta_height_raw.tif").exists()


def _measured_sqnr(tmp_path: Path, out: Path, hf_name: str, gguf_name: str) -> float:
    """Measured sqnr_db for one tensor pair, recomputed directly from weights."""
    from weight_atlas.loaders.gguf_loader import GGUFLoader
    from weight_atlas.loaders.safetensors_loader import SafetensorsLoader

    ref_path = tmp_path / "ref.safetensors"
    q40_path = tmp_path / "q40.gguf"
    a = {h.name: h.load() for h in SafetensorsLoader().open(ref_path)}
    b = {h.name: h.load() for h in GGUFLoader().open(q40_path)}
    av = a[hf_name].reshape(-1)
    bv = b[gguf_name].reshape(-1)
    d = bv - av
    return float(10.0 * np.log10((av**2).sum() / (d**2).sum())) if (d**2).sum() > 0 else np.inf


# ---------------------------------------------------------------------------
# Edit preset (edit signatures / abliteration)
# ---------------------------------------------------------------------------


def _edit_spec(*, u1_coherence: bool = False) -> object:
    """Load the default spec and enable edit-preset knobs for the test."""
    from weight_atlas.core.types import load_default_spec

    spec = load_default_spec()
    spec.edit["u1_coherence"] = u1_coherence
    return spec


def _write_abliteration_surrogate(
    path: Path,
    base_tensors: dict[str, np.ndarray],
    n_layers: int,
    start_layer: int,
    end_layer: int,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Write B = A + Δ with the abliteration surrogate Δ = d̂(Wd̂)ᵀ.

    A rank-1 edit sharing one direction ``d̂`` applied to ``mlp.down_proj``
    across layers ``[start_layer, end_layer]``. Everything else is unchanged
    (rel_l2 = 0). Returns the written tensors.
    """
    rng = np.random.default_rng(seed)
    tensors = {name: arr.copy() for name, arr in base_tensors.items()}
    # Shared abliteration direction in the mlp_down output space.
    dhat = rng.normal(0, 1, (32,)).astype(np.float64)
    dhat = dhat / np.linalg.norm(dhat)
    for layer in range(start_layer, end_layer + 1):
        name = f"model.layers.{layer}.mlp.down_proj.weight"
        w = base_tensors[name].astype(np.float64)
        delta = np.outer(dhat, w @ dhat)
        tensors[name] = (w + delta).astype(np.float32)
    from safetensors.numpy import save_file

    save_file(tensors, str(path))
    return tensors


def _write_spike_model(path: Path, base_tensors: dict[str, np.ndarray]) -> None:
    """Write B with weight-space spikes mirroring the M4 localization fixture.

    - layer 2 ``mlp.down_proj`` scaled by 100 (dominant rel_l2 spike)
    - layer 3 ``self_attn.o_proj`` receives a rank-1 +2.0 perturbation
    """
    tensors = {name: arr.copy() for name, arr in base_tensors.items()}
    rng = np.random.default_rng(11)
    tensors["model.layers.2.mlp.down_proj.weight"] = (
        base_tensors["model.layers.2.mlp.down_proj.weight"] * 100.0
    ).astype(np.float32)
    u = rng.normal(0, 1, (32,)).astype(np.float32)
    v = rng.normal(0, 1, (32,)).astype(np.float32)
    tensors["model.layers.3.self_attn.o_proj.weight"] = (
        base_tensors["model.layers.3.self_attn.o_proj.weight"]
        + 2.0 * np.outer(u, v)
    ).astype(np.float32)
    from safetensors.numpy import save_file

    save_file(tensors, str(path))


class TestEditPreset:
    @pytest.fixture(autouse=True)
    def _register(self):
        import weight_atlas.loaders.safetensors_loader  # noqa: F401
        import weight_atlas.render.matplotlib_sheet  # noqa: F401
        yield

    @pytest.fixture
    def base_tensors(self, tmp_path):
        return make_fake_model(tmp_path / "base.safetensors", n_layers=32, hidden=32, seed=42)

    def _run_edit(self, tmp_path, base_tensors, b_path, *, u1=False, jobs=1, n_layers=32):
        from weight_atlas.paired import run_paired

        a = tmp_path / "base.safetensors"
        out = tmp_path / "edit_out"
        run_paired(
            a,
            b_path,
            out,
            _edit_spec(u1_coherence=u1),
            fp_a=None,
            fp_b=None,
            preset="edit",
            jobs=jobs,
        )
        return json.loads((out / "compare_summary.json").read_text()), out

    def test_abliteration_surrogate_localized(self, tmp_path, base_tensors):
        """Δ = d̂(Wd̂)ᵀ on mlp_down layers 14–28 → low_rank_localized, band, coherence."""
        b = tmp_path / "abliterated.safetensors"
        _write_abliteration_surrogate(b, base_tensors, n_layers=32, start_layer=14, end_layer=28)

        summary, out = self._run_edit(tmp_path, base_tensors, b, u1=True, n_layers=32)

        sig = summary["edit_signature"]
        assert sig["classification"] == "low_rank_localized"
        # Rank-1 edit: delta_stable_rank ≈ 1 within a small tolerance.
        rank = sig["stats"]["median_delta_stable_rank"]
        assert rank == pytest.approx(1.0, abs=0.2)
        # Band covers layers 14..28 with mlp_down concentrated.
        assert sig["bands"], "expected at least one edit band"
        assert sig["bands"][0]["start_layer"] == 14
        assert sig["bands"][0]["end_layer"] == 28
        assert "mlp_down" in sig["bands"][0]["slots"]
        # Shared d̂ direction ⇒ pairwise u1 coherence near 1.
        assert sig["stats"]["u1_coherence"] > 0.9

        # Artefacts present.
        assert (out / "edit_rel_l2.png").exists()
        assert (out / "edit_rank.png").exists()
        assert (out / "edit_profile.png").exists()
        assert (out / "field_edit_rel_l2_raw.tif").exists()
        assert (out / "field_edit_delta_stable_rank_raw.tif").exists()
        assert (out / "field_edit_spectral_share_raw.tif").exists()
        assert (out / "field_edit_dspec_raw.tif").exists()

    def test_edit_second_run_byte_identical(self, tmp_path, base_tensors):
        """Edit preset: two runs (jobs=1 vs jobs=2) → identical manifest."""
        b = tmp_path / "abliterated.safetensors"
        _write_abliteration_surrogate(b, base_tensors, n_layers=32, start_layer=14, end_layer=28)

        out1 = tmp_path / "edit_out1"
        out2 = tmp_path / "edit_out2"
        from weight_atlas.core.types import load_default_spec
        from weight_atlas.paired import run_paired

        spec = load_default_spec()
        spec.edit["u1_coherence"] = True
        for o, j in ((out1, 1), (out2, 2)):
            run_paired(tmp_path / "base.safetensors", b, o, spec, preset="edit", jobs=j)

        m1 = json.loads((out1 / "manifest.json").read_text())
        m2 = json.loads((out2 / "manifest.json").read_text())
        assert set(m1) == set(m2)
        for name in m1:
            assert m1[name] == m2[name], f"{name} differs across jobs"

    def test_spike_model_hotspot_ranking(self, tmp_path, base_tensors):
        """Weight-space spikes → hotspot_ranking_rel_l2 top-2 = (2, mlp_down), (3, attn_o)."""
        b = tmp_path / "spike.safetensors"
        _write_spike_model(b, base_tensors)

        summary, _ = self._run_edit(tmp_path, base_tensors, b)
        top = summary["edit_signature"]["hotspot_ranking_rel_l2"][:2]
        assert (top[0]["layer"], top[0]["slot"]) == (2, "mlp_down")
        assert (top[1]["layer"], top[1]["slot"]) == (3, "attn_o")

    def test_q40_fixture_full_rank_uniform(self, tmp_path):
        """Q4_0 quantization → full_rank_uniform (quantization-like, no bands)."""
        n_layers = 4
        tensors = make_fake_model(tmp_path / "ref.safetensors", n_layers=n_layers, hidden=32, seed=42)
        q40 = _make_q40(tmp_path, tensors, n_layers)

        from weight_atlas.core.types import load_default_spec
        from weight_atlas.paired import run_paired

        out = tmp_path / "edit_q40"
        run_paired(tmp_path / "ref.safetensors", q40, out, load_default_spec(), preset="edit", jobs=1)
        summary = json.loads((out / "compare_summary.json").read_text())

        sig = summary["edit_signature"]
        assert sig["classification"] == "full_rank_uniform"
        assert sig["bands"] == []
        # Noise-floor policy: loader differs (safetensors vs gguf) → mismatched.
        assert summary["noise_floor"]["policy"] == "mismatched"
