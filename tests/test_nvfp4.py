"""NVFP4 decoder tests: E4M3 table, round-trip vs reference layout, loader E2E.

The decoder mirrors compressed-tensors v0.18.0's nvfp4-pack-quantized
format (verified against source, ``compressors/nvfp4/base.py`` +
``helpers.py``): FP4 E2M1 weights (nibble-packed, low nibble = even
column) x per-group-16 E4M3 scales (FULL bytes, dtype F8_E4M3, tensor
name ``weight_scale``) x per-tensor fp32 ``weight_global_scale``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from weight_atlas.loaders import mxfp4, nvfp4
from weight_atlas.loaders.nvfp4 import e4m3_to_float
from weight_atlas.loaders.safetensors_loader import SafetensorsLoader

_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _pack_fp4(values: np.ndarray) -> np.ndarray:
    """Pack (m, k) E2M1-grid values two-per-uint8, low nibble = even column.

    Mirrors compressed-tensors ``pack_fp4_to_uint8``: index 0..7 magnitude
    (0, 0.5, 1, 1.5, 2, 3, 4, 6), sign in bit 3; ``packed = idx0 | (idx1 << 4)``.
    """
    sign = (values < 0).astype(np.uint8) << 3
    idx = np.zeros(values.shape, dtype=np.uint8)
    mag = np.abs(values)
    for i, g in enumerate(_E2M1):
        idx[mag == g] = i
    nib = idx | sign
    m, k = nib.shape
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    assert packed.shape == (m, k // 2)
    return packed


def _e4m3_bytes(values: np.ndarray) -> np.ndarray:
    """Exact E4M3 bit patterns for values that are representable E4M3
    (test uses powers of two and 448-magnitude values)."""
    import ml_dtypes

    return values.astype(np.float64).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)


class TestE4M3Decode:
    def test_matches_ml_dtypes_all_256_bytes(self) -> None:
        import ml_dtypes

        all_bytes = np.arange(256, dtype=np.uint8)
        ref = all_bytes.view(ml_dtypes.float8_e4m3fn).astype(np.float64)
        mine = e4m3_to_float(all_bytes)
        both_nan = np.isnan(ref) & np.isnan(mine)
        assert (both_nan | np.isclose(ref, mine, rtol=0, atol=0)).all()

    def test_spot_values(self) -> None:
        cases = {0x40: 2.0, 0x50: 8.0, 0x38: 1.0, 0x7E: 448.0, 0x01: 2.0**-9, 0x00: 0.0}
        for b, expect in cases.items():
            assert e4m3_to_float(np.array([b], dtype=np.uint8))[0] == pytest.approx(expect)


class TestRoundTrip:
    def test_pack_dequant_roundtrip(self) -> None:
        """E2M1-grid values x exact-E4M3 scales x global scale survive
        pack->dequant EXACTLY (no rounding left anywhere)."""
        rng = np.random.default_rng(11)
        m, k = 8, 64
        grid = _E2M1[rng.integers(0, 8, size=(m, k))] * np.where(rng.random((m, k)) < 0.5, -1, 1)
        group_scales = 2.0 ** rng.integers(-3, 3, size=(m, k // nvfp4.NVFP4_GROUP_SIZE)).astype(np.float32)
        gs = 0.05
        dense = grid * np.repeat(group_scales, nvfp4.NVFP4_GROUP_SIZE, axis=1) * gs

        packed = _pack_fp4(grid)
        scale_bytes = _e4m3_bytes(group_scales)
        out = nvfp4.dequantize_nvfp4(packed, scale_bytes, np.array([gs], dtype=np.float32))
        np.testing.assert_array_equal(out, dense.astype(np.float32))

    def test_global_scale_applied(self) -> None:
        rng = np.random.default_rng(3)
        grid = _E2M1[rng.integers(0, 8, size=(2, 32))] * np.where(rng.random((2, 32)) < 0.5, -1, 1)
        packed = _pack_fp4(grid)
        sb = _e4m3_bytes(np.ones((2, 2), dtype=np.float32))
        a = nvfp4.dequantize_nvfp4(packed, sb, 1.0)
        b = nvfp4.dequantize_nvfp4(packed, sb, 2.0)
        np.testing.assert_array_equal(b, a * 2.0)

    def test_shape_mismatch_raises(self) -> None:
        packed = np.zeros((4, 16), dtype=np.uint8)  # k=32 -> groups=2
        sb = np.zeros((4, 3), dtype=np.uint8)  # WRONG group count
        with pytest.raises(ValueError, match="scale shape"):
            nvfp4.dequantize_nvfp4(packed, sb, 1.0)

    def test_odd_input_dim_raises(self) -> None:
        packed = np.zeros((2, 3), dtype=np.uint8)  # k=6, not a multiple of 16
        sb = np.zeros((2, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="not a multiple"):
            nvfp4.dequantize_nvfp4(packed, sb, 1.0)


class TestLoaderE2E:
    def _make_nvfp4_file(self, tmp_path: Path, m: int, k: int, seed: int) -> tuple[Path, np.ndarray]:
        rng = np.random.default_rng(seed)
        grid = _E2M1[rng.integers(0, 8, size=(m, k))] * np.where(rng.random((m, k)) < 0.5, -1, 1)
        group_scales = 2.0 ** rng.integers(-2, 2, size=(m, k // nvfp4.NVFP4_GROUP_SIZE)).astype(np.float32)
        gs = 0.125
        dense = (grid * np.repeat(group_scales, nvfp4.NVFP4_GROUP_SIZE, axis=1) * gs).astype(np.float32)
        tensors = {
            "blk.0.attn_q.weight_packed": _pack_fp4(grid),
            "blk.0.attn_q.weight_scale": _e4m3_bytes(group_scales),
            "blk.0.attn_q.weight_global_scale": np.array([gs], dtype=np.float32),
            "blk.0.norm.weight": np.ones(8, dtype=np.float32),
        }
        path = tmp_path / "nvfp4_model.safetensors"
        save_file(tensors, str(path))
        return path, dense

    def test_nvfp4_safetensors_roundtrip(self, tmp_path: Path) -> None:
        path, dense = self._make_nvfp4_file(tmp_path, m=6, k=64, seed=21)
        loader = SafetensorsLoader()
        handles = loader.open(path)
        by_name = {h.name: h for h in handles}
        assert "blk.0.attn_q.weight" in by_name  # merged dense name
        for gone in ("blk.0.attn_q.weight_packed", "blk.0.attn_q.weight_scale",
                     "blk.0.attn_q.weight_global_scale"):
            assert gone not in by_name
        assert "blk.0.norm.weight" in by_name  # untouched

        h = by_name["blk.0.attn_q.weight"]
        assert h.dtype == "FP4_NVFP4"
        assert h.shape == (6, 64)
        np.testing.assert_array_equal(h.load(), dense)

    def test_scan_pipeline_end_to_end(self, tmp_path: Path) -> None:
        """The merged handle flows through the full stats pipeline."""
        from weight_atlas.core.types import load_default_spec
        from weight_atlas.scan import scan

        path, _ = self._make_nvfp4_file(tmp_path, m=16, k=128, seed=5)
        out = tmp_path / "scan"
        scan(path, out, load_default_spec(), jobs=1)
        import json

        fp = json.loads((out / "fingerprint.json").read_text())
        assert "blk.0.attn_q.weight" in fp["tensors"]
        assert fp["tensors"]["blk.0.attn_q.weight"]["dtype"] == "FP4_NVFP4"
        # provenance anchored
        assert fp["model"]["sources"][0]["file"].endswith(".safetensors")

    def test_hf_naming_merges_triple(self, tmp_path: Path) -> None:
        """Unsloth/HF naming (verified against the real Flash-Next NVFP4
        plefp8 export): ``<base>.weight`` U8 packed + ``<base>.weight_scale``
        (F8_E4M3 full bytes) + ``<base>.weight_scale_2`` (F32 scalar)."""
        import ml_dtypes

        rng = np.random.default_rng(41)
        m, k = 6, 64
        grid = _E2M1[rng.integers(0, 8, size=(m, k))] * np.where(rng.random((m, k)) < 0.5, -1, 1)
        group_scales = 2.0 ** rng.integers(-2, 2, size=(m, k // nvfp4.NVFP4_GROUP_SIZE)).astype(np.float32)
        gs = 0.125
        dense = (grid * np.repeat(group_scales, nvfp4.NVFP4_GROUP_SIZE, axis=1) * gs).astype(np.float32)

        tensors = {
            "blk.0.attn_q.weight": _pack_fp4(grid),
            "blk.0.attn_q.weight_scale": _e4m3_bytes(group_scales).view(ml_dtypes.float8_e4m3fn),
            "blk.0.attn_q.weight_scale_2": np.array([gs], dtype=np.float32),
            "blk.0.norm.weight": np.ones(8, dtype=np.float32),
        }
        path = tmp_path / "nvfp4_hf.safetensors"
        save_file(tensors, str(path))

        loader = SafetensorsLoader()
        handles = loader.open(path)
        by_name = {h.name: h for h in handles}
        assert "blk.0.attn_q.weight" in by_name
        for gone in ("blk.0.attn_q.weight_scale", "blk.0.attn_q.weight_scale_2"):
            assert gone not in by_name, gone
        assert "blk.0.norm.weight" in by_name

        h = by_name["blk.0.attn_q.weight"]
        assert h.dtype == "FP4_NVFP4"
        assert h.shape == (m, k)
        np.testing.assert_array_equal(h.load(), dense)

    def test_mxfp4_still_works_alongside(self, tmp_path: Path) -> None:
        """The format discriminator must not break the MXFP4 pair path."""
        rng = np.random.default_rng(31)
        m, k = 4, 64
        grid = _E2M1[rng.integers(0, 8, size=(m, k))] * np.where(rng.random((m, k)) < 0.5, -1, 1)
        scales = 2.0 ** rng.integers(-2, 2, size=(m, k // mxfp4.DEFAULT_GROUP_SIZE))
        dense = (grid * np.repeat(scales, mxfp4.DEFAULT_GROUP_SIZE, axis=1)).astype(np.float32)

        packed = _pack_fp4(grid)
        # MXFP4 scales are E8M0 bytes: 2**(b-127)
        scale_bytes = (np.log2(scales) + 127).astype(np.uint8)
        tensors = {
            "blk.0.mlp.weight_packed": packed,
            "blk.0.mlp.weight_scale": scale_bytes,
        }
        path = tmp_path / "mxfp4_model.safetensors"
        save_file(tensors, str(path))

        loader = SafetensorsLoader()
        handles = loader.open(path)
        by_name = {h.name: h for h in handles}
        assert "blk.0.mlp.weight" in by_name
        assert by_name["blk.0.mlp.weight"].dtype == "FP4_MXFP4"
        np.testing.assert_array_equal(by_name["blk.0.mlp.weight"].load(), dense)
