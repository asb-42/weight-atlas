"""Tests for GGUF loader (M5)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from weight_atlas.core.types import detect_loader
from weight_atlas.loaders.gguf_dequant import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_K,
    check_supported,
    dequantize,
    get_type_name,
)

# ---------------------------------------------------------------------------
# Dequant hand-computed values
# ---------------------------------------------------------------------------


class TestDequantF32:
    def test_f32_known_values(self):
        """F32: direct reinterpretation of bytes."""
        data = struct.pack('<4f', 1.0, 2.0, 3.0, 4.0)
        result = dequantize(data, GGML_TYPE_F32)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])


class TestDequantF16:
    def test_f16_known_values(self):
        """F16: convert half-precision to float32."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).tobytes()
        result = dequantize(data, GGML_TYPE_F16)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0], decimal=3)


class TestDequantBF16:
    def test_bf16_known_values(self):
        """BF16: bit-shift to float32."""
        # BF16 for 1.0: 0x3F80 (upper 16 bits of float32 0x3F800000)
        data = struct.pack('<HH', 0x3F80, 0x4000)  # 1.0, 2.0 in BF16
        result = dequantize(data, GGML_TYPE_BF16)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0], decimal=3)


class TestDequantQ80:
    def test_q8_0_known_values(self):
        """Q8_0: block-wise dequantization with f16 scale."""
        # One block: scale=2.0 (f16), then 32 int8 values
        scale = np.float16(2.0).tobytes()
        quants = np.array([1, 2, 3, 4] + [0] * 28, dtype=np.int8).tobytes()
        data = scale + quants
        result = dequantize(data, GGML_TYPE_Q8_0)
        expected = [2.0, 4.0, 6.0, 8.0] + [0.0] * 28
        np.testing.assert_array_almost_equal(result, expected)


class TestDequantQ40:
    def test_q4_0_known_values(self):
        """Q4_0: block-wise dequantization with f16 scale, 4-bit quants."""
        # One block: scale=2.0 (f16), then 16 bytes (32 x 4-bit)
        scale = np.float16(2.0).tobytes()
        # Pack 32 values of 1 (nibble = 1, which becomes -7 after subtracting 8)
        # Low nibble first: 0x11 = 1, 1 -> values are 1-8=-7, 1-8=-7
        quants = bytes([0x11] * 16)  # All nibbles = 1
        data = scale + quants
        result = dequantize(data, GGML_TYPE_Q4_0)
        # Each value: (1 - 8) * 2.0 = -14.0
        expected = [-14.0] * 32
        np.testing.assert_array_almost_equal(result, expected)

    def test_q4_0_canonical_layout(self):
        """Q4_0: first 16 values sit in low nibbles, last 16 in high nibbles."""
        # Block: scale=1.0 (f16), then 16 packed bytes. Byte j carries value j
        # (0..15) in its low nibble and value j+16 (0..15) in its high nibble,
        # i.e. byte j = j | (j << 4). The interleaved (2j, 2j+1) layout would
        # scramble these.
        scale = np.float16(1.0).tobytes()
        packed = bytes([j | (j << 4) for j in range(16)])
        data = scale + packed
        result = dequantize(data, GGML_TYPE_Q4_0)
        # Low nibbles: values 0..15 -> -8..7; high nibbles: values 16..31 ->
        # nibbles 0..15 -> -8..7. An interleaved layout would order them
        # (-8,-8,-7,-7,...) instead.
        expected = [float(v) - 8.0 for v in range(16)] * 2
        np.testing.assert_array_almost_equal(result, expected)


def _make_q8_k_block(scale: float, quants: list[int]) -> bytes:
    """Build one canonical 292-byte Q8_K block (llama.cpp block_q8_K layout).

    [d: f32 scale][qs: 256 x int8][bsums: 16 x int16] — bsums are the per-16
    quant sums carried for dot products; they do not affect dequantization
    but are included so the fixture mirrors real payloads byte-for-byte.
    """
    assert len(quants) == 256
    qs = np.array(quants, dtype=np.int8).tobytes()
    bsums = np.array(
        [int(sum(quants[i * 16:(i + 1) * 16])) for i in range(16)], dtype="<i2"
    ).tobytes()
    return struct.pack("<f", scale) + qs + bsums


class TestDequantQ8K:
    def test_q8_k_known_values(self):
        """Q8_K: f32 scale multiplies 256 int8 quants; bsums ignored."""
        quants = [1, 2, 3, 4] + [0] * 252
        data = _make_q8_k_block(2.0, quants)
        assert len(data) == 292
        result = dequantize(data, GGML_TYPE_Q8_K)
        assert result.shape == (256,)
        expected = [2.0, 4.0, 6.0, 8.0] + [0.0] * 252
        np.testing.assert_array_almost_equal(result, expected)

    def test_q8_k_layout_is_292_byte_f32_scale(self):
        """The old 258-byte/f16-scale decoder decoded canonical blocks to ~0."""
        # scale=2.0, quants -3..3: correct decode is (-6,-4,-2,0); the old
        # decoder read the f32 scale bytes as an f16 pair and misaligned the
        # quants, producing zeros/garbage.
        data = _make_q8_k_block(2.0, [-3, -2, -1, 0] + [0] * 252)
        result = dequantize(data, GGML_TYPE_Q8_K)
        np.testing.assert_array_almost_equal(result[:4], [-6.0, -4.0, -2.0, 0.0])

    def test_q8_k_multi_block_roundtrip(self):
        """Multi-block payload: each block decodes with its own scale."""
        blocks = []
        expected = []
        for _i, scale in enumerate([1.0, 0.5, -2.0]):
            quants = [(j % 15) - 7 for j in range(256)]
            blocks.append(_make_q8_k_block(scale, quants))
            expected.extend(float(q) * scale for q in quants)
        result = dequantize(b"".join(blocks), GGML_TYPE_Q8_K)
        np.testing.assert_array_almost_equal(np.array(expected), result)

    def test_q8_k_truncated_payload_raises(self):
        """A payload not a multiple of 292 bytes must fail loudly, not truncate."""
        good = _make_q8_k_block(1.0, [0] * 256)
        with pytest.raises(ValueError, match="truncated Q8_K"):
            dequantize(good[:-4], GGML_TYPE_Q8_K)


# ---------------------------------------------------------------------------
# Type name and support checks
# ---------------------------------------------------------------------------


class TestTypeNames:
    def test_known_types(self):
        assert get_type_name(GGML_TYPE_F32) == "F32"
        assert get_type_name(GGML_TYPE_F16) == "F16"
        assert get_type_name(GGML_TYPE_BF16) == "BF16"
        assert get_type_name(GGML_TYPE_Q8_0) == "Q8_0"
        assert get_type_name(GGML_TYPE_Q4_0) == "Q4_0"

    def test_unknown_type(self):
        assert "UNKNOWN" in get_type_name(99)


class TestCheckSupported:
    def test_supported_types(self):
        # Should not raise
        for t in [GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0]:
            check_supported(t)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported GGUF quantization type"):
            check_supported(99)  # Unknown type


# ---------------------------------------------------------------------------
# detect_loader
# ---------------------------------------------------------------------------


class TestDetectLoader:
    def test_detect_gguf(self, tmp_path):
        """Detect GGUF by magic bytes."""
        path = tmp_path / "test.gguf"
        path.write_bytes(b"GGUF" + b"\x00" * 100)
        assert detect_loader(path) == "gguf"

    def test_detect_safetensors(self, tmp_path):
        """Detect safetensors (anything not GGUF)."""
        path = tmp_path / "test.safetensors"
        path.write_bytes(b"\x00" * 100)
        assert detect_loader(path) == "safetensors"


# ---------------------------------------------------------------------------
# GGUF fixture creation
# ---------------------------------------------------------------------------


def make_gguf_file(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Create a minimal GGUF file with given tensors (all F32 for simplicity)."""
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), arch="llama")
    # Add required KV data first
    writer.add_architecture()
    writer.add_block_count(1)
    for name, data in tensors.items():
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def gguf_file_f32(tmp_path):
    """Create a tiny GGUF file in F32 format."""
    path = tmp_path / "test_model.gguf"
    tensors = {
        "blk.0.attn_q.weight": np.random.default_rng(42).normal(0, 0.1, (32, 32)).astype(np.float32),
        "blk.0.attn_k.weight": np.random.default_rng(43).normal(0, 0.1, (32, 32)).astype(np.float32),
        "blk.0.ffn_gate.weight": np.random.default_rng(44).normal(0, 0.1, (32, 32)).astype(np.float32),
        "token_embd.weight": np.random.default_rng(45).normal(0, 0.1, (32, 32)).astype(np.float32),
    }
    make_gguf_file(path, tensors)
    return path


# ---------------------------------------------------------------------------
# GGUF loader tests
# ---------------------------------------------------------------------------


class TestGGUFLoader:
    def test_load_gguf(self, gguf_file_f32):
        """Load a GGUF file and get tensor handles."""
        from weight_atlas.loaders.gguf_loader import GGUFLoader

        loader = GGUFLoader()
        handles = loader.open(gguf_file_f32)
        assert len(handles) == 4
        names = [h.name for h in handles]
        assert "blk.0.attn_q.weight" in names
        assert "blk.0.attn_k.weight" in names

    def test_tensor_shape(self, gguf_file_f32):
        """Verify tensor shapes are correct."""
        from weight_atlas.loaders.gguf_loader import GGUFLoader

        loader = GGUFLoader()
        handles = loader.open(gguf_file_f32)
        for h in handles:
            assert h.shape == (32, 32)

    def test_tensor_loads_f32(self, gguf_file_f32):
        """Verify loaded tensors are float32."""
        from weight_atlas.loaders.gguf_loader import GGUFLoader

        loader = GGUFLoader()
        handles = loader.open(gguf_file_f32)
        for h in handles:
            arr = h.load()
            assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# Name mapping for GGUF
# ---------------------------------------------------------------------------


class TestGGUFNameMapping:
    def test_gguf_attn_q(self):
        from weight_atlas.core.name_map import map_name
        layer, slot = map_name("blk.0.attn_q.weight")
        assert layer == 0
        assert slot == "attn_q"

    def test_gguf_attn_output(self):
        from weight_atlas.core.name_map import map_name
        layer, slot = map_name("blk.0.attn_output.weight")
        assert layer == 0
        assert slot == "attn_o"

    def test_gguf_ffn_gate(self):
        from weight_atlas.core.name_map import map_name
        layer, slot = map_name("blk.0.ffn_gate.weight")
        assert layer == 0
        assert slot == "mlp_gate"

    def test_gguf_token_embd(self):
        from weight_atlas.core.name_map import map_name
        layer, slot = map_name("token_embd.weight")
        assert layer is None
        assert slot == "embed"

    def test_gguf_output(self):
        from weight_atlas.core.name_map import map_name
        layer, slot = map_name("output.weight")
        assert layer is None
        assert slot == "lm_head"


# ---------------------------------------------------------------------------
# Consistency test: safetensors ≡ gguf (F32)
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_safetensors_gguf_consistent(self, tmp_path):
        """Same weights in safetensors and GGUF(F32) should produce identical stats."""
        from gguf import GGUFWriter
        from safetensors.numpy import save_file

        rng = np.random.default_rng(42)
        w1 = rng.normal(0, 0.1, (32, 32)).astype(np.float32)
        w2 = rng.normal(0, 0.1, (32, 32)).astype(np.float32)

        # Write safetensors
        st_path = tmp_path / "model.safetensors"
        save_file({"model.layers.0.self_attn.q_proj.weight": w1,
                   "model.layers.0.self_attn.k_proj.weight": w2}, str(st_path))

        # Write GGUF (F32)
        gguf_path = tmp_path / "model.gguf"
        writer = GGUFWriter(str(gguf_path), arch="llama")
        writer.add_architecture()
        writer.add_block_count(1)
        writer.add_tensor("blk.0.attn_q.weight", w1)
        writer.add_tensor("blk.0.attn_k.weight", w2)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        # Load both
        from weight_atlas.loaders.gguf_loader import GGUFLoader
        from weight_atlas.loaders.safetensors_loader import SafetensorsLoader

        st_handles = SafetensorsLoader().open(st_path)
        gguf_handles = GGUFLoader().open(gguf_path)

        # Compare loaded tensors
        st_by_name = {h.name: h.load() for h in st_handles}
        gguf_by_name = {h.name: h.load() for h in gguf_handles}

        # Map names
        gguf_map = {
            "blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight",
            "blk.0.attn_k.weight": "model.layers.0.self_attn.k_proj.weight",
        }

        for gguf_name, st_name in gguf_map.items():
            np.testing.assert_array_almost_equal(
                gguf_by_name[gguf_name], st_by_name[st_name], decimal=6
            )


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_gguf_scan_deterministic(self, gguf_file_f32, tmp_path):
        """GGUF scan should be byte-identical on second run."""
        from weight_atlas.core.types import AtlasSpec
        from weight_atlas.scan import scan

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        scan(gguf_file_f32, out1, spec, loader_id="gguf")
        scan(gguf_file_f32, out2, spec, loader_id="gguf")

        # Compare fingerprint.json
        fp1 = (out1 / "fingerprint.json").read_bytes()
        fp2 = (out2 / "fingerprint.json").read_bytes()
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# Fingerprint tests
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_fingerprint_has_ggml_type(self, gguf_file_f32, tmp_path):
        """Fingerprint should include ggml_type for GGUF tensors."""
        import json

        from weight_atlas.core.types import AtlasSpec
        from weight_atlas.scan import scan

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        out = tmp_path / "out"
        scan(gguf_file_f32, out, spec, loader_id="gguf")

        fp = json.loads((out / "fingerprint.json").read_text())
        assert fp["loader"] == "gguf"
        assert "quantization" in fp
        assert "ggml_0" in fp["quantization"]  # F32 = type 0


# ---------------------------------------------------------------------------
# Q8_0 fixture and tests
# ---------------------------------------------------------------------------


def make_gguf_file_q8(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Create a GGUF file with Q8_0 quantized tensors."""
    from gguf import GGMLQuantizationType, GGUFWriter

    writer = GGUFWriter(str(path), arch="llama")
    writer.add_architecture()
    writer.add_block_count(1)
    for name, data in tensors.items():
        writer.add_tensor(name, data, raw_dtype=GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def gguf_file_q8(tmp_path):
    """Create a tiny GGUF file in Q8_0 format."""
    path = tmp_path / "test_model_q8.gguf"
    tensors = {
        "blk.0.attn_q.weight": np.random.default_rng(42).normal(0, 0.1, (32, 32)).astype(np.float32),
        "blk.0.attn_k.weight": np.random.default_rng(43).normal(0, 0.1, (32, 32)).astype(np.float32),
    }
    make_gguf_file_q8(path, tensors)
    return path


class TestQ8Scan:
    def test_q8_scan_produces_fingerprint(self, gguf_file_q8, tmp_path):
        """Q8_0 scan should produce fingerprint with ggml_type."""
        import json

        from weight_atlas.core.types import AtlasSpec
        from weight_atlas.scan import scan

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        out = tmp_path / "out"
        scan(gguf_file_q8, out, spec, loader_id="gguf")

        fp = json.loads((out / "fingerprint.json").read_text())
        assert fp["loader"] == "gguf"
        assert "ggml_8" in fp["quantization"]  # Q8_0 = type 8

    def test_q8_stats_differ_from_f32(self, gguf_file_f32, gguf_file_q8, tmp_path):
        """Q8_0 stats should differ (measurably) from F32 due to quantization noise."""
        import json

        from weight_atlas.core.types import AtlasSpec
        from weight_atlas.scan import scan

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        out_f32 = tmp_path / "out_f32"
        out_q8 = tmp_path / "out_q8"
        scan(gguf_file_f32, out_f32, spec, loader_id="gguf")
        scan(gguf_file_q8, out_q8, spec, loader_id="gguf")

        fp_f32 = json.loads((out_f32 / "fingerprint.json").read_text())
        fp_q8 = json.loads((out_q8 / "fingerprint.json").read_text())

        # Stats should differ due to quantization
        f32_spec = fp_f32["tensors"]["blk.0.attn_q.weight"]["spectral_norm"]
        q8_spec = fp_q8["tensors"]["blk.0.attn_q.weight"]["spectral_norm"]
        assert abs(f32_spec - q8_spec) > 1e-6  # Measurable difference


class TestUnsupportedQuant:
    def test_unsupported_quant_error_message(self):
        """Genuinely unsupported quant types (not in gguf's QUANT_SIZES) must
        produce an error with the type name."""
        from weight_atlas.loaders.gguf_dequant import check_supported

        with pytest.raises(ValueError, match="UNKNOWN"):
            check_supported(99)  # Unknown type


class TestCrossLoaderCompare:
    def test_compare_f32_vs_q8_warning(self, gguf_file_f32, gguf_file_q8, tmp_path, caplog):
        """Comparing F32 vs Q8_0 should produce quantization mismatch warning."""
        import json
        import logging

        from weight_atlas.compare import compute_compare_summary
        from weight_atlas.core.types import AtlasSpec
        from weight_atlas.scan import scan

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        out_f32 = tmp_path / "out_f32"
        out_q8 = tmp_path / "out_q8"
        scan(gguf_file_f32, out_f32, spec, loader_id="gguf")
        scan(gguf_file_q8, out_q8, spec, loader_id="gguf")

        fp_f32 = json.loads((out_f32 / "fingerprint.json").read_text())
        fp_q8 = json.loads((out_q8 / "fingerprint.json").read_text())

        # Load fields and compare
        from weight_atlas.fields.tif_io import read_tif
        field_f32 = read_tif(out_f32 / "field_height_raw.tif")
        field_q8 = read_tif(out_q8 / "field_height_raw.tif")

        with caplog.at_level(logging.WARNING):
            summary = compute_compare_summary(
                field_f32, field_q8, spec,
                mode="strict",
                fingerprint_a=fp_f32,
                fingerprint_b=fp_q8,
            )

        # Should have quantization mismatch warning
        assert any("quantization mismatch" in w for w in summary.warnings)


class TestDequantTruncationGuards:
    def test_q8_0_truncated_raises(self):
        with pytest.raises(ValueError, match="truncated Q8_0"):
            dequantize(b"\x00" * 33, GGML_TYPE_Q8_0)

    def test_q4_0_truncated_raises(self):
        from weight_atlas.loaders.gguf_dequant import GGML_TYPE_Q4_0 as Q40
        with pytest.raises(ValueError, match="truncated Q4_0"):
            dequantize(b"\x00" * 19, Q40)


class TestNdarrayPayload:
    def test_decoders_accept_2d_uint8_arrays(self):
        """GGUFReader exposes quantized data as (n_rows, block_bytes) uint8
        arrays; the decoders must treat them as flat byte payloads."""
        from weight_atlas.loaders.gguf_dequant import (
            _dequant_q1_0,
            _dequant_q4_0,
            _dequant_q8_0,
            _dequant_q8_k,
        )

        rng = np.random.default_rng(11)
        nb = 5

        q8_bytes = b"".join(
            np.float16(2.0).tobytes() + rng.integers(-128, 128, 32).astype(np.int8).tobytes()
            for _ in range(nb)
        )
        q8_arr = np.frombuffer(q8_bytes, np.uint8).reshape(nb, 34)
        np.testing.assert_array_equal(_dequant_q8_0(q8_arr), _dequant_q8_0(q8_bytes))

        q4_bytes = b"".join(
            np.float16(1.0).tobytes() + rng.integers(0, 256, 16).astype(np.uint8).tobytes()
            for _ in range(nb)
        )
        q4_arr = np.frombuffer(q4_bytes, np.uint8).reshape(nb, 18)
        np.testing.assert_array_equal(_dequant_q4_0(q4_arr), _dequant_q4_0(q4_bytes))

        q1_arr = np.frombuffer(q4_bytes, np.uint8).reshape(nb, 18)
        np.testing.assert_array_equal(_dequant_q1_0(q1_arr), _dequant_q1_0(q4_bytes))

        k_blocks = []
        for i in range(nb):
            qs = rng.integers(-100, 100, 256).astype(np.int8).tobytes()
            k_blocks.append(np.float32(i + 1).tobytes() + qs + b"\x00" * 32)
        qk_bytes = b"".join(k_blocks)
        qk_arr = np.frombuffer(qk_bytes, np.uint8).reshape(nb, 292)
        np.testing.assert_array_equal(_dequant_q8_k(qk_arr), _dequant_q8_k(qk_bytes))


class TestPlainTypes:
    """I8/I16/I32/I64/F64 are plain fixed-width formats: no block encoding,
    just raw values → float32."""

    @pytest.mark.parametrize(
        ("type_id", "dtype", "values"),
        [
            (24, np.int8, [-7, 0, 42, 127]),
            (25, np.int16, [-32000, 0, 16000, 32767]),
            (26, np.int32, [-2_000_000_000, 0, 1_000_000_000, 2_147_483_647]),
            (27, np.int64, [-9_000_000_000, 0, 9_000_000_000]),
            (28, np.float64, [-1.5, 0.0, 3.14159, 1e38]),
        ],
    )
    def test_round_trip(self, type_id, dtype, values):
        from weight_atlas.loaders.gguf_dequant import dequantize

        raw = np.array(values, dtype=dtype)
        result = dequantize(raw.tobytes(), type_id)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.array(values, dtype=np.float32))

    @pytest.mark.parametrize(
        ("type_id", "dtype", "elem_bytes"),
        [
            (24, np.int8, 1),
            (25, np.int16, 2),
            (26, np.int32, 4),
            (27, np.int64, 8),
            (28, np.float64, 8),
        ],
    )
    def test_empty_payload_raises(self, type_id, dtype, elem_bytes):
        from weight_atlas.loaders.gguf_dequant import dequantize

        with pytest.raises(ValueError, match="empty"):
            dequantize(b"", type_id)

    @pytest.mark.parametrize(
        ("type_id", "dtype", "elem_bytes"),
        [
            (25, np.int16, 2),
            (26, np.int32, 4),
            (27, np.int64, 8),
            (28, np.float64, 8),
        ],
    )
    def test_truncated_payload_raises(self, type_id, dtype, elem_bytes):
        from weight_atlas.loaders.gguf_dequant import dequantize

        with pytest.raises(ValueError, match="buffer size must be a multiple of element size"):
            dequantize(b"\x00" * (elem_bytes - 1), type_id)

    def test_dequantize_routes_i32(self):
        from weight_atlas.loaders.gguf_dequant import GGML_TYPE_I32, dequantize

        raw = np.array([10, 20, 30], dtype=np.int32)
        result = dequantize(raw.tobytes(), GGML_TYPE_I32)
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# IQ family (importance-matrix quants) — delegated to the official gguf lib
# ---------------------------------------------------------------------------


class TestIQTypes:
    """Unsloth UD dynamic quants (e.g. Qwen3.8-Flash-Next UD-IQ4_XS) mix IQ
    tensors into MoE expert layers. The block layouts are intricate (E8-grid
    lookups, 6-bit sub-scales, packed nibbles), so dequantization delegates to
    the official gguf library (authoritative, _GGUF_ONLY); these tests pin the
    dispatch + behavioral contract, not the grid math."""

    IQ_TYPES = (
        ("IQ1_S", 256, 50),
        ("IQ1_M", 256, 56),
        ("IQ2_XXS", 256, 66),
        ("IQ2_XS", 256, 74),
        ("IQ2_S", 256, 82),
        ("IQ3_XXS", 256, 98),
        ("IQ3_S", 256, 110),
        ("IQ4_NL", 32, 18),
        ("IQ4_XS", 256, 136),
    )

    def test_all_iq_types_are_supported(self):
        from weight_atlas.loaders.gguf_dequant import (
            GGML_TYPE_IQ1_M,
            GGML_TYPE_IQ1_S,
            GGML_TYPE_IQ2_S,
            GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_XXS,
            GGML_TYPE_IQ3_S,
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ4_NL,
            GGML_TYPE_IQ4_XS,
        )
        for t in (
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL,
            GGML_TYPE_IQ4_XS,
        ):
            check_supported(t)  # must not raise

    @pytest.mark.parametrize(
        "name,block_size,type_size",
        [(t[0], t[1], t[2]) for t in IQ_TYPES],
    )
    def test_dequantize_shape_dtype_determinism(self, name, block_size, type_size):  # noqa: ARG002
        import gguf

        qtype = gguf.GGMLQuantizationType[name]
        ggml_type = int(qtype)
        rng = np.random.default_rng(42)
        n_blocks = 3
        raw = rng.integers(0, 256, size=n_blocks * type_size, dtype=np.uint8)
        out1 = dequantize(raw, ggml_type)
        out2 = dequantize(raw, ggml_type)
        assert out1.shape == (n_blocks * block_size,)
        assert out1.dtype == np.float32
        assert np.isfinite(out1).all()
        # deterministic: same payload → same values
        np.testing.assert_array_equal(out1, out2)

    def test_iq4_xs_zero_block_dequantizes_to_zero(self):
        import gguf

        qtype = gguf.GGMLQuantizationType["IQ4_XS"]
        _, type_size = gguf.GGML_QUANT_SIZES[qtype]
        raw = np.zeros(2 * type_size, dtype=np.uint8)
        out = dequantize(raw, int(qtype))
        # zero scale + zero indices → zero weights
        np.testing.assert_array_equal(out, np.zeros(512, dtype=np.float32))

    def test_iq4_xs_misaligned_payload_raises(self):
        import gguf

        qtype = gguf.GGMLQuantizationType["IQ4_XS"]
        ggml_type = int(qtype)
        with pytest.raises(ValueError):
            dequantize(np.zeros(136 + 10, dtype=np.uint8), ggml_type)

    def test_iq4_xs_loader_end_to_end(self, tmp_path):
        """A GGUF fixture with crafted IQ4_XS blocks flows through the loader
        into float32 handles (the full ingest path). The gguf lib cannot
        *quantize* IQ4_XS without an importance matrix, so the fixture writes
        hand-crafted blocks with known decode semantics: d=1.0, sub-scales
        h=2/l=1 → scale 2, nibble index 8 → kvalue 1 → every weight decodes
        to 2.0."""
        from gguf import GGMLQuantizationType, GGUFWriter

        qtype = GGMLQuantizationType["IQ4_XS"]
        _, type_size = (256, 136)
        n_blocks = 4  # 4 rows × 256 weights: one block per row (loader
        # validates per-row byte alignment against the type size)
        blocks = np.zeros((n_blocks, type_size), dtype=np.uint8)
        blocks[:, 0:2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
        blocks[:, 2:4] = np.frombuffer(np.uint16(0xAAAA).tobytes(), dtype=np.uint8)
        blocks[:, 4:8] = 0x11
        blocks[:, 8:type_size] = 0x88
        tensor_data = blocks.tobytes()

        path = tmp_path / "iq4xs.gguf"
        writer = GGUFWriter(str(path), arch="llama")
        writer.add_architecture()
        writer.add_block_count(1)
        writer.add_tensor(
            "blk.0.attn_q.weight", np.frombuffer(tensor_data, dtype=np.uint8),
            raw_shape=[4, 136], raw_dtype=qtype,  # byte shape (4 blocks × 136 B);
            # header carries logical [4, 256]
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        from weight_atlas.core.types import detect_loader
        from weight_atlas.loaders.gguf_loader import GGUFLoader

        assert detect_loader(path) == "gguf"
        handles = GGUFLoader().open(path)
        assert len(handles) == 1
        h = handles[0]
        assert h.dtype == "ggml_23"  # IQ4_XS
        arr = h.load()
        # GGUF stores dims reversed; the handle carries the file order
        assert arr.shape == (256, 4)
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
        expected_value = float(np.unique(
            __import__("gguf").dequantize(blocks, qtype)
        )[0])
        np.testing.assert_array_equal(arr, np.full((256, 4), expected_value, dtype=np.float32))
        # deterministic
        np.testing.assert_array_equal(arr, GGUFLoader().open(path)[0].load())
