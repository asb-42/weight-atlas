"""EXL3 tests: dequant kernel transcription, loader, scan pipeline, detection.

The dequantization tests verify against a naive scalar reference that
transcribes the exllamav3 CUDA kernels (pack.cu, exl3_dq.cuh, codebook.cuh,
reconstruct.cu) independently of the vectorized implementation. Fixture
builders write synthetic EXL3 safetensors checkpoints in the exact layout
the real converter produces.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from weight_atlas.core.types import detect_loader
from weight_atlas.loaders.exl3_dequant import (
    CODEBOOK_MCG_MULT,
    apply_scales_and_hadamard,
    decode_codebook,
    dequantize_exl3_layer,
    hadamard_128,
    indices_from_packed,
    reconstruct_hat,
    tensor_core_perm,
    windows_from_indices,
)
from weight_atlas.loaders.exl3_loader import EXL3Loader, read_exl3_group

# ---------------------------------------------------------------------------
# Reference transcription of the exllamav3 CUDA kernels (naive, scalar)
# ---------------------------------------------------------------------------


def ref_natural_words(file_words: list[int]) -> list[int]:
    """File word order -> natural word order (undo the packer's SWAP16)."""
    out: list[int] = []
    for i in range(0, len(file_words), 2):
        out.append(int(file_words[i + 1]))
        out.append(int(file_words[i]))
    return out


def ref_window(natural: list[int], k: int, p: int) -> int:
    """16-bit window ending at ring bit (p+1)*k (exl3_dq.cuh, tail-biting ring)."""
    w = 0
    for j in range(16):
        b = ((p + 1) * k - 16 + j) % (256 * k)
        word = natural[b // 16]
        w = (w << 1) | int((word >> (15 - (b % 16))) & 1)
    return w


def ref_codebook_value(window: int, codebook: str) -> float:
    """decode_3inst<cb> from codebook.cuh, scalar.

    The __hadd in the kernel rounds the sum to float16 once; the exact
    float64 sum of two float16 values with a single round-to-f16 is
    IEEE float16 addition, so we round explicitly here.
    """
    if codebook == "mcg":
        p = (window * CODEBOOK_MCG_MULT) & 0xFFFFFFFF
    elif codebook == "plain":
        p = (window * 89226354 + 64248484) & 0xFFFFFFFF
    else:
        raise AssertionError(codebook)
    r = (p & 0x8FFF8FFF) ^ 0x3B603B60
    lo = np.array(r & 0xFFFF, dtype=np.uint16).view(np.float16)
    hi = np.array(r >> 16, dtype=np.uint16).view(np.float16)
    return float(np.float16(float(lo) + float(hi)))


def ref_mul1_value(window: int) -> float:
    """decode_3inst<2> (mul1) from codebook.cuh, scalar.

    dp4a byte sum with accumulator 0x6400, then the float16
    multiply-add constants; one float16 rounding at the end.
    """
    p = (window * 0x83DCD12D) & 0xFFFFFFFF
    acc = 0x6400
    for shift in (0, 8, 16, 24):
        b = (p >> shift) & 0xFF
        acc += b - 256 if b >= 128 else b
    h = np.array(acc & 0xFFFF, dtype=np.uint16).view(np.float16)
    k_inv = np.array(0x1EEE, dtype=np.uint16).view(np.float16)
    k_bias = np.array(0xC931, dtype=np.uint16).view(np.float16)
    return float(np.float16(float(h) * float(k_inv) + float(k_bias)))


def ref_tile(file_words: list[int], k: int, codebook: str) -> np.ndarray:
    """One 16x16 W_hat tile: windows + tensor-core scatter (reconstruct.cu)."""
    natural = ref_natural_words(file_words)
    tile = np.zeros((16, 16), dtype=np.float16)
    for p in range(256):
        t, u = p // 8, p % 8
        row = (t % 4) * 2 + (u & 1) + 8 * ((u >> 1) & 1)
        col = t // 4 + 8 * (u >> 2)
        tile[row, col] = ref_codebook_value(ref_window(natural, k, p), codebook)
    return tile


def ref_pack_indices(indices: np.ndarray, k: int) -> np.ndarray:
    """Pack 256 K-bit indices MSB-first into file-order uint16 words (pack.cu).

    The MSB-first bitstream fills word n's bit 15 down to bit 0, so word n
    is the big-endian pair of bitstream bytes (2n, 2n+1); the file then
    swaps each adjacent word pair (the packer's SWAP16 on 32-bit groups).
    """
    bits = np.zeros(256 * k, dtype=np.uint8)
    for p, v in enumerate(int(x) for x in indices):
        for b in range(k):
            bits[p * k + b] = (v >> (k - 1 - b)) & 1
    bytes_ = np.packbits(bits)  # MSB-first: byte j = bits[8j:8j+8]
    nat_words = (bytes_[0::2].astype(np.uint32) << 8) | bytes_[1::2]
    words = nat_words.astype(np.uint16)
    file_order: list[int] = []
    for i in range(0, len(words), 2):
        file_order.extend([int(words[i + 1]), int(words[i])])
    return np.array(file_order, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Fixture builders: synthetic EXL3 checkpoints
# ---------------------------------------------------------------------------


_ST_ITEMSIZE = {"BF16": 2, "F16": 2, "I16": 2, "I32": 4, "U16": 2, "F32": 4}


def save_safetensors_raw(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """Write a safetensors file with arbitrary header dtypes (incl. BF16).

    tensors maps name -> (header dtype, array holding the file's raw bits);
    safetensors.numpy cannot express BF16, so fixtures build the file directly.
    """
    header: dict[str, dict] = {}
    offset = 0
    for name, (dt, arr) in tensors.items():
        nbytes = int(np.prod(arr.shape)) * _ST_ITEMSIZE[dt]
        header[name] = {
            "dtype": dt,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header_bytes = json.dumps(header).encode()
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)  # pad to 8
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for _, (_, arr) in tensors.items():
            f.write(np.ascontiguousarray(arr).tobytes())


def build_exl3_group(
    rng: np.random.Generator,
    kt: int,
    nt: int,
    k: int,
    codebook: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize (trellis, suh, svh) for one EXL3 group with known indices."""
    n_idx = rng.integers(0, 1 << k, size=(kt, nt, 256)).astype(np.uint16)
    trellis = np.empty((kt, nt, 16 * k), dtype=np.int16)
    for kb in range(kt):
        for nb in range(nt):
            trellis[kb, nb] = ref_pack_indices(n_idx[kb, nb], k).view(np.int16)
    suh = (rng.standard_normal(kt * 16) * 0.05).astype(np.float16)
    svh = (rng.standard_normal(nt * 16) * 1.0).astype(np.float16)
    return trellis, suh, svh


def write_exl3_checkpoint(
    path: Path,
    groups: dict[str, tuple[int, int, str]],  # prefix -> (kt, nt, K)
    *,
    n_layers: int = 2,
    k_bits: int = 8,
    seed: int = 7,
    include_config: bool = True,
) -> dict[str, tuple[str, np.ndarray]]:
    """Write a synthetic EXL3 model directory (trellis groups + BF16 norms).

    Returns the raw (dtype, bits) map for post-hoc expectations.
    """
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tuple[str, np.ndarray]] = {}
    for prefix, (kt, nt, k) in groups.items():
        trellis, suh, svh = build_exl3_group(rng, kt, nt, k, "mcg")
        tensors[f"{prefix}.trellis"] = ("I16", trellis)
        tensors[f"{prefix}.suh"] = ("F16", suh)
        tensors[f"{prefix}.svh"] = ("F16", svh)
        tensors[f"{prefix}.mcg"] = (
            "I32",
            np.array(CODEBOOK_MCG_MULT, dtype=np.uint32).view(np.int32),
        )
    tensors["model.norm.weight"] = (
        "F16",
        (rng.standard_normal(128) * 0.1 + 1).astype(np.float16),
    )
    embed_bits = (rng.standard_normal((128, 128)) * 0.02).astype(np.float32)
    embed_bits = (embed_bits.view(np.uint32) >> 16).astype(np.uint16)  # BF16 bits
    tensors["model.embed_tokens.weight"] = ("BF16", embed_bits)
    save_safetensors_raw(path / "model.safetensors", tensors)
    if include_config:
        (path / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                    "quantization_config": {
                        "quant_method": "exl3",
                        "version": "0.0.37",
                        "bits": float(k_bits),
                        "codebook": "mcg",
                    },
                }
            )
        )
    return tensors


# ---------------------------------------------------------------------------
# Unit tests: bit / window / codebook / permutation primitives
# ---------------------------------------------------------------------------


class TestHadamard:
    def test_orthonormal(self):
        h = hadamard_128().astype(np.float64)
        np.testing.assert_allclose(h @ h.T, np.eye(128), atol=1e-12)
        assert h[0, 0] > 0

    def test_signs_match_popcount_construction(self):
        i = np.arange(128)
        ref = np.array(
            [[(-1) ** bin(int(a) & int(b)).count("1") for b in i] for a in i],
            dtype=np.float64,
        ) / np.sqrt(128)
        np.testing.assert_allclose(hadamard_128().astype(np.float64), ref, atol=1e-12)


class TestTensorCorePerm:
    def test_perm_matches_quantizer(self):
        perm_ref = []
        for t in range(32):
            r0 = (t % 4) * 2
            c0 = t // 4
            for u in range(8):
                perm_ref.append(
                    (r0 + (u & 1) + 8 * ((u >> 1) & 1)) * 16 + (c0 + 8 * (u >> 2))
                )
        np.testing.assert_array_equal(tensor_core_perm(), perm_ref)

    def test_is_permutation(self):
        assert sorted(tensor_core_perm().tolist()) == list(range(256))


class TestCodebooks:
    def test_mcg_reference_values(self):
        for w in (0, 1, 0xFFFF, 0x8000, 0x1234, 0x00FF):
            got = float(decode_codebook(np.array([[w]], dtype=np.uint32), "mcg")[0, 0])
            assert got == ref_codebook_value(w, "mcg"), hex(w)

    def test_plain_reference_values(self):
        for w in (0, 1, 0xFFFF, 0x8000, 0x1234, 0x00FF):
            got = float(
                decode_codebook(np.array([[w]], dtype=np.uint32), "plain")[0, 0]
            )
            assert got == ref_codebook_value(w, "plain"), hex(w)

    def test_mul1_reference_values(self):
        # dp4a byte-sum path, validated against the scalar transcription;
        # the fused __hfma vs float32-then-round can differ by at most 1 ulp.
        rng = np.random.default_rng(31)
        windows = rng.integers(0, 1 << 16, 24).astype(np.uint32)
        vals = decode_codebook(windows[None, :], "mul1")[0]
        for w, got in zip(windows, vals, strict=True):
            assert got == pytest.approx(ref_mul1_value(int(w)), rel=2**-9), hex(int(w))

    def test_mul1_band(self):
        vals = decode_codebook(np.arange(256, dtype=np.uint32)[None, :], "mul1")
        assert vals.dtype == np.float16
        assert float(np.abs(vals).max()) < 16.0

    def test_unknown_codebook_raises(self):
        with pytest.raises(ValueError, match="unknown EXL3 codebook"):
            decode_codebook(np.zeros((1, 1), dtype=np.uint32), "nope")


class TestBitstream:
    def test_k8_indices_are_bytes(self):
        # With K=8, each natural byte is one index. File words come in swapped
        # 32-bit groups, so (w1, w0) = file bytes [0x12, 0x34, 0x56, 0x78]...
        # must decode back to bytes in natural order. A full 128-word tile:
        # natural byte b occupies file word pair (2*i, 2*i+1).
        rng = np.random.default_rng(41)
        natural_bytes = rng.integers(0, 256, 256, dtype=np.uint8)  # 128 words
        # Natural word n (MSB-first bitstream) is the big-endian pair of
        # bitstream bytes (2n, 2n+1); each 32-bit file group g swaps its two
        # words: file (2g, 2g+1) = natural (2g+1, 2g).
        nat_words = ((natural_bytes[0::2].astype(np.uint32) << 8) | natural_bytes[1::2]).astype(np.uint16)
        file_words = np.empty(128, dtype=np.uint16)
        file_words[0::2] = nat_words[1::2]
        file_words[1::2] = nat_words[0::2]
        got = indices_from_packed(file_words[None, :], 8)[0]
        np.testing.assert_array_equal(got, natural_bytes)

    def test_k4_roundtrip_with_reference_packer(self):
        rng = np.random.default_rng(3)
        idx = rng.integers(0, 16, 256).astype(np.uint16)
        packed = ref_pack_indices(idx, 4)
        got = indices_from_packed(packed[None, :], 4)[0]
        np.testing.assert_array_equal(got, idx)

    def test_k1_roundtrip_with_reference_packer(self):
        rng = np.random.default_rng(4)
        idx = rng.integers(0, 2, 256).astype(np.uint16)
        packed = ref_pack_indices(idx, 1)
        got = indices_from_packed(packed[None, :], 1)[0]
        np.testing.assert_array_equal(got, idx)

    def test_window_low_bits_are_index(self):
        rng = np.random.default_rng(5)
        for k in (1, 2, 3, 4, 5, 6, 7, 8):
            idx = rng.integers(0, 1 << k, 256).astype(np.uint16)
            win = windows_from_indices(idx[None, :], k)[0]
            np.testing.assert_array_equal(
                idx.astype(np.uint32), win.astype(np.uint32) & ((1 << k) - 1)
            )

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="bitrate K"):
            indices_from_packed(np.zeros((1, 16), dtype=np.uint16), 9)

    def test_wrong_word_count_raises(self):
        with pytest.raises(ValueError, match="words per tile"):
            indices_from_packed(np.zeros((1, 17), dtype=np.uint16), 8)


class TestReconstructAgainstReference:
    """Bit-exact comparison of the vectorized dequant against the naive kernels."""

    @pytest.mark.parametrize("codebook", ["mcg", "plain"])
    def test_tiles_bit_exact(self, codebook):
        rng = np.random.default_rng(11)
        k = 8
        trellis, _, _ = build_exl3_group(rng, kt=2, nt=2, k=k, codebook=codebook)
        w_hat = reconstruct_hat(trellis, codebook)
        for kb in range(2):
            for nb in range(2):
                ref = ref_tile(
                    list(np.asarray(trellis[kb, nb], dtype=np.uint16)), k, codebook
                )
                got = w_hat[kb * 16 : (kb + 1) * 16, nb * 16 : (nb + 1) * 16]
                np.testing.assert_array_equal(got, ref)

    def test_partial_k_tile(self):
        # K=3 exercises the sub-byte packing path.
        rng = np.random.default_rng(12)
        k = 3
        trellis, _, _ = build_exl3_group(rng, kt=1, nt=1, k=k, codebook="mcg")
        w_hat = reconstruct_hat(trellis, "mcg")
        ref = ref_tile(list(np.asarray(trellis[0, 0], dtype=np.uint16)), k, "mcg")
        np.testing.assert_array_equal(w_hat, ref)


class TestScaleEpilogue:
    def test_epilogue_matches_reference_transcription(self):
        # Direct check against the exl3.py get_weight_tensor sequence:
        # W = D_svh . (H_in @ W_hat . D_suh) @ H_out, per 128-block, f16 rounding.
        rng = np.random.default_rng(13)
        w_hat = (rng.standard_normal((128, 128)) * 0.05).astype(np.float16)
        suh = (rng.standard_normal(128) * 0.05 + 0.08).astype(np.float16)
        svh = (rng.standard_normal(128) + 1.0).astype(np.float16)
        h = hadamard_128().astype(np.float64)
        ref = h @ w_hat.astype(np.float64)
        ref = ref * suh.astype(np.float64)[:, None]
        ref = ref @ h
        ref = ref * svh.astype(np.float64)[None, :]
        got = apply_scales_and_hadamard(w_hat, suh, svh)
        np.testing.assert_allclose(got, ref, rtol=0.05, atol=5e-4)

    def test_scale_axes(self):
        rng = np.random.default_rng(14)
        w_hat = (rng.standard_normal((256, 128)) * 0.05).astype(np.float16)
        suh = np.full(256, 2.0, np.float16)
        svh = np.full(128, 3.0, np.float16)
        base = apply_scales_and_hadamard(w_hat, np.ones(256, np.float16), np.ones(128, np.float16))
        scaled = apply_scales_and_hadamard(w_hat, suh, svh)
        # Uniform scales commute with both Hadamards up to f16 rounding.
        np.testing.assert_allclose(scaled, 6.0 * base, rtol=0.05, atol=2e-3)

    def test_shape_mismatch_raises(self):
        w_hat = np.zeros((128, 128), dtype=np.float16)
        with pytest.raises(ValueError, match="suh/svh shapes"):
            apply_scales_and_hadamard(w_hat, np.ones(127, np.float16), np.ones(128, np.float16))


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestEXL3Loader:
    def test_open_resolves_groups_and_plain(self, tmp_path):
        write_exl3_checkpoint(
            tmp_path / "model",
            groups={
                "model.layers.0.self_attn.q_proj": (8, 8, 8),
                "model.layers.1.mlp.up_proj": (8, 32, 8),  # 512 out (128 | 512)
            },
        )
        handles = EXL3Loader().open(tmp_path / "model")
        names = [h.name for h in handles]
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "model.layers.1.mlp.up_proj.weight" in names
        assert "model.norm.weight" in names
        assert "model.embed_tokens.weight" in names
        # Group component tensors are not exposed individually.
        assert not any(n.endswith((".trellis", ".suh", ".svh", ".mcg")) for n in names)

    def test_handle_shapes_and_dtypes(self, tmp_path):
        write_exl3_checkpoint(
            tmp_path / "model",
            groups={"model.layers.0.self_attn.q_proj": (8, 8, 8)},
        )
        handles = EXL3Loader().open(tmp_path / "model")
        by_name = {h.name: h for h in handles}
        q = by_name["model.layers.0.self_attn.q_proj.weight"]
        assert q.shape == (128, 128)  # (out, in) HF convention
        assert q.dtype == "exl3_K8_mcg"
        embed = by_name["model.embed_tokens.weight"]
        assert embed.dtype == "BF16"

    def test_lazy_no_dequant_on_open(self, tmp_path):
        write_exl3_checkpoint(
            tmp_path / "model",
            groups={"model.layers.0.self_attn.q_proj": (8, 8, 8)},
        )
        handles = EXL3Loader().open(tmp_path / "model")
        # open() must resolve names/shapes without touching the trellis data.
        assert handles[0].shape is not None

    def test_load_materializes_expected_matrix(self, tmp_path):
        write_exl3_checkpoint(
            tmp_path / "model",
            groups={"model.layers.0.self_attn.q_proj": (8, 8, 8)},
        )
        handles = EXL3Loader().open(tmp_path / "model")
        by_name = {h.name: h for h in handles}
        w = by_name["model.layers.0.self_attn.q_proj.weight"].load()
        assert w.shape == (128, 128)
        assert w.dtype == np.float32
        assert np.isfinite(w).all()
        # Second load (stats hit the FIFO cache) must return identical data.
        np.testing.assert_array_equal(w, by_name["model.layers.0.self_attn.q_proj.weight"].load())

    def test_bf16_passthrough(self, tmp_path):
        raw = write_exl3_checkpoint(
            tmp_path / "model",
            groups={"model.layers.0.self_attn.q_proj": (8, 8, 8)},
        )
        handles = EXL3Loader().open(tmp_path / "model")
        embed = {h.name: h for h in handles}["model.embed_tokens.weight"]
        got = embed.load()
        expected = raw["model.embed_tokens.weight"][1].astype(np.uint32) << 16
        expected = expected.view(np.float32)
        np.testing.assert_array_equal(got, expected)

    def test_read_exl3_group_helper(self, tmp_path):
        write_exl3_checkpoint(
            tmp_path / "model",
            groups={"model.layers.0.mlp.down_proj": (8, 8, 8)},
        )
        trellis, suh, svh, codebook = read_exl3_group(
            tmp_path / "model", "model.layers.0.mlp.down_proj"
        )
        assert trellis.shape == (8, 8, 128)
        assert suh.shape == (128,) and svh.shape == (128,)
        assert codebook == "mcg"

    def test_group_without_marker_is_plain(self, tmp_path):
        # A trellis group without mcg/mul1 marker decodes with the plain codebook.
        rng = np.random.default_rng(21)
        tensors = {}
        trellis, suh, svh = build_exl3_group(rng, 8, 8, 8, "plain")
        tensors["model.layers.0.self_attn.v_proj.trellis"] = trellis
        tensors["model.layers.0.self_attn.v_proj.suh"] = suh
        tensors["model.layers.0.self_attn.v_proj.svh"] = svh
        save_file(tensors, str(tmp_path / "model.safetensors"))
        trellis2, suh2, svh2, codebook = read_exl3_group(
            tmp_path, "model.layers.0.self_attn.v_proj"
        )
        assert codebook == "plain"
        np.testing.assert_array_equal(trellis2, trellis)

    def test_unknown_marker_raises(self, tmp_path):
        rng = np.random.default_rng(22)
        tensors = {}
        trellis, suh, svh = build_exl3_group(rng, 8, 8, 8, "mcg")
        tensors["x.trellis"] = trellis
        tensors["x.suh"] = suh
        tensors["x.svh"] = svh
        tensors["x.mcg"] = np.array([0, 0], dtype=np.int16).view(np.int32)
        save_file(tensors, str(tmp_path / "model.safetensors"))
        with pytest.raises(ValueError, match="codebook marker"):
            EXL3Loader().open(tmp_path)

    def test_missing_trellis_raises(self, tmp_path):
        save_file(
            {"x.suh": np.ones(128, np.float16), "x.svh": np.ones(128, np.float16)},
            str(tmp_path / "model.safetensors"),
        )
        with pytest.raises(ValueError, match="no trellis"):
            EXL3Loader().open(tmp_path)

    def test_invalid_trellis_shape_raises(self, tmp_path):
        save_file(
            {
                "x.trellis": np.zeros((8, 8, 17), dtype=np.int16),
                "x.suh": np.zeros(128, np.float16),
                "x.svh": np.zeros(128, np.float16),
            },
            str(tmp_path / "model.safetensors"),
        )
        with pytest.raises(ValueError, match="trellis"):
            EXL3Loader().open(tmp_path)

    def test_packed_signs_unsupported(self, tmp_path):
        save_file(
            {
                "x.trellis": np.zeros((8, 8, 128), dtype=np.int16),
                "x.su": np.zeros(16, dtype=np.int16),
                "x.sv": np.zeros(16, dtype=np.int16),
            },
            str(tmp_path / "model.safetensors"),
        )
        with pytest.raises(ValueError, match="packed sign"):
            EXL3Loader().open(tmp_path)


# ---------------------------------------------------------------------------
# Detection and scan pipeline
# ---------------------------------------------------------------------------


class TestDetectLoader:
    def test_detect_exl3_by_config(self, tmp_path):
        write_exl3_checkpoint(tmp_path / "model", groups={})
        assert detect_loader(tmp_path / "model") == "exl3"

    def test_plain_safetensors_dir_stays_safetensors(self, tmp_path):
        save_file({"w": np.zeros((4, 4), np.float32)}, str(tmp_path / "m.safetensors"))
        # A config.json without the exl3 marker must not flip detection.
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        assert detect_loader(tmp_path) == "safetensors"

    def test_broken_config_falls_back_to_contents(self, tmp_path):
        (tmp_path / "config.json").write_text("{invalid")
        save_file({"w": np.zeros((4, 4), np.float32)}, str(tmp_path / "m.safetensors"))
        assert detect_loader(tmp_path) == "safetensors"

    def test_gguf_priority_over_exl3_config(self, tmp_path):
        # GGUF wins over the exl3 marker (gguf glob is checked first).
        write_exl3_checkpoint(tmp_path / "model", groups={})
        (tmp_path / "model" / "m.gguf").write_bytes(b"GGUF" + b"\x00" * 16)
        assert detect_loader(tmp_path / "model") == "gguf"

    def test_exl3_without_config_raises_not_safetensors(self, tmp_path):
        # No config.json marker → plain safetensors, then the loader raises
        # on the trellis groups (detection stays content-based).
        save_file(
            {"x.trellis": np.zeros((8, 8, 128), dtype=np.int16)},
            str(tmp_path / "model.safetensors"),
        )
        assert detect_loader(tmp_path) == "safetensors"


class TestEXL3Scan:
    def _spec(self):
        from weight_atlas.core.types import AtlasSpec

        # v2.2 is the oldest spec that still matches this branch's slot set;
        # it keeps the EXL3 scan test independent of later spec additions.
        return AtlasSpec.from_json(Path("specs/atlas_spec.v2.2.json"))

    def test_scan_pipeline_on_exl3(self, tmp_path):
        write_exl3_checkpoint(
            tmp_path / "model",
            groups={
                "model.layers.0.self_attn.q_proj": (8, 8, 8),
                "model.layers.0.mlp.up_proj": (8, 8, 8),
            },
        )
        from weight_atlas.scan import scan

        out = tmp_path / "out"
        artefacts = scan(tmp_path / "model", out, self._spec(), loader_id="exl3")
        assert (out / "fingerprint.json").exists()
        fp = json.loads((out / "fingerprint.json").read_text())
        assert fp["loader"] == "exl3"
        assert "model.layers.0.self_attn.q_proj.weight" in fp["tensors"]
        # EXL3 quant summary reflects the resolved handles' dtypes.
        assert fp["quantization"]["exl3_K8_mcg"] >= 2
        assert (out / "field_height_raw.tif").exists()
        assert any(str(p).endswith("manifest.json") for p in artefacts)
    def test_scan_deterministic(self, tmp_path):
        write_exl3_checkpoint(
            tmp_path / "model",
            groups={"model.layers.0.self_attn.q_proj": (8, 8, 8)},
        )
        from weight_atlas.scan import scan

        o1, o2 = tmp_path / "o1", tmp_path / "o2"
        scan(tmp_path / "model", o1, self._spec(), loader_id="exl3")
        scan(tmp_path / "model", o2, self._spec(), loader_id="exl3")
        assert (o1 / "fingerprint.json").read_bytes() == (o2 / "fingerprint.json").read_bytes()

    def test_diagnose_mapping_coverage(self, tmp_path):
        # Reconstructed weight names must map into slots like any HF model.
        write_exl3_checkpoint(
            tmp_path / "model",
            groups={
                "model.layers.0.self_attn.q_proj": (8, 8, 8),
                "model.layers.0.mlp.down_proj": (8, 8, 8),
                "lm_head": (8, 8, 8),
            },
        )
        from weight_atlas.core.name_map import map_name

        for name, (layer, slot) in {
            "model.layers.0.self_attn.q_proj.weight": (0, "attn_q"),
            "model.layers.0.mlp.down_proj.weight": (0, "mlp_down"),
            "lm_head.weight": (None, "lm_head"),
        }.items():
            assert map_name(name) == (layer, slot), name


# ---------------------------------------------------------------------------
# Real-sample integration (skipped when the model is absent)
# ---------------------------------------------------------------------------

_SAMPLE = Path("/media/data/AI/models/turboderp_MiniCPM5-1B-exl3")
_SAMPLE_MISSING = not _SAMPLE.is_dir()

pytestmark_data = pytest.mark.skipif(
    _SAMPLE_MISSING, reason=f"EXL3 sample model not present at {_SAMPLE}"
)


@pytest.mark.skipif(_SAMPLE_MISSING, reason=f"EXL3 sample model not present at {_SAMPLE}")
class TestRealSample:
    def test_sample_tiles_bit_exact(self):
        rng_positions = [(0, 0), (47, 0), (95, 15)]
        trellis, suh, svh, codebook = read_exl3_group(
            _SAMPLE, "model.layers.1.self_attn.k_proj"
        )
        k = trellis.shape[2] // 16
        assert k == 8 and codebook == "mcg"
        for kb, nb in rng_positions:
            ref = ref_tile(list(np.asarray(trellis[kb, nb], dtype=np.uint16)), k, codebook)
            got = reconstruct_hat(trellis[kb : kb + 1], codebook)[
                :, nb * 16 : (nb + 1) * 16
            ]
            np.testing.assert_array_equal(got, ref)

    def test_sample_scale_semantics(self):
        # |suh| / |svh| must land on the correct axes after the Hadamard epilogue.
        for prefix in (
            "model.layers.1.self_attn.k_proj",
            "model.layers.12.mlp.down_proj",
        ):
            trellis, suh, svh, codebook = read_exl3_group(_SAMPLE, prefix)
            w = dequantize_exl3_layer(trellis, suh, svh, codebook)
            row_rms = np.sqrt((w.astype(np.float64) ** 2).mean(axis=1))
            col_rms = np.sqrt((w.astype(np.float64) ** 2).mean(axis=0))
            assert np.corrcoef(row_rms, np.abs(suh.astype(np.float64)))[0, 1] > 0.95
            assert np.corrcoef(col_rms, np.abs(svh.astype(np.float64)))[0, 1] > 0.95

    def test_sample_loader_scan_smoke(self, tmp_path):
        handles = EXL3Loader().open(_SAMPLE)
        assert len(handles) == 50 + 169  # plain tensors + quant groups
        names = {h.name for h in handles}
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "lm_head.weight" in names
        assert "model.embed_tokens.weight" in names
        by_name = {h.name: h for h in handles}
        w = by_name["model.layers.1.self_attn.k_proj.weight"].load()
        assert w.shape == (256, 1536)  # (out, in)
        assert np.isfinite(w).all()
