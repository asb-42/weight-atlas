"""PyTorch loader tests: pure-python unpickler + BDH layout expansion.

The fixture builder hand-assembles the PyTorch checkpoint pickle (protocol 2,
same opcode layout torch writes) so tests run without torch installed.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from weight_atlas.core.name_map import map_name
from weight_atlas.core.types import AtlasSpec, TensorStats, detect_loader
from weight_atlas.fields.rasterizer import rasterize_bdh_lattice, rasterize_flat
from weight_atlas.loaders import pytorch_loader  # noqa: F401 — registers "pytorch"
from weight_atlas.loaders.pytorch_loader import PyTorchLoader

# ── Minimal PyTorch .pt fixture builder ────────────────────────────────────


def _binunicode(s: str) -> bytes:
    b = s.encode("utf-8")
    return b"X" + struct.pack("<I", len(b)) + b


def _global(mod: str, name: str) -> bytes:
    return b"c" + mod.encode() + b"\n" + name.encode() + b"\n"


def _int32(v: int) -> bytes:
    return b"J" + struct.pack("<i", v)


def _tuple(items: list[bytes]) -> bytes:
    """Encode a tuple from already-encoded item bytes (MARK+TUPLE for n>=4)."""
    body = b"".join(items)
    n = len(items)
    if n == 0:
        return b")"
    if n <= 3:
        return body + {1: b"\x85", 2: b"\x86", 3: b"\x87"}[n]
    return b"(" + body + b"t"


def _tensor_rebuild(key: str, numel: int, size: tuple[int, ...], stride: tuple[int, ...]) -> bytes:
    """Encode one torch._utils._rebuild_tensor_v2 call for storage ``key``."""
    out = _global("torch._utils", "_rebuild_tensor_v2")
    out += b"("  # MARK: args tuple
    out += b"("  # MARK: storage persistent-id tuple
    out += _binunicode("storage")
    out += _global("torch", "FloatStorage")
    out += _binunicode(key)
    out += _binunicode("cpu")
    out += _int32(numel)
    out += b"t"  # TUPLE
    out += b"Q"  # BINPERSID → persistent_load
    out += b"K\x00"  # BININT1 storage_offset
    out += _tuple([_int32(x) for x in size])
    out += _tuple([_int32(x) for x in stride])
    out += b"\x89"  # NEWFALSE requires_grad
    out += b")"  # EMPTY_TUPLE backward_hooks
    out += b"t"  # TUPLE (args)
    out += b"R"  # REDUCE
    return out


def _build_pt_bytes(
    cfg: dict[str, int],
    tensors: dict[str, tuple[tuple[int, ...], np.ndarray]],
) -> bytes:
    """Assemble a minimal PyTorch checkpoint pickle: {cfg, model_state}."""
    out = b"\x80\x02"  # PROTO 2
    out += b"}"  # top-level dict
    out += b"("  # MARK: top-level items
    # cfg value: a plain dict
    out += _binunicode("cfg")
    out += b"}" + b"("
    for k, v in cfg.items():
        out += _binunicode(k) + _int32(v)
    out += b"u"
    # model_state: OrderedDict via GLOBAL + REDUCE, populated via SETITEMS
    out += _binunicode("model_state")
    out += _global("collections", "OrderedDict")
    out += b")R"  # EMPTY_TUPLE + REDUCE → OrderedDict()
    out += b"("  # MARK: key/value items
    for storage_key, (name, (size, data)) in enumerate(tensors.items()):
        out += _binunicode(name)
        stride = tuple(int(np.prod(size[i + 1 :])) for i in range(len(size)))
        out += _tensor_rebuild(str(storage_key), data.size, size, stride)
    out += b"u"  # SETITEMS on the OrderedDict
    out += b"u"  # SETITEMS on the top-level dict (cfg, model_state)
    out += b"."  # STOP
    return out


def _write_pt(path: Path, cfg: dict[str, int], tensors: dict[str, tuple[tuple[int, ...], np.ndarray]]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", _build_pt_bytes(cfg, tensors))
        for i, (_name, (_size, data)) in enumerate(tensors.items()):
            zf.writestr(f"archive/data/{i}", data.astype("<f4").tobytes())
    return path


@pytest.fixture()
def bdh_pt(tmp_path: Path) -> Path:
    """BDH-layout checkpoint: nh=2, D=4, mult=3 → unit=2, N=6 per head."""
    rng = np.random.default_rng(42)
    nh, d, mult, vocab = 2, 4, 3, 8
    n = mult * (d // nh)
    tensors = {
        "decoder": ((nh * n, d), rng.standard_normal((nh * n, d), dtype=np.float32)),
        "encoder": ((nh, d, n), rng.standard_normal((nh, d, n), dtype=np.float32)),
        "encoder_v": ((nh, d, n), rng.standard_normal((nh, d, n), dtype=np.float32)),
        "lm_head": ((d, vocab), rng.standard_normal((d, vocab), dtype=np.float32)),
        "attn.freqs": ((1, 1, 1, n), rng.standard_normal((1, 1, 1, n), dtype=np.float32)),
        "embed.weight": ((vocab, d), rng.standard_normal((vocab, d), dtype=np.float32)),
    }
    cfg = {
        "n_head": nh,
        "n_embd": d,
        "mlp_internal_dim_multiplier": mult,
        "vocab_size": vocab,
        "n_layer": 1,
        "block_size": 8,
    }
    return _write_pt(tmp_path / "bdh_test.pt", cfg, tensors)


# ── Tests ──────────────────────────────────────────────────────────────────


def test_detect_loader_pt_magic(bdh_pt: Path) -> None:
    assert detect_loader(bdh_pt) == "pytorch"


def test_bdh_expansion_names_and_shapes(bdh_pt: Path) -> None:
    loader = PyTorchLoader()
    handles = loader.open(bdh_pt)
    by_name = {h.name: h for h in handles}

    # Monolithic
    assert by_name["encoder"].shape == (2, 4, 6)
    assert by_name["decoder"].shape == (12, 4)
    # Per-head (blk.{h}.Name)
    assert by_name["blk.0.encoder"].shape == (4, 6)
    assert by_name["blk.1.decoder"].shape == (6, 4)
    # Per-unit: 3 units x 2 heads per core tensor
    for u in range(3):
        for h in range(2):
            assert by_name[f"encoder.u{u}.h{h}"].shape == (4, 2)
            assert by_name[f"decoder.u{u}.h{h}"].shape == (2, 4)
            assert by_name[f"encoder.u{u}.h{h}"].expert_id == u
    # Non-core tensors stay monolithic
    assert by_name["embed.weight"].shape == (8, 4)
    assert by_name["lm_head"].shape == (4, 8)
    assert "blk.0.embed.weight" not in by_name
    assert loader.metadata["bdh_n_heads"] == "2"
    assert loader.metadata["bdh_unit"] == "2"
    assert loader.metadata["bdh_mult"] == "3"


def test_bdh_slice_values(bdh_pt: Path) -> None:
    loader = PyTorchLoader()
    handles = {h.name: h for h in loader.open(bdh_pt)}
    enc = handles["encoder"].load()
    dec = handles["decoder"].load()
    # encoder (nh, D, N): head slice along axis 0, unit slice along axis 2
    assert np.array_equal(handles["blk.1.encoder"].load(), enc[1])
    assert np.array_equal(handles["encoder.u2.h0"].load(), enc[0][:, 4:6])
    # decoder (nh*N, D) head-major: head 1 rows 6:12, unit 1 → rows 8:10
    assert np.array_equal(handles["blk.1.decoder"].load(), dec[6:12])
    assert np.array_equal(handles["decoder.u1.h1"].load(), dec[6 + 2 : 6 + 4])
    # Cache: repeated loads share the storage array (identical values)
    assert np.array_equal(handles["encoder.u0.h0"].load(), enc[0][:, 0:2])


def test_plain_pt_passthrough(tmp_path: Path) -> None:
    tensors = {
        "dense.weight": ((4, 4), np.arange(16, dtype=np.float32).reshape(4, 4)),
    }
    pt = _write_pt(tmp_path / "plain.pt", {"n_layer": 1}, tensors)
    loader = PyTorchLoader()
    handles = loader.open(pt)
    assert [h.name for h in handles] == ["dense.weight"]
    assert handles[0].shape == (4, 4)
    assert np.array_equal(handles[0].load(), np.arange(16, dtype=np.float32).reshape(4, 4))
    assert loader.metadata == {}


def test_bdh_name_mapping() -> None:
    # Monolithic + per-unit names map to bdh slots (layer None);
    # per-head names ride the gguf layer pattern.
    assert map_name("encoder") == (None, "bdh_encoder")
    assert map_name("encoder_v") == (None, "bdh_encoder_v")
    assert map_name("decoder") == (None, "bdh_decoder")
    assert map_name("encoder.u0.h1") == (None, "bdh_encoder")
    assert map_name("decoder.u3.h0") == (None, "bdh_decoder")
    assert map_name("attn.freqs") == (None, "rope_freqs")
    assert map_name("embed.weight") == (None, "embed")
    assert map_name("lm_head") == (None, "lm_head")
    assert map_name("blk.0.encoder") == (0, "bdh_encoder")
    assert map_name("blk.7.encoder_v") == (7, "bdh_encoder_v")
    assert map_name("blk.3.decoder") == (3, "bdh_decoder")
    # Generic HF/GGUF names must be unaffected by the bdh rules
    assert map_name("model.layers.5.self_attn.q_proj.weight")[1] == "attn_q"
    assert map_name("blk.2.attn_q")[1] == "attn_q"
    # T5-style encoder stacks keep their pre-bdh mapping (layer via "layer.0")
    assert map_name("encoder.block.0.layer.0.weight") == (0, "other")
    assert map_name("model.encoder.dense.weight") == (None, "other")


def test_bdh_lattice_rasterization(bdh_pt: Path, tmp_path: Path) -> None:
    loader = PyTorchLoader()
    loader.open(bdh_pt)
    spec = AtlasSpec.from_json(Path("specs/atlas_spec.v2.4.json"))

    stats = [
        TensorStats(name="encoder.u0.h0", shape=(4, 2), spectral_norm=1.0, stable_rank=0.5),
        TensorStats(name="encoder.u1.h0", shape=(4, 2), spectral_norm=2.0, stable_rank=0.6),
        TensorStats(name="encoder.u0.h1", shape=(4, 2), spectral_norm=3.0, stable_rank=0.7),
        TensorStats(name="decoder.u2.h1", shape=(2, 4), spectral_norm=4.0, stable_rank=0.8),
    ]
    panels = rasterize_bdh_lattice(stats, spec, "spectral_norm")
    assert [p.slot for p in panels] == ["bdh_encoder", "bdh_decoder"]
    enc = panels[0]
    assert enc.data.shape == (2, 3)  # heads x units
    assert enc.row_labels == ["0", "1"]
    assert enc.col_labels == ["0", "1", "2"]
    assert enc.data[0, 0] == 1.0 and enc.data[0, 1] == 2.0
    assert enc.data[1, 0] == 3.0 and np.isnan(enc.data[1, 1])
    dec = panels[1]
    assert dec.data.shape == (2, 3)
    assert np.isnan(dec.data[0, 2]) and dec.data[1, 2] == 4.0


def test_flat_field_skips_unit_handles(bdh_pt: Path) -> None:
    spec = AtlasSpec.from_json(Path("specs/atlas_spec.v2.4.json"))
    stats = [
        TensorStats(name="encoder", shape=(2, 4, 6), spectral_norm=9.0),
        TensorStats(name="embed.weight", shape=(8, 4), spectral_norm=1.0),
        TensorStats(name="encoder.u0.h0", shape=(4, 2), spectral_norm=99.0, expert_id=0),
    ]
    field = rasterize_flat(stats, spec, "spectral_norm")
    assert field is not None
    # The unit handle must not collapse into the bdh_encoder column
    idx = field.col_labels.index("bdh_encoder")
    assert field.data[0, idx] == 9.0
