"""EXL3 loader: safetensors checkpoints with EXL3 trellis groups (registry-ID ``exl3``).

EXL3 (exllamav3, QTIP-derived trellis quantization) stores each quantized
linear as a group ``<prefix>.{trellis, suh, svh, [mcg|mul1]}``; everything
else (embeddings, norms, ...) is stored as plain safetensors tensors --
including BF16, decoded through the shared safetensors helpers. Quantized
groups dequantize lazily to original-basis float32 weights under their
canonical HuggingFace names (``<prefix>.weight``, transposed to the
nn.Linear (out, in) convention).

Group component tensors (suh/svh/trellis/mcg/mul1) are not exposed; the
group resolves to a single reconstructed weight tensor. Handles memoize
through ``TensorHandle.load()`` (the scan pipeline calls ``clear()`` per
tensor to bound RAM), so this loader adds no caching of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from weight_atlas.core.registry import register_loader
from weight_atlas.core.types import TensorHandle
from weight_atlas.loaders.exl3_dequant import (
    CODEBOOK_MCG_MULT,
    CODEBOOK_MUL1_MULT,
    MAX_K,
    EXL3FormatError,
    dequantize_exl3_layer,
)
from weight_atlas.loaders.safetensors_loader import (
    _discover_files,
    _from_raw,
    _read_header_full,
    _validate_offsets,
)

_GROUP_SUFFIXES = (".trellis", ".suh", ".svh", ".mcg", ".mul1", ".su", ".sv")
_MARKER_BITS = {"mcg": CODEBOOK_MCG_MULT, "mul1": CODEBOOK_MUL1_MULT}

# Header dtypes readable as native bits: packed integers (trellis words)
# plus F16 for the per-channel scale tensors. Decoding packed data through
# ``_from_raw`` (which reinterprets to float32) would corrupt bit patterns.
_RAW_BITS_DTYPES = {"I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64", "F16"}
_INT_DTYPES = {
    "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
    "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64,
    "F16": np.float16,
}


def _read_bits(f: Path, base: int, info: dict[str, Any]) -> np.ndarray:
    """Read a tensor's payload in its native dtype (no float reinterpretation).

    Trellis words are packed bit patterns and suh/svh are float16 — both
    must survive byte-identical; ``_from_raw`` (float32 decode) is reserved
    for plain passthrough tensors.
    """
    dtype = str(info["dtype"])
    if dtype not in _RAW_BITS_DTYPES:
        raise EXL3FormatError(f"expected a packed/float16 dtype, got {dtype!r}")
    raw_start, raw_end = info["data_offsets"]
    shape = tuple(int(x) for x in info["shape"])
    with open(f, "rb") as fh:
        fh.seek(base + int(raw_start))
        raw = fh.read(int(raw_end) - int(raw_start))
    arr = np.frombuffer(raw, dtype=_INT_DTYPES[dtype])
    return arr.reshape(shape)


def _group_geometry(trellis_info: dict[str, Any]) -> tuple[int, int, int]:
    """(in_features, out_features, K) from the trellis shape, with validation."""
    shape = tuple(int(x) for x in trellis_info["shape"])
    if len(shape) != 3 or shape[2] % 16 != 0 or not 1 <= shape[2] // 16 <= MAX_K:
        raise EXL3FormatError(f"invalid trellis shape {shape}")
    k = shape[2] // 16
    in_f, out_f = shape[0] * 16, shape[1] * 16
    if in_f % 128 or out_f % 128:
        raise EXL3FormatError(
            f"EXL3 features must be multiples of 128, got {in_f}x{out_f}"
        )
    return in_f, out_f, k


def _marker_value(path: Path, base: int, info: dict[str, Any]) -> int:
    with open(path, "rb") as f:
        f.seek(base + int(info["data_offsets"][0]))
        return int(np.frombuffer(f.read(4), dtype="<u4").item())


def _codebook_of(
    comp: dict[str, tuple[Path, dict[str, Any]]],
    headers: dict[Path, tuple[dict[str, dict[str, Any]], int]],
) -> str:
    """Codebook id for a group, validated against its marker tensor value."""
    mcg = comp.get("mcg")
    mul1 = comp.get("mul1")
    if mcg is not None and mul1 is not None:
        raise EXL3FormatError("EXL3 group carries both mcg and mul1 markers")
    for key, entry in (("mcg", mcg), ("mul1", mul1)):
        if entry is None:
            continue
        f, info = entry
        value = _marker_value(f, headers[f][1], info)
        if value != _MARKER_BITS[key]:
            raise EXL3FormatError(
                f"unknown {key} codebook marker 0x{value:08X} "
                f"(expected 0x{_MARKER_BITS[key]:08X})"
            )
        return key
    return "plain"


def _scan_groups(
    files: list[Path],
    headers: dict[Path, tuple[dict[str, dict[str, Any]], int]],
) -> tuple[
    dict[str, dict[str, tuple[Path, dict[str, Any]]]],
    list[tuple[Path, str, dict[str, Any]]],
]:
    """Split tensor names into EXL3 groups (by prefix) and plain tensors."""
    groups: dict[str, dict[str, tuple[Path, dict[str, Any]]]] = {}
    plain: list[tuple[Path, str, dict[str, Any]]] = []
    seen: dict[str, Path] = {}
    for f in files:
        infos, _ = headers[f]
        for name, info in infos.items():
            if name in seen:
                raise ValueError(f"duplicate tensor name {name!r} in {f} and {seen[name]}")
            seen[name] = f
            for suffix in _GROUP_SUFFIXES:
                if name.endswith(suffix):
                    groups.setdefault(name[: -len(suffix)], {})[suffix[1:]] = (f, info)
                    break
            else:
                plain.append((f, name, info))
    return groups, plain


def _resolve_comp(
    prefix: str,
    comp: dict[str, tuple[Path, dict[str, Any]]],
    headers: dict[Path, tuple[dict[str, dict[str, Any]], int]],
) -> tuple[
    tuple[Path, dict[str, Any]],
    tuple[Path, dict[str, Any]],
    tuple[Path, dict[str, Any]],
    int,
    int,
    int,
    str,
]:
    """Validate one group; return its component entries + geometry + codebook."""
    if "trellis" not in comp:
        raise EXL3FormatError(f"EXL3 group {prefix!r} has no trellis tensor")
    trellis_entry = comp["trellis"]
    suh_entry = comp.get("suh")
    svh_entry = comp.get("svh")
    if suh_entry is None or svh_entry is None:
        if "su" in comp or "sv" in comp:
            raise EXL3FormatError(
                f"EXL3 group {prefix!r} uses packed sign tensors (su/sv); "
                "only unpacked suh/svh are supported"
            )
        raise EXL3FormatError(f"EXL3 group {prefix!r} is missing suh/svh")
    in_f, out_f, k = _group_geometry(trellis_entry[1])
    codebook = _codebook_of(comp, headers)
    for info, n in ((suh_entry[1], in_f), (svh_entry[1], out_f)):
        if tuple(int(x) for x in info["shape"]) != (n,):
            raise EXL3FormatError(
                f"{prefix}: scale tensor shape {info['shape']} != ({n},)"
            )
    return trellis_entry, suh_entry, svh_entry, in_f, out_f, k, codebook


def _load_group_weight(
    headers: dict[Path, tuple[dict[str, dict[str, Any]], int]],
    trellis_entry: tuple[Path, dict[str, Any]],
    suh_entry: tuple[Path, dict[str, Any]],
    svh_entry: tuple[Path, dict[str, Any]],
    codebook: str,
) -> np.ndarray:
    """Dequantize one EXL3 group to float32, HF (out, in) convention."""
    f, info = trellis_entry
    trellis = _read_bits(f, headers[f][1], info).astype(np.int16, copy=False)
    f, info = suh_entry
    suh = _read_bits(f, headers[f][1], info).astype(np.float16, copy=False)
    f, info = svh_entry
    svh = _read_bits(f, headers[f][1], info).astype(np.float16, copy=False)
    w = dequantize_exl3_layer(trellis, suh, svh, codebook)  # (in, out)
    return np.ascontiguousarray(w.T)


@register_loader("exl3")
class EXL3Loader:
    """Lazy EXL3 loader; groups dequantize on demand via TensorHandle memoization."""

    format_id = "exl3"

    def source_files(self, path: Path) -> list[Path]:
        """Same discovery as ``open`` — the shards the scan hashes."""
        return _discover_files(path)

    def open(self, path: Path) -> list[TensorHandle]:
        files = _discover_files(path)
        headers = {f: _read_header_full(f) for f in files}
        for f in files:
            _validate_offsets(f, headers[f][0], headers[f][1])
        self.metadata: dict[str, str] = {}
        for f in files:
            # Surface the file-level __metadata__ block like the safetensors
            # loader does; it is not a tensor and must not reach _scan_groups.
            self.metadata.update(headers[f][0].pop("__metadata__", {}))
        groups, plain = _scan_groups(files, headers)

        handles = [
            self._plain_handle(f, name, info, headers[f][1])
            for f, name, info in plain
        ]
        handles += [
            self._group_handle(prefix, groups[prefix], headers)
            for prefix in sorted(groups)
        ]
        return handles

    def _plain_handle(
        self,
        f: Path,
        name: str,
        info: dict[str, Any],
        base: int,
    ) -> TensorHandle:
        def load() -> np.ndarray:
            raw_start, raw_end = info["data_offsets"]
            with open(f, "rb") as fh:
                fh.seek(base + int(raw_start))
                raw = fh.read(int(raw_end) - int(raw_start))
            return _from_raw(raw, str(info["dtype"]), tuple(int(x) for x in info["shape"]))

        return TensorHandle(
            name=name,
            shape=tuple(int(x) for x in info["shape"]),
            dtype=str(info["dtype"]),
            loader=load,
        )

    def _group_handle(
        self,
        prefix: str,
        comp: dict[str, tuple[Path, dict[str, Any]]],
        headers: dict[Path, tuple[dict[str, dict[str, Any]], int]],
    ) -> TensorHandle:
        trellis_entry, suh_entry, svh_entry, in_f, out_f, k, codebook = _resolve_comp(
            prefix, comp, headers
        )
        name = f"{prefix}.weight"

        def load() -> np.ndarray:
            return _load_group_weight(headers, trellis_entry, suh_entry, svh_entry, codebook)

        return TensorHandle(
            name=name,
            shape=(out_f, in_f),
            dtype=f"exl3_K{k}_{codebook}",
            loader=load,
        )


def read_exl3_group(
    model_path: Path, prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Read one EXL3 quant group directly: (trellis, suh, svh, codebook id)."""
    files = _discover_files(model_path)
    headers = {f: _read_header_full(f) for f in files}
    groups, _ = _scan_groups(files, headers)
    if prefix not in groups:
        raise KeyError(f"no EXL3 group {prefix!r} under {model_path}")
    trellis_entry, suh_entry, svh_entry, _in_f, _out_f, _k, codebook = _resolve_comp(
        prefix, groups[prefix], headers
    )
    f, info = trellis_entry
    trellis = _read_bits(f, headers[f][1], info).astype(np.int16, copy=False)
    f, info = suh_entry
    suh = _read_bits(f, headers[f][1], info).astype(np.float16, copy=False)
    f, info = svh_entry
    svh = _read_bits(f, headers[f][1], info).astype(np.float16, copy=False)
    return trellis, suh, svh, codebook
