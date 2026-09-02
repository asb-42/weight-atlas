"""Safetensors loader: mmap, sharded, registry-ID ``safetensors``.

Reads tensor payloads at the byte level (rather than via ``safetensors``'
``get_tensor``) so that BF16 tensors load reliably without relying on the
numpy backend's dtype registry, and so that MXFP4 ``weight_packed`` +
``weight_scale`` pairs can be dequantized into a single float32 weight
(see :mod:`weight_atlas.loaders.mxfp4`).
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from weight_atlas.core.registry import register_loader
from weight_atlas.core.types import TensorHandle
from weight_atlas.loaders.mxfp4 import (
    dequantize_mxfp4,
    is_packed_tensor,
    scale_name_for_packed,
    weight_name_for_packed,
)

_HEADER_SIZE_FMT = "<Q"
_HEADER_SIZE_SIZE = 8
# A crafted or corrupt file can claim a multi-GB header and OOM the worker on
# f.read(); real headers max out in the tens of MB even for huge sharded MoE.
_MAX_HEADER_BYTES = 512 * 1024 * 1024

# safetensors dtype string -> little-endian numpy dtype.
_DTYPE_MAP: dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "U64": np.dtype("<u8"),
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "I16": np.dtype("<i2"),
    "U16": np.dtype("<u2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
    "C64": np.dtype("<c8"),
}
_BF16 = "BF16"

# Kimi-K3 MoE expert id extraction for dequantized handles.
_KIMI_EXPERT_RE = re.compile(r"block_sparse_moe\.experts\.(\d+)")


def _read_header(path: Path) -> dict[str, dict[str, Any]]:
    """Read the safetensors JSON header without loading tensor data."""
    header, _ = _read_header_full(path)
    return header


def _read_header_full(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Return ``(header, data_section_offset)`` for a safetensors file.

    ``data_section_offset`` is the absolute file offset where tensor payloads
    begin (immediately after the 8-byte length + JSON header).
    """
    file_len = path.stat().st_size
    with open(path, "rb") as f:
        size = struct.unpack(_HEADER_SIZE_FMT, f.read(_HEADER_SIZE_SIZE))[0]
        if size > _MAX_HEADER_BYTES:
            raise ValueError(
                f"{path}: safetensors header claims {size} bytes "
                f"(limit {_MAX_HEADER_BYTES}) — corrupt or malicious file"
            )
        if _HEADER_SIZE_SIZE + size > file_len:
            raise ValueError(
                f"{path}: safetensors header ({size} bytes) exceeds file size "
                f"({file_len} bytes)"
            )
        data = json.loads(f.read(size))
    header = {k: v for k, v in data.items() if isinstance(v, dict)}
    return header, _HEADER_SIZE_SIZE + size


def _validate_offsets(path: Path, header: dict[str, dict[str, Any]], data_offset: int) -> None:
    """Reject headers whose tensor offsets fall outside the data section.

    A negative start would seek backwards into the JSON header and decode
    header bytes as weights; an oversized end yields a truncated read that
    fails later with a cryptic reshape error.
    """
    data_len = path.stat().st_size - data_offset
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            start, end = info["data_offsets"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: malformed header entry for {name!r}") from exc
        start, end = int(start), int(end)
        if not (0 <= start <= end <= data_len):
            raise ValueError(
                f"{path}: tensor {name!r} has invalid data_offsets "
                f"[{start}, {end}] against a {data_len}-byte data section"
            )


def _read_tensor_raw(path: Path, header: dict, name: str, data_offset: int) -> bytes:
    """Read the raw little-endian payload bytes for a single tensor."""
    start, end = header[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_offset + start)
        return f.read(end - start)


def _read_tensor_uint8(path: Path, header: dict, name: str, data_offset: int) -> np.ndarray:
    """Read a tensor's payload as a uint8 array (used for MXFP4 pairs)."""
    raw = _read_tensor_raw(path, header, name, data_offset)
    shape = tuple(int(x) for x in header[name]["shape"])
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def _from_raw(raw: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """Interpret raw little-endian bytes as a float32 array (BF16 aware).

    BF16 (16-bit: sign|8-bit exp|7-bit mantissa) widens to float32 by shifting
    the bit pattern left 16 — no bf16 numpy dtype registration required.
    """
    if dtype == _BF16:
        bits = np.frombuffer(raw, dtype=np.dtype("<u2"))
        f32 = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
        return f32.reshape(shape).copy()
    np_dtype = _DTYPE_MAP.get(dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}")
    arr = np.frombuffer(raw, dtype=np_dtype)
    return arr.reshape(shape).astype(np.float32, copy=False)


def _load_tensor(path: Path, name: str) -> np.ndarray:
    """Materialise a single tensor as float32 (self-contained, header re-read)."""
    header, data_offset = _read_header_full(path)
    raw = _read_tensor_raw(path, header, name, data_offset)
    return _from_raw(raw, str(header[name]["dtype"]), tuple(int(x) for x in header[name]["shape"]))


def _load_mxfp4_pair(path: Path, header: dict, packed_name: str, scale_name: str, data_offset: int) -> np.ndarray:
    """Dequantize an MXFP4 ``weight_packed`` + ``weight_scale`` pair to float32."""
    packed = _read_tensor_uint8(path, header, packed_name, data_offset)
    scale = _read_tensor_uint8(path, header, scale_name, data_offset)
    return dequantize_mxfp4(packed, scale)


def _discover_files(path: Path) -> list[Path]:
    """Return sorted list of .safetensors files for a file or directory."""
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")
    return files


@register_loader("safetensors")
class SafetensorsLoader:
    """Memory-mapped safetensors loader supporting sharded models."""

    format_id = "safetensors"

    def source_files(self, path: Path) -> list[Path]:
        """Same discovery as ``open`` — the shards the scan hashes."""
        return _discover_files(path)

    def open(self, path: Path) -> list[TensorHandle]:
        files = _discover_files(path)
        seen: dict[str, Path] = {}
        self.metadata: dict[str, str] = {}
        # Row-major payload ranges for row-block streaming (giant tensors).
        self._tensor_ranges: dict[str, tuple[Path, str, tuple[int, ...], int, int, int]] = {}

        handles: list[TensorHandle] = []
        for f in files:
            header, data_offset = _read_header_full(f)
            self.metadata.update(header.pop("__metadata__", {}))
            _validate_offsets(f, header, data_offset)
            entries: list[tuple[str, tuple[int, ...], str, int, int]] = []
            for name, info in header.items():
                if name in seen:
                    raise ValueError(f"duplicate tensor name {name!r} in {f} and {seen[name]}")
                seen[name] = f
                shape = tuple(int(x) for x in info["shape"])
                dtype = str(info["dtype"])
                start, end = info["data_offsets"]
                entries.append((name, shape, dtype, start, end))
                if len(shape) == 2 and dtype in ("F32", "F16", "BF16"):
                    self._tensor_ranges[name] = (f, dtype, shape, data_offset + start, end - start, 0)

            # Names in this file for MXFP4 pair resolution.
            names_in_file = {e[0] for e in entries}
            consumed: set[str] = set()

            for name, shape, dtype, _start, _end in entries:
                if name in consumed:
                    continue  # already merged into a dequantized MXFP4 handle
                # MXFP4 packed tensor with a scale sibling in the same shard.
                if is_packed_tensor(name):
                    base = weight_name_for_packed(name)
                    scale_name = scale_name_for_packed(name)
                    if scale_name in names_in_file:
                        m, k2 = shape[0], shape[1]
                        handles.append(
                            TensorHandle(
                                name=base,
                                shape=(m, k2 * 2),
                                dtype="FP4_MXFP4",
                                expert_id=_kimi_expert_id(base),
                                loader=partial(_load_mxfp4_pair, f, header, name, scale_name, data_offset),
                            )
                        )
                        consumed.add(name)
                        consumed.add(scale_name)
                        continue

                handles.append(
                    TensorHandle(
                        name=name,
                        shape=shape,
                        dtype=dtype,
                        loader=partial(_load_named, f, header, name, data_offset, dtype, shape),
                    )
                )

        return handles

    def iter_row_blocks(
        self, handle: TensorHandle, rows_per_block: int
    ) -> Iterator[np.ndarray] | None:
        """Yield float32 row blocks of a 2-D F32/F16/BF16 tensor from its
        row-major file payload (memory-mapped reads, no materialization)."""
        info = self._tensor_ranges.get(handle.name)
        if info is None:
            return None
        f, dtype, shape, abs_start, _length, _zero = info
        rows, cols = handle.shape
        itemsize = {"F32": 4, "F16": 2, "BF16": 2}[dtype]

        def _gen() -> Iterator[np.ndarray]:
            with open(f, "rb") as fh:
                for lo in range(0, rows, rows_per_block):
                    hi = min(lo + rows_per_block, rows)
                    fh.seek(abs_start + lo * cols * itemsize)
                    raw = fh.read((hi - lo) * cols * itemsize)
                    if dtype == "BF16":
                        bits = np.frombuffer(raw, dtype="<u2")
                        yield (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(hi - lo, cols)
                    elif dtype == "F16":
                        yield np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(hi - lo, cols)
                    else:
                        yield np.frombuffer(raw, dtype="<f4").reshape(hi - lo, cols)

        return _gen()


def _load_named(path: Path, header: dict, name: str, data_offset: int, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = _read_tensor_raw(path, header, name, data_offset)
    return _from_raw(raw, dtype, shape)


def _kimi_expert_id(name: str) -> int | None:
    m = _KIMI_EXPERT_RE.search(name)
    return int(m.group(1)) if m else None
