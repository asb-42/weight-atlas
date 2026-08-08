"""Safetensors loader: mmap, sharded, registry-ID ``safetensors``."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from weight_atlas.core.registry import register_loader
from weight_atlas.core.types import TensorHandle

_HEADER_SIZE_FMT = "<Q"
_HEADER_SIZE_SIZE = 8


def _read_header(path: Path) -> dict[str, dict[str, Any]]:
    """Read the safetensors JSON header without loading tensor data."""
    with open(path, "rb") as f:
        size = struct.unpack(_HEADER_SIZE_FMT, f.read(_HEADER_SIZE_SIZE))[0]
        data = json.loads(f.read(size))
    return {k: v for k, v in data.items() if isinstance(v, dict)}


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

    def open(self, path: Path) -> list[TensorHandle]:
        files = _discover_files(path)
        seen: dict[str, Path] = {}
        entries: list[tuple[Path, str, tuple[int, ...], str]] = []
        for f in files:
            header = _read_header(f)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                if name in seen:
                    raise ValueError(
                        f"duplicate tensor name {name!r} in {f} and {seen[name]}"
                    )
                seen[name] = f
                shape = tuple(int(x) for x in info["shape"])
                dtype = str(info["dtype"])
                entries.append((f, name, shape, dtype))

        handles: list[TensorHandle] = []
        for f, name, shape, dtype in entries:
            handles.append(
                TensorHandle(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    loader=_make_loader(f, name),
                )
            )
        return handles


def _make_loader(f_path: Path, t_name: str) -> Callable[[], np.ndarray]:
    return lambda: _load_tensor(f_path, t_name)


def _load_tensor(path: Path, name: str) -> np.ndarray:
    """Materialise a single tensor as float32 via the mmap-backed file."""
    with safe_open(path, framework="np") as fp:
        arr = fp.get_tensor(name)
        return np.asarray(arr, dtype=np.float32)
