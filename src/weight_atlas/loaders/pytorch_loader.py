"""PyTorch loader: pure-python unpickler, registry-ID ``pytorch``.

Reads ``.pt`` checkpoint files (PyTorch ZIP format) without requiring the
``torch`` package.  The loader parses the pickle in ``data.pkl``, extracts
tensor metadata (name, shape, dtype, storage key), and reads raw float32
payloads from the ``data/`` files inside the ZIP archive.

Only model weights are returned; optimizer state, step counters, and other
checkpoint metadata are discarded.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from weight_atlas.core.registry import register_loader
from weight_atlas.core.types import TensorHandle

# ── Unpickler helpers ──────────────────────────────────────────────────────


class _StorageTypeStub:
    """Minimal stand-in for ``torch.FloatStorage`` / ``torch.IntStorage`` etc.

    The pickle stream references these classes via ``GLOBAL`` opcodes; the
    stub records the dtype so the rebuild function can validate storage size.
    """

    _DTYPE_MAP: dict[str, str] = {
        "FloatStorage": "float32",
        "DoubleStorage": "float64",
        "HalfStorage": "float16",
        "BFloat16Storage": "bfloat16",
        "LongStorage": "int64",
        "IntStorage": "int32",
        "ShortStorage": "int16",
        "ByteStorage": "uint8",
    }

    def __init__(self, name: str) -> None:
        self.dtype = self._DTYPE_MAP.get(name, "unknown")


class _MinimalPtUnpickler(pickle.Unpickler):
    """Unpickler that captures tensor metadata without importing torch.

    Handles the subset of pickle opcodes that PyTorch uses for its
    ``data.pkl`` archives: ``GLOBAL`` references to
    ``torch._utils._rebuild_tensor_v2`` and ``torch.FloatStorage`` / etc.,
    plus ``BINPERSID`` persistent-load tuples.

    After ``.load()``, the ``.tensors`` list contains one dict per tensor
    (in pickle opcode order) with keys: ``zip_path``, ``dtype``, ``shape``,
    ``offset``, ``numel``.
    """

    def __init__(self, file: io.BytesIO, prefix: str) -> None:
        super().__init__(file)
        self.prefix = prefix
        self.tensors: list[dict] = []

    def persistent_load(self, saved_id: object) -> object:
        # PyTorch encodes storage references as tuples:
        #   ('storage', StorageTypeStub, key_str, location_str, numel)
        # _rebuild_tensor_v2 unpacks this tuple, so we pass it through.
        return saved_id

    def find_class(self, module: str, name: str) -> object:
        # torch storage types → stub
        if module == "torch" and "Storage" in name:
            return type("StorageStub", (), {"__init__": lambda self, n=name: setattr(self, "name", n)})()

        # Rebuild function → capture tensor metadata
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return self._rebuild_tensor_v2

        # Parameter rebuilders → passthrough (return first arg)
        if module == "torch._utils" and "parameter" in name.lower():
            return lambda *a, **kw: a[0] if a else None

        # collections.OrderedDict → stdlib
        if module == "collections" and name == "OrderedDict":
            return OrderedDict

        # Any other torch module → no-op
        if module.startswith("torch"):
            return lambda *a, **kw: None

        return super().find_class(module, name)

    def _rebuild_tensor_v2(
        self,
        storage: object,
        storage_offset: int,
        size: Sequence[int],
        stride: Sequence[int],
        requires_grad: bool,
        backward_hooks: object,
        metadata: object = None,
    ) -> None:
        if not isinstance(storage, tuple) or len(storage) < 5:
            return
        _tag, stype, key, _location, numel = storage[:5]
        dtype = getattr(stype, "dtype", "float32")
        self.tensors.append({
            "zip_path": f"{self.prefix}data/{key}",
            "dtype": dtype,
            "shape": tuple(int(x) for x in size),
            "offset": int(storage_offset),
            "numel": int(numel),
        })


# ── Loader ─────────────────────────────────────────────────────────────────


def _discover_pt_files(path: Path) -> list[Path]:
    """Find ``.pt`` files in a directory, or return ``[path]`` for a single file."""
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no .pt files in {path}")
    return files


# Dtype string → numpy dtype
_NUMPY_DTYPES: dict[str, type] = {
    "float32": np.float32,
    "float64": np.float64,
    "float16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "uint8": np.uint8,
}


def _parse_pt_file(pt_path: Path) -> list[tuple[str, dict]]:
    """Parse a .pt file and return [(name, tensor_metadata)] for model weights."""
    with zipfile.ZipFile(pt_path) as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not pkl_names:
            raise ValueError(f"no data.pkl in {pt_path}")
        pkl_name = pkl_names[0]
        prefix = pkl_name[: -len("data.pkl")]
        pkl_bytes = zf.read(pkl_name)

    # Unpickle to capture tensor metadata AND the result dict
    up = _MinimalPtUnpickler(io.BytesIO(pkl_bytes), prefix)
    result = up.load()

    # Extract model_state keys from the unpickled result
    model_state = {}
    if isinstance(result, dict):
        model_state = result.get("model_state", {})

    model_keys = list(model_state.keys()) if isinstance(model_state, OrderedDict) else []

    # The first N tensors in up.tensors correspond to model_state keys,
    # because model_state is fully processed before optimizer_state in the
    # pickle stream (SETITEMS order).
    results: list[tuple[str, dict]] = []
    for i, name in enumerate(model_keys):
        if i >= len(up.tensors):
            break
        md = up.tensors[i]
        results.append((name, md))

    return results


@register_loader("pytorch")
class PyTorchLoader:
    """Loader for PyTorch ``.pt`` checkpoint files.

    Reads the ZIP archive, parses ``data.pkl`` to extract model weight
    metadata, and returns lazy ``TensorHandle`` objects that read raw
    float32 payloads from the ``data/`` files on demand.
    """

    format_id = "pytorch"

    def open(self, path: Path) -> list[TensorHandle]:
        files = _discover_pt_files(path)
        handles: list[TensorHandle] = []

        for pt_path in files:
            for name, md in _parse_pt_file(pt_path):
                if md["dtype"] != "float32":
                    continue
                if md["numel"] == 0:
                    continue

                np_dtype = _NUMPY_DTYPES.get(md["dtype"], np.float32)
                handles.append(
                    TensorHandle(
                        name=name,
                        shape=md["shape"],
                        dtype=md["dtype"],
                        loader=_make_loader(pt_path, md["zip_path"], md["numel"], np_dtype, md["shape"]),
                    )
                )

        return handles


def _make_loader(
    pt_path: Path,
    zip_path: str,
    numel: int,
    np_dtype: type,
    shape: tuple[int, ...],
) -> Callable[[], np.ndarray]:
    """Create a lazy loader closure for a tensor stored in a ZIP entry."""
    _path = pt_path
    _entry = zip_path
    _n = numel
    _dt = np_dtype
    _shape = shape

    def _load() -> np.ndarray:
        with zipfile.ZipFile(_path) as zf:
            raw = zf.read(_entry)
        return np.frombuffer(raw, dtype=_dt, count=_n).reshape(_shape)

    return _load
