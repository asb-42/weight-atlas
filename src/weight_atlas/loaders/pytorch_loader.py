"""PyTorch loader: pure-python unpickler, registry-ID ``pytorch``.

Reads ``.pt`` checkpoint files (PyTorch ZIP format) without requiring the
``torch`` package.  The loader parses the pickle in ``data.pkl``, extracts
tensor metadata (name, shape, dtype, storage key), and reads raw float32
payloads from the ``data/`` files inside the ZIP archive.

Only model weights are returned; optimizer state, step counters, and other
checkpoint metadata are discarded.

BDH layout expansion: when the checkpoint matches the BDH (Dragon Hatchling)
layout — ``encoder``/``encoder_v`` as ``(nh, D, N)`` with the neuron axis
last, ``decoder`` as ``(nh*N, D)`` head-major, plus a cfg with
``n_head``/``n_embd``/``mlp_internal_dim_multiplier`` — the loader emits
three granularities:

- monolithic tensors (``encoder``, ``decoder``, ...) — whole-tensor stats,
- per-head tensors (``blk.{h}.encoder`` ...) — populate the main raster as
  heads x slots,
- per-lattice-unit tensors (``encoder.u{u}.h{h}`` ..., ``expert_id=u``) —
  one unit = ``n_embd // n_head`` neurons per head (the route lattice);
  feed :func:`weight_atlas.fields.rasterizer.rasterize_bdh_lattice` panels.

Per-unit and per-head handles slice from an instance-level float32 cache of
each storage entry (the ZIP payload is read once per storage), so peak RAM
is the full model in float32.
"""

from __future__ import annotations

import io
import pickle
import re
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from weight_atlas.core.registry import register_loader
from weight_atlas.core.types import TensorHandle

# Head index suffix on BDH per-unit tensor names.
_UNIT_HEAD_RE = re.compile(r"\.h(\d+)$")

_BDH_CORE_KEYS = ("encoder", "encoder_v", "decoder")


# ── Unpickler helpers ──────────────────────────────────────────────────────


class _StorageTypeStub:
    """Stand-in for ``torch.FloatStorage`` etc. pushed by GLOBAL opcodes.

    The pickle stream references these classes without instantiating them
    (they end up inside the persistent-id tuple), so the stub only needs to
    carry the dtype as a class attribute.
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


def _storage_stub_class(name: str) -> type:
    return type(name, (_StorageTypeStub,), {"dtype": _StorageTypeStub._DTYPE_MAP.get(name, "unknown")})


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
        # torch storage types → stub classes carrying the dtype
        if module == "torch" and "Storage" in name:
            return _storage_stub_class(name)

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


# ── Checkpoint parsing ─────────────────────────────────────────────────────


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


def _parse_pt_file(pt_path: Path) -> tuple[dict, list[tuple[str, dict]]]:
    """Parse a .pt file into (cfg, [(name, tensor_metadata)]) for model weights."""
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

    cfg: dict = {}
    model_keys: list[str] = []
    if isinstance(result, dict):
        cfg = result.get("cfg", {}) or {}
        model_state = result.get("model_state", {})
        if isinstance(model_state, OrderedDict):
            model_keys = list(model_state.keys())

    # The first N tensors in up.tensors correspond to model_state keys,
    # because model_state is fully processed before optimizer_state in the
    # pickle stream (SETITEMS order).
    named: list[tuple[str, dict]] = []
    for i, name in enumerate(model_keys):
        if i >= len(up.tensors):
            break
        named.append((name, up.tensors[i]))

    return cfg, named


# ── BDH layout expansion ───────────────────────────────────────────────────


def _bdh_layout(cfg: dict, shapes: dict[str, tuple[int, ...]]) -> dict | None:
    """Detect the BDH layout and return expansion parameters, else None.

    Requires: cfg with ``n_head``/``n_embd``/``mlp_internal_dim_multiplier``,
    3D ``encoder``/``encoder_v`` ``(nh, D, N)`` and 2D ``decoder``
    ``(nh*N, D)`` whose dims agree with the cfg.
    """
    try:
        nh = int(cfg["n_head"])
        d = int(cfg["n_embd"])
        mult = int(cfg["mlp_internal_dim_multiplier"])
    except (KeyError, TypeError, ValueError):
        return None
    if nh <= 0 or d <= 0 or mult <= 0 or d % nh != 0:
        return None
    enc = shapes.get("encoder")
    enc_v = shapes.get("encoder_v")
    dec = shapes.get("decoder")
    if enc is None or enc_v is None or dec is None:
        return None
    n = mult * (d // nh)
    if enc != (nh, d, n) or enc_v != (nh, d, n) or dec != (nh * n, d):
        return None
    return {"n_heads": nh, "d_model": d, "mult": mult, "unit": d // nh, "n_per_head": n}


def _range_loader(
    cache: dict[str, np.ndarray],
    pt_path: Path,
    zip_path: str,
    np_dtype: type,
    numel: int,
    offset: int,
    shape: tuple[int, ...],
) -> Callable[[], np.ndarray]:
    """Loader closure for one contiguous element range of a storage.

    The first access reads the whole ZIP entry, converts to float32, and
    memoizes it on the owning loader instance; later slices are views.
    """
    _key = zip_path
    _start = offset
    _count = int(np.prod(shape)) if shape else 1

    def _load() -> np.ndarray:
        arr = cache.get(_key)
        if arr is None:
            with zipfile.ZipFile(pt_path) as zf:
                raw = zf.read(_key)
            arr = np.frombuffer(raw, dtype=np_dtype, count=numel).astype(np.float32)
            cache[_key] = arr
        return arr.reshape(-1)[_start : _start + _count].reshape(shape)

    return _load


def _head_column_loader(
    cache: dict[str, np.ndarray],
    pt_path: Path,
    zip_path: str,
    np_dtype: type,
    numel: int,
    head_start: int,
    head_shape: tuple[int, int],
    col_lo: int,
    col_hi: int,
) -> Callable[[], np.ndarray]:
    """Loader for a column slice of one head block of ``encoder``/``encoder_v``.

    The (nh, D, N) tensor is contiguous per head; the neuron axis is the
    last dim, so a lattice unit is ``head[:, col_lo:col_hi]`` — a strided
    view materialized as a contiguous copy.
    """
    _key = zip_path
    _start = head_start
    _count = head_shape[0] * head_shape[1]

    def _load() -> np.ndarray:
        arr = cache.get(_key)
        if arr is None:
            with zipfile.ZipFile(pt_path) as zf:
                raw = zf.read(_key)
            arr = np.frombuffer(raw, dtype=np_dtype, count=numel).astype(np.float32)
            cache[_key] = arr
        head = arr.reshape(-1)[_start : _start + _count].reshape(head_shape)
        return np.ascontiguousarray(head[:, col_lo:col_hi])

    return _load


def _monolithic_loader(
    pt_path: Path,
    zip_path: str,
    numel: int,
    np_dtype: type,
    shape: tuple[int, ...],
) -> Callable[[], np.ndarray]:
    """Loader closure reading a full storage entry and reshaping (no cache)."""
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


# ── Loader ─────────────────────────────────────────────────────────────────


@register_loader("pytorch")
class PyTorchLoader:
    """Loader for PyTorch ``.pt`` checkpoint files.

    Reads the ZIP archive, parses ``data.pkl`` to extract model weight
    metadata, and returns lazy ``TensorHandle`` objects.  BDH-layout
    checkpoints are expanded per head and per route-lattice unit (see
    module docstring); other layouts pass through monolithically.
    """

    format_id = "pytorch"

    def __init__(self) -> None:
        # Full-storage float32 cache shared by per-head/per-unit slices.
        self._storage_cache: dict[str, np.ndarray] = {}
        self.metadata: dict[str, str] = {}

    def open(self, path: Path) -> list[TensorHandle]:
        files = _discover_pt_files(path)
        handles: list[TensorHandle] = []

        for pt_path in files:
            cfg, named = _parse_pt_file(pt_path)
            shapes = {name: md["shape"] for name, md in named}
            layout = _bdh_layout(cfg, shapes)
            if layout:
                self._open_bdh(pt_path, named, layout, handles)
            else:
                self._open_plain(pt_path, named, handles)

        return handles

    def _open_plain(self, pt_path: Path, named: list[tuple[str, dict]], handles: list[TensorHandle]) -> None:
        for name, md in named:
            if md["dtype"] != "float32" or md["numel"] == 0:
                continue
            handles.append(
                TensorHandle(
                    name=name,
                    shape=md["shape"],
                    dtype=md["dtype"],
                    loader=_monolithic_loader(
                        pt_path, md["zip_path"], md["numel"],
                        _NUMPY_DTYPES.get(md["dtype"], np.float32), md["shape"],
                    ),
                )
            )

    def _open_bdh(
        self,
        pt_path: Path,
        named: list[tuple[str, dict]],
        layout: dict,
        handles: list[TensorHandle],
    ) -> None:
        nh, n, unit = layout["n_heads"], layout["n_per_head"], layout["unit"]
        self.metadata.update({
            "bdh_n_heads": str(nh),
            "bdh_unit": str(unit),
            "bdh_mult": str(layout["mult"]),
            "bdh_decoder_layout": "head-major (h*N + n)",
        })
        for name, md in named:
            if md["dtype"] != "float32" or md["numel"] == 0:
                continue
            dtype = md["dtype"]
            shape = md["shape"]
            np_dtype = _NUMPY_DTYPES.get(dtype, np.float32)
            base = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "loader": _monolithic_loader(pt_path, md["zip_path"], md["numel"], np_dtype, shape),
            }
            if name in _BDH_CORE_KEYS:
                handles.append(TensorHandle(**base))
                d_model = layout["d_model"]
                # encoder/encoder_v (nh, D, N): head h is the contiguous
                # element range [h*D*N, (h+1)*D*N); a lattice unit is a
                # column slice of that head. decoder (nh*N, D) is head-major:
                # head h is rows [h*N, (h+1)*N), unit (h, u) is the
                # contiguous range [(h*N + u*unit)*D, ...).
                head_elems = d_model * n
                for h in range(nh):
                    head_start = md["offset"] + h * head_elems
                    if name == "decoder":
                        head_shape = (n, d_model)
                        handles.append(
                            TensorHandle(
                                name=f"blk.{h}.{name}",
                                shape=head_shape,
                                dtype=dtype,
                                loader=_range_loader(
                                    self._storage_cache, pt_path, md["zip_path"],
                                    np_dtype, md["numel"], head_start, head_shape,
                                ),
                            )
                        )
                    else:
                        head_shape = (d_model, n)
                        handles.append(
                            TensorHandle(
                                name=f"blk.{h}.{name}",
                                shape=head_shape,
                                dtype=dtype,
                                loader=_range_loader(
                                    self._storage_cache, pt_path, md["zip_path"],
                                    np_dtype, md["numel"], head_start, head_shape,
                                ),
                            )
                        )
                # Per route-lattice unit: one unit = `unit` neurons per head.
                for u in range(layout["mult"]):
                    for h in range(nh):
                        if name == "decoder":
                            unit_shape = (unit, d_model)
                            unit_start = md["offset"] + (h * n + u * unit) * d_model
                            handles.append(
                                TensorHandle(
                                    name=f"{name}.u{u}.h{h}",
                                    shape=unit_shape,
                                    dtype=dtype,
                                    expert_id=u,
                                    loader=_range_loader(
                                        self._storage_cache, pt_path, md["zip_path"],
                                        np_dtype, md["numel"], unit_start, unit_shape,
                                    ),
                                )
                            )
                        else:
                            handles.append(
                                TensorHandle(
                                    name=f"{name}.u{u}.h{h}",
                                    shape=(d_model, unit),
                                    dtype=dtype,
                                    expert_id=u,
                                    loader=_head_column_loader(
                                        self._storage_cache, pt_path, md["zip_path"],
                                        np_dtype, md["numel"],
                                        md["offset"] + h * head_elems, (d_model, n),
                                        u * unit, (u + 1) * unit,
                                    ),
                                )
                            )
            else:
                handles.append(TensorHandle(**base))
