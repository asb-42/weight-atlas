"""GGUF loader: mmap, lazy TensorHandles, registry-ID ``gguf``.

Supports MoE models with 3D stacked expert tensors (ffn_*_exps).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from weight_atlas.core.registry import register_loader
from weight_atlas.core.types import TensorHandle
from weight_atlas.loaders.gguf_dequant import dequantize

# GGUF magic number
GGUF_MAGIC = b"GGUF"

# GGUF MoE expert tensor patterns (3D stacked)
_MOE_EXPS_PATTERNS = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]


def _is_gguf(path: Path) -> bool:
    """Check if a file is a GGUF file by reading magic bytes."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == GGUF_MAGIC
    except OSError:
        return False


def _discover_gguf_files(path: Path) -> list[Path]:
    """Return sorted list of .gguf files for a file or directory."""
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.gguf"))
    if not files:
        raise FileNotFoundError(f"no .gguf files in {path}")
    return files


def _is_moe_exps_tensor(name: str) -> bool:
    """Check if a tensor name is a MoE expert tensor (3D stacked)."""
    return any(pattern in name for pattern in _MOE_EXPS_PATTERNS)


class _SharedExpertDequant:
    """Dequantize a 3D stacked GGUF expert tensor exactly once, then slice.

    All per-expert sub-handles of one stacked tensor share a single
    ``_SharedExpertDequant``, so a layer with ``n_experts`` experts performs one
    full dequantization instead of ``n_experts`` (previously each sub-handle
    re-dequantized the whole 3D tensor on every ``load()`` — quadratic for MoE).

    Shape convention: GGUF reports the *logical* shape via ``tensor.shape`` with
    the expert axis LAST (e.g. ``(hidden, hidden, n_experts)``). Quantized
    tensors report a different ``tensor.data.shape`` (the byte layout, e.g.
    ``(n_blocks, ..., block_bytes)``), so we must reshape the dequantized float
    output to the LOGICAL ``tensor.shape`` and slice the last (expert) axis —
    the gguf dequantizer returns elements in logical-shape order.
    """

    def __init__(
        self,
        data: np.ndarray,
        ggml_type: int,
        logical_shape: tuple[int, ...],
        n_children: int,
    ) -> None:
        if len(logical_shape) != 3:
            raise ValueError(
                f"expected a 3D stacked expert tensor, got shape {logical_shape}"
            )
        self._data = data
        self._ggml_type = ggml_type
        self._shape = tuple(logical_shape)
        self._arr: np.ndarray | None = None
        self._lock = threading.Lock()
        self._n_children = n_children
        self._children_done = 0

    def _ensure(self) -> np.ndarray:
        with self._lock:
            if self._arr is None:
                arr = dequantize(self._data.tobytes(), self._ggml_type)
                if arr.size != int(np.prod(self._shape)):
                    raise ValueError(
                        f"dequantized element count {arr.size} does not match logical "
                        f"shape {self._shape}"
                    )
                self._arr = arr.reshape(self._shape).astype(np.float32)
            return self._arr

    def slice(self, expert_id: int) -> np.ndarray:
        """Return the 2D float32 slice for one expert (last axis = expert axis)."""
        return self._ensure()[:, :, expert_id]

    def release_child(self) -> None:
        """Called by each expert sub-handle's ``TensorHandle.clear()``.

        A stacked parent dequantized to float32 can be ~1 GB (256 experts ×
        2048×512); holding every parent for the whole scan would keep the
        entire model in RAM (~4 bytes/param). Once every sub-handle has
        finished, the 3D array is dropped so memory stays proportional to the
        parents currently being processed. The lock guarantees the array is
        never freed while a sub-handle is still using its slice: a handle only
        clears after its statistics are computed, and numpy slice views keep
        the underlying buffer alive even after ``_arr`` is reset to ``None``.
        """
        with self._lock:
            self._children_done += 1
            if self._children_done >= self._n_children:
                self._arr = None


@register_loader("gguf")
class GGUFLoader:
    """Memory-mapped GGUF loader with lazy dequantization.

    Supports MoE models with 3D stacked expert tensors by splitting them
    into per-expert sub-handles.
    """

    format_id = "gguf"

    def open(self, path: Path) -> list[TensorHandle]:
        try:
            from gguf import GGUFReader
        except ImportError as exc:  # pragma: no cover - gguf is a core dep
            raise ImportError(
                "GGUF model scanning requires the 'gguf' package. Install it with "
                "`pip install gguf` (it is a core weight-atlas dependency, so "
                "reinstall the project to get it)."
            ) from exc

        files = _discover_gguf_files(path)
        handles: list[TensorHandle] = []

        for f in files:
            reader = GGUFReader(str(f))
            for tensor in reader.tensors:
                name = tensor.name
                # Logical shape as reported by gguf (experts LAST for 3D MoE).
                shape = tuple(int(x) for x in tensor.shape)
                ggml_type = tensor.tensor_type.value
                data = tensor.data  # raw bytes / memmap (byte layout for quantized)

                # Check if this is a 3D MoE expert tensor using the LOGICAL shape
                # (tensor.shape), NOT data.shape which is the quantized byte layout.
                if _is_moe_exps_tensor(name) and len(shape) == 3:
                    # Split into per-expert sub-handles sharing one dequantization.
                    # Expert axis is the LAST dim of the logical shape.
                    n_experts = shape[-1]
                    expert_shape = shape[:-1]  # (hidden, hidden)
                    shared = _SharedExpertDequant(data, ggml_type, shape, n_experts)
                    for expert_id in range(n_experts):
                        handles.append(
                            TensorHandle(
                                name=f"{name}[{expert_id}]",
                                shape=expert_shape,
                                dtype=f"ggml_{ggml_type}",
                                loader=self._make_shared_loader(shared, expert_id),
                                expert_id=expert_id,
                                # Once the last expert sub-handle is cleared, the
                                # shared parent releases its ~1 GB float32 array.
                                on_clear=shared.release_child,
                            )
                        )
                else:
                    handles.append(
                        TensorHandle(
                            name=name,
                            shape=shape,
                            dtype=f"ggml_{ggml_type}",
                            loader=self._make_loader(data, ggml_type, shape),
                        )
                    )

        return handles

    def _make_loader(
        self, data: np.ndarray, ggml_type: int, shape: tuple[int, ...]
    ) -> Callable[[], np.ndarray]:
        return lambda: self._load_tensor(data, ggml_type, shape)

    def _make_shared_loader(
        self, shared: _SharedExpertDequant, expert_id: int
    ) -> Callable[[], np.ndarray]:
        """Build a zero-arg loader that slices one expert from a shared dequant."""

        def loader() -> np.ndarray:
            return shared.slice(expert_id)

        return loader

    def _load_tensor(self, data: np.ndarray, ggml_type: int, shape: tuple[int, ...]) -> np.ndarray:
        """Materialise a single tensor as float32."""
        arr = dequantize(data.tobytes(), ggml_type)
        return arr.reshape(shape).astype(np.float32)
