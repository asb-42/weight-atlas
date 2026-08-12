"""GGUF loader: mmap, lazy TensorHandles, registry-ID ``gguf``.

Supports MoE models with 3D stacked expert tensors (ffn_*_exps).
"""

from __future__ import annotations

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

    Shape convention: the gguf library reports ``tensor.shape`` with the expert
    axis LAST (e.g. ``(hidden, hidden, n_experts)``) while ``tensor.data.shape``
    (the actual memory layout) has the expert axis FIRST (e.g.
    ``(n_experts, hidden, hidden)``). Everything here is derived from
    ``actual_shape`` (the data layout) and experts are sliced along axis 0, so
    slicing stays correct regardless of the header/report shape ordering.
    """

    def __init__(
        self,
        data: np.ndarray,
        ggml_type: int,
        actual_shape: tuple[int, ...],
    ) -> None:
        if len(actual_shape) != 3:
            raise ValueError(
                f"expected a 3D stacked expert tensor, got shape {actual_shape}"
            )
        self._data = data
        self._ggml_type = ggml_type
        self._shape = tuple(actual_shape)
        self._arr: np.ndarray | None = None

    def _ensure(self) -> np.ndarray:
        if self._arr is None:
            arr = dequantize(self._data.tobytes(), self._ggml_type)
            self._arr = arr.reshape(self._shape).astype(np.float32)
        return self._arr

    def slice(self, expert_id: int) -> np.ndarray:
        """Return the 2D float32 slice for one expert (axis 0 = expert axis)."""
        return self._ensure()[expert_id, :, :]


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
                shape = tuple(int(x) for x in tensor.shape)
                ggml_type = tensor.tensor_type.value
                data = tensor.data  # This is the raw bytes or memmap

                # Check if this is a 3D MoE expert tensor
                # Use data.shape since GGUF may report shape differently from memory layout
                actual_shape = data.shape if hasattr(data, 'shape') else shape
                if _is_moe_exps_tensor(name) and len(actual_shape) == 3:
                    # Split into per-expert sub-handles sharing one dequantization
                    n_experts = actual_shape[0]
                    expert_shape = actual_shape[1:]  # (hidden, hidden)
                    shared = _SharedExpertDequant(data, ggml_type, actual_shape)
                    for expert_id in range(n_experts):
                        handles.append(
                            TensorHandle(
                                name=f"{name}[{expert_id}]",
                                shape=expert_shape,
                                dtype=f"ggml_{ggml_type}",
                                loader=lambda s=shared, e=expert_id: s.slice(e),
                                expert_id=expert_id,
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

    def _load_tensor(self, data: np.ndarray, ggml_type: int, shape: tuple[int, ...]) -> np.ndarray:
        """Materialise a single tensor as float32."""
        arr = dequantize(data.tobytes(), ggml_type)
        return arr.reshape(shape).astype(np.float32)
