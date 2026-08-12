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


@register_loader("gguf")
class GGUFLoader:
    """Memory-mapped GGUF loader with lazy dequantization.

    Supports MoE models with 3D stacked expert tensors by splitting them
    into per-expert sub-handles.
    """

    format_id = "gguf"

    def open(self, path: Path) -> list[TensorHandle]:
        from gguf import GGUFReader

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
                    # Split into per-expert sub-handles
                    n_experts = actual_shape[0]
                    expert_shape = actual_shape[1:]  # (hidden, hidden)
                    for expert_id in range(n_experts):
                        handles.append(
                            TensorHandle(
                                name=f"{name}[{expert_id}]",
                                shape=expert_shape,
                                dtype=f"ggml_{ggml_type}",
                                loader=self._make_expert_loader(data, ggml_type, shape, expert_id),
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

    def _make_expert_loader(
        self, data: np.ndarray, ggml_type: int, shape: tuple[int, ...], expert_id: int
    ) -> Callable[[], np.ndarray]:
        return lambda: self._load_expert_tensor(data, ggml_type, shape, expert_id)

    def _load_tensor(self, data: np.ndarray, ggml_type: int, shape: tuple[int, ...]) -> np.ndarray:
        """Materialise a single tensor as float32."""
        arr = dequantize(data.tobytes(), ggml_type)
        return arr.reshape(shape).astype(np.float32)

    def _load_expert_tensor(
        self, data: np.ndarray, ggml_type: int, shape: tuple[int, ...], expert_id: int
    ) -> np.ndarray:
        """Materialise a single expert slice from a 3D stacked tensor.

        GGUF stores 3D expert tensors as (hidden, hidden, n_experts) where
        the last dimension is the expert index.
        """
        # Load the full 3D tensor
        arr = dequantize(data.tobytes(), ggml_type)
        arr_3d = arr.reshape(shape)
        # Return the 2D slice for this expert (last dimension)
        return arr_3d[:, :, expert_id].astype(np.float32)
