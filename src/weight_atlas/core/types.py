"""Core data types: TensorHandle, TensorStats, Field2D, ExpertPanel, AtlasSpec."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class TensorHandle:
    """Lazy handle to a tensor inside a model file.

    ``load()`` materialises the tensor as float32 only when called.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        loader: Callable[[], np.ndarray],
        expert_id: int | None = None,
    ) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self._loader = loader
        self.expert_id = expert_id  # For MoE expert tensors

    def load(self) -> np.ndarray:
        arr = self._loader()
        return arr.astype(np.float32, copy=False)


@dataclass
class TensorStats:
    """Computed statistics for a single tensor."""
    name: str
    shape: tuple[int, ...]
    frobenius: float = 0.0
    spectral_norm: float = 0.0
    effective_rank: float = 0.0
    stable_rank: float = 0.0
    kurtosis: float = 0.0
    sparsity: float = 0.0
    expert_id: int | None = None  # For MoE expert tensors


@dataclass
class Field2D:
    """A 2D scalar field rasterised from tensor statistics."""
    channel: str
    data: np.ndarray
    row_labels: list[str] = field(default_factory=list)
    col_labels: list[str] = field(default_factory=list)
    spec_version: int = 1


@dataclass
class ExpertPanel:
    """A 2D field for MoE expert statistics (Layer × Expert).

    Each slot (gate/up/down) has its own panel.
    """
    slot: str  # mlp_gate, mlp_up, or mlp_down
    channel: str  # height, tint, rough
    data: np.ndarray  # Shape: (n_layers, n_experts)
    row_labels: list[str] = field(default_factory=list)
    col_labels: list[str] = field(default_factory=list)
    spec_version: int = 1

    @property
    def n_layers(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_experts(self) -> int:
        return int(self.data.shape[1])


@dataclass
class AtlasSpec:
    """Versioned cartography convention loaded from atlas_spec.v2.3.json."""
    spec_version: int
    slots: list[str]
    channels: dict[str, Any]
    grid: dict[str, Any]
    sheet: dict[str, Any]
    seeds: dict[str, Any]
    blender: dict[str, Any] = field(default_factory=dict)
    compare: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path) -> AtlasSpec:
        with open(path) as f:
            raw = json.load(f)
        return cls(
            spec_version=raw["spec_version"],
            slots=list(raw["slots"]),
            channels=dict(raw["channels"]),
            grid=dict(raw["grid"]),
            sheet=dict(raw["sheet"]),
            seeds=dict(raw["seeds"]),
            blender=dict(raw.get("blender", {})),
            compare=dict(raw.get("compare", {})),
        )

    def channel_stat(self, channel: str) -> str:
        return str(self.channels[channel]["stat"])

    def channel_scale(self, channel: str) -> dict[str, Any]:
        return dict(self.channels[channel]["scale"])


# GGUF magic bytes
_GGUF_MAGIC = b"GGUF"


def detect_loader(path: Path) -> str:
    """Detect loader type from file magic bytes.

    Returns:
        "gguf" if file starts with GGUF magic, "safetensors" otherwise.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == _GGUF_MAGIC:
            return "gguf"
        return "safetensors"
    except OSError:
        return "safetensors"
