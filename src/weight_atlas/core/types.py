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

    ``load()`` materialises the tensor as float32 on first call and memoizes
    the result, so computing N statistics from the same handle performs a
    single load/dequantization instead of N (the scan pipeline computes 6-7
    stats per tensor — without memoization that multiplies I/O and runtime by
    the stat count).
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
        self._cache: np.ndarray | None = None
        self._loaded = False

    def load(self) -> np.ndarray:
        if not self._loaded:
            self._cache = self._loader().astype(np.float32, copy=False)
            self._loaded = True
        assert self._cache is not None
        return self._cache


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
    kernel_norm: float = 0.0
    expert_id: int | None = None  # For MoE expert tensors


@dataclass
class Field2D:
    """A 2D scalar field rasterised from tensor statistics."""
    channel: str
    data: np.ndarray
    row_labels: list[str] = field(default_factory=list)
    col_labels: list[str] = field(default_factory=list)
    spec_version: int = 1
    model_name: str = ""  # display name shown in sheet titles


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
    vision_slots: list[str] = field(default_factory=list)
    vision_channels: dict[str, Any] = field(default_factory=dict)

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
            vision_slots=list(raw.get("vision_slots", [])),
            vision_channels=dict(raw.get("vision_channels", {})),
        )

    def channel_stat(self, channel: str) -> str:
        return str(self.channels[channel]["stat"])

    def channel_scale(self, channel: str) -> dict[str, Any]:
        return dict(self.channels[channel]["scale"])


# Canonical default atlas spec — the single source of truth used by BOTH the
# CLI and the web UI. All shipped spec files must agree with this version or
# scans produced by one entrypoint can never be compared against the other
# (compare/align.py hard-rejects spec_version mismatches).
DEFAULT_SPEC_NAME = "atlas_spec.v2.3.json"
DEFAULT_SPEC_VERSION = 3


def get_default_spec_path() -> Path:
    """Absolute path to the canonical default atlas spec (CWD-independent)."""
    return Path(__file__).resolve().parent.parent.parent.parent / "specs" / DEFAULT_SPEC_NAME


def load_default_spec() -> AtlasSpec:
    """Load the canonical default spec, asserting its version is current.

    Raises RuntimeError if the shipped default spec is stale (spec_version
    differs from DEFAULT_SPEC_VERSION), so drift between the spec files is
    caught at startup instead of producing incompatible fingerprints.
    """
    spec = AtlasSpec.from_json(get_default_spec_path())
    if spec.spec_version != DEFAULT_SPEC_VERSION:
        raise RuntimeError(
            f"canonical default spec {get_default_spec_path()} has spec_version "
            f"{spec.spec_version}; expected {DEFAULT_SPEC_VERSION}. "
            "Stale spec file — reconcile the shipped specs."
        )
    return spec


# GGUF magic bytes
_GGUF_MAGIC = b"GGUF"


def detect_loader(path: Path) -> str:
    """Detect loader type from file magic bytes or directory contents.

    Returns:
        "gguf" for GGUF files / directories containing ``*.gguf``,
        "safetensors" for safetensors files / directories containing
        ``*.safetensors``.

    Raises:
        FileNotFoundError: for directories with no recognizable shards, or an
            unreadable path — instead of silently defaulting to safetensors.
    """
    try:
        if path.is_dir():
            if any(path.glob("*.gguf")):
                return "gguf"
            if any(path.glob("*.safetensors")):
                return "safetensors"
            raise FileNotFoundError(
                f"cannot detect loader for directory {path}: no .gguf or "
                ".safetensors files. Pass --loader explicitly."
            )
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == _GGUF_MAGIC:
            return "gguf"
        return "safetensors"
    except OSError as exc:
        raise FileNotFoundError(f"cannot detect loader for {path}: {exc}") from exc
