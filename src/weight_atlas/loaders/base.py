"""Loader protocol."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from weight_atlas.core.types import TensorHandle


class Loader(Protocol):
    format_id: str

    def open(self, path: Path) -> Sequence[TensorHandle]:
        """Open a model path and return lazy tensor handles."""
        ...
