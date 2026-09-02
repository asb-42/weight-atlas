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

    def source_files(self, path: Path) -> list[Path]:
        """Source model files backing ``open(path)`` — name-sorted, stable.

        Used for scan-time provenance hashing (Phase 0 scan sharing):
        the scan records per-file SHA-256 of exactly these files. Must
        return the same list for the same input (deterministic), and the
        same files ``open`` consumes. Default: the path itself if it is
        a file, else every regular file under it, name-sorted.
        """
        if path.is_file():
            return [path]
        return sorted(p for p in path.rglob("*") if p.is_file())
