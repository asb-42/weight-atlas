"""Statistic protocol."""

from __future__ import annotations

from typing import Protocol

from weight_atlas.core.types import TensorHandle


class Statistic(Protocol):
    stat_id: str

    def compute(self, t: TensorHandle) -> float:
        """Compute a scalar statistic for a tensor. Pure, seeded."""
        ...
