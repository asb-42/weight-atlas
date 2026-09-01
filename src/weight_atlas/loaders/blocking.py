"""Row-block streaming for giant tensors (n-gram embedding tables).

A 51B-element table materialized as float32 is 204 GiB — ``load()`` OOMs
before any statistic runs (see docs/2026-09-01_ngram-table-streaming.md).
This module exposes ``iter_row_blocks``: a uniform per-format dispatcher that
yields float32 row blocks without materializing the tensor, so the streaming
statistics path (:mod:`weight_atlas.stats.streaming`) can accumulate
everything block-wise.

Row blocks are always the FIRST axis of the handle shape in *file order*
(GGUF dims come reversed; the caller sees whatever ``handle.shape`` says).
Formats without block support yield ``None`` from the dispatcher — callers
fall back to ``load()`` (full materialization), which is fine for formats
that never carry giant tensors.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np

from weight_atlas.core.types import TensorHandle

_ROWS_PER_BLOCK_DEFAULT = 16_384  # ≈ 128 MiB fp32 at D=2048


def iter_row_blocks(
    handle: TensorHandle,
    loader: Any,
    rows_per_block: int = _ROWS_PER_BLOCK_DEFAULT,
) -> Iterator[np.ndarray] | None:
    """Yield float32 row blocks of ``handle`` without full materialization.

    Returns ``None`` when neither the loader nor a default path supports
    block streaming for this tensor (caller falls back to ``load()``).
    """
    if len(handle.shape) != 2:
        return None
    rows_per_block = max(1, rows_per_block)

    block_source = getattr(loader, "iter_row_blocks", None)
    if callable(block_source):
        return cast(
            "Iterator[np.ndarray] | None", block_source(handle, rows_per_block)
        )

    # Default fallback: full materialization, then yield blocks from the
    # array. Memory-honest (the array IS the materialization) — tests that
    # assert peak-RAM bounds must provide a real block-streaming loader.
    arr = handle.load()

    def _from_array() -> Iterator[np.ndarray]:
        for lo in range(0, arr.shape[0], rows_per_block):
            yield arr[lo : lo + rows_per_block]

    return _from_array()
