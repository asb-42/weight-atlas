"""Distribution-shape statistics: percentiles, outlier fractions, dynamic range.

Adopted from alesha-pro/atlas (MIT, see docs/2026-08-31_atlas-alesha-pro-analysis.md)
and re-implemented deterministically: their scanner subsamples tensors above 16M
elements with an unseeded ``torch.randint``; here the subsample is a seeded
``numpy.random.default_rng`` draw so two scans of the same checkpoint produce
identical fingerprints.

Like the shared SVD spectrum (:mod:`weight_atlas.stats.spectrum`), the summary
is computed once per tensor and cached on the handle — the scan pipeline reads
nine derived scalars from one chunked pass instead of nine separate passes.
All accumulation is chunked float64 over the float32 payload (same discipline
as :mod:`weight_atlas.stats.shape_moments`) to bound memory on large MoE
tensors.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator

import numpy as np

from weight_atlas.core.registry import register_stat
from weight_atlas.core.types import TensorHandle

_CHUNK = 2**20  # 1M elements
_SAMPLE_CAP = 16_000_000
_DEFAULT_SEED = 0

# |w| percentile ladder (order matches the alesha-pro jsonl: p50…p9999).
PERCENTILE_LEVELS = (0.5, 0.9, 0.99, 0.999, 0.9999)


def _flat_chunks(x: np.ndarray) -> Iterator[np.ndarray]:
    flat = x.reshape(-1)
    for i in range(0, flat.size, _CHUNK):
        yield flat[i : i + _CHUNK].astype(np.float64)


def _summary(x: np.ndarray, seed: int) -> dict[str, float]:
    """Chunked distribution summary of the flattened tensor.

    Deterministic for any (input, seed): the percentile sample above
    ``_SAMPLE_CAP`` elements is a seeded draw, never an unseeded one.
    """
    flat = x.reshape(-1)
    n = flat.size
    if n == 0:
        return {
            "mean": float("nan"), "std": float("nan"), "absmax": float("nan"),
            "absmean": float("nan"), "p50": float("nan"), "p90": float("nan"),
            "p99": float("nan"), "p999": float("nan"), "p9999": float("nan"),
            "outlier_3s": float("nan"), "outlier_4s": float("nan"),
            "outlier_6s": float("nan"), "dyn_range": float("nan"),
        }

    mean = 0.0
    for chunk in _flat_chunks(flat):
        mean += float(chunk.sum())
    mean /= n

    m2 = 0.0
    absmean = 0.0
    amax = 0.0
    for chunk in _flat_chunks(flat):
        d = chunk - mean
        m2 += float((d * d).sum())
        a = np.abs(chunk)
        absmean += float(a.sum())
        if a.size:
            amax = max(amax, float(a.max()))
    m2 /= n
    absmean /= n
    std = float(np.sqrt(m2))

    # Percentiles: exact up to the sample cap; above it, a seeded subsample
    # (chunked build — np.quantile sorts, so keep the array bounded).
    if n <= _SAMPLE_CAP:
        sample = np.abs(flat).astype(np.float64, copy=False)
    else:
        rng = np.random.default_rng(seed)
        sample = np.abs(flat[rng.integers(0, n, size=_SAMPLE_CAP)].astype(np.float64))

    pcts = np.quantile(sample, PERCENTILE_LEVELS).tolist() if sample.size else [0.0] * len(PERCENTILE_LEVELS)
    p50 = pcts[0]

    out3 = out4 = out6 = 0.0
    if std > 0:
        n3 = n4 = n6 = 0
        for chunk in _flat_chunks(flat):
            d = np.abs(chunk - mean)
            n3 += int(np.count_nonzero(d > 3 * std))
            n4 += int(np.count_nonzero(d > 4 * std))
            n6 += int(np.count_nonzero(d > 6 * std))
        out3, out4, out6 = n3 / n, n4 / n, n6 / n

    return {
        "mean": mean,
        "std": std,
        "absmax": amax,
        "absmean": absmean,
        "p50": p50,
        "p90": pcts[1],
        "p99": pcts[2],
        "p999": pcts[3],
        "p9999": pcts[4],
        "outlier_3s": out3,
        "outlier_4s": out4,
        "outlier_6s": out6,
        "dyn_range": amax / p50 if p50 > 0 else float("inf"),
    }


_summary_cache: weakref.WeakKeyDictionary[TensorHandle, dict[str, float]] = weakref.WeakKeyDictionary()


def distribution_summary(t: TensorHandle, seed: int = _DEFAULT_SEED) -> dict[str, float]:
    """Cached distribution summary for a handle (one pass per tensor)."""
    cached = _summary_cache.get(t)
    if cached is None:
        cached = _summary(t.load(), seed)
        _summary_cache[t] = cached
    return cached


def _make_summary_stat(field: str, description: str) -> type:
    class _SummaryStat:
        def __init__(self, seed: int = _DEFAULT_SEED) -> None:
            self._seed = seed

        def compute(self, t: TensorHandle) -> float:
            return distribution_summary(t, seed=self._seed)[field]

    _SummaryStat.__name__ = field
    _SummaryStat.__qualname__ = field
    _SummaryStat.__doc__ = f"{field}: {description} (shared distribution summary)."
    return _SummaryStat


_DISTRIBUTION_STATS = (
    ("mean", "mean of the weight values"),
    ("std", "standard deviation of the weight values"),
    ("absmax", "largest absolute weight"),
    ("absmean", "mean absolute weight"),
    ("p50", "50th percentile of |w|"),
    ("p90", "90th percentile of |w|"),
    ("p99", "99th percentile of |w|"),
    ("p999", "99.9th percentile of |w|"),
    ("p9999", "99.99th percentile of |w|"),
    ("outlier_3s", "fraction of weights beyond 3 std"),
    ("outlier_4s", "fraction of weights beyond 4 std"),
    ("outlier_6s", "fraction of weights beyond 6 std"),
    ("dyn_range", "absmax / p50 (dynamic range of |w|)"),
)

for _field, _desc in _DISTRIBUTION_STATS:
    register_stat(_field)(_make_summary_stat(_field, _desc))


def _amax_ratios(x: np.ndarray) -> tuple[float, float]:
    """max/median of per-row and per-column amax for a 2-D tensor.

    The outlier-channel problem as two numbers: a large ratio means one
    channel dominates the per-channel scale and destroys per-channel
    quantization.
    """
    row_amax = np.abs(x).max(axis=1).astype(np.float64)
    col_amax = np.abs(x).max(axis=0).astype(np.float64)
    row_med = float(np.median(row_amax))
    col_med = float(np.median(col_amax))
    row_ratio = float(row_amax.max()) / row_med if row_med > 0 else float("inf")
    col_ratio = float(col_amax.max()) / col_med if col_med > 0 else float("inf")
    return row_ratio, col_ratio


_ratios_cache: weakref.WeakKeyDictionary[TensorHandle, tuple[float, float]] = weakref.WeakKeyDictionary()


def amax_ratios(t: TensorHandle) -> tuple[float, float]:
    """Cached (row_amax_ratio, col_amax_ratio); NaN for non-2-D tensors.

    NaN is the established "not applicable, never zero" representation: the
    query API serializes it as ``null``.
    """
    cached = _ratios_cache.get(t)
    if cached is None:
        x = t.load()
        cached = _amax_ratios(x) if x.ndim == 2 else (float("nan"), float("nan"))
        _ratios_cache[t] = cached
    return cached


@register_stat("row_amax_ratio")
class RowAmaxRatio:
    """max/median of per-row amax (2-D only; NaN otherwise)."""

    stat_id = "row_amax_ratio"

    def compute(self, t: TensorHandle) -> float:
        return amax_ratios(t)[0]


@register_stat("col_amax_ratio")
class ColAmaxRatio:
    """max/median of per-column amax (2-D only; NaN otherwise)."""

    stat_id = "col_amax_ratio"

    def compute(self, t: TensorHandle) -> float:
        return amax_ratios(t)[1]
