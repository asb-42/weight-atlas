"""Map per-slot tensor statistics to fractal (fBm) parameters.

The atlas raster is (layers × slots). For the fractal terrain renderer we
want each slot column to inherit its own fractal character from the *real*
per-tensor statistics of that slot — not a plain heightmap. This module
aggregates per-slot stats (median across layers) and maps them onto fBm
parameters in a deterministic, spec-driven way.

Determinism: median aggregation is deterministic; mappings are pure functions
(clamp/scale), no RNG.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weight_atlas.core.name_map import map_name

# Stat keys available per tensor (fingerprint).
_STAT_KEYS = ("kurtosis", "sparsity", "effective_rank", "spectral_norm")

_DEFAULTS: dict[str, float] = {
    "octaves": 4.0,
    "persistence": 0.5,
    "lacunarity": 2.0,
    "base_freq": 1.0,
}


def slot_stat_medians(
    fingerprint: Mapping[str, Any],
    slots: list[str],
) -> dict[str, dict[str, float]]:
    """Aggregate per-tensor stats into per-slot medians (across layers).

    Returns ``{slot: {kurtosis, sparsity, effective_rank, spectral_norm}}``.
    Non-layer tensors (embed, lm_head) and missing combos are skipped; a slot
    with no tensors gets NaN values. Deterministic.
    """
    tensors = fingerprint.get("tensors", {})
    per_slot: dict[str, dict[str, list[float]]] = {
        s: {k: [] for k in _STAT_KEYS} for s in slots
    }
    for name, ts in tensors.items():
        if not isinstance(ts, dict):
            continue
        layer, slot = map_name(str(name))
        if layer is None or slot not in per_slot:
            continue
        for k in _STAT_KEYS:
            v = ts.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                per_slot[slot][k].append(float(v))

    out: dict[str, dict[str, float]] = {}
    for s in slots:
        row: dict[str, float] = {}
        for k in _STAT_KEYS:
            vals = per_slot[s][k]
            row[k] = float(np.median(vals)) if vals else float("nan")
        out[s] = row
    return out


def load_fingerprint_stats(
    out_dir: Path,
    slots: list[str],
) -> dict[str, dict[str, float]]:
    """Read fingerprint.json from a scan output dir and aggregate per slot."""
    fp_path = out_dir / "fingerprint.json"
    if fp_path.exists():
        try:
            fp = json.loads(fp_path.read_text())
            if isinstance(fp, dict):
                return slot_stat_medians(fp, slots)
        except (OSError, ValueError):
            pass
    return {s: {k: float("nan") for k in _STAT_KEYS} for s in slots}


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _scale(v: float, v_lo: float, v_hi: float, out_lo: float, out_hi: float) -> float:
    if not np.isfinite(v) or v_hi <= v_lo:
        return out_lo
    t = (v - v_lo) / (v_hi - v_lo)
    return out_lo + t * (out_hi - out_lo)


def stats_to_params(
    slot_stats: dict[str, dict[str, float]],
    mapping: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Map per-slot stat medians to fBm params.

    ``mapping`` comes from the spec ``fractal`` block and carries the
    source-stat → target-param range table, e.g.:

        "mapping": {
          "octaves":    {"stat": "effective_rank", "lo": 4, "hi": 8},
          "persistence":{"stat": "kurtosis",       "lo": 0.35, "hi": 0.7},
          "lacunarity": {"stat": "sparsity",       "lo": 1.8, "hi": 2.4},
          "base_freq":  {"stat": "spectral_norm",  "lo": 1.0, "hi": 2.5}
        }

    Each target is linearly scaled from the stat's observed min→max (across
    slots) into the target range, then clamped. Slots with NaN stats fall
    back to the midpoint of the target range. Deterministic.
    """
    mapping = mapping or {}
    targets: dict[str, dict[str, float]] = {}

    # Observed min/max per source stat across slots (finite only).
    obs_lo: dict[str, float] = {}
    obs_hi: dict[str, float] = {}
    for stat in _STAT_KEYS:
        vals = [
            stats[stat] for stats in slot_stats.values()
            if stat in stats and np.isfinite(stats[stat])
        ]
        obs_lo[stat] = float(np.min(vals)) if vals else float("nan")
        obs_hi[stat] = float(np.max(vals)) if vals else float("nan")

    for target, cfg in mapping.items():
        stat = cfg.get("stat")
        if stat not in _STAT_KEYS:
            continue
        out_lo = float(cfg.get("lo", 0.0))
        out_hi = float(cfg.get("hi", 1.0))
        mid = 0.5 * (out_lo + out_hi)
        targets[target] = {}
        for slot, stats in slot_stats.items():
            v = stats.get(stat, float("nan"))
            targets[target][slot] = _clamp(
                _scale(v, obs_lo[stat], obs_hi[stat], out_lo, out_hi)
                if np.isfinite(v) and np.isfinite(obs_lo[stat]) else mid,
                out_lo,
                out_hi,
            )
    return targets


def slot_fractal_params(
    out_dir: Path,
    slots: list[str],
    fractal_cfg: Mapping[str, Any],
    seed: int,
) -> dict[str, dict[str, float]]:
    """Full pipeline: fingerprint → per-slot medians → per-slot fBm params.

    Returns ``{slot: {octaves, persistence, lacunarity, base_freq, seed}}``.
    The ``seed`` per slot is derived deterministically from the base seed and
    the slot index, so the noise lattice differs per slot while staying
    reproducible.
    """
    stats = load_fingerprint_stats(out_dir, slots)
    mapping = fractal_cfg.get("mapping", {}) if fractal_cfg else {}
    params = stats_to_params(slot_stats=stats, mapping=mapping)
    out: dict[str, dict[str, float]] = {}
    for i, slot in enumerate(slots):
        out[slot] = {
            "octaves": max(1, int(round(params.get("octaves", {}).get(slot, _DEFAULTS["octaves"])))),
            "persistence": params.get("persistence", {}).get(slot, _DEFAULTS["persistence"]),
            "lacunarity": params.get("lacunarity", {}).get(slot, _DEFAULTS["lacunarity"]),
            "base_freq": params.get("base_freq", {}).get(slot, _DEFAULTS["base_freq"]),
            "seed": int(seed) + i * 1009,
        }
    return out
