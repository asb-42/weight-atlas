"""Shared compare pipeline: the single orchestration used by CLI and web API.

The CLI and the API worker previously carried private copies of this logic
that had already diverged (row labels, noise-floor veil, channel filtering).
Both now delegate here so identical inputs produce identical artefacts from
either entrypoint.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from weight_atlas.core.types import AtlasSpec

ProgressFn = Callable[[float, str], None]


def discover_channels_from_manifest(manifest: dict[str, str]) -> list[str]:
    """Channel names from manifest keys (``field_<channel>_{raw,smooth}.tif``)."""
    channels: set[str] = set()
    for key in manifest:
        if not key.startswith("field_") or not key.endswith(".tif"):
            continue
        core = key[len("field_"):-len(".tif")]
        if core.endswith("_raw"):
            channels.add(core[:-len("_raw")])
        elif core.endswith("_smooth"):
            channels.add(core[:-len("_smooth")])
    return sorted(channels)


def run_compare(
    dir_a: Path,
    dir_b: Path,
    out: Path,
    spec: AtlasSpec,
    *,
    mode: str = "strict",
    interp: str | None = None,
    row_labels_a: list[str] | None = None,
    row_labels_b: list[str] | None = None,
    noise_floor_dir: Path | None = None,
    progress: ProgressFn | None = None,
) -> list[Path]:
    """Compare two scan output directories and write all compare artefacts.

    Args:
        dir_a, dir_b: scan output directories (manifest.json + field_*.tif)
        out: output directory for delta/summary artefacts (created as needed)
        spec: atlas specification
        mode: "strict" or "aligned"
        interp: aligned-mode row resampling; None → spec's aligned_interp
        row_labels_a/b: optional real layer labels from the scans' fingerprints
        noise_floor_dir: optional calibration compare directory for the
            delta-sheet noise-floor veil
        progress: optional ``(fraction, message)`` callback

    Returns:
        List of artefact paths written. Empty when no channels matched.
    """
    from weight_atlas.compare import compute_compare_summary, hotspot_ranking
    from weight_atlas.fields.smoothing import smooth, upsample
    from weight_atlas.fields.tif_io import read_tif, write_tif

    def _report(pct: float, msg: str) -> None:
        if progress is not None:
            progress(float(pct), msg)

    out.mkdir(parents=True, exist_ok=True)

    fp_a_path = dir_a / "fingerprint.json"
    fp_b_path = dir_b / "fingerprint.json"
    fp_a = json.loads(fp_a_path.read_text()) if fp_a_path.exists() else None
    fp_b = json.loads(fp_b_path.read_text()) if fp_b_path.exists() else None

    manifest_path = dir_a / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {dir_a}")
    manifest = json.loads(manifest_path.read_text())
    channels = discover_channels_from_manifest(manifest)
    # Only compare channels the alignment/compare infrastructure supports —
    # the main spec.channels. Vision (vision_*) and MoE expert-panel
    # (expert_*) fields use their own taxonomies; feeding them through
    # compute_compare_summary would KeyError on spec.channel_scale.
    channels = [c for c in channels if c in spec.channels]

    summary_channels: dict[str, Any] = {}
    summary = None
    all_artefacts: list[Path] = []
    total = max(1, len(channels))

    for i, channel in enumerate(channels):
        _report(0.1 + 0.8 * (i / total), f"Comparing {channel} field...")
        field_a_path = dir_a / f"field_{channel}_raw.tif"
        field_b_path = dir_b / f"field_{channel}_raw.tif"

        if not field_a_path.exists() or not field_b_path.exists():
            continue

        field_a = read_tif(field_a_path)
        field_b = read_tif(field_b_path)

        summary = compute_compare_summary(
            field_a, field_b, spec,
            mode=mode,
            interp=interp,
            fingerprint_a=fp_a,
            fingerprint_b=fp_b,
            row_labels_a=row_labels_a,
            row_labels_b=row_labels_b,
        )
        summary_channels[channel] = summary.channels[channel]

        delta_path = out / f"delta_{channel}_raw.tif"
        write_tif(delta_path, summary.channels[channel].delta)
        all_artefacts.append(delta_path)

        # Additive M9 artefacts: per-channel |delta| rasters that the
        # noise-floor veil consumes.
        field_delta_raw = out / f"field_delta_{channel}_raw.tif"
        write_tif(field_delta_raw, summary.channels[channel].delta)
        all_artefacts.append(field_delta_raw)

        field_delta_smooth = out / f"field_delta_{channel}_smooth.tif"
        write_tif(
            field_delta_smooth,
            smooth(
                upsample(summary.channels[channel].delta, int(spec.grid.get("upsample", 1))),
                float(spec.grid.get("smooth_sigma", 1.0)),
            ),
        )
        all_artefacts.append(field_delta_smooth)

        # Render delta sheet PNGs so the compare report has visuals.
        # The side-effect import registers the "delta" renderer; without it
        # (e.g. plain CLI runs) get_renderer raises KeyError and sheets are
        # silently skipped.
        try:
            import weight_atlas.compare.render  # noqa: F401 — registers "delta"
            from weight_atlas.core.registry import get_renderer

            renderer_cls = get_renderer("delta")
        except KeyError:
            renderer_cls = None  # delta renderer not registered
        if renderer_cls is not None:
            rendered = renderer_cls().render(
                summary.channels[channel].delta,
                spec,
                out / "render",
                channel=channel,
                row_labels=summary.aligned_row_labels,
                col_labels=summary.aligned_col_labels,
                mode=mode,
                model_a=dir_a.name,
                model_b=dir_b.name,
                noise_floor_dir=noise_floor_dir,
            )
            all_artefacts.extend(rendered)

    if not summary_channels or summary is None:
        return []

    _report(0.95, "Writing compare summary...")
    channel_details: dict[str, Any] = {}
    compare_summary: dict[str, Any] = {
        "mode": mode,
        "spec_version": spec.spec_version,
        "model_a": summary.model_a,
        "model_b": summary.model_b,
        "loaders": {
            "a": summary.model_a.get("loader", "unknown"),
            "b": summary.model_b.get("loader", "unknown"),
        },
        "warnings": summary.warnings,
        "alignment": summary.alignment,
        "channels": channel_details,
    }
    for ch_name, ch_delta in summary_channels.items():
        ranking = hotspot_ranking(
            ch_delta, col_labels=summary.aligned_col_labels, top_k=5
        )
        channel_details[ch_name] = {
            "rel_l2": ch_delta.rel_l2,
            "cosine_sim": ch_delta.cosine_sim,
            "hotspot_layer": ch_delta.hotspot_layer,
            "hotspot_slot": ch_delta.hotspot_slot,
            "hotspot_value": ch_delta.hotspot_value,
            "argmax": list(ch_delta.argmax),
            "hotspot_ranking": [
                {"layer": r[0], "slot": r[1], "abs_delta": r[2]} for r in ranking
            ],
        }

    summary_path = out / "compare_summary.json"
    with open(summary_path, "w") as f:
        json.dump(compare_summary, f, indent=2, sort_keys=True)
        f.write("\n")
    all_artefacts.append(summary_path)

    return all_artefacts
