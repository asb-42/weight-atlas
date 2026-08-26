"""MoE Expert Panel comparison: compare Layer × Expert panels between two models."""

from __future__ import annotations

from dataclasses import dataclass

from weight_atlas.compare.align import align
from weight_atlas.compare.delta import ChannelDelta, compute_delta
from weight_atlas.core.types import AtlasSpec, ExpertPanel


@dataclass
class PanelCompareResult:
    """Result of comparing two expert panels."""

    slot: str
    channel: str
    status: str  # "compared" or "skipped"
    reason: str = ""
    delta: ChannelDelta | None = None


def compare_expert_panels(
    panels_a: list[ExpertPanel],
    panels_b: list[ExpertPanel],
    spec: AtlasSpec,
    *,
    mode: str = "strict",
) -> list[PanelCompareResult]:
    """Compare expert panels between two models.

    Args:
        panels_a: Expert panels from model A
        panels_b: Expert panels from model B
        spec: Atlas specification
        mode: Comparison mode ("strict" or "aligned")

    Returns:
        List of PanelCompareResult objects
    """
    results: list[PanelCompareResult] = []

    # Create lookup by (slot, channel)
    panels_a_dict = {(p.slot, p.channel): p for p in panels_a}
    panels_b_dict = {(p.slot, p.channel): p for p in panels_b}

    # Get all unique (slot, channel) combinations
    all_keys = set(panels_a_dict.keys()) | set(panels_b_dict.keys())

    for key in sorted(all_keys):
        slot, channel = key
        panel_a = panels_a_dict.get(key)
        panel_b = panels_b_dict.get(key)

        if panel_a is None or panel_b is None:
            results.append(PanelCompareResult(
                slot=slot,
                channel=channel,
                status="skipped",
                reason=f"Panel missing in {'B' if panel_a else 'A'}",
            ))
            continue

        # Check shape compatibility
        if panel_a.data.shape != panel_b.data.shape:
            results.append(PanelCompareResult(
                slot=slot,
                channel=channel,
                status="skipped",
                reason=f"Shape mismatch: {panel_a.data.shape} vs {panel_b.data.shape}",
            ))
            continue

        # Align and compute delta; honour the spec's aligned_interp like the
        # main compare path does.
        from weight_atlas.compare.delta import _get_aligned_interp

        aligned = align(
            panel_a.data, panel_b.data, spec,
            mode=mode,
            interp=_get_aligned_interp(spec),
        )
        # Hotspot slots report real expert IDs when the panels carry them
        # (rasterize_expert_panels fills col_labels with the expert ids);
        # fall back to positional indices for legacy panels.
        aligned.col_labels = (
            list(panel_a.col_labels)
            or [str(i) for i in range(panel_a.data.shape[1])]
        )
        delta = compute_delta(aligned, channel, spec)

        results.append(PanelCompareResult(
            slot=slot,
            channel=channel,
            status="compared",
            delta=delta,
        ))

    return results


def discover_panels_from_manifest(manifest: dict[str, str]) -> list[str]:
    """Discover expert panel field names from manifest.

    Returns:
        List of panel field names (e.g., ["expert_mlp_gate_height", "expert_mlp_up_height", ...])
    """
    panels: set[str] = set()
    for key in manifest:
        if not key.startswith("field_expert_") or not key.endswith(".tif"):
            continue
        # Extract panel field name: field_expert_<slot>_<channel>_<raw/small>.tif
        core = key[len("field_expert_"):-len(".tif")]
        # Remove _raw or _smooth suffix
        if core.endswith("_raw"):
            panels.add(core[:-len("_raw")])
        elif core.endswith("_smooth"):
            panels.add(core[:-len("_smooth")])
    return sorted(panels)
