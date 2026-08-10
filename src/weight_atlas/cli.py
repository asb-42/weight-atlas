"""CLI entry point for weight-atlas.

Uses argparse (stdlib) to avoid an extra dependency. The spec permits
click or argparse and asks for a justification: argparse keeps the
core dependency footprint minimal and is sufficient for the flat
command hierarchy (scan, render, compare).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from weight_atlas.core.registry import get_renderer
from weight_atlas.core.types import AtlasSpec
from weight_atlas.render import (
    blender,  # noqa: F401 — registers renderer
    matplotlib_sheet,  # noqa: F401 — registers renderer
)
from weight_atlas.scan import scan as run_scan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weight-atlas",
        description="LLM weight fingerprinting and topographic visualization",
    )
    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser("scan", help="Scan a model into artefacts (auto-detects format)")
    scan.add_argument("path", type=Path, help="Path to model file or directory (.safetensors or .gguf)")
    scan.add_argument("--out", type=Path, required=True, help="Output directory")
    scan.add_argument("--spec", type=Path, default=None, help="Path to atlas spec JSON")
    scan.add_argument("--loader", choices=["safetensors", "gguf"], default=None,
                      help="Loader to use (default: auto-detect)")

    render = sub.add_parser("render", help="Render PNGs from scan artefacts")
    render.add_argument("out_dir", type=Path, help="Directory containing scan artefacts")
    render.add_argument("--renderer", default="sheet", help="Renderer plugin id")
    render.add_argument("--field", default="height", help="Field to render (default: height, use expert_mlp_down for MoE)")

    compare = sub.add_parser("compare", help="Compare two scanned models quantitatively")
    compare.add_argument("dir_a", type=Path, help="Directory containing scan artefacts for model A")
    compare.add_argument("dir_b", type=Path, help="Directory containing scan artefacts for model B")
    compare.add_argument("--out", type=Path, required=True, help="Output directory for comparison artefacts")
    compare.add_argument("--mode", choices=["strict", "aligned"], default="strict",
                         help="Comparison mode (default: strict)")
    compare.add_argument("--spec", type=Path, default=None, help="Path to atlas spec JSON")

    activity = sub.add_parser("activity", help="Capture forward-pass activations (fMRI mode)")
    activity.add_argument("model_path", type=Path, help="Path to HuggingFace model directory")
    activity.add_argument("--out", type=Path, required=True, help="Output directory")
    activity.add_argument("--protocol", default="v1", help="Protocol version (default: v1)")
    activity.add_argument("--device", default="cpu", help="Device (default: cpu)")
    activity.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="Data type")
    activity.add_argument("--seed", type=int, default=0, help="Random seed for determinism")
    activity.add_argument("--max-layers", type=int, default=None, help="Max layers to capture")

    diagnose = sub.add_parser("diagnose", help="Diagnose tensor name mapping coverage")
    diagnose.add_argument("path", type=Path, help="Path to model file or directory")
    diagnose.add_argument("--spec", type=Path, default=None, help="Path to atlas spec JSON")
    diagnose.add_argument("--loader", choices=["safetensors", "gguf"], default=None,
                          help="Loader to use (default: auto-detect)")
    diagnose.add_argument("--threshold", type=float, default=0.8,
                          help="Warning threshold for in_slots ratio (default: 0.8)")

    return parser


def _cmd_scan(args: argparse.Namespace) -> int:
    spec_path = args.spec or Path("specs/atlas_spec.v2.json")
    spec = AtlasSpec.from_json(spec_path)


    artefacts = run_scan(args.path, args.out, spec, loader_id=args.loader)

    # Check mapping coverage and warn if < 80%
    from weight_atlas.core.registry import get_loader
    from weight_atlas.core.types import detect_loader
    from weight_atlas.scan import _build_fingerprint
    loader_id = args.loader or detect_loader(args.path)
    loader = get_loader(loader_id)()
    handles = list(loader.open(args.path))
    from weight_atlas.scan import _make_handles
    stats = [_make_handles(h) for h in handles]
    fp = _build_fingerprint(stats, spec, loader_id, handles)
    mc = fp.get("mapping_coverage", {})
    ratio = mc.get("ratio", 1.0)
    if ratio < 0.8:
        print(f"WARNING: mapping coverage {ratio:.1%} < 80% "
              f"({mc.get('in_slots', 0)}/{mc.get('total', 0)} tensors in slots). "
              f"Run 'weight-atlas diagnose {args.path}' for details.",
              file=sys.stderr)

    for a in artefacts:
        print(a)
    return 0


def _discover_channels_from_manifest(manifest: dict[str, str]) -> list[str]:
    """Discover channel names from manifest keys (field_<channel>_.tif).

    Uses manifest.json as source of truth for which channels exist.
    """
    channels: set[str] = set()
    for key in manifest:
        if not key.startswith("field_") or not key.endswith(".tif"):
            continue
        # Strip prefix and suffix to get channel_smooth or channel_raw
        core = key[len("field_"):-len(".tif")]
        # Remove _raw or _smooth suffix
        if core.endswith("_raw"):
            channels.add(core[:-len("_raw")])
        elif core.endswith("_smooth"):
            channels.add(core[:-len("_smooth")])
    return sorted(channels)


def _cmd_render(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest.json not found in {out_dir}; run scan first", file=sys.stderr)
        return 1

    # Load spec from scan artefacts: look for a recorded spec_version.
    # We reconstruct from the default spec; in future the scan could emit
    # the spec it used.
    spec = AtlasSpec.from_json(Path("specs/atlas_spec.v2.json"))

    renderer_cls = get_renderer(args.renderer)
    renderer = renderer_cls()

    from weight_atlas.core.types import Field2D
    from weight_atlas.fields.tif_io import read_tif

    # Discover channels from manifest (source of truth)
    manifest = json.loads(manifest_path.read_text())
    channels = _discover_channels_from_manifest(manifest)

    produced: list[Path] = []
    for channel in channels:
        # Render the smooth version if available, otherwise raw
        smooth_path = out_dir / f"field_{channel}_smooth.tif"
        raw_path = out_dir / f"field_{channel}_raw.tif"
        tif = smooth_path if smooth_path.exists() else raw_path
        if not tif.exists():
            continue

        data = read_tif(tif)

        # Apply channel scaling for better visualization
        from weight_atlas.fields.scaling import apply_scale
        ch_spec = spec.channels.get(channel, {})
        if "scale" in ch_spec:
            data = apply_scale(data, ch_spec["scale"])

        field = Field2D(
            channel=channel,
            data=data,
            spec_version=spec.spec_version,
        )
        # Add scatter overlay for embedding density field
        scatter_path = out_dir / "embedding_scatter.npy"
        if args.renderer == "blender":
            produced.extend(renderer.render(field, spec, out_dir / "render", field_name=args.field))
        else:
            produced.extend(renderer.render(field, spec, out_dir / "render", scatter_path=scatter_path if scatter_path.exists() else None))

    for p in produced:
        print(p)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Run comparison between two scanned model directories."""
    from weight_atlas.compare import compute_compare_summary, hotspot_ranking
    from weight_atlas.fields.tif_io import read_tif

    spec_path = args.spec or Path("specs/atlas_spec.v2.json")
    spec = AtlasSpec.from_json(spec_path)



    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    # Load fingerprints if available
    fp_a_path = args.dir_a / "fingerprint.json"
    fp_b_path = args.dir_b / "fingerprint.json"
    fp_a = json.loads(fp_a_path.read_text()) if fp_a_path.exists() else None
    fp_b = json.loads(fp_b_path.read_text()) if fp_b_path.exists() else None

    # Discover channels from manifest (use dir_a as reference)
    manifest_path = args.dir_a / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest.json not found in {args.dir_a}; run scan first", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    channels = _discover_channels_from_manifest(manifest)

    # For each channel, load raw fields, compute delta
    summary_channels = {}
    all_artefacts: list[Path] = []

    for channel in channels:
        field_a_path = args.dir_a / f"field_{channel}_raw.tif"
        field_b_path = args.dir_b / f"field_{channel}_raw.tif"

        if not field_a_path.exists() or not field_b_path.exists():
            continue

        field_a = read_tif(field_a_path)
        field_b = read_tif(field_b_path)

        # Get row labels from fingerprint (layer order)
        row_labels_a = _get_row_labels_from_fingerprint(fp_a) if fp_a else None
        row_labels_b = _get_row_labels_from_fingerprint(fp_b) if fp_b else None

        summary = compute_compare_summary(
            field_a, field_b, spec,
            mode=args.mode,
            fingerprint_a=fp_a,
            fingerprint_b=fp_b,
            row_labels_a=row_labels_a,
            row_labels_b=row_labels_b,
        )
        summary_channels[channel] = summary.channels[channel]

        # Write delta field as TIFF
        delta_path = out / f"delta_{channel}_raw.tif"
        from weight_atlas.fields.tif_io import write_tif
        write_tif(delta_path, summary.channels[channel].delta)
        all_artefacts.append(delta_path)

        # Render delta sheet
        try:
            renderer = get_renderer("delta")()
            rendered = renderer.render(
                summary.channels[channel].delta,
                spec,
                out / "render",
                channel=channel,
                row_labels=summary.aligned_row_labels,
                col_labels=summary.aligned_col_labels,
                mode=args.mode,
            )
            all_artefacts.extend(rendered)
        except KeyError:
            pass  # delta renderer not registered

    # Compare expert panels (MoE)
    from weight_atlas.compare.panel import compare_expert_panels
    from weight_atlas.core.types import ExpertPanel

    panels_a = []
    panels_b = []

    # Load panels from both directories
    for channel in channels:
        for slot in ["mlp_gate", "mlp_up", "mlp_down"]:
            panel_a_path = args.dir_a / f"field_expert_{slot}_{channel}_raw.tif"
            panel_b_path = args.dir_b / f"field_expert_{slot}_{channel}_raw.tif"

            if panel_a_path.exists() and panel_b_path.exists():
                from weight_atlas.fields.tif_io import read_tif
                data_a = read_tif(panel_a_path)
                data_b = read_tif(panel_b_path)
                panels_a.append(ExpertPanel(slot=slot, channel=channel, data=data_a))
                panels_b.append(ExpertPanel(slot=slot, channel=channel, data=data_b))

    if panels_a and panels_b:
        compare_expert_panels(panels_a, panels_b, spec, mode=args.mode)

    # Write compare_summary.json
    compare_summary = {
        "mode": args.mode,
        "spec_version": spec.spec_version,
        "model_a": summary.model_a,
        "model_b": summary.model_b,
        "loaders": {"a": summary.model_a.get("loader", "unknown"), "b": summary.model_b.get("loader", "unknown")},
        "warnings": summary.warnings,
        "channels": {},
    }
    for ch_name, ch_delta in summary_channels.items():
        ranking = hotspot_ranking(ch_delta, col_labels=summary.aligned_col_labels, top_k=5)
        compare_summary["channels"][ch_name] = {
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

    for p in all_artefacts:
        print(p)
    return 0


def _get_row_labels_from_fingerprint(fp: dict | None) -> list[str] | None:
    """Extract ordered row labels from a fingerprint.

    Returns layer indices sorted numerically.
    """
    if fp is None:
        return None
    tensors = fp.get("tensors", {})
    layers: set[int] = set()
    from weight_atlas.core.name_map import map_name
    for name in tensors:
        layer, _ = map_name(name)
        if layer is not None:
            layers.add(layer)
    return [str(layer_idx) for layer_idx in sorted(layers)]



def _cmd_activity(args: argparse.Namespace) -> int:
    """Run activity capture (fMRI mode)."""
    from weight_atlas.activity import capture_activity, load_protocol
    from weight_atlas.activity.capture import CaptureConfig

    protocol = load_protocol(args.protocol)
    config = CaptureConfig(
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        max_layers=args.max_layers,
    )

    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    metadata = capture_activity(model, tokenizer, protocol, config, args.out)

    print(f"Activity capture complete. Protocol: {metadata['protocol_version']}")
    print(f"Protocol hash: {metadata['protocol_hash']}")
    print(f"States: {metadata['states']}")
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "scan":
        return _cmd_scan(args)
    if args.command == "render":
        return _cmd_render(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "activity":
        return _cmd_activity(args)
    if args.command == "diagnose":
        return _cmd_diagnose(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())



def _cmd_diagnose(args: argparse.Namespace) -> int:
    """Diagnose tensor name mapping coverage for a model."""
    from weight_atlas.core.registry import get_loader
    from weight_atlas.core.types import detect_loader




    loader_id = args.loader or detect_loader(args.path)
    loader = get_loader(loader_id)()
    handles = list(loader.open(args.path))

    # Build mapping report
    from weight_atlas.core.name_map import map_name
    total = len(handles)
    slot_counts: dict[str, int] = {}
    unmapped: list[str] = []
    for h in handles:
        layer, slot = map_name(h.name)
        slot_counts[slot] = slot_counts.get(slot, 0) + 1
        if slot == "other":
            unmapped.append(h.name)

    in_slots = total - len(unmapped)
    ratio = in_slots / total if total > 0 else 0.0

    print(f"Model: {args.path}")
    print(f"Loader: {loader_id}")
    print(f"Total tensors: {total}")
    print(f"In slots: {in_slots} ({ratio:.1%})")
    print(f"Unmapped: {len(unmapped)}")
    print()
    print("Slot distribution:")
    for slot, count in sorted(slot_counts.items(), key=lambda x: -x[1]):
        print(f"  {slot:20s} {count:4d}")
    if unmapped:
        print()
        print("Unmapped tensors:")
        for name in unmapped:
            print(f"  {name}")

    # Warning if below threshold
    if ratio < args.threshold:
        print()
        print(f"WARNING: in_slots ratio {ratio:.1%} < {args.threshold:.0%} threshold.",
              file=sys.stderr)
        return 1
    return 0
