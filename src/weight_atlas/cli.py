"""CLI entry point for weight-atlas.

Uses argparse (stdlib) to avoid an extra dependency. The spec permits
click or argparse and asks for a justification: argparse keeps the
core dependency footprint minimal and is sufficient for the flat
command hierarchy (scan, render, compare).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from weight_atlas.core.registry import get_renderer, list_loaders
from weight_atlas.core.types import AtlasSpec, load_default_spec
from weight_atlas.render import (
    blender,  # noqa: F401 — registers renderer
    fractal,  # noqa: F401 — registers renderer
    matplotlib_sheet,  # noqa: F401 — registers renderer
)
from weight_atlas.scan import scan as run_scan


def _load_spec(spec_path: Path | None) -> AtlasSpec:
    """Load an explicitly-provided spec, else the canonical default."""
    if spec_path is not None:
        return AtlasSpec.from_json(spec_path)
    return load_default_spec()


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
    scan.add_argument("--loader", choices=sorted(list_loaders()), default=None,
                      help="Loader to use (default: auto-detect)")
    scan.add_argument("--jobs", type=int, default=None,
                      help="Parallel statistics workers (default: min(8, cpu_count)); "
                           "results are identical for any value")
    scan.add_argument("--quant-probe", action="store_true",
                      help="Also measure RTN quantizability per tensor (SQNR for INT8 "
                           "per-channel, INT4 group-128, FP8 e4m3); opt-in, adds ~6 "
                           "passes over all weights")

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
    compare.add_argument("--interp", choices=["linear", "nearest"], default=None,
                         help="Aligned-mode row resampling: linear (bilinear) or nearest "
                              "(nearest-layer matching). Default: spec compare.aligned_interp")
    compare.add_argument("--spec", type=Path, default=None, help="Path to atlas spec JSON")
    compare.add_argument("--noise-floor", type=Path, default=None,
                         help="Directory of a calibration compare job; cells with |delta| at or "
                              "below the calibration |delta| get a noise-floor grey veil on the "
                              "delta sheets")

    paired = sub.add_parser("paired", aliases=["qimpact"], help="Paired tensor-difference analysis between two weight snapshots (M9): quant-impact or edit-signatures")
    paired.set_defaults(alias="paired")
    paired.add_argument("scan_a", type=Path, help="Directory containing scan artefacts for model A")
    paired.add_argument("scan_b", type=Path, help="Directory containing scan artefacts for model B")
    paired.add_argument("--weights-a", type=Path, required=True, help="Path to model A weights (GGUF/safetensors)")
    paired.add_argument("--weights-b", type=Path, required=True, help="Path to model B weights (GGUF/safetensors)")
    paired.add_argument("--out", type=Path, required=True, help="Output directory for paired artefacts")
    paired.add_argument("--preset", choices=["quant", "edit"], default="quant",
                        help="Analysis preset: quant (quantization impact, default) or edit (edit signatures / abliteration)")
    paired.add_argument("--ref-side", choices=["a", "b"], default="a",
                        help="Reference side for SQNR/rel-L2 (default: a)")
    paired.add_argument("--mode", default="strict",
                        help="Paired analysis is strict-only (default: strict); "
                             "any other value is rejected")
    paired.add_argument("--jobs", type=int, default=None,
                        help="Parallel measurement workers (default: min(8, cpu_count)); "
                             "results are identical for any value")
    paired.add_argument("--spec", type=Path, default=None, help="Path to atlas spec JSON")

    activity = sub.add_parser("activity", help="Capture forward-pass activations (fMRI mode)")
    activity.add_argument("model_path", type=Path, help="Path to HuggingFace model directory")
    activity.add_argument("--out", type=Path, required=True, help="Output directory")
    activity.add_argument("--protocol", default="v1", help="Protocol version (default: v1)")
    activity.add_argument("--device", default="cpu", help="Device (default: cpu)")
    activity.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="Data type")
    activity.add_argument("--seed", type=int, default=0, help="Random seed for determinism")
    activity.add_argument("--max-layers", type=int, default=None, help="Max layers to capture")
    activity.add_argument("--probes", default="",
                          help="Comma-separated opt-in activity probes: actq,fragility,linattn "
                               "(default: none — pinned baseline capture)")

    diagnose = sub.add_parser("diagnose", help="Diagnose tensor name mapping coverage")
    diagnose.add_argument("path", type=Path, help="Path to model file or directory")
    diagnose.add_argument("--spec", type=Path, default=None, help="Path to atlas spec JSON")
    diagnose.add_argument("--loader", choices=sorted(list_loaders()), default=None,
                          help="Loader to use (default: auto-detect)")
    diagnose.add_argument("--threshold", type=float, default=0.8,
                          help="Warning threshold for in_slots ratio (default: 0.8)")

    serve = sub.add_parser("serve", help="Run the web UI (LAN-reachable by default)")
    serve.add_argument("--host", default="0.0.0.0",
                       help="Interface to bind (default: 0.0.0.0 = all interfaces / LAN; use 127.0.0.1 for localhost-only)")
    serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")

    return parser


def _cmd_scan(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)

    try:
        artefacts = run_scan(args.path, args.out, spec, loader_id=args.loader, jobs=args.jobs,
                             quant_probe=args.quant_probe)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Warn when mapping coverage is poor. The fingerprint was already written by
    # run_scan — do not re-open the loader or recompute statistics here.
    fp_path = args.out / "fingerprint.json"
    if fp_path.exists():
        with open(fp_path) as f:
            fp = json.load(f)
        mc = fp.get("mapping_coverage", {})
        ratio = float(mc.get("in_slots", 1.0))
        if ratio < 0.8:
            print(
                f"WARNING: mapping coverage {ratio:.1%} < 80% "
                f"({mc.get('unmapped', 0)} of {fp.get('model', {}).get('n_tensors', 0)} "
                "tensors unmapped). Run 'weight-atlas diagnose "
                f"{args.path}' for details.",
                file=sys.stderr,
            )

    for a in artefacts:
        print(a)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest.json not found in {out_dir}; run scan first", file=sys.stderr)
        return 1

    # Load spec from scan artefacts: look for a recorded spec_version.
    # We reconstruct from the default spec; in future the scan could emit
    # the spec it used.
    spec = load_default_spec()

    renderer_cls = get_renderer(args.renderer)
    renderer = renderer_cls()

    from weight_atlas.compare.pipeline import discover_channels_from_manifest
    from weight_atlas.fields.rasterizer import load_channel_field

    # Discover channels from manifest (source of truth)
    manifest = json.loads(manifest_path.read_text())
    channels = discover_channels_from_manifest(manifest)

    produced: list[Path] = []
    for channel in channels:
        field = load_channel_field(out_dir, channel, spec, model_name=out_dir.name)
        if field is None:
            continue

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
    from weight_atlas.compare.pipeline import run_compare

    spec = _load_spec(args.spec)

    fp_a_path = args.dir_a / "fingerprint.json"
    fp_b_path = args.dir_b / "fingerprint.json"
    fp_a = json.loads(fp_a_path.read_text()) if fp_a_path.exists() else None
    fp_b = json.loads(fp_b_path.read_text()) if fp_b_path.exists() else None

    if not (args.dir_a / "manifest.json").exists():
        print(f"manifest.json not found in {args.dir_a}; run scan first", file=sys.stderr)
        return 1

    artefacts = run_compare(
        args.dir_a, args.dir_b, args.out, spec,
        mode=args.mode,
        interp=args.interp,
        row_labels_a=_get_row_labels_from_fingerprint(fp_a),
        row_labels_b=_get_row_labels_from_fingerprint(fp_b),
        noise_floor_dir=args.noise_floor,
    )
    if not artefacts:
        print(
            "no channels to compare (no matching field_*_raw.tif artefacts "
            f"in {args.dir_a} and {args.dir_b})",
            file=sys.stderr,
        )
        return 0

    for p in artefacts:
        print(p)
    return 0


def _cmd_paired(args: argparse.Namespace) -> int:
    """Run paired tensor-difference analysis (quant or edit preset)."""
    from weight_atlas.paired import run_paired

    spec = _load_spec(args.spec)

    fp_a_path = args.scan_a / "fingerprint.json"
    fp_b_path = args.scan_b / "fingerprint.json"
    fp_a = json.loads(fp_a_path.read_text()) if fp_a_path.exists() else None
    fp_b = json.loads(fp_b_path.read_text()) if fp_b_path.exists() else None

    try:
        artefacts = run_paired(
            args.weights_a,
            args.weights_b,
            args.out,
            spec,
            fp_a=fp_a,
            fp_b=fp_b,
            ref_side=args.ref_side,
            jobs=args.jobs,
            mode=args.mode,
            preset=args.preset,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for a in artefacts:
        print(a)
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
    probes = tuple(p.strip() for p in args.probes.split(",") if p.strip())
    config = CaptureConfig(
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        max_layers=args.max_layers,
        probes=probes,
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


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the web UI on the given interface."""
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required for the web UI. Install with: pip install -e '.[web]'",
            file=sys.stderr,
        )
        return 1

    # Timestamped terminal output so serve logs can be correlated in time —
    # with long-running scan jobs the default untimestaped uvicorn lines are
    # useless for "when did this happen" questions.
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt=datefmt,
    )
    # uvicorn replaces its own loggers' handlers at startup (log_config);
    # give both uvicorn formatters timestamps too.
    log_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s %(levelprefix)s %(message)s",
                "datefmt": datefmt,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                "datefmt": datefmt,
            },
        },
        "handlers": {
            "default": {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"},
            "access": {"formatter": "access", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"},
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }

    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not loopback:
        print(
            f"WARNING [{time.strftime(datefmt)}]: serving on {args.host} exposes the web UI to the LAN. "
            "Weight Atlas has no authentication and its API can read scan "
            "directories and serve artefacts. Run only on a trusted network "
            "or behind a firewall/VPN.",
            file=sys.stderr,
        )

    url_host = "127.0.0.1" if loopback else "<this-machine-lan-ip>"
    print(f"Web UI [{time.strftime(datefmt)}]: http://{url_host}:{args.port}  (Ctrl+C to stop)")
    # Factory mode: uvicorn calls create_app() per worker instead of importing a
    # module-level app, so the job worker thread only starts on lifespan startup
    # (no import-time side effects; safe under --reload).
    uvicorn.run(
        "weight_atlas.api.main:create_app",
        log_config=log_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )
    return 0

def _cmd_diagnose(args: argparse.Namespace) -> int:
    """Diagnose tensor name mapping coverage for a model."""
    from weight_atlas.core.registry import get_loader
    from weight_atlas.core.types import detect_loader

    try:
        loader_id = args.loader or detect_loader(args.path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
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
    if args.command in ("paired", "qimpact"):
        return _cmd_paired(args)
    if args.command == "activity":
        return _cmd_activity(args)
    if args.command == "diagnose":
        return _cmd_diagnose(args)
    if args.command == "serve":
        return _cmd_serve(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
