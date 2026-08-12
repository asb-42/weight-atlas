"""Scan pipeline: load → stats → fields → artefacts + manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from weight_atlas.core.name_map import map_name
from weight_atlas.core.registry import get_loader
from weight_atlas.core.types import AtlasSpec, TensorHandle, TensorStats, detect_loader
from weight_atlas.fields.rasterizer import detect_moe, rasterize, rasterize_expert_panels
from weight_atlas.fields.scaling import apply_scale, log1p
from weight_atlas.fields.tif_io import write_tif
from weight_atlas.loaders import (
    gguf_loader,  # noqa: F401 — triggers registration
    safetensors_loader,  # noqa: F401 — triggers registration
)
from weight_atlas.stats.norms import EffectiveRank, FrobeniusNorm, SpectralNorm
from weight_atlas.stats.shape_moments import Kurtosis, Sparsity
from weight_atlas.stats.stable_rank import StableRank


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_handles(tensor: TensorHandle) -> TensorStats:
    """Compute all registered statistics for one tensor."""
    return TensorStats(
        name=tensor.name,
        shape=tensor.shape,
        frobenius=FrobeniusNorm().compute(tensor),
        spectral_norm=SpectralNorm().compute(tensor),
        effective_rank=EffectiveRank().compute(tensor),
        stable_rank=StableRank().compute(tensor),
        kurtosis=Kurtosis().compute(tensor),
        sparsity=Sparsity().compute(tensor),
        expert_id=tensor.expert_id,
    )


def scan(
    model_path: Path,
    out: Path,
    spec: AtlasSpec,
    *,
    loader_id: str | None = None,
) -> list[Path]:
    """Run the full scan pipeline.

    Produces:
    - fingerprint.json (sorted keys, indent 2)
    - field_<channel>_raw.tif
    - field_<channel>_smooth.tif
    - field_expert_<slot>_{raw,smooth}.tif (for MoE models)
    - manifest.json (sha256 per artefact)

    Args:
        model_path: path to model file or directory
        out: output directory
        spec: atlas specification
        loader_id: override loader (default: auto-detect)
    """
    out.mkdir(parents=True, exist_ok=True)

    # Auto-detect loader if not specified
    if loader_id is None:
        loader_id = detect_loader(model_path)

    loader = get_loader(loader_id)()
    handles = list(loader.open(model_path))
    stats = [_make_handles(h) for h in handles]

    fingerprint = _build_fingerprint(stats, spec, loader_id, handles)

    # Compute scaling metadata for fingerprint (v2.1)
    scaling_meta = _compute_scaling_metadata(stats, spec)
    if scaling_meta:
        fingerprint["scaling"] = scaling_meta

    fp_path = out / "fingerprint.json"
    with open(fp_path, "w") as f:
        json.dump(fingerprint, f, indent=2, sort_keys=True)
        f.write("\n")

    artefacts: list[Path] = [fp_path]
    for channel, ch_spec in spec.channels.items():
        stat_key = ch_spec["stat"]
        field_raw = rasterize(stats, spec, stat_key)
        raw_path = out / f"field_{channel}_raw.tif"
        write_tif(raw_path, field_raw.data)
        artefacts.append(raw_path)

        # v2.1 pipeline: apply pre-transform (e.g. log1p) then robust_scale
        pre = ch_spec.get("pre")
        data = field_raw.data
        if pre == "log1p":
            data = log1p(data)
        scaled = apply_scale(data, ch_spec["scale"])
        from weight_atlas.fields.degenerations import diagnose_fields
        from weight_atlas.fields.smoothing import smooth, upsample

        up = upsample(scaled, int(spec.grid["upsample"]))
        smoothed = smooth(up, float(spec.grid["smooth_sigma"]))
        smooth_path = out / f"field_{channel}_smooth.tif"
        write_tif(smooth_path, smoothed)
        artefacts.append(smooth_path)

        # MoE expert panels
        expert_panels = rasterize_expert_panels(stats, spec, stat_key)
        for panel in expert_panels:
            panel_raw_path = out / f"field_expert_{panel.slot}_{channel}_raw.tif"
            write_tif(panel_raw_path, panel.data)
            artefacts.append(panel_raw_path)

            # v2.1 pipeline: apply pre-transform then robust_scale
            panel_data = panel.data
            if pre == "log1p":
                panel_data = log1p(panel_data)
            scaled_panel = apply_scale(panel_data, ch_spec["scale"])
            up_panel = upsample(scaled_panel, int(spec.grid["upsample"]))
            smoothed_panel = smooth(up_panel, float(spec.grid["smooth_sigma"]))
            panel_smooth_path = out / f"field_expert_{panel.slot}_{channel}_smooth.tif"
            write_tif(panel_smooth_path, smoothed_panel)
            artefacts.append(panel_smooth_path)

    # Embedding projection (PCA or UMAP)
    embedding_spec = getattr(spec, 'embedding', {})
    if embedding_spec:
        method = embedding_spec.get('method', 'pca')
        grid_size = embedding_spec.get('grid', 256)
        n_components = embedding_spec.get('components', 3)
        subsample = embedding_spec.get('subsample_scatter', 5000)
        seeds = embedding_spec.get('seeds', {'pca': 0, 'umap': 0})

        # Find embedding tensor (handle HF, GGUF, and prefixed VLM naming e.g.
        # Kimi K3's ``language_model.model.embed_tokens.weight``).
        embed_tensor = None
        for h in handles:
            if h.name.endswith(('model.embed_tokens.weight', 'token_embd.weight')):
                embed_tensor = h
                break

        if embed_tensor is not None:
            embeddings = embed_tensor.load()  # (V, D)

            if method == 'umap':
                from weight_atlas.embedding.umap import compute_umap
                projected, umap_meta = compute_umap(
                    embeddings,
                    n_components=2,
                    seed=seeds.get('umap', 0),
                )
                # Save UMAP result
                np.save(out / 'embedding_umap.npy', projected.astype(np.float32))
                artefacts.append(out / 'embedding_umap.npy')
                embedding_meta = umap_meta
            else:
                # PCA (default)
                from weight_atlas.embedding.pca import (
                    compute_pca,
                    embedding_to_density,
                    project_with_pca,
                )
                components, explained_variance, mean = compute_pca(
                    embeddings,
                    n_components=n_components,
                    seed=seeds.get('pca', 0),
                )
                projected = project_with_pca(embeddings, components, mean)

                # Save PCA result
                np.save(out / 'embedding_pca.npy', projected.astype(np.float32))
                artefacts.append(out / 'embedding_pca.npy')

                # Create density field
                density = embedding_to_density(
                    projected[:, :2],
                    grid_size=grid_size,
                    subsample=subsample,
                    seed=seeds.get('pca', 0),
                )

                # Write density TIFFs
                raw_path = out / 'field_embed_density_raw.tif'
                write_tif(raw_path, density)
                artefacts.append(raw_path)

                from weight_atlas.fields.degenerations import diagnose_fields
                from weight_atlas.fields.smoothing import smooth, upsample

                scaled = apply_scale(density, {'type': 'log1p'})
                up = upsample(scaled, int(spec.grid['upsample']))
                smoothed = smooth(up, float(spec.grid['smooth_sigma']))
                smooth_path = out / 'field_embed_density_smooth.tif'
                write_tif(smooth_path, smoothed)
                artefacts.append(smooth_path)

                # Save scatter coordinates (subsampled for visualization)
                subsample_scatter = embedding_spec.get('subsample_scatter', 5000)
                scatter_seed = seeds.get('pca', 0)
                rng = np.random.default_rng(scatter_seed)
                n_points = projected.shape[0]
                if n_points > subsample_scatter:
                    scatter_indices = rng.choice(n_points, size=subsample_scatter, replace=False)
                    scatter_coords = projected[scatter_indices, :2]
                else:
                    scatter_coords = projected[:, :2]

                np.save(out / 'embedding_scatter.npy', scatter_coords.astype(np.float32))
                artefacts.append(out / 'embedding_scatter.npy')

                embedding_meta = {
                    'method': 'pca',
                    'explained_variance': explained_variance.tolist(),
                    'n_components': n_components,
                    'sign_convention': 'max_abs_positive',
                    'scatter_subsample': subsample_scatter,
                    'scatter_seed': scatter_seed,
                }

            # Save embedding metadata
            with open(out / 'embedding_meta.json', 'w') as f:
                json.dump(embedding_meta, f, indent=2)
            artefacts.append(out / 'embedding_meta.json')

    # Degeneration checks on raw fields
    fields_for_diag = {}
    for channel in spec.channels:
        raw_path = out / f"field_{channel}_raw.tif"
        if raw_path.exists():
            from weight_atlas.fields.tif_io import read_tif
            fields_for_diag[channel] = read_tif(raw_path)
    if fields_for_diag:
        degen_report = diagnose_fields(fields_for_diag)
        if degen_report.warnings:
            fingerprint["warnings"] = fingerprint.get("warnings", []) + degen_report.warnings

    manifest = {str(p.relative_to(out)): _sha256(p) for p in artefacts}
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return artefacts + [manifest_path]


def _build_fingerprint(
    stats: Iterable[TensorStats],
    spec: AtlasSpec,
    loader_id: str,
    handles: list[TensorHandle] | None = None,
) -> dict:
    """Build the fingerprint dict from computed tensor statistics."""
    # Get tool version from package metadata
    try:
        tool_version = importlib.metadata.version("weight-atlas")
    except importlib.metadata.PackageNotFoundError:
        tool_version = "0.2.0"

    out: dict = {
        "spec_version": spec.spec_version,
        "tool_version": tool_version,
        "loader": loader_id,
        "model": {"n_tensors": 0, "n_layers": 0},
        "tensors": {},
    }
    layers: set[int] = set()

    # Build ggml_type mapping from handles if available
    ggml_types: dict[str, str] = {}
    if handles:
        for h in handles:
            if h.dtype.startswith("ggml_"):
                ggml_types[h.name] = h.dtype

    for ts in stats:
        layer, slot = map_name(ts.name)
        if layer is not None:
            layers.add(layer)
        tensor_info = {
            "shape": list(ts.shape),
            "frobenius": ts.frobenius,
            "spectral_norm": ts.spectral_norm,
            "effective_rank": ts.effective_rank,
            "stable_rank": ts.stable_rank,
            "kurtosis": ts.kurtosis,
            "sparsity": ts.sparsity,
        }
        # Add ggml_type if present
        if ts.name in ggml_types:
            tensor_info["ggml_type"] = ggml_types[ts.name]
        out["tensors"][ts.name] = tensor_info

    out["model"]["n_tensors"] = len(out["tensors"])
    out["model"]["n_layers"] = len(layers)

    # Add mapping coverage (name audit)
    n_mapped = sum(1 for name in out["tensors"] if map_name(name)[1] != "other")
    n_total = len(out["tensors"])
    n_unmapped = n_total - n_mapped
    out["mapping_coverage"] = {
        "in_slots": round(n_mapped / n_total, 4) if n_total > 0 else 0.0,
        "in_other": round(n_unmapped / n_total, 4) if n_total > 0 else 0.0,
        "unmapped": n_unmapped,
        "unmapped_tensors": [name for name in out["tensors"] if map_name(name)[1] == "other"][:20],
    }

    # Add quantization summary for GGUF
    if loader_id == "gguf" and ggml_types:
        quant_summary: dict[str, int] = {}
        for ggml_type in ggml_types.values():
            quant_summary[ggml_type] = quant_summary.get(ggml_type, 0) + 1
        out["quantization"] = quant_summary

    # Add MoE info
    moe_info = detect_moe(stats)
    if moe_info:
        out["model"]["moe"] = moe_info

    return out


def _compute_scaling_metadata(stats: Iterable[TensorStats], spec: AtlasSpec) -> dict | None:
    """Compute scaling metadata for fingerprint.json (v2.1).

    For each channel, records the robust scale parameters and the raw/clip bounds.
    Only applies when spec uses robust_scale.
    """
    # Check if any channel uses robust_scale
    has_robust = any(
        ch_spec["scale"]["type"] in ("robust_scale", "quantile_clip")
        for ch_spec in spec.channels.values()
    )
    if not has_robust:
        return None

    # Build per-channel stat arrays from stats
    channels_meta: dict[str, dict] = {}
    for channel, ch_spec in spec.channels.items():
        stat_key = ch_spec["stat"]
        scale_type = ch_spec["scale"]["type"]
        if scale_type not in ("robust_scale", "quantile_clip"):
            continue

        # Collect all values for this stat across tensors
        vals_list: list[float] = []
        for ts in stats:
            v = getattr(ts, stat_key, None)
            if v is not None and np.isfinite(v):
                vals_list.append(float(v))

        if not vals_list:
            continue

        arr = np.array(vals_list, dtype=np.float64)
        raw_min = float(np.min(arr))
        raw_max = float(np.max(arr))
        # Apply pre-transform (e.g. log1p) before computing clip bounds (v2.1)
        pre = ch_spec.get("pre")
        if pre == "log1p":
            arr = np.log1p(np.maximum(arr, 0.0))

        lower = float(ch_spec["scale"].get("lower", ch_spec["scale"].get("lo", 0.01)))
        upper = float(ch_spec["scale"].get("upper", ch_spec["scale"].get("hi", 0.99)))

        q_lo = float(np.quantile(arr, lower))
        q_hi = float(np.quantile(arr, upper))

        channels_meta[channel] = {
            "q_lo": round(q_lo, 4),
            "q_hi": round(q_hi, 4),
            "raw_min": round(raw_min, 4),
            "raw_max": round(raw_max, 4),
        }

    if not channels_meta:
        return None

    return {
        "method": "robust_scale",
        "params": {"lower": 0.01, "upper": 0.99},
        "channels": channels_meta,
    }
