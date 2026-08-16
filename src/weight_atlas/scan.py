"""Scan pipeline: load → stats → fields → artefacts + manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from weight_atlas.core.name_map import map_name
from weight_atlas.core.registry import get_loader
from weight_atlas.core.types import AtlasSpec, TensorHandle, TensorStats, detect_loader
from weight_atlas.fields.rasterizer import (
    detect_moe,
    detect_vision,
    rasterize,
    rasterize_expert_panels,
    rasterize_vision,
)
from weight_atlas.fields.scaling import apply_scale, log1p
from weight_atlas.fields.tif_io import write_tif
from weight_atlas.loaders import (
    gguf_loader,  # noqa: F401 — triggers registration
    safetensors_loader,  # noqa: F401 — triggers registration
)
from weight_atlas.stats.norms import (
    EffectiveRank,
    FrobeniusNorm,
    KernelNorm,
    SpectralNorm,
)
from weight_atlas.stats.shape_moments import Kurtosis, Sparsity
from weight_atlas.stats.stable_rank import StableRank

# Tensors whose float32 materialization is >= this many bytes are computed
# serially (one at a time) during the stats phase to keep peak RAM bounded on
# models with very large tensors (see ``scan``).
_BIG_TENSOR_THRESHOLD_BYTES = 1 << 30  # 1 GiB


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
        kernel_norm=KernelNorm().compute(tensor),
        expert_id=tensor.expert_id,
    )


def _resolve_jobs(jobs: int | None) -> int:
    """Resolve the stats worker count: explicit value, else min(8, cpu_count)."""
    if jobs is not None and jobs > 0:
        return jobs
    import os
    return max(1, min(8, os.cpu_count() or 1))


def _stats_for_handle(h: TensorHandle) -> TensorStats:
    """Compute all statistics for one tensor.

    BLAS threading is capped once by ``scan()`` around the whole parallel
    section (see ``scan``), never per tensor: resizing OpenBLAS's internal
    thread pool from several worker threads at once can deadlock inside
    OpenBLAS (observed on large MoE scans), so the limit must be applied a
    single time before the pool starts, not re-entered for every tensor.
    """
    return _make_handles(h)


def scan(
    model_path: Path,
    out: Path,
    spec: AtlasSpec,
    *,
    loader_id: str | None = None,
    progress: Callable[[float, str], None] | None = None,
    jobs: int | None = None,
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
        progress: optional ``(fraction, message)`` callback reported as each
            phase of the pipeline completes (loading, statistics, rasterizing,
            smoothing, expert panels, embedding, manifest).
        jobs: number of parallel statistics workers (default: min(8, CPUs)).
            Each tensor's statistics are computed independently and
            deterministically, so results are identical for any ``jobs``.
    """
    def _report(pct: float, msg: str) -> None:
        if progress is not None:
            progress(float(pct), msg)

    out.mkdir(parents=True, exist_ok=True)

    # Auto-detect loader if not specified
    if loader_id is None:
        loader_id = detect_loader(model_path)

    _report(0.0, "Opening model...")
    loader = get_loader(loader_id)()
    _report(0.02, "Reading tensor metadata...")
    handles = list(loader.open(model_path))

    # Compute per-tensor statistics (the expensive SVD steps), optionally in
    # parallel across tensors. Every handle's memoized payload is released
    # right after its statistics are computed so the whole model is never held
    # in RAM (~4 bytes/parameter for a 35B MoE would be ~140 GB).
    #
    # Memory bound: computing statistics materializes each tensor as float32
    # and several stats build temporary copies (kurtosis, sparsity, SVD), so a
    # single worker can transiently use ~3-4x the tensor's float32 size. With 8
    # parallel workers on a model whose largest tensors are multi-GB (e.g. Kimi
    # K3's 4.7 GB lm_head/embed_tokens), peak RSS reached ~120 GB and the
    # process was OOM-killed. The few multi-GB tensors are therefore computed
    # serially (one at a time) while the many small tensors still run in
    # parallel for throughput. Per-tensor stats are deterministic regardless of
    # processing order, so output is byte-identical to a fully serial scan.
    n_total = len(handles)
    report_every = max(1, n_total // 40) if n_total else 1
    jobs_n = _resolve_jobs(jobs)

    # Tensors that materialize to >= 1 GiB float32 are handled serially to keep
    # peak RAM bounded even on models with very large tensors.
    big_threshold = _BIG_TENSOR_THRESHOLD_BYTES
    big_idxs = [
        i for i, h in enumerate(handles)
        if int(np.prod(h.shape)) * 4 >= big_threshold
    ]
    small_idxs = [
        i for i, h in enumerate(handles)
        if int(np.prod(h.shape)) * 4 < big_threshold
    ]

    stats: list[TensorStats] = [None] * n_total  # type: ignore[list-item]

    def _report_stats(i: int) -> None:
        if i % report_every == 0 or i == n_total - 1:
            _report(
                0.04 + 0.36 * ((i + 1) / n_total),
                f"Computing statistics ({i + 1}/{n_total})...",
            )

    def _run_stats(idx_iter: Iterable[int], parallel: bool) -> None:
        items = list(idx_iter)
        if parallel and jobs_n > 1 and len(items) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=jobs_n) as ex:
                for i, ts in ex.map(lambda i: (i, _stats_for_handle(handles[i])), items):
                    handles[i].clear()
                    stats[i] = ts
                    _report_stats(i)
        else:
            for i in items:
                ts = _stats_for_handle(handles[i])
                handles[i].clear()
                stats[i] = ts
                _report_stats(i)

    # Small tensors first (parallel), then the few multi-GB tensors serially.
    use_pool = jobs_n > 1 and n_total > 1
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - optional dep
        threadpool_limits = None
    if use_pool and threadpool_limits is not None:
        # The stats thread pool runs numpy/BLAS concurrently. Cap the BLAS
        # thread pool ONCE around the whole section: entering/exiting
        # ``threadpool_limits`` from several worker threads at once resizes
        # OpenBLAS's internal pool concurrently and can deadlock inside
        # OpenBLAS (observed on large MoE scans). A single cap keeps every
        # BLAS call single-threaded with no pool resizing mid-flight.
        try:
            with threadpool_limits(limits=1):
                _run_stats(small_idxs, parallel=True)
                _run_stats(big_idxs, parallel=False)
        except RuntimeError:
            # threadpoolctl present but no supported BLAS loaded → there is no
            # shared thread pool to resize, so parallel numpy is safe.
            _run_stats(small_idxs, parallel=True)
            _run_stats(big_idxs, parallel=False)
    else:
        # jobs=1, or no threadpoolctl → cannot cap BLAS. Concurrent
        # multithreaded OpenBLAS calls from several Python threads risk the
        # same deadlock, so stats run serially (safe, deterministic).
        _run_stats(small_idxs, parallel=False)
        _run_stats(big_idxs, parallel=False)

    assert all(s is not None for s in stats)

    _report(0.42, "Building fingerprint...")
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
    fields_for_diag: dict[str, np.ndarray] = {}
    n_channels = len(spec.channels)
    for ci, (channel, ch_spec) in enumerate(spec.channels.items()):
        chan_lo = 0.46 + 0.34 * (ci / n_channels)
        chan_hi = 0.46 + 0.34 * ((ci + 1) / n_channels)
        stat_key = ch_spec["stat"]
        _report(chan_lo, f"Rasterizing {channel} field ({stat_key})...")
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

        _report(chan_lo + 0.55 * (chan_hi - chan_lo), f"Smoothing {channel} field...")
        up = upsample(scaled, int(spec.grid["upsample"]))
        smoothed = smooth(up, float(spec.grid["smooth_sigma"]))
        smooth_path = out / f"field_{channel}_smooth.tif"
        write_tif(smooth_path, smoothed)
        artefacts.append(smooth_path)

    # MoE expert panels. Expert tensors are the vast majority of a MoE model's
    # tensors, so the panels use the spec's ``expert_channels`` (cheap O(n)
    # statistics: frobenius/kurtosis/sparsity) instead of the SVD-based main
    # channels — the shared spectrum is reserved for the (few) dense tensors.
    panel_channels = spec.expert_channels if spec.expert_channels else spec.channels
    for channel, ch_spec in panel_channels.items():
        stat_key = ch_spec["stat"]
        _report(0.93, f"Generating expert panels ({stat_key})...")
        expert_panels = rasterize_expert_panels(stats, spec, stat_key)
        for panel in expert_panels:
            panel_raw_path = out / f"field_expert_{panel.slot}_{channel}_raw.tif"
            write_tif(panel_raw_path, panel.data)
            artefacts.append(panel_raw_path)

            # v2.1 pipeline: apply pre-transform then robust_scale
            pre = ch_spec.get("pre")
            panel_data = panel.data
            if pre == "log1p":
                panel_data = log1p(panel_data)
            scaled_panel = apply_scale(panel_data, ch_spec["scale"])
            up_panel = upsample(scaled_panel, int(spec.grid["upsample"]))
            smoothed_panel = smooth(up_panel, float(spec.grid["smooth_sigma"]))
            panel_smooth_path = out / f"field_expert_{panel.slot}_{channel}_smooth.tif"
            write_tif(panel_smooth_path, smoothed_panel)
            artefacts.append(panel_smooth_path)

    # Vision tower fields (VLM models): a separate sheet with its own slot
    # taxonomy and statistics, so multimodal models show a distinct fingerprint
    # instead of having their vision tensors silently dropped.
    if spec.vision_slots and spec.vision_channels:
        from weight_atlas.fields.smoothing import smooth, upsample

        n_vis = len(spec.vision_channels)
        for vi, (channel, ch_spec) in enumerate(spec.vision_channels.items()):
            vis_lo = 0.80 + 0.05 * (vi / n_vis)
            stat_key = ch_spec["stat"]
            _report(vis_lo, f"Rasterizing vision {channel} field ({stat_key})...")
            vision_field = rasterize_vision(stats, spec, stat_key)
            if vision_field is None:
                continue  # text-only model — no vision tensors
            field_name = f"vision_{channel}"
            raw_path = out / f"field_{field_name}_raw.tif"
            write_tif(raw_path, vision_field.data)
            artefacts.append(raw_path)

            pre = ch_spec.get("pre")
            data = vision_field.data
            if pre == "log1p":
                data = log1p(data)
            scaled = apply_scale(data, ch_spec["scale"])
            up = upsample(scaled, int(spec.grid["upsample"]))
            smoothed = smooth(up, float(spec.grid["smooth_sigma"]))
            smooth_path = out / f"field_{field_name}_smooth.tif"
            write_tif(smooth_path, smoothed)
            artefacts.append(smooth_path)

            fields_for_diag[field_name] = vision_field.data

    # Embedding projection (PCA or UMAP)
    embedding_spec = getattr(spec, 'embedding', {})
    if embedding_spec:
        _report(0.80, "Projecting embeddings...")
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
            _report(0.84, f"Projecting embeddings ({method})...")

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
    _report(0.93, "Checking field degenerations...")
    for channel in spec.channels:
        raw_path = out / f"field_{channel}_raw.tif"
        if raw_path.exists():
            from weight_atlas.fields.tif_io import read_tif
            fields_for_diag[channel] = read_tif(raw_path)
    if fields_for_diag:
        degen_report = diagnose_fields(fields_for_diag)
        if degen_report.warnings:
            fingerprint["warnings"] = fingerprint.get("warnings", []) + degen_report.warnings

    _report(0.97, "Writing manifest...")
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

    # Build ggml_type/dtype mapping from handles if available.
    ggml_types: dict[str, str] = {}
    dtypes: dict[str, str] = {}
    if handles:
        for h in handles:
            if h.dtype.startswith("ggml_"):
                ggml_types[h.name] = h.dtype
            dtypes[h.name] = h.dtype

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
            "kernel_norm": ts.kernel_norm,
        }
        # Add ggml_type if present
        if ts.name in ggml_types:
            tensor_info["ggml_type"] = ggml_types[ts.name]
        # Add on-disk dtype for type_map (GGUF ggml_type, safetensors dtype)
        if ts.name in dtypes:
            tensor_info["dtype"] = dtypes[ts.name]
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

    # Add vision-tower info (VLM models) — mapped tensors, block count, and the
    # number of global tensors (patch_embed / pos_embed / projector). Text-only
    # models get no ``model.vision`` block.
    vision_info = detect_vision(stats)
    if vision_info:
        out["model"]["vision"] = vision_info
        out["mapping_coverage"]["vision_tensors"] = vision_info["n_tensors"]

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
    params: dict[str, float] | None = None
    for channel, ch_spec in spec.channels.items():
        stat_key = ch_spec["stat"]
        scale_type = ch_spec["scale"]["type"]
        if scale_type not in ("robust_scale", "quantile_clip"):
            continue
        # Record the actual quantile bounds from the spec (not hardcoded).
        lower = float(ch_spec["scale"].get("lower", ch_spec["scale"].get("lo", 0.01)))
        upper = float(ch_spec["scale"].get("upper", ch_spec["scale"].get("hi", 0.99)))
        if params is None:
            params = {"lower": lower, "upper": upper}

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
        "params": params or {"lower": 0.01, "upper": 0.99},
        "channels": channels_meta,
    }
