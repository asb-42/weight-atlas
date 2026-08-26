"""Activity capture: Forward-pass hooks for Activity Mode.

Uses plain PyTorch forward hooks (no NNsight/TransformerLens dependency).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CaptureConfig:
    """Configuration for activity capture."""
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 0
    max_layers: int | None = None


@dataclass
class LayerActivity:
    """Activity captured from a single layer."""
    layer_idx: int
    residual_rms: np.ndarray  # (seq_len,) RMS per position
    expert_usage: np.ndarray | None = None  # (seq_len, n_experts) or None


@dataclass
class StateActivity:
    """Activity captured for a single protocol state."""
    state_name: str
    layers: list[LayerActivity] = field(default_factory=list)


def _first_hidden(output: Any) -> Any:
    """First tensor of a module output: plain tuple or transformers ModelOutput.

    ``ModelOutput`` is an OrderedDict subclass, not a tuple, so
    ``isinstance(output, tuple)`` misses it and the raw object would flow
    into arithmetic below.
    """
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "to_tuple"):
        return output.to_tuple()[0]
    return output


def capture_activity(
    model: Any,
    tokenizer: Any,
    protocol: Any,
    config: CaptureConfig,
    out_dir: Path,
) -> dict[str, Any]:
    """Capture activity for all protocol states.

    Process-global torch state (thread count, deterministic-algorithms flag,
    RNG state) is saved before the run and restored afterwards, so calling
    this from a process that does other torch work never leaks configuration.

    Args:
        model: HuggingFace model instance
        tokenizer: HuggingFace tokenizer instance
        protocol: ActivityProtocol instance
        config: Capture configuration
        out_dir: Output directory

    Returns:
        Metadata dict
    """
    # cuBLAS refuses to run deterministically on CUDA >= 10.2 unless this is
    # set before the first CUDA context is created. Harmless on CPU-only runs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch  # type: ignore[import-not-found]

    old_num_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_rng_state = torch.random.get_rng_state()
    try:
        return _capture(model, tokenizer, protocol, config, out_dir)
    finally:
        torch.set_num_threads(old_num_threads)
        torch.use_deterministic_algorithms(old_deterministic)
        torch.random.set_rng_state(old_rng_state)


def _capture(
    model: Any,
    tokenizer: Any,
    protocol: Any,
    config: CaptureConfig,
    out_dir: Path,
) -> dict[str, Any]:
    """Capture body: runs with deterministic settings applied."""
    import json

    import torch  # type: ignore[import-not-found]

    # Set deterministic settings
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)

    # Set dtype
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(config.dtype, torch.float32)

    was_training = bool(getattr(model, "training", False))

    # Move model to device and dtype
    model = model.to(device=config.device, dtype=dtype)
    model.eval()

    # Storage for activations
    activations: dict[int, list[np.ndarray]] = {}
    router_activations: dict[int, list[np.ndarray]] = {}

    # Register hooks
    hooks = []

    def make_hook(layer_idx: int) -> Any:
        def hook(module: Any, layer_input: Any, output: Any) -> None:
            hidden = _first_hidden(output)
            # .float() upcast: NumPy has no bfloat16 (`.numpy()` raises for
            # bf16), and the reduction is more precise in float32 anyway.
            rms = torch.sqrt((hidden.float() ** 2).mean(dim=-1)).detach().cpu().numpy()
            if layer_idx not in activations:
                activations[layer_idx] = []
            activations[layer_idx].append(rms)
        return hook

    def make_router_hook(layer_idx: int) -> Any:
        def hook(module: Any, layer_input: Any, output: Any) -> None:
            # Router output is (batch, seq_len, n_experts)
            router_out = _first_hidden(output)
            softmax_out = (
                torch.softmax(router_out.float(), dim=-1).detach().cpu().numpy()
            )
            if layer_idx not in router_activations:
                router_activations[layer_idx] = []
            router_activations[layer_idx].append(softmax_out)
        return hook

    # Find layers and register hooks
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Cannot find model.layers")

    max_layers = config.max_layers or len(layers)
    for i, layer in enumerate(layers[:max_layers]):
        # Hook on layer output
        hooks.append(layer.register_forward_hook(make_hook(i)))

        # Hook on router if present (MoE)
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            hooks.append(layer.mlp.gate.register_forward_hook(make_router_hook(i)))

    # Capture activity for each state
    state_activities: dict[str, StateActivity] = {}

    try:
        for state in protocol.states:
            inputs = tokenizer(
                state.content,
                return_tensors="pt",
                max_length=state.max_len,
                truncation=True,
                padding="max_length",
            ).to(config.device)

            # Positions where the attention mask is 0 are padding: their
            # activations are meaningless and would pollute the Layer x
            # Position field (with padding="max_length" every state is
            # full-length, so the NaN-for-absent-position machinery below
            # could otherwise never fire).
            attn_mask = inputs.get("attention_mask")
            valid_positions = (
                attn_mask[0].detach().cpu().numpy().astype(bool)
                if attn_mask is not None
                else None
            )

            with torch.no_grad():
                model(**inputs)

            # Store state activity
            state_act = StateActivity(state_name=state.name)
            for layer_idx in sorted(activations.keys()):
                layer_rms = activations[layer_idx][-1]  # Last forward pass
                # Average over batch dim, excluding padded positions
                if layer_rms.ndim > 1:
                    if valid_positions is not None:
                        layer_rms = layer_rms[:, valid_positions]
                    layer_rms = layer_rms.mean(axis=0)

                expert_usage = None
                if layer_idx in router_activations:
                    expert_usage = router_activations[layer_idx][-1]
                    if expert_usage.ndim > 2:
                        if valid_positions is not None:
                            expert_usage = expert_usage[:, valid_positions, :]
                        expert_usage = expert_usage.mean(axis=0)

                state_act.layers.append(LayerActivity(
                    layer_idx=layer_idx,
                    residual_rms=layer_rms,
                    expert_usage=expert_usage,
                ))

            state_activities[state.name] = state_act

            # Clear activations for next state
            activations.clear()
            router_activations.clear()

    finally:
        # Remove hooks and restore the caller's module state
        for hook in hooks:
            hook.remove()
        if hasattr(model, "train"):
            model.train(was_training)

    # Save artefacts
    out_dir.mkdir(parents=True, exist_ok=True)

    for state_name, state_act in state_activities.items():
        # Residual RMS field: Layer × Position
        n_layers = len(state_act.layers)
        max_len = max(layer_act.residual_rms.shape[0] for layer_act in state_act.layers)
        residual_field = np.full((n_layers, max_len), np.nan, dtype=np.float64)

        has_experts = any(layer_act.expert_usage is not None for layer_act in state_act.layers)
        expert_field = None
        if has_experts:
            n_experts = max(
                (layer_act.expert_usage.shape[1] for layer_act in state_act.layers if layer_act.expert_usage is not None),
                default=0,
            )
            if n_experts > 0:
                expert_field = np.full((n_layers, n_experts), np.nan, dtype=np.float64)

        for layer_act in state_act.layers:
            row = layer_act.layer_idx
            seq_len = layer_act.residual_rms.shape[0]
            residual_field[row, :seq_len] = layer_act.residual_rms

            if expert_field is not None and layer_act.expert_usage is not None:
                # Average over positions to get per-expert usage
                usage = layer_act.expert_usage.mean(axis=0)
                expert_field[row, :len(usage)] = usage

        # Write residual field
        residual_raw_path = out_dir / f"field_activity_{state_name}_residual_raw.tif"
        from weight_atlas.fields.tif_io import write_tif
        write_tif(residual_raw_path, residual_field)

        # Write expert field if present
        if expert_field is not None:
            expert_raw_path = out_dir / f"field_activity_{state_name}_experts_raw.tif"
            write_tif(expert_raw_path, expert_field)

    # Save metadata
    metadata = {
        "protocol_version": protocol.version,
        "protocol_hash": protocol.protocol_hash,
        "device": config.device,
        "dtype": config.dtype,
        "seed": config.seed,
        "n_layers": len(layers),
        "states": [s.name for s in protocol.states],
        "torch_version": torch.__version__,
    }

    with open(out_dir / "activity_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Build manifest
    artefacts = list(out_dir.glob("*.tif")) + [
        out_dir / "activity_meta.json",
    ]
    manifest = {str(p.relative_to(out_dir)): _sha256(p) for p in artefacts}
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return metadata


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
