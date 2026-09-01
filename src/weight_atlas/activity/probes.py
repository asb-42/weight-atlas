"""Optional activity probes (P3, alesha-pro adoption — see
docs/2026-08-31_atlas-alesha-pro-analysis.md §4/§7): activation quantizability,
per-layer INT4 fragility, and GDN linear-attention state.

These probes extend the pinned Activity-Mode protocol additively: each writes
its own JSON artefact (`activity_<probe>.json`) and never changes the
protocol hash or the residual/expert fields.

All probes are **opt-in** (`CaptureConfig.probes`, CLI `--probes`): each adds
at least one extra forward pass per state, and the fragility probe runs one
forward per layer (expensive by design).

torch is imported lazily (the `activity` extra); every probe degrades
gracefully when the model does not match its assumptions (e.g. no Gated
DeltaNet modules → the linattn probe writes no artefact).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

# (module-name suffix, site label) — activation-SQNR probe points. Sites are
# the inputs of the big Linear projections (the tensors an activation-quantized
# deployment would feed).
ACTQ_SITES: tuple[tuple[str, str], ...] = (
    ("self_attn.q_proj", "attn.q_in"),
    ("self_attn.k_proj", "attn.k_in"),
    ("self_attn.v_proj", "attn.v_in"),
    ("self_attn.o_proj", "attn.o_in"),
    ("mlp.gate_proj", "mlp.gate_in"),
    ("mlp.up_proj", "mlp.up_in"),
    ("mlp.down_proj", "mlp.down_in"),
)

KNOWN_PROBES = ("actq", "fragility", "linattn")

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Pure math (numpy — testable without torch; torch call sites below mirror
# these formulas exactly)
# ---------------------------------------------------------------------------


def half_life_from_g(g_mean: float) -> float | None:
    """Half-life (in tokens) of a linear-attention memory decaying at ``g``.

    The gated-delta-rule decay is ``exp(g)`` per token with ``g <= 0``; the
    memory halves after ``ln(2) / -g`` tokens. Returns None for a non-decaying
    gate (``g >= 0``) — not applicable, never a fake infinity.
    """
    if g_mean is None or not math.isfinite(g_mean) or g_mean >= 0.0:
        return None
    return _LN2 / -g_mean


def kl_and_cos(base_logits: Any, quant_logits: Any, pos_slice: slice) -> dict[str, float]:
    """KL(base ‖ quantized) over ``pos_slice`` tokens + full-logit cosine.

    ``base_logits``/``quant_logits``: (batch, seq, vocab) float arrays.
    Mirrors the fragility-probe reduction so the torch call site stays thin.
    """
    import numpy as np

    base = np.asarray(base_logits, dtype=np.float64)[:, :-1, :]
    quant = np.asarray(quant_logits, dtype=np.float64)[:, :-1, :]
    base_lp = base - base.max(axis=-1, keepdims=True)
    base_lp = base_lp - np.log(np.exp(base_lp).sum(axis=-1, keepdims=True))
    q_lp = quant - quant.max(axis=-1, keepdims=True)
    q_lp = q_lp - np.log(np.exp(q_lp).sum(axis=-1, keepdims=True))
    kl = float(
        (
            np.exp(base_lp[:, pos_slice, :]) * (base_lp[:, pos_slice, :] - q_lp[:, pos_slice, :])
        ).sum(axis=-1).mean()
    )
    cos = float(
        np.sum(base[:, pos_slice, :] * quant[:, pos_slice, :])
        / max(
            1e-12,
            float(np.linalg.norm(base[:, pos_slice, :])) * float(np.linalg.norm(quant[:, pos_slice, :])),
        )
    )
    return {"kl": kl, "logit_cos": cos}


# ---------------------------------------------------------------------------
# Activation-SQNR probe (P3.7)
# ---------------------------------------------------------------------------


@dataclass
class ActqCollector:
    """Accumulates (layer, site) → ΣINT8-SQNR, ΣFP8-SQNR, n.

    The torch hook calls :meth:`add` with the two measured dB values; the
    dump is the per-site per-layer mean series (alesha's flow-collector
    aggregation).
    """

    layers: list[int] = field(default_factory=list)
    sums: dict[tuple[int, str], list[float]] = field(default_factory=dict)

    def add(self, layer: int, site: str, int8_db: float, fp8_db: float) -> None:
        if layer not in self.layers:
            self.layers.append(layer)
        key = (layer, site)
        entry = self.sums.setdefault(key, [0.0, 0.0, 0])
        entry[0] += int8_db
        entry[1] += fp8_db
        entry[2] += 1

    def dump(self) -> dict[str, Any]:
        layers = sorted(self.layers)
        sites: dict[str, dict[str, Any]] = {}
        for layer in layers:
            for _, site in ACTQ_SITES:
                entry = self.sums.get((layer, site))
                if entry is None:
                    continue
                s = sites.setdefault(site, {"int8_db": [], "fp8_db": [], "layers": layers})
                s["int8_db"].append(round(entry[0] / entry[2], 6))
                s["fp8_db"].append(round(entry[1] / entry[2], 6))
        return {"sites": sites}

    def hooks(self, model: Any) -> list[Any]:
        """Register INT8/FP8-SQNR pre-hooks on every matching Linear input.

        torch-side math mirrors stats.sqnr's RTN schemes: INT8 per-token
        dynamic (per last-dim amax/127), FP8 e4m3 global amax/448 via the
        hardware cast. Returns the hook handles for the caller's finally.
        """
        import torch

        def int8_dyn_sqnr(x: Any) -> float:
            flat = x.detach().reshape(-1, x.shape[-1]).float()
            s = flat.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127.0
            xq = (flat / s).round().clamp(-127, 127) * s
            denom = (flat - xq).pow(2).sum().clamp(min=1e-20)
            return float(10 * torch.log10(flat.pow(2).sum() / denom))

        def fp8_sqnr(x: Any) -> float:
            flat = x.detach().reshape(-1, x.shape[-1]).float()
            s = flat.abs().amax().clamp(min=1e-12) / 448.0
            try:
                xq = (
                    (flat / s)
                    .clamp(-448.0, 448.0)
                    .to(torch.float8_e4m3fn)
                    .float()
                    * s
                )
            except (RuntimeError, TypeError):  # pre-2.1 CPU: no fp8 cast
                return float("nan")
            denom = (flat - xq).pow(2).sum().clamp(min=1e-20)
            return float(10 * torch.log10(flat.pow(2).sum() / denom))

        handles: list[Any] = []
        collector = self

        def make_hook(layer: int, site: str) -> Any:
            def hook(_module: Any, args: Any) -> None:
                x = args[0]
                if x.dim() < 2 or x.shape[-1] < 64:
                    return  # tiny activations: SQNR is meaningless noise
                collector.add(layer, site, int8_dyn_sqnr(x), fp8_sqnr(x))

            return hook

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise ValueError("Cannot find model.layers")

        for i, layer in enumerate(layers):
            for name, mod in layer.named_modules():
                for suffix, site in ACTQ_SITES:
                    if name.endswith(suffix):
                        handles.append(mod.register_forward_pre_hook(make_hook(i, site)))
        return handles


# ---------------------------------------------------------------------------
# Per-layer INT4 fragility probe (P3.8)
# ---------------------------------------------------------------------------


def fake_int4_g128_torch(w: Any, group: int = 128) -> Any:
    """Fake-quantize the last dim in 128-groups (INT4 symmetric, amax/7).

    Same scheme as :func:`weight_atlas.stats.sqnr.int4_group128_sqnr` —
    torch-side twin for weight swapping. Rows whose length is not a multiple
    of the group size keep their tail unquantized (the scan-side stat reports
    NaN there; here the model must still run).
    """
    wd = w.detach()
    n = wd.shape[-1]
    g = n // group * group
    if g == 0:
        return wd
    wf = wd.float()
    x = wf[..., :g].reshape(-1, group)
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 7.0
    xq = ((x / s).round().clamp(-7, 7) * s).reshape(wf.shape[:-1] + (g,))
    out = wf.clone()
    out[..., :g] = xq
    return out.to(wd.dtype)


def _iter_linears(module: Any) -> Iterator[tuple[str, Any]]:
    for name, mod in module.named_modules():
        if mod.__class__.__name__ == "Linear":
            yield name, mod


def layer_fragility(
    model: Any,
    input_ids: Any,
    *,
    pos_fraction: float = 0.75,
    max_layers: int | None = None,
) -> dict[str, Any]:
    """KL(base ‖ INT4-g128 layer) per layer + logit cosine.

    One extra forward per layer (the expensive probe): every Linear inside a
    single decoder layer is fake-quantized, the logits are compared against
    the unquantized baseline on the last ``1 - pos_fraction`` tokens, then the
    original weights are restored. Deterministic given the model + inputs.
    """
    import torch

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Cannot find model.layers")
    n_layers = len(layers) if max_layers is None else min(max_layers, len(layers))

    with torch.no_grad():
        base = model(input_ids=input_ids, use_cache=False).logits.float()
    pos = slice(int(base.shape[1] * pos_fraction), None)

    kl: list[float] = []
    cos: list[float] = []
    for i in range(n_layers):
        lins = list(_iter_linears(layers[i]))
        saved = [(m, m.weight.data) for _, m in lins]
        try:
            with torch.no_grad():
                for _, m in lins:
                    m.weight.data = fake_int4_g128_torch(m.weight.data)
                out = model(input_ids=input_ids, use_cache=False).logits.float()
            metrics = kl_and_cos(
                base.detach().cpu().numpy(),
                out.detach().cpu().numpy(),
                pos,
            )
        finally:
            for m, w in saved:
                m.weight.data = w
        kl.append(round(metrics["kl"], 6))
        cos.append(round(metrics["logit_cos"], 6))
    return {"n_layers": n_layers, "kl": kl, "logit_cos": cos, "pos_fraction": pos_fraction}


# ---------------------------------------------------------------------------
# GDN linear-attention state probe (P3.9)
# ---------------------------------------------------------------------------


def find_gdn_modules(model: Any) -> list[tuple[str, Any]]:
    """Decoder modules implementing Gated DeltaNet (``chunk_gated_delta_rule``).

    Qwen3-Next / Qwen3.8-family linear-attention branches. Empty list = model
    has none (dense attention stack) → the caller skips the probe.
    """
    found = []
    for name, mod in model.named_modules():
        if callable(getattr(mod, "chunk_gated_delta_rule", None)):
            found.append((name, mod))
    return found


class GDNCollector:
    """Write-gate β, decay g and recurrent-state RMS per GDN layer.

    Mirrors alesha's live collector: ``in_proj_b`` output → sigmoid mean (β),
    ``in_proj_a`` output → g = -exp(A_log) · softplus(dt + dt_bias) mean, and
    the wrapped ``chunk_gated_delta_rule`` returns the final state whose RMS
    is accumulated. Dump includes the per-layer half-life in tokens.
    """

    def __init__(self) -> None:
        self.beta: dict[int, list[float]] = {}
        self.g: dict[int, list[float]] = {}
        self.state: dict[int, list[float]] = {}
        self._orig: dict[int, Any] = {}
        self._mods: dict[int, Any] = {}
        self.handles: list[Any] = []

    def attach(self, model: Any) -> bool:
        gdns = find_gdn_modules(model)
        if not gdns:
            return False
        import torch
        import torch.nn.functional as torch_f

        collector = self
        for i, (_name, la) in enumerate(gdns):
            self._mods[i] = la
            self._orig[i] = la.chunk_gated_delta_rule

            in_b = getattr(la, "in_proj_b", None)
            in_a = getattr(la, "in_proj_a", None)
            if in_b is not None:
                def beta_hook(_m: Any, _args: Any, out: Any, _i: int = i) -> None:
                    collector.beta.setdefault(_i, []).append(
                        float(torch.sigmoid(out.detach().float()).mean())
                    )

                self.handles.append(in_b.register_forward_hook(beta_hook))
            if in_a is not None and hasattr(la, "A_log") and hasattr(la, "dt_bias"):
                def g_hook(_m: Any, _args: Any, out: Any, _i: int = i, _la: Any = la) -> None:
                    dt = torch_f.softplus(out.detach().float() + _la.dt_bias.detach().float())
                    g = -_la.A_log.detach().float().exp() * dt
                    collector.g.setdefault(_i, []).append(float(g.mean()))

                self.handles.append(in_a.register_forward_hook(g_hook))

            def wrapped(q: Any, k: Any, v: Any, _i: int = i, **kw: Any) -> Any:
                kw["output_final_state"] = True
                out, state = self._orig[_i](q, k, v, **kw)
                if state is not None:
                    self.state.setdefault(_i, []).append(
                        float(state.detach().float().pow(2).mean().sqrt())
                    )
                return out, state

            la.chunk_gated_delta_rule = wrapped
        return True

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        for i, la in self._mods.items():
            la.chunk_gated_delta_rule = self._orig[i]

    @staticmethod
    def _mean(src: dict[int, list[float]], i: int) -> float | None:
        vals = src.get(i)
        if not vals:
            return None
        return sum(vals) / len(vals)

    def dump(self) -> dict[str, Any]:
        layers = sorted(self._mods)
        out: dict[str, Any] = {
            "n_layers": len(layers),
            "beta_open": [],
            "g_mean": [],
            "state_rms": [],
            "half_life_tokens": [],
        }
        for i in layers:
            beta = self._mean(self.beta, i)
            g = self._mean(self.g, i)
            state = self._mean(self.state, i)
            out["beta_open"].append(round(beta, 6) if beta is not None else None)
            out["g_mean"].append(round(g, 6) if g is not None else None)
            out["state_rms"].append(round(state, 6) if state is not None else None)
            hl = half_life_from_g(g) if g is not None else None
            out["half_life_tokens"].append(round(hl, 3) if hl is not None else None)
        return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def validate_probes(probes: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Normalize + validate a probe list (CLI/config boundary)."""
    unknown = [p for p in probes if p not in KNOWN_PROBES]
    if unknown:
        raise ValueError(
            f"Unknown probes: {unknown} (known: {list(KNOWN_PROBES)})"
        )
    return tuple(sorted(set(probes)))


def run_probes(
    model: Any,
    tokenizer: Any,
    protocol: Any,
    config: Any,
    inputs_by_state: dict[str, Any],
    out_dir: Any,
) -> list[str]:
    """Run the enabled probes; return the artefact file names written.

    ``inputs_by_state`` carries the tokenized batches from the main capture
    (probes reuse them — one tokenizer pass). Every probe owns its hooks and
    removes them; every artefact is additive to the pinned protocol.
    """
    import json

    written: list[str] = []

    def _write(name: str, payload: dict[str, Any]) -> None:
        path = out_dir / name
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        written.append(name)

    if "actq" in config.probes:
        actq_collector = ActqCollector()
        handles = actq_collector.hooks(model)
        try:
            import torch

            for state in protocol.states:
                inputs = inputs_by_state.get(state.name)
                if inputs is None:
                    continue
                with torch.no_grad():
                    model(**inputs)
        finally:
            for h in handles:
                h.remove()
        _write("activity_actq.json", actq_collector.dump())

    if "fragility" in config.probes:
        first = protocol.states[0]
        inputs = inputs_by_state.get(first.name)
        if inputs is not None and "input_ids" in inputs:
            frag = layer_fragility(model, inputs["input_ids"])
            _write("activity_fragility.json", frag)

    if "linattn" in config.probes:
        gdn_collector = GDNCollector()
        if gdn_collector.attach(model):
            try:
                import torch

                for state in protocol.states:
                    inputs = inputs_by_state.get(state.name)
                    if inputs is None:
                        continue
                    with torch.no_grad():
                        model(**inputs)
            finally:
                gdn_collector.remove()
            _write("activity_linattn.json", gdn_collector.dump())
        else:
            _write("activity_linattn.json", {"n_layers": 0, "note": "no GDN modules found"})

    return written
