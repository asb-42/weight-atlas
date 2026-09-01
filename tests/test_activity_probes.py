"""P3 activity probes: actq collector, fragility math, GDN state (alesha-pro adoption)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch optional (activity extra)")

# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_halving_matches_definition(self) -> None:
        from weight_atlas.activity.probes import half_life_from_g

        # g = -ln(2)/8 → memory decays to 1/2 after exactly 8 tokens
        assert half_life_from_g(-np.log(2) / 8) == pytest.approx(8.0)

    def test_non_decaying_g_is_none(self) -> None:
        from weight_atlas.activity.probes import half_life_from_g

        assert half_life_from_g(0.0) is None
        assert half_life_from_g(0.5) is None
        assert half_life_from_g(float("nan")) is None
        assert half_life_from_g(None) is None


class TestKlAndCos:
    def test_identical_logits_give_zero_kl_and_one_cos(self) -> None:
        from weight_atlas.activity.probes import kl_and_cos

        rng = np.random.default_rng(1)
        logits = rng.standard_normal((2, 16, 32))
        out = kl_and_cos(logits, logits, slice(8, None))
        assert out["kl"] == pytest.approx(0.0, abs=1e-10)
        assert out["logit_cos"] == pytest.approx(1.0, abs=1e-10)

    def test_noise_gives_positive_kl(self) -> None:
        from weight_atlas.activity.probes import kl_and_cos

        rng = np.random.default_rng(2)
        base = rng.standard_normal((1, 32, 64))
        quant = base + rng.standard_normal(base.shape) * 0.5
        out = kl_and_cos(base, quant, slice(0, None))
        assert out["kl"] > 0.0
        assert 0.0 < out["logit_cos"] < 1.0


# ---------------------------------------------------------------------------
# Actq collector (aggregation only — the torch SQNR formulas are the
# stats.sqnr twins and are pinned there)
# ---------------------------------------------------------------------------


class TestActqCollector:
    def test_aggregation_mean_per_layer(self) -> None:
        from weight_atlas.activity.probes import ActqCollector

        c = ActqCollector()
        c.add(0, "attn.q_in", 40.0, 30.0)
        c.add(0, "attn.q_in", 42.0, 32.0)
        c.add(1, "mlp.up_in", 38.0, 28.0)
        dump = c.dump()
        q = dump["sites"]["attn.q_in"]
        assert q["layers"] == [0, 1]
        assert q["int8_db"][0] == pytest.approx(41.0)  # mean over 2 batches
        assert q["fp8_db"][0] == pytest.approx(31.0)
        assert dump["sites"]["mlp.up_in"]["int8_db"] == [38.0]

    def test_empty_dump(self) -> None:
        from weight_atlas.activity.probes import ActqCollector

        assert ActqCollector().dump() == {"sites": {}}


# ---------------------------------------------------------------------------
# Fake-INT4 (torch twin of the scan-side scheme — weight swap mechanics)
# ---------------------------------------------------------------------------


class TestFakeInt4:
    def test_groups_quantize_to_amax_grid(self) -> None:
        torch = pytest.importorskip("torch")
        from weight_atlas.activity.probes import fake_int4_g128_torch

        # One group of 128 values with amax 7.0 → dequantized values are
        # multiples of 1.0 (amax/7), within [-7, 7].
        w = torch.linspace(-7.0, 7.0, 128).reshape(1, 128)
        q = fake_int4_g128_torch(w)
        assert q.shape == w.shape
        steps = (q / 1.0).round()
        assert torch.allclose(q, steps * 1.0, atol=1e-6)
        assert q.abs().max() <= 7.0 + 1e-6

    def test_untidy_tail_kept(self) -> None:
        from weight_atlas.activity.probes import fake_int4_g128_torch

        w = torch.randn(4, 100)
        q = fake_int4_g128_torch(w)  # 100 % 128 != 0 → tail unquantized
        assert torch.equal(q[:, 128:], w[:, 128:])  # nothing beyond g=0 anyway
        # g = 100 // 128 * 128 = 0 → whole tensor unchanged
        assert torch.equal(q, w)

    def test_zero_group_row_unchanged(self) -> None:
        from weight_atlas.activity.probes import fake_int4_g128_torch

        w = torch.randn(4, 50)
        assert torch.equal(fake_int4_g128_torch(w), w)


# ---------------------------------------------------------------------------
# GDN collector mechanics (stub modules, no real GDN needed)
# ---------------------------------------------------------------------------


class _StubGDN:
    """Minimal Gated DeltaNet stand-in for hook-attachment mechanics."""

    def __init__(self) -> None:
        self.dt_bias = torch.zeros(4)
        self.A_log = torch.zeros(4)
        self.in_proj_b = torch.nn.Linear(4, 4)
        self.in_proj_a = torch.nn.Linear(4, 4)
        self._chunk = lambda q, k, v, **kw: (  # noqa: E731
            (q, kw.get("output_final_state", False)), None
        )

    def chunk_gated_delta_rule(self, q, k, v, **kw):
        return self._chunk(q, k, v, **kw)


class _StubModel:
    def __init__(self) -> None:
        self.layers = [torch.nn.Module()]  # placeholder; named_modules walks self
        self.gdn = _StubGDN()

    def named_modules(self, memo=None, prefix=""):
        yield "gdn", self.gdn
        for name, child in self._named_children():
            yield f"{prefix}{name}", child

    def _named_children(self):
        yield "gdn", self.gdn
        yield "layers", self.layers[0]


class TestGDNCollector:
    def test_find_gdn_modules(self) -> None:
        from weight_atlas.activity.probes import find_gdn_modules

        model = _StubModel()
        found = find_gdn_modules(model)
        assert len(found) == 1 and found[0][0] == "gdn"

    def test_attach_dump_remove_roundtrip(self) -> None:
        from weight_atlas.activity.probes import GDNCollector

        model = _StubModel()
        gdn = model.gdn
        collector = GDNCollector()
        assert collector.attach(model) is True

        # simulate a forward: feed in_proj_b/in_proj_a outputs like the hooks do
        x = torch.randn(1, 4)
        out_b = gdn.in_proj_b(x)
        out_a = gdn.in_proj_a(x)
        collector.beta.setdefault(0, []).append(float(torch.sigmoid(out_b).mean()))
        g = -gdn.A_log.exp() * torch.nn.functional.softplus(out_a + gdn.dt_bias)
        collector.g.setdefault(0, []).append(float(g.mean()))

        # wrapped chunk returns (out, state) and records state RMS
        q = torch.randn(1, 2, 4)
        out, state = gdn.chunk_gated_delta_rule(q, q, q, output_final_state=True)
        assert state is not None
        collector.state.setdefault(0, []).append(float(state.pow(2).mean().sqrt()))

        dump = collector.dump()
        assert dump["n_layers"] == 1
        assert 0.0 < dump["beta_open"][0] < 1.0
        assert dump["g_mean"][0] < 0.0
        assert dump["half_life_tokens"][0] is not None and dump["half_life_tokens"][0] > 0

        collector.remove()
        # original function restored
        assert gdn.chunk_gated_delta_rule(q, q, q) == gdn._chunk(q, q, q)

    def test_attach_on_dense_model_is_false(self) -> None:
        from weight_atlas.activity.probes import GDNCollector

        collector = GDNCollector()
        assert collector.attach(torch.nn.Sequential()) is False
        assert collector.dump()["n_layers"] == 0


# ---------------------------------------------------------------------------
# Probe selection + orchestration contract
# ---------------------------------------------------------------------------


class TestProbeSelection:
    def test_validate_unknown_raises(self) -> None:
        from weight_atlas.activity.probes import validate_probes

        with pytest.raises(ValueError, match="Unknown probes"):
            validate_probes(("actq", "teleport"))

    def test_validate_dedupes_and_sorts(self) -> None:
        from weight_atlas.activity.probes import validate_probes

        assert validate_probes(("linattn", "actq", "actq")) == ("actq", "linattn")

    def test_run_probes_writes_artefacts(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from weight_atlas.activity.probes import run_probes

        # Dense stub model: actq + linattn find nothing meaningful, fragility
        # needs model.layers — stub raises → skipped via inputs_by_state=None.
        class _Dense:
            layers = []

        model = _Dense()
        protocol = SimpleNamespace(states=[SimpleNamespace(name="rest")])
        config = SimpleNamespace(probes=("actq", "linattn"))
        written = run_probes(model, None, protocol, config, {}, tmp_path)
        assert "activity_actq.json" in written
        assert "activity_linattn.json" in written
        lin = json.loads((tmp_path / "activity_linattn.json").read_text())
        assert lin["n_layers"] == 0 and "no GDN modules" in lin["note"]
