"""Stats checkpoint journal: resume after crash, identity guard, torn lines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from weight_atlas.core.types import TensorHandle
from weight_atlas.scan import (
    _CHECKPOINT_NAME,
    _journal_append,
    _journal_load,
    _journal_open,
    _model_identity,
)
from weight_atlas.scan import scan as run_scan


def _fake_handles(n: int, seed: int = 11) -> list[TensorHandle]:
    rng = np.random.default_rng(seed)
    handles = []
    for i in range(n):
        rows = int(rng.integers(8, 64))
        cols = int(rng.integers(8, 64))
        x = rng.standard_normal((rows, cols)).astype(np.float32)
        handles.append(TensorHandle(f"t{i:04d}.weight", x.shape, "float32", lambda x=x: x))
    return handles


class TestIdentity:
    def test_same_model_same_identity(self) -> None:
        h = _fake_handles(8)
        assert _model_identity(h, 0, 0, False) == _model_identity(h, 0, 0, False)

    def test_seed_change_changes_identity(self) -> None:
        h = _fake_handles(8)
        assert _model_identity(h, 0, 0, False) != _model_identity(h, 1, 0, False)
        assert _model_identity(h, 0, 0, False) != _model_identity(h, 0, 7, False)

    def test_probe_flag_changes_identity(self) -> None:
        h = _fake_handles(8)
        assert _model_identity(h, 0, 0, False) != _model_identity(h, 0, 0, True)

    def test_tensor_change_changes_identity(self) -> None:
        h1 = _fake_handles(8)
        h2 = _fake_handles(8, seed=12)
        assert _model_identity(h1, 0, 0, False) != _model_identity(h2, 0, 0, False)


class TestJournalLoad:
    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        h = _fake_handles(4)
        assert _journal_load(tmp_path / "nope.jsonl", _model_identity(h, 0, 0, False)) == {}

    def test_foreign_identity_rejected(self, tmp_path: Path) -> None:
        h = _fake_handles(4)
        ident = _model_identity(h, 0, 0, False)
        path = tmp_path / _CHECKPOINT_NAME
        with open(path, "w") as f:
            f.write(json.dumps({"_checkpoint_header": {"identity": "other-model"}}) + "\n")
            f.write(json.dumps({"name": "t0000.weight", "shape": [8, 8], "frobenius": 1.0}) + "\n")
        assert _journal_load(path, ident) == {}

    def test_torn_tail_skipped(self, tmp_path: Path) -> None:
        h = _fake_handles(4)
        ident = _model_identity(h, 0, 0, False)
        path = tmp_path / _CHECKPOINT_NAME
        good = {"name": "t0000.weight", "shape": [8, 8], "frobenius": 2.0}
        with open(path, "w") as f:
            f.write(json.dumps({"_checkpoint_header": {"identity": ident}}) + "\n")
            f.write(json.dumps(good) + "\n")
            f.write('{"name": "t0001.weight", "shape": [8, 8], "frobeni')  # torn
        loaded = _journal_load(path, ident)
        assert set(loaded) == {"t0000.weight"}
        assert loaded["t0000.weight"].frobenius == 2.0

    def test_roundtrip_full_tensorstats(self, tmp_path: Path) -> None:
        from weight_atlas.scan import _journal_append

        h = _fake_handles(1)
        ident = _model_identity(h, 0, 0, True)
        path = tmp_path / _CHECKPOINT_NAME
        from weight_atlas.scan import _journal_open

        fh = _journal_open(path, ident, has_entries=False)
        from weight_atlas.scan import _stats_for_handle

        ts = _stats_for_handle(h[0], quant_probe=True)
        _journal_append(fh, ts)
        fh.close()
        loaded = _journal_load(path, ident)
        assert set(loaded) == {h[0].name}
        rt = loaded[h[0].name]
        for field in ("frobenius", "spectral_norm", "stable_rank", "kurtosis",
                      "sqnr_int8_ch", "sqnr_fp8_e4m3", "row_amax_ratio", "sv_decay"):
            a, b = getattr(ts, field), getattr(rt, field)
            if isinstance(a, float) and np.isnan(a):
                assert np.isnan(b)
            else:
                assert a == b


class TestScanResume:
    def _write_partial_journal(
        self, out: Path, model_path: Path, tensor_names: list[str], stats_list
    ) -> None:
        """Write a valid journal with entries for ``tensor_names``."""
        from tests.fixtures import make_fake_model  # noqa: F401

        # Build the identity the same way scan() would: from the real handles.
        from weight_atlas.core.registry import get_loader
        from weight_atlas.core.types import detect_loader

        loader = get_loader(detect_loader(model_path))()
        handles = loader.open(model_path)
        ident = _model_identity(handles, 0, 0, False)
        path = out / _CHECKPOINT_NAME
        fh = _journal_open(path, ident, has_entries=False)
        by_name = {ts.name: ts for ts in stats_list}
        for name in tensor_names:
            _journal_append(fh, by_name[name])
        fh.close()

    def test_resume_produces_identical_fingerprint(self, tmp_path: Path) -> None:
        """Scan → capture some stats → delete fingerprint → rescan with a
        journal containing those stats → identical fingerprint (values are
        pure functions; the journal short-circuits computation)."""
        from tests.fixtures import make_fake_model
        from weight_atlas.core.types import load_default_spec
        from weight_atlas.scan import _stats_for_handle

        model_path = tmp_path / "fake.safetensors"
        make_fake_model(model_path, n_layers=2)
        spec = load_default_spec()
        out = tmp_path / "out"

        # Reference fingerprint (full scan)
        run_scan(model_path, out, spec, jobs=1)
        fp_ref = (out / "fingerprint.json").read_bytes()
        # journal was discarded on success
        assert not (out / _CHECKPOINT_NAME).exists()

        # Simulate a crash: rebuild a journal with a subset of real stats
        from weight_atlas.core.registry import get_loader
        from weight_atlas.core.types import detect_loader

        handles = get_loader(detect_loader(model_path))().open(model_path)
        subset = [_stats_for_handle(h) for h in handles[:5]]
        out2 = tmp_path / "out2"
        out2.mkdir()
        self._write_partial_journal(out2, model_path, [ts.name for ts in subset], subset)

        run_scan(model_path, out2, spec, jobs=1)
        fp2 = (out2 / "fingerprint.json").read_bytes()
        assert fp2 == fp_ref  # resumed scan is byte-identical
        # journal consumed again
        assert not (out2 / _CHECKPOINT_NAME).exists()

    def test_resume_with_probe_mismatch_ignores_journal(self, tmp_path: Path) -> None:
        """A journal written without the probe must NOT be used by a
        probe scan (identity includes the probe flag)."""
        from tests.fixtures import make_fake_model
        from weight_atlas.core.types import load_default_spec
        from weight_atlas.scan import _stats_for_handle

        model_path = tmp_path / "fake.safetensors"
        make_fake_model(model_path, n_layers=2)
        spec = load_default_spec()
        out = tmp_path / "out"
        out.mkdir()

        handles = get_loader_handles(model_path)
        subset = [_stats_for_handle(h) for h in handles[:5]]
        self._write_partial_journal(out, model_path, [ts.name for ts in subset], subset)

        run_scan(model_path, out, spec, jobs=1, quant_probe=True)
        fp = json.loads((out / "fingerprint.json").read_text())
        # probe fields must be real (not NaN) despite the non-probe journal
        some = next(iter(fp["tensors"].values()))
        assert not np.isnan(some["sqnr_int8_ch"])

    def test_fresh_flag_discards_journal(self, tmp_path: Path) -> None:
        from tests.fixtures import make_fake_model
        from weight_atlas.core.types import load_default_spec
        from weight_atlas.scan import _stats_for_handle

        model_path = tmp_path / "fake.safetensors"
        make_fake_model(model_path, n_layers=2)
        spec = load_default_spec()
        out = tmp_path / "out"
        out.mkdir()

        handles = get_loader_handles(model_path)
        subset = [_stats_for_handle(h) for h in handles[:5]]
        self._write_partial_journal(out, model_path, [ts.name for ts in subset], subset)

        run_scan(model_path, out, spec, jobs=1, fresh=True)
        # complete scan, journal consumed
        assert (out / "fingerprint.json").exists()
        assert not (out / _CHECKPOINT_NAME).exists()


def get_loader_handles(model_path: Path) -> list[TensorHandle]:
    from weight_atlas.core.registry import get_loader
    from weight_atlas.core.types import detect_loader

    return get_loader(detect_loader(model_path))().open(model_path)


class TestFailureKeepsJournal:
    def test_journal_survives_crash_mid_scan(self, tmp_path: Path) -> None:
        """A scan that dies mid-stats leaves the journal with the completed
        tensors — the resume payload."""
        import weight_atlas.scan as scan_mod
        from tests.fixtures import make_fake_model
        from weight_atlas.core.types import load_default_spec

        model_path = tmp_path / "fake.safetensors"
        make_fake_model(model_path, n_layers=3)
        spec = load_default_spec()
        out = tmp_path / "out"

        real = scan_mod._stats_for_handle
        calls = {"n": 0}

        def exploding(h, *a, **k):
            calls["n"] += 1
            if calls["n"] > 5:
                raise RuntimeError("simulated crash")
            return real(h, *a, **k)

        monkey = pytest.MonkeyPatch()
        monkey.setattr(scan_mod, "_stats_for_handle", exploding)
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                run_scan(model_path, out, spec, jobs=1)
        finally:
            monkey.undo()

        path = out / _CHECKPOINT_NAME
        assert path.exists(), "journal must survive the crash"
        assert calls["n"] > 5
        loaded = _journal_load(path, _model_identity(
            get_loader_handles(model_path), 0, 0, False
        ))
        assert len(loaded) >= 5, "completed tensors must be journalled"
