"""Loader tests: metadata-only discovery, duplicate detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import SLOTS, make_fake_model
from weight_atlas.loaders.safetensors_loader import SafetensorsLoader


def test_loader_discovers_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "m.safetensors"
        make_fake_model(model, n_layers=4, hidden=32)
        loader = SafetensorsLoader()
        handles = loader.open(model)
        # 4 layers * 9 slots + embed + lm_head
        assert len(handles) == 4 * len(SLOTS) + 2
        names = {h.name for h in handles}
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "lm_head.weight" in names


def test_loader_lazy_no_load():
    """Loader must not call get_tensor (i.e., not materialise data)."""
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "m.safetensors"
        make_fake_model(model, n_layers=2, hidden=8)
        loader = SafetensorsLoader()
        handles = loader.open(model)
        for h in handles:
            # shape is known without loading
            assert len(h.shape) == 2
            assert h.shape == (8, 8)


def test_loader_rejects_duplicate_names():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Create two files with the same tensor name.
        from safetensors.numpy import save_file
        a = np.zeros((2, 2), dtype=np.float32)
        save_file({"dup": a}, tmp / "a.safetensors")
        save_file({"dup": a}, tmp / "b.safetensors")
        loader = SafetensorsLoader()
        with pytest.raises(ValueError, match="duplicate tensor name"):
            loader.open(tmp)


def test_loader_directory_glob_sorted():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from safetensors.numpy import save_file
        # a.safetensors contains tensor 'y'; z.safetensors contains tensor 'x'.
        # Files are globbed sorted, so a.safetensors is read first → 'y' before 'x'.
        save_file({"x": np.zeros((2, 2), dtype=np.float32)}, tmp / "z.safetensors")
        save_file({"y": np.zeros((2, 2), dtype=np.float32)}, tmp / "a.safetensors")
        loader = SafetensorsLoader()
        handles = loader.open(tmp)
        names = [h.name for h in handles]
        assert names == ["y", "x"]  # sorted by filename (a before z), not tensor name
