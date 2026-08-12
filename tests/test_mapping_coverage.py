"""Tests for mapping_coverage format in fingerprint.json."""

from __future__ import annotations

from pathlib import Path

import pytest

from weight_atlas.scan import _build_fingerprint


@pytest.fixture
def spec():
    from weight_atlas.core.types import AtlasSpec
    return AtlasSpec.from_json(Path("specs/atlas_spec.v2.1.json"))


def test_mapping_coverage_format(spec):
    """mapping_coverage must have in_slots (ratio), in_other (ratio), unmapped (count), unmapped_tensors (list)."""
    from weight_atlas.core.types import TensorStats

    stats = [
        TensorStats(name="blk.0.attn_q.weight", shape=(4096, 4096)),
        TensorStats(name="blk.0.attn_k.weight", shape=(4096, 4096)),
        TensorStats(name="blk.0.ffn_gate.weight", shape=(4096, 12288)),
        TensorStats(name="unknown_tensor.weight", shape=(100,)),
    ]
    fp = _build_fingerprint(stats, spec, "gguf")

    mc = fp["mapping_coverage"]
    assert "in_slots" in mc
    assert "in_other" in mc
    assert "unmapped" in mc
    assert "unmapped_tensors" in mc

    # in_slots and in_other should be ratios (floats between 0 and 1)
    assert 0.0 <= mc["in_slots"] <= 1.0
    assert 0.0 <= mc["in_other"] <= 1.0

    # unmapped should be a count
    assert isinstance(mc["unmapped"], int)
    assert mc["unmapped"] == 1

    # unmapped_tensors should be a list
    assert isinstance(mc["unmapped_tensors"], list)
    assert "unknown_tensor.weight" in mc["unmapped_tensors"]


def test_mapping_coverage_all_mapped(spec):
    """When all tensors are mapped, in_slots should be 1.0."""
    from weight_atlas.core.types import TensorStats

    stats = [
        TensorStats(name="blk.0.attn_q.weight", shape=(4096, 4096)),
        TensorStats(name="blk.0.attn_k.weight", shape=(4096, 4096)),
    ]
    fp = _build_fingerprint(stats, spec, "gguf")

    mc = fp["mapping_coverage"]
    assert mc["in_slots"] == 1.0
    assert mc["in_other"] == 0.0
    assert mc["unmapped"] == 0
    assert mc["unmapped_tensors"] == []


def test_mapping_coverage_none_mapped(spec):
    """When no tensors are mapped, in_slots should be 0.0."""
    from weight_atlas.core.types import TensorStats

    stats = [
        TensorStats(name="unknown_tensor_1.weight", shape=(100,)),
        TensorStats(name="unknown_tensor_2.weight", shape=(200,)),
    ]
    fp = _build_fingerprint(stats, spec, "gguf")

    mc = fp["mapping_coverage"]
    assert mc["in_slots"] == 0.0
    assert mc["in_other"] == 1.0
    assert mc["unmapped"] == 2
    assert len(mc["unmapped_tensors"]) == 2
