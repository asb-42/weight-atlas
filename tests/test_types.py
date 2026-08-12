"""Tests for core/types.py: AtlasSpec loading."""

from __future__ import annotations

from pathlib import Path

from weight_atlas.core.types import (
    DEFAULT_SPEC_NAME,
    DEFAULT_SPEC_VERSION,
    AtlasSpec,
    get_default_spec_path,
    load_default_spec,
)


def test_load_spec():
    spec = AtlasSpec.from_json(Path("specs/atlas_spec.v2.2.json"))
    assert spec.spec_version == 2
    assert "attn_q" in spec.slots
    assert spec.channel_stat("height") == "spectral_norm"
    assert spec.channel_stat("tint") == "stable_rank"
    assert spec.channel_scale("tint")["type"] == "robust_scale"
    assert spec.channel_scale("rough")["type"] == "rank_scale"
    assert spec.channel_scale("height")["type"] == "rank_scale"
    assert spec.sheet["contour_levels"] == 12
    assert spec.seeds["svd"] == 0


def test_default_spec_is_canonical_and_absolute():
    """The default spec resolved by both CLI and web must be the canonical one.

    Regression test for the spec-version divergence: if the canonical default
    spec file drifts (or a stale spec is picked up), load_default_spec raises.
    """
    path = get_default_spec_path()
    assert path.name == DEFAULT_SPEC_NAME
    assert path.is_absolute()
    assert path.exists()

    spec = load_default_spec()
    assert spec.spec_version == DEFAULT_SPEC_VERSION
    # The canonical spec must carry the expanded Kimi K3 slot set.
    assert "expert" in spec.slots
    assert "attn_kv_a" in spec.slots
    assert "vision_qkv" in spec.slots
