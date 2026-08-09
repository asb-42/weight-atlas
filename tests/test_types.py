"""Tests for core/types.py: AtlasSpec loading."""

from __future__ import annotations

from pathlib import Path

from weight_atlas.core.types import AtlasSpec


def test_load_spec():
    spec = AtlasSpec.from_json(Path("specs/atlas_spec.v2.json"))
    assert spec.spec_version == 2
    assert "attn_q" in spec.slots
    assert spec.channel_stat("height") == "spectral_norm"
    assert spec.channel_stat("tint") == "stable_rank"
    assert spec.channel_scale("tint")["type"] == "log1p"
    assert spec.channel_scale("rough")["type"] == "quantile_clip"
    assert spec.sheet["contour_levels"] == 12
    assert spec.seeds["svd"] == 0
