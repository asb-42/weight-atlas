"""Tests for mapping_coverage format in fingerprint.json."""

from __future__ import annotations

from pathlib import Path

import pytest

from weight_atlas.scan import _build_fingerprint


@pytest.fixture
def spec():
    from weight_atlas.core.types import AtlasSpec
    return AtlasSpec.from_json(Path("specs/atlas_spec.v2.2.json"))


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


def test_mapping_coverage_mtp_draft_head_mapped(spec):
    """Qwen3-Next MTP draft head tensors (blk.N.nextn.*) must map to mtp slots."""
    from weight_atlas.core.types import TensorStats

    stats = [
        TensorStats(name="blk.64.nextn.eh_proj.weight", shape=(10240, 5120)),
        TensorStats(name="blk.64.nextn.enorm.weight", shape=(5120,)),
        TensorStats(name="blk.64.nextn.hnorm.weight", shape=(5120,)),
        TensorStats(name="blk.64.nextn.shared_head_norm.weight", shape=(5120,)),
    ]
    fp = _build_fingerprint(stats, spec, "gguf")

    mc = fp["mapping_coverage"]
    assert mc["in_slots"] == 1.0
    assert mc["in_other"] == 0.0
    assert mc["unmapped"] == 0
    assert mc["unmapped_tensors"] == []


def test_mapping_coverage_attn_sinks_mapped(spec):
    """GPT-OSS / Qwen3-Next attention-sink registers (blk.N.attn_sinks) must map."""
    from weight_atlas.core.types import TensorStats

    stats = [
        TensorStats(name=f"blk.{i}.attn_sinks.weight", shape=(64,))
        for i in range(36)
    ]
    fp = _build_fingerprint(stats, spec, "gguf")

    mc = fp["mapping_coverage"]
    assert mc["in_slots"] == 1.0
    assert mc["in_other"] == 0.0
    assert mc["unmapped"] == 0
    assert mc["unmapped_tensors"] == []


def test_mapping_coverage_flash_next_hf_families(spec):
    """Qwen3.8-Flash-Next HF export (Unsloth plefp8): PLE projections,
    indexer norms, hyper-connection mixers and MTP tower must all map —
    and MTP must stay off the language raster (layer None)."""
    from weight_atlas.core.types import TensorStats

    names = [
        "model.language_model.layers.1.ple.key_proj.weight",
        "model.language_model.layers.1.ple.value_proj.weight",
        "model.language_model.layers.1.ple.conv1d.weight",
        "model.language_model.layers.1.ple.norm_conv.weight",
        "model.language_model.layers.1.ple.norm_key.weight",
        "model.language_model.layers.1.ple.norm_query.weight",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_30.weight",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
        "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight",
        "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight",
        "model.language_model.layers.3.attn_hyper_connection.block_inject_weight.weight",
        "model.language_model.layers.3.attn_hyper_connection.hc_norm.weight",
        "model.language_model.layers.3.mlp_hyper_connection.input_mix_weight_up.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    ]
    stats = [TensorStats(name=n, shape=(32, 32)) for n in names]
    fp = _build_fingerprint(stats, spec, "safetensors")

    mc = fp["mapping_coverage"]
    assert mc["in_slots"] == 1.0, mc["unmapped_tensors"]
    assert mc["unmapped"] == 0

    from weight_atlas.core.name_map import map_name

    # PLE table material is non-layer (like the GGUF giant table)
    layer, slot = map_name(
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_30.weight"
    )
    assert (layer, slot) == (None, "ngram_embd")
    # ...while per-layer PLE projections keep their layer cells
    assert map_name("model.language_model.layers.1.ple.key_proj.weight") == (1, "ngram_key")


def test_mapping_mtp_never_collides_with_language_rows(spec):
    """MTP draft-tower tensors must map layer-None: mtp.layers.N would
    otherwise overwrite language row N (raster cells are last-wins)."""
    from weight_atlas.core.name_map import map_name

    for n in (
        "mtp.layers.3.self_attn.q_proj.weight",
        "mtp.layers.3.mlp.shared_expert.down_proj.weight",
        "mtp.layers.3.attn_hyper_connection.hc_norm.weight",
        "mtp.fc_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ):
        layer, slot = map_name(n)
        assert layer is None, f"{n} got language row {layer}"
        assert slot != "other" or "fc_embedding" in n or "pre_fc_norm" in n, n
