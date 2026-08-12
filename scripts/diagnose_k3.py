"""Diagnose Kimi K3 model (waste/k3) for processing / mapping issues.

Performs metadata-only checks (no tensor data is loaded):
1. Enumerates all tensor names from model.safetensors.index.json
2. Runs weight-atlas map_name() to compute mapping coverage / slot distribution
3. Categorizes unmapped tensors by structural prefix
4. Checks layer-index collisions (vision tower vs language model)
5. Cross-validates shard safetensors headers against the index weight_map
6. Reports tensor dtypes (esp. MXFP4 packed/scale) to assess processing impact
"""

from __future__ import annotations

import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

MODEL_DIR = Path("/media/data/coding/waste/k3")
INDEX = MODEL_DIR / "model.safetensors.index.json"

# Add weight-atlas to path
sys.path.insert(0, "/media/data/coding/weight-atlas/src")
from weight_atlas.core.name_map import map_name  # noqa: E402

_HDR = "===="
_DASH = "----"


def read_st_header(path: Path) -> dict:
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size))


def main() -> int:
    idx = json.loads(INDEX.read_text())
    weight_map: dict[str, str] = idx["weight_map"]
    names = list(weight_map.keys())
    print(f"Total tensors in index: {len(names)}")
    print(f"total_size (bytes): {idx['metadata']['total_size']}")

    # ---- 1/2. Mapping coverage via weight-atlas name map ----
    slot_counts: Counter[str] = Counter()
    unmapped: list[str] = []
    layer_counts: Counter[int] = Counter()
    for name in names:
        layer, slot = map_name(name)
        slot_counts[slot] += 1
        if slot == "other":
            unmapped.append(name)
        if layer is not None:
            layer_counts[layer] += 1

    n = len(names)
    in_slots = n - len(unmapped)
    print(f"\n{_HDR} Mapping coverage")
    print(f"in_slots: {in_slots} ({in_slots / n:.1%})  other: {len(unmapped)} ({len(unmapped) / n:.1%})")
    print("\nSlot distribution:")
    for slot, count in slot_counts.most_common():
        print(f"  {slot:16s} {count:6d}")

    # ---- Categorize unmapped tensors by structural prefix ----
    print(f"\n{_HDR} Unmapped ('other') tensor categories")
    cat_re = [
        ("block_sparse_moe.experts (packed+scale)", r"block_sparse_moe\.experts\.\d+\."),
        ("block_sparse_moe.shared_experts", r"block_sparse_moe\.shared_experts?"),
        ("block_sparse_moe.router / gate", r"block_sparse_moe\.(router|gate|norm|shared_expert_gate)"),
        ("MLA self_attn (q_a/q_b/kv_a/kv_b)", r"self_attn\.(q_a|q_b|kv_a|kv_b|o_proj|q_norm|k_norm)"),
        ("linear_attn", r"linear_attn"),
        ("kimi attention blocks", r"(situ|attention\.block|attn_res)"),
        ("vision_tower", r"vision_tower"),
        ("mm_projector / patch_merger", r"(mm_projector|patch_merger|merge)"),
        ("language_model prefixes", r"language_model"),
        ("media / multimodal", r"(media|image|video|audio)"),
        ("rotary / position", r"(rotary|position|rope)"),
        ("norm (rms/layer)", r"norm"),
    ]
    cat_counts: Counter[str] = Counter()
    for name in unmapped:
        matched = False
        for label, pat in cat_re:
            if re.search(pat, name):
                cat_counts[label] += 1
                matched = True
                break
        if not matched:
            cat_counts["(other / uncategorised)"] += 1
    for label, count in cat_counts.most_common():
        print(f"  {label:45s} {count:6d}")

    # ---- Sample unmapped names ----
    print(f"\n{_HDR} Sample unmapped names")
    for name in unmapped[:25]:
        print(f"  {name}")

    # ---- What actually maps (compact) ----
    print(f"\n{_HDR} Mapped tensor summary ({in_slots} tensors)")
    mapped_names = [name for name in names if map_name(name)[1] != "other"]
    # Full-attention layers: have q_proj/k_proj/v_proj/o_proj all mapped
    fa_layers = sorted(
        {
            map_name(n)[0]
            for n in mapped_names
            if n.endswith("self_attn.q_proj.weight")
        }
    )
    print(f"layers with mapped q_proj/k_proj/v_proj (full-attention): {len(fa_layers)} -> {fa_layers}")
    print("example mapped names:")
    for name in mapped_names[:8]:
        layer, slot = map_name(name)
        print(f"  [{layer}] {slot:12s} {name}")

    # ---- Detailed unmapped samples by category ----
    print(f"\n{_HDR} Detailed unmapped samples by category")
    detailed = [
        ("MLA self_attn", r"self_attn"),
        ("block_sparse_moe.shared_experts", r"shared_expert"),
        ("block_sparse_moe.router/gate/norm", r"(router|gate|moe_|norm)"),
        ("vision_tower", r"vision_tower"),
        ("mm_projector / patch_merger", r"(mm_projector|patch_merger|merge)"),
        ("language_model non-moe prefixes", r"language_model"),
        ("block_sparse_moe.experts", r"block_sparse_moe\.experts"),
    ]
    shown: set[str] = set()
    for label, pat in detailed:
        cnt = 0
        print(f"\n  {_DASH} {label} {_DASH}")
        for name in unmapped:
            if name in shown or not re.search(pat, name):
                continue
            if label == "language_model non-moe prefixes" and "block_sparse_moe" in name:
                continue
            print(f"    {name}")
            shown.add(name)
            cnt += 1
            if cnt >= 20:
                print(f"    ... ({cnt}+ shown of category)")
                break
        if cnt == 0:
            print("    (none)")

    # ---- Layer index analysis ----
    print(f"\n{_HDR} Layer index analysis")
    print(f"layers referenced: {len(layer_counts)} distinct indices")
    print(f"min layer: {min(layer_counts)}  max layer: {max(layer_counts)}")

    def family(name: str) -> str:
        if name.startswith("vision_tower"):
            return "vision"
        if "block_sparse_moe" in name:
            return "moe"
        if "linear_attn" in name or "self_attn" in name:
            return "language"
        if "mm_projector" in name or "patch_merger" in name:
            return "projector"
        return "language/other"

    fam_layers: dict[str, set[int]] = defaultdict(set)
    for name in names:
        layer = map_name(name)[0]
        if layer is not None:
            fam_layers[family(name)].add(layer)
    for fam, layers in sorted(fam_layers.items()):
        print(f"  {fam:12s} layers: {len(layers)}  range {min(layers)}-{max(layers)}")

    # ---- Shard cross-validation ----
    print(f"\n{_HDR} Shard header cross-validation vs index")
    shards = sorted(MODEL_DIR.glob("model-*.safetensors"))
    print(f"shard files present: {len(shards)}")

    index_to_shard: dict[str, set[str]] = defaultdict(set)
    for name, shard in weight_map.items():
        index_to_shard[shard].add(name)

    missing_in_shard: list[str] = []
    orphan_in_shard: list[str] = []
    header_counts = Counter()
    for shard in shards:
        header = read_st_header(shard)
        header_keys = {k for k in header if k != "__metadata__"}
        header_counts[len(header_keys)] += 1
        expected = index_to_shard.get(shard.name, set())
        missing = expected - header_keys
        orphan = header_keys - expected
        if missing:
            missing_in_shard.append(f"{shard.name}: missing {len(missing)} (e.g. {sorted(missing)[:3]})")
        if orphan:
            orphan_in_shard.append(f"{shard.name}: orphan {len(orphan)} (e.g. {sorted(orphan)[:3]})")

    print(f"shards whose header keys match index exactly: {header_counts.most_common(1)}")
    if missing_in_shard:
        print(f"SHARDS MISSING TENSORS ({len(missing_in_shard)}):")
        for m in missing_in_shard[:10]:
            print(f"  {m}")
    else:
        print("No shards missing index-referenced tensors.")
    if orphan_in_shard:
        print(f"SHARDS WITH ORPHAN TENSORS ({len(orphan_in_shard)}):")
        for o in orphan_in_shard[:10]:
            print(f"  {o}")
    else:
        print("No orphan tensors in shards (all accounted for by index).")

    # ---- Dtype / format analysis ----
    print(f"\n{_HDR} Dtype / format analysis")
    # Which shards contain MoE tensors
    moe_shards = []
    for s in shards:
        h = read_st_header(s)
        if any("block_sparse_moe" in k for k in h if k != "__metadata__"):
            moe_shards.append(s)
    print(f"shards containing MoE tensors: {len(moe_shards)} (e.g. {[s.name for s in moe_shards[:3]]})")

    if moe_shards:
        h = read_st_header(moe_shards[0])
        packed_dtypes: Counter[str] = Counter()
        scale_dtypes: Counter[str] = Counter()
        shapes: Counter[tuple] = Counter()
        moe_sample = []
        for k, v in h.items():
            if k == "__metadata__" or "block_sparse_moe" not in k:
                continue
            if k.endswith("weight_packed"):
                packed_dtypes[v["dtype"]] += 1
                shapes[tuple(v["shape"])] += 1
                if len(moe_sample) < 6:
                    moe_sample.append((k, tuple(v["shape"]), v["dtype"]))
            elif k.endswith("weight_scale"):
                scale_dtypes[v["dtype"]] += 1
        print(f"  {moe_shards[0].name} packed dtypes: {dict(packed_dtypes)}")
        print(f"  {moe_shards[0].name} scale dtypes: {dict(scale_dtypes)}")
        print(f"  packed shapes: {list(shapes.items())[:6]}")
        print("  sample MoE entries:")
        for s in moe_sample:
            print(f"    {s}")

    # Also check a non-MoE shard's self_attn dtypes
    non_moe = [s for s in shards if s not in moe_shards]
    if non_moe:
        h = read_st_header(non_moe[0])
        sa_dtypes: Counter[str] = Counter()
        for k, v in h.items():
            if k == "__metadata__":
                continue
            if "self_attn" in k or "input_layernorm" in k or "post_attention_layernorm" in k:
                sa_dtypes[v["dtype"]] += 1
        print(f"  non-MoE shard {non_moe[0].name} self_attn/norm dtypes: {dict(sa_dtypes)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
