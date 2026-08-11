"""Diagnose per-slot distribution of tensor statistics.

Usage: python scripts/diagnose_distribution.py <fingerprint.json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from weight_atlas.core.name_map import map_name


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_distribution.py <fingerprint.json>")
        sys.exit(1)

    fp = json.loads(Path(sys.argv[1]).read_text())
    tensors = fp["tensors"]

    # Group by slot
    by_slot = defaultdict(list)
    for name, t in tensors.items():
        layer, slot = map_name(name)
        by_slot[slot].append({
            "name": name,
            "layer": layer,
            "spectral_norm": t.get("spectral_norm"),
            "stable_rank": t.get("stable_rank"),
            "kurtosis": t.get("kurtosis"),
        })

    print(f"Total tensors: {len(tensors)}")
    print(f"Slots: {len(by_slot)}")
    print()

    for slot in sorted(by_slot.keys()):
        items = by_slot[slot]
        sn = [x["spectral_norm"] for x in items if x["spectral_norm"] is not None]
        sr = [x["stable_rank"] for x in items if x["stable_rank"] is not None]
        ku = [x["kurtosis"] for x in items if x["kurtosis"] is not None]

        print(f"=== {slot}: n={len(items)} ===")

        if sn:
            sn = np.array(sn)
            print(f"  spectral_norm: min={sn.min():.3f}, max={sn.max():.3f}, "
                  f"median={np.median(sn):.3f}, p99={np.percentile(sn,99):.3f}, "
                  f"p1={np.percentile(sn,1):.3f}")
            # Show top 5 outliers
            top_idx = np.argsort(sn)[-5:][::-1]
            print(f"    top 5: {sn[top_idx].tolist()}")

        if sr:
            sr = np.array(sr)
            print(f"  stable_rank:   min={sr.min():.3f}, max={sr.max():.3f}, "
                  f"median={np.median(sr):.3f}, p99={np.percentile(sr,99):.3f}")

        if ku:
            ku = np.array(ku)
            print(f"  kurtosis:      min={ku.min():.3f}, max={ku.max():.3f}, "
                  f"median={np.median(ku):.3f}, p99={np.percentile(ku,99):.3f}")
            # Show outliers
            top_idx = np.argsort(ku)[-5:][::-1]
            bot_idx = np.argsort(ku)[:5]
            print(f"    top 5: {ku[top_idx].tolist()}")
            print(f"    bot 5: {ku[bot_idx].tolist()}")

        print()


if __name__ == "__main__":
    main()
