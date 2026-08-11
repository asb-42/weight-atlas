"""Debug script to diagnose visualization issues.

Run: python scripts/debug_visualization.py <path_to_scan_directory>
"""
import sys
import json
from pathlib import Path
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_visualization.py <scan_directory>")
        sys.exit(1)

    scan_dir = Path(sys.argv[1])
    if not scan_dir.exists():
        print(f"Directory not found: {scan_dir}")
        sys.exit(1)

    # Check fingerprint
    fp_path = scan_dir / "fingerprint.json"
    if fp_path.exists():
        with open(fp_path) as f:
            fp = json.load(f)
        print(f"Spec version: {fp.get('spec_version', 'N/A')}")
        scaling = fp.get('scaling', {})
        if scaling:
            print(f"Scaling method: {scaling.get('method', 'N/A')}")
            print(f"Scaling params: {scaling.get('params', 'N/A')}")
            for ch, meta in scaling.get('channels', {}).items():
                print(f"  {ch}: q_lo={meta.get('q_lo')}, q_hi={meta.get('q_hi')}, "
                      f"raw_min={meta.get('raw_min')}, raw_max={meta.get('raw_max')}")
        else:
            print("WARNING: No scaling block in fingerprint - scan did not use v2.1 spec!")
    else:
        print("WARNING: No fingerprint.json found")

    # Check TIFF files
    for tif in sorted(scan_dir.glob("field_*.tif")):
        from weight_atlas.fields.tif_io import read_tif
        data = read_tif(tif)
        finite = data[np.isfinite(data)]
        print(f"\n{tif.name}:")
        print(f"  Shape: {data.shape}")
        print(f"  Finite: {finite.size}/{data.size} ({finite.size/data.size*100:.1f}%)")
        if finite.size > 0:
            print(f"  Min: {np.min(finite):.6f}")
            print(f"  Max: {np.max(finite):.6f}")
            print(f"  Mean: {np.mean(finite):.6f}")
            print(f"  Std: {np.std(finite):.6f}")
            print(f"  Median: {np.median(finite):.6f}")
            # Check distribution
            p1 = np.percentile(finite, 1)
            p99 = np.percentile(finite, 99)
            print(f"  1st percentile: {p1:.6f}")
            print(f"  99th percentile: {p99:.6f}")
            print(f"  Range (99-1): {p99-p1:.6f}")
            # Check if data is clustered
            if finite.size > 10:
                hist, edges = np.histogram(finite, bins=10)
                print(f"  Histogram (10 bins):")
                for i, (h, e) in enumerate(zip(hist, edges)):
                    print(f"    [{e:.4f}, {edges[i+1]:.4f}): {h} ({h/finite.size*100:.1f}%)")

    # Check render directory
    render_dir = scan_dir / "render"
    if render_dir.exists():
        print(f"\nRendered PNGs:")
        for png in sorted(render_dir.glob("*.png")):
            print(f"  {png.name} ({png.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
