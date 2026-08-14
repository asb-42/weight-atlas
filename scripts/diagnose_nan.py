#!/usr/bin/env python3
"""
weight-atlas NaN / Slot-Gap Diagnose
====================================
Analysiert eine fingerprint.json (oder die TIFF-Raster) und zeigt:
  - Welche Slots pro Layer fehlen (NaN)
  - mapping_coverage
  - Visuelle Karte der Lücken
  - Verdacht auf Architektur-Muster (Vision-Tower, QK-Norm, MoE, etc.)

Usage:
    python diagnose_nan.py ./artefacts/fingerprint.json
    python diagnose_nan.py ./artefacts/  # sucht automatisch fingerprint.json
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict, Counter

try:
    import numpy as np
    from PIL import Image
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[WARN] numpy oder PIL nicht verfügbar – TIFF-Analyse übersprungen.")


def load_fingerprint(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_mapping_coverage(fp: dict):
    """Prüft den mapping_coverage-Block."""
    mc = fp.get("mapping_coverage")
    if not mc:
        print("[!] Kein 'mapping_coverage' in fingerprint.json gefunden.")
        return

    print("\n" + "=" * 60)
    print("MAPPING COVERAGE")
    print("=" * 60)
    total = mc.get("total_tensors", 0)
    mapped = mc.get("mapped_tensors", 0)
    pct = mc.get("coverage_percent", 0)
    print(f"  Tensoren gesamt:   {total}")
    print(f"  Gemappt:           {mapped}")
    print(f"  Coverage:          {pct:.1f}%")

    unmapped = mc.get("unmapped_names", [])
    if unmapped:
        print(f"\n  Unmapped Tensoren ({len(unmapped)}):")
        for name in unmapped[:20]:
            print(f"    - {name}")
        if len(unmapped) > 20:
            print(f"    ... und {len(unmapped) - 20} weitere")

    missing_slots = mc.get("missing_slots", [])
    if missing_slots:
        print(f"\n  Fehlende Slots ({len(missing_slots)}):")
        for slot in missing_slots:
            print(f"    - {slot}")


def analyze_tensor_list(fp: dict):
    """Analysiert die Tensor-Liste nach Layer × Slot."""
    tensors = fp.get("tensors", fp.get("stats", []))
    if not tensors:
        print("[!] Keine Tensor-Liste in fingerprint.json gefunden.")
        return None, None

    # Baue Raster-Struktur
    by_layer_slot = defaultdict(dict)
    all_slots = set()
    all_layers = set()

    for t in tensors:
        layer = t.get("layer")
        slot = t.get("slot")
        if layer is None or slot is None:
            continue
        by_layer_slot[layer][slot] = t
        all_slots.add(slot)
        all_layers.add(layer)

    layers = sorted(all_layers)
    slots = sorted(all_slots, key=lambda s: (
        0 if s.startswith("embed") else
        1 if s.startswith("norm") else
        2 if "attn" in s else
        3 if "mlp" in s else
        4 if s.startswith("lm_head") else
        5
    ))

    return layers, slots, by_layer_slot


def build_nan_raster(layers, slots, by_layer_slot):
    """Erzeugt eine Bool-Matrix: True = NaN / fehlend."""
    raster = []
    for layer in layers:
        row = []
        for slot in slots:
            t = by_layer_slot[layer].get(slot)
            if t is None:
                row.append(True)
            else:
                # Prüfe auf NaN in den Stats
                sn = t.get("spectral_norm")
                is_nan = sn is None or (isinstance(sn, float) and np.isnan(sn)) if HAS_NUMPY else sn is None
                row.append(is_nan)
        raster.append(row)
    return np.array(raster) if HAS_NUMPY else raster


def print_nan_map(layers, slots, nan_raster):
    """ASCII-Visualisierung der NaN-Lücken."""
    print("\n" + "=" * 60)
    print("NaN-KARTE (Layer × Slot)")
    print("=" * 60)
    print("Legende:  █ = Daten vorhanden   · = NaN / fehlend")
    print()

    # Header
    header = "Layer │ " + " ".join(f"{s[:4]:>4}" for s in slots)
    print(header)
    print("─" * len(header))

    for i, layer in enumerate(layers):
        if HAS_NUMPY:
            row_str = " ".join("████" if not nan_raster[i, j] else "····" for j in range(len(slots)))
        else:
            row_str = " ".join("████" if not nan_raster[i][j] else "····" for j in range(len(slots)))
        print(f" {layer:>4} │ {row_str}")

    # Zusammenfassung pro Slot
    print("\n" + "─" * 40)
    print("NaN-Statistik pro Slot:")
    for j, slot in enumerate(slots):
        if HAS_NUMPY:
            nan_count = nan_raster[:, j].sum()
        else:
            nan_count = sum(1 for i in range(len(layers)) if nan_raster[i][j])
        pct = 100 * nan_count / len(layers)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {slot:20s} {bar} {nan_count:>3}/{len(layers)} ({pct:>5.1f}%)")


def detect_patterns(layers, slots, by_layer_slot):
    """Erkennt verdächtige Architektur-Muster."""
    print("\n" + "=" * 60)
    print("MUSTER-ERKENNUNG")
    print("=" * 60)

    # 1. Vision-Tower am Anfang?
    first_layers = [l for l in layers if l < 3]
    vision_slots = ["vision", "img", "pixel", "patch", "visual"]
    for layer in first_layers:
        for slot in by_layer_slot[layer]:
            if any(v in slot.lower() for v in vision_slots):
                print(f"  [Vision] Slot '{slot}' in Layer {layer} gefunden.")

    # 2. QK-Norm nur in bestimmten Layern?
    qk_norm_layers = sorted(
        l for l in layers
        if "attn_q_norm" in by_layer_slot[l] or "attn_k_norm" in by_layer_slot[l]
    )
    if qk_norm_layers:
        print(f"  [QK-Norm] Vorhanden in Layern: {qk_norm_layers[:10]}{'...' if len(qk_norm_layers) > 10 else ''}")
        if qk_norm_layers[0] > 0:
            print(f"           -> Fehlt in den ersten {qk_norm_layers[0]} Layer(n). Das erklärt vertikale Lücken.")

    # 3. MoE-Router / Expert-Muster?
    has_router = any("router" in s for s in slots)
    has_expert = any("expert" in s for s in slots)
    if has_router:
        print("  [MoE] Router-Slot erkannt – Modell hat Mixture-of-Experts.")
    if has_expert:
        print("  [MoE] Expert-Slots erkannt – Expert-Panels sollten vorhanden sein.")

    # 4. Lücken in Embed / LM-Head?
    embed_layers = [l for l in layers if "embed" in by_layer_slot[l]]
    lm_head_layers = [l for l in layers if "lm_head" in by_layer_slot[l]]
    if len(embed_layers) <= 1:
        print(f"  [Embed] Nur in Layer {embed_layers} – typisch (nur Layer 0).")
    if len(lm_head_layers) <= 1:
        print(f"  [LM-Head] Nur in Layer {lm_head_layers} – typisch (nur letzter Layer).")

    # 5. Gleichmäßige Lücken = wahrscheinlich Spec-Lücke
    for slot in slots:
        present = sorted(l for l in layers if slot in by_layer_slot[l])
        if not present:
            print(f"  [WARN] Slot '{slot}' komplett fehlend – Spec-Mismatch!")
            continue
        gaps = [present[i+1] - present[i] for i in range(len(present)-1)]
        if gaps and max(gaps) > 2:
            # Prüfe, ob es ein regelmäßiges Muster gibt
            if len(set(gaps)) == 1:
                print(f"  [Muster] Slot '{slot}' nur jeden {gaps[0]}. Layer (regelmäßig).")
            else:
                print(f"  [Lücke] Slot '{slot}' hat unregelmäßige Lücken: max Abstand = {max(gaps)} Layer.")


def analyze_tiffs(artefact_dir: Path):
    """Optional: Lädt die TIFF-Raster und zeigt NaN-Positionen."""
    if not HAS_NUMPY:
        return

    for tiff_name in ["field_height_raw.tif", "field_tint_raw.tif", "field_rough_raw.tif"]:
        tiff_path = artefact_dir / tiff_name
        if not tiff_path.exists():
            continue

        img = np.array(Image.open(tiff_path))
        nan_mask = np.isnan(img)
        if not nan_mask.any():
            continue

        print(f"\n  [TIFF] {tiff_name}:")
        print(f"         NaN-Pixel: {nan_mask.sum():,} / {img.size:,} ({100*nan_mask.sum()/img.size:.2f}%)")

        # Finde vertikale Linien (NaN-Spalten)
        nan_cols = np.where(nan_mask.all(axis=0))[0]
        if len(nan_cols):
            print(f"         Komplett-NaN-Spalten: {len(nan_cols)} (wahrscheinlich fehlende Slots)")

        # Finde horizontale Linien
        nan_rows = np.where(nan_mask.all(axis=1))[0]
        if len(nan_rows):
            print(f"         Komplett-NaN-Zeilen: {len(nan_rows)} (wahrscheinlich fehlende Layer)")


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if target.is_dir():
        fp_path = target / "fingerprint.json"
        if not fp_path.exists():
            # Suche rekursiv
            candidates = list(target.rglob("fingerprint.json"))
            if candidates:
                fp_path = candidates[0]
            else:
                print("[!] Keine fingerprint.json gefunden.")
                sys.exit(1)
    else:
        fp_path = target

    print(f"Lade: {fp_path}")
    fp = load_fingerprint(fp_path)
    artefact_dir = fp_path.parent

    # 1. Mapping Coverage
    analyze_mapping_coverage(fp)

    # 2. Tensor-Raster-Analyse
    result = analyze_tensor_list(fp)
    if result:
        layers, slots, by_layer_slot = result
        print(f"\nGefunden: {len(layers)} Layer × {len(slots)} Slots")
        nan_raster = build_nan_raster(layers, slots, by_layer_slot)
        print_nan_map(layers, slots, nan_raster)
        detect_patterns(layers, slots, by_layer_slot)

    # 3. TIFF-Analyse
    analyze_tiffs(artefact_dir)

    print("\n" + "=" * 60)
    print("Empfehlung:")
    print("  - Wenn Coverage < 100%: Prüfe das Name-Mapping (GGUF vs. Safetensors).")
    print("  - Wenn Slots regelmäßig fehlen: Spec erweitern oder 'optional_slots' einführen.")
    print("  - Wenn NaN-Lücken stören: Im Renderer auf neutralgrau (0.5) statt Weiß setzen.")
    print("=" * 60)


if __name__ == "__main__":
    main()
