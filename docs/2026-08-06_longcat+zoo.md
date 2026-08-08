Build-Spec

0. Festlegungen (bindend; Änderung nur nach explizitem Ok)

- Python 3.11+, Packaging via pyproject.toml + uv. Extras: web, blender, gguf, umap – Core bleibt schlank.
- Core-Dependencies: numpy, safetensors, scipy, matplotlib, tifffile (Float32-TIFF als Heightmap-Format – Blender liest das nativ, keine C-Builds; EXR nur falls später nötig).
- Trennung Analyse/Präsentation: Kanonisches Artefakt ist renderer-unabhängig: fingerprint.parquet + field_<channel>.tif + atlas_spec.v1.json. - Blender und matplotlib sind nur Renderer-Plugins; Vergleiche laufen später auf den Feldern, nicht auf den Bildern.
- UI: FastAPI + Jinja2 + HTMX + wenig Alpine.js. Kein Node-Build, keine SPA. (Erweiterbarkeit kommt aus der Backend-Architektur, nicht aus Frontend-Magie.)
- Determinismus vor Schönheit: Jeder RNG seeded aus der Spec, keine Zeitstempel/Metadaten in Artefakten, jede Skalierungsentscheidung in der Spec versioniert.

1. Repo-Struktur

weight-atlas/
├── pyproject.toml
├── specs/atlas_spec.v1.json          # versionierte Kartografie-Konvention
├── src/weight_atlas/
│   ├── core/
│   │   ├── registry.py               # Plugin-Registry (Loader/Stats/Renderer)
│   │   ├── types.py                  # TensorHandle, TensorStats, Field2D, AtlasSpec
│   │   └── name_map.py               # Tensorname → (layer, slot), pro Modellfamilie
│   ├── loaders/   base.py · safetensors_loader.py · (gguf: Backlog)
│   ├── stats/     base.py · norms.py · spectral.py · shape_moments.py
│   ├── fields/    rasterizer.py · scaling.py · smoothing.py · tif_io.py
│   ├── render/    base.py · matplotlib_sheet.py · blender/{terrain.blend, render_terrain.py, wrapper.py}
│   ├── api/       main.py · jobs.py · routes.py        # M3
│   ├── ui/        templates/ · static/                 # M3
│   └── cli.py
├── tests/  (fixtures: winzige geseedete Fake-Modelle, keine echten Downloads)
└── docs/   BACKLOG.md · ARCHITECTURE.md

2. Kern-Interfaces (so bauen, Abweichungen nur nach Rückfrage)

class Loader(Protocol):
    format_id: str
    def open(self, path: Path) -> Sequence[TensorHandle]   # lazy/mmap, nie Voll-Load

class Statistic(Protocol):
    stat_id: str
    def compute(self, t: TensorHandle) -> float            # pure, seeded

class Renderer(Protocol):
    renderer_id: str
    def render(self, field: Field2D, spec: AtlasSpec, out: Path) -> list[Path]

- TensorHandle.load() materialisiert erst auf Aufruf, als float32.
- Registry-Dekoratoren: @register.loader("safetensors") usw. Neue Visualisierung = neues Renderer-Plugin; neuer Vergleich (M4+) = Funktion auf Field2D – nichts davon erfordert Core-Änderungen.
- Raster-Konvention (Spec v1): Zeile = Layer, Spalte = Komponenten-Slot [embed, attn_q/k/v/o, mlp_up/gate/down, norm, router, lm_head]. Unbekannte Tensoren → Warnung + other, nie still ignorieren.
- Kanäle: Höhe = log1p(spektral), Tönung = effective_rank (quantil-geclippt 1–99 %), Rauheit (Mikro-Displacement) = kurtosis. MoE-Experten: eigenes Sheet pro Slot (Zeile = Layer, Spalte = Expert-ID) – ab M1 zulassen, aber erst in M2+ rendern.

3. Milestones + Definition of Done

M0 – Scaffolding. Repo, pyproject, CI (lint+test), Registry, types.py, CLI-Gerüst (--help).
DoD: CI grün; weight-atlas --help läuft; Registry-Test grün.

M1 – Vertikale Slice ohne Blender. safetensors-Loader (mmap, sharded); Stats: Frobenius, Spektralnorm+eff. Rang (randomized SVD, numpy-eigen, Seed fix), Kurtosis, Sparsity; Rasterizer; Scaling/Smoothing; Float32-TIFF-Export; matplotlib-Sheet (Hillshade + Isolinien + hypsometrische Tönung, rohe und geglättete Version).
DoD: CLI scan <file> erzeugt Fingerprint + Feld + Sheet für ein Fixture-Modell; zweiter Lauf = identische SHA-256; Stats-Unit-Tests gegen handgerechnete Werte an einer Fixmatrix.

M2 – Blender-Renderer. terrain.blend + render_terrain.py (bpy): Grid 1024², Displace via TIFF, Höhenlinien im Shader, Licht NW fix, ortho Top-View; Wrapper ruft headless auf (blender -b … -P … --), Pfad via Env-Var WEIGHT_ATLAS_BLENDER.
DoD: gleicher Feld+Spec-Input → pixelidentisches PNG; CI testet nur den Wrapper (Dry-Run), Blender-Render lokal dokumentiert mit Smoke-Test.

M3 – Web-UI. Verzeichnis-Scan, Job-Queue (In-Prozess-Worker + SQLite-Status), Seiten: Modellliste, Detail (Sheet, Terrain, Stats-Tabelle, Spec), Job-Progress.
DoD: kompletter Flow „Ordner wählen → Job → Artefakte ansehen“ ohne manuelle Dateioperationen; keine neuen Frontend-Dependencies.

M4+ – Backlog (nur in docs/BACKLOG.md eintragen, nicht bauen): GGUF-Loader, Δ-Karten/Erosions-Tönung für Abliteration, Mesh-Export (OBJ) als diffbares Fingerprint-Artefakt, UMAP-Embedding-Sheet, MoE-Experten-Panel-Render, Aktivitäts-/„fMRI“-Modus via NNsight, Morph-Animation A→B.

4. Rules of Engagement

- Ein Milestone nach dem anderen. Nach jedem Milestone: Kurzreport (Umgesetzt/Abweichungen/Offene Fragen) + Stopp, Weiterarbeit erst nach Ok. Report speichern unter docs/reports/{datum}_progress-report.md
- Conventional Commits pro Task; vor jedem Commit: Lint + Tests grün.
- Scope-Disziplin: Neue Ideen → BACKLOG.md, nicht in den laufenden Milestone.
- Optionale Dependencies nur hinter Lazy-Imports; Core muss ohne Blender/GPU/Node installierbar und testbar bleiben.
- Tests nur gegen geseedete Fixture-Modelle (Generator in tests/fixtures.py); CI lädt nie echte Gewichte.
- Öffentliche Protokolle mit Docstring; jede Nicht-Offensichtlichkeit (z. B. SVD-Iterationen, Clip-Quantile) in ARCHITECTURE.md begründet.

----

Milestones M0 + M1

Rolle: Du implementierst ein neues Open-Source-Tool. Spezifikation ist bindend; Abweichungen nur als dokumentierte Entscheidung in docs/ARCHITECTURE.md mit Begründung, andernfalls Rückfrage.
Ziel: Vertikale Slice: *.safetensors einlesen → Tensor-Statistiken → 2D-Felder (renderer-unabhängig) → topografisches 2D-Blatt. CLI-only, deterministisch, getestet. Keine UI, kein Blender in diesem Auftrag (kommt in M2/M3).

Kontext & Festlegungen

- Python ≥3.11, uv, Layout src/weight_atlas/, CLI-Entry weight-atlas.
- Deps Core: numpy, safetensors, scipy, matplotlib, tifffile. Dev: pytest, ruff, mypy. Keine weiteren Dependencies ohne Rückfrage.
- Kanonische Artefakte (renderer-unabhängig): fingerprint.json, field_<channel>_{raw,smooth}.tif, manifest.json (sha256 je Artefakt).
- Determinismus: alle RNGs seeded; keine Zeitstempel in Artefakten; PNG-Metadaten fix (Software: weight-atlas, Creation Time: 1970-01-01T00:00:00Z); TIFF byte-identisch bei Zweitlauf (Test!).
- Raster-Konvention: Zeile = Layer, Spalte = Slot in Spec-Reihenfolge; fehlende Werte = NaN, nie implizit auffüllen.

M0 – Scaffolding (in dieser Reihenfolge, Commit pro Task)

- Repo + pyproject.toml (Extras anlegen: web, blender – leer, nur reserviert), ruff/mypy-Config, GitHub-Actions (lint, mypy, test).
- core/registry.py: Registry mit register_loader/stat/renderer + Lookups; Duplikat-ID → ValueError; Tests.
- core/types.py: TensorHandle (name, shape, dtype, load() -> np.ndarray float32, lazy), TensorStats, Field2D (channel, np.ndarray, row/col-Labels, spec_version), AtlasSpec (pydantic oder dataclass+validiert, lädt JSON).
- core/name_map.py: Regex-Mapping Llama/Qwen-Stil (self_attn.q_proj→attn_q, mlp.gate_proj→mlp_gate, input_layernorm→norm_attn, embed_tokens→embed, lm_head, unknown→other); Layer-Index extrahieren; Tests mit echten Namensmustern.
- cli.py mit --help (click oder argparse; Entscheidung begründen).
- specs/atlas_spec.v1.json exakt wie unten anlegen.

{
  "spec_version": 1,
  "slots": ["embed","attn_q","attn_k","attn_v","attn_o","mlp_gate","mlp_up",
            "mlp_down","norm_attn","norm_mlp","router","lm_head","other"],
  "channels": {
    "height": {"stat": "spectral_norm",  "scale": {"type": "log1p"}},
    "tint":   {"stat": "effective_rank", "scale": {"type": "quantile_clip", "lo": 0.01, "hi": 0.99}},
    "rough":  {"stat": "kurtosis",       "scale": {"type": "log1p"}}
  },
  "grid":  {"upsample": 8, "smooth_sigma": 1.0},
  "sheet": {"contour_levels": 12, "light_azdeg": 315, "light_altdeg": 45, "dpi": 150},
  "seeds": {"svd": 0}
}

M1 – Vertikale Slice

1. Loader loaders/safetensors_loader.py: safe_open(framework="np"), mmap; Input = Datei oder Verzeichnis (glob *.safetensors, sortiert; Duplikatnamen → Fehler). Registry-ID safetensors.

2. Stats (stats/, je Klasse mit stat_id, pure, seeded wo nötig):
- frobenius (Akkumulation float64, chunked über flatten-Blöcke),
- spectral_norm + effective_rank: kleine Matrizen (≤512²) exakt via np.linalg.svd; sonst randomisierte SVD (Halko: k=16, p=4, q=2 Power-Iterations, Ω aus default_rng(spec.seeds.svd)); eff. Rang = exp(−Σ pᵢ·ln pᵢ), p = s/Σs; 1-D-Tensoren: spectral = L2, effrank = 1. Bias-Vektoren (1-D, nicht norm) trotzdem rechnen.
- kurtosis (Fisher), sparsity (Anteil |w|<1e-3).
- Unit-Tests gegen Handwerte: diag(3,4)→spectral 4; I₈→effrank 8; bekannte kleine Matrizen für frobenius/kurtosis/sparsity.

3. Fields: rasterizer.py (Stats-Tabelle → Matrizen je Channel, rows=Layer, cols=Slots, NaN wo fehlt), scaling.py (log1p, quantile_clip), smoothing.py (bilinear upsample scipy.ndimage.zoom + Gaussian; raw immer zusätzlich behalten), tif_io.py (float32, byte-deterministisch).

4. Renderer render/matplotlib_sheet.py (Registry-ID sheet): Hillshade (LightSource, az/alt aus Spec), hypsometrische Tönung (Colormap grün→braun→weiß, fix), Contours (n=Spec), Achsenbeschriftung Layer/Slot, Colorbar; raw und smooth als PNG. Agg-Backend, fixe dpi/figsize.

5. CLI:
- weight-atlas scan PATH --out DIR [--spec SPEC] → fingerprint.json (sorted keys, indent 2) + alle field_*.tif + manifest.json.
- weight-atlas render OUT_DIR --renderer sheet → PNGs aus den Artefakten (Render liest nur Artefakte, nie Gewichte – wichtige Architekturregel!).

6. Tests: Fixture-Generator tests/fixtures.py (Fake-Modell: 4 Layer, hidden 32, alle Slots, seeded randn mit slot-abhängiger Skala, als safetensors in tmp_path); Rasterizer-Test (Shape, NaN-Position); Determinismus-Test: scan+render zweimal → alle sha256 identisch (manifest-Vergleich).

7. Doku: README.md (Quickstart), docs/ARCHITECTURE.md (Pipeline, Konventionen, Begründungen: TIFF statt EXR, effrank aus trunc. Spektrum = akzeptierte Bias, PNG-Metadaten-Fix).

Definition of Done (selbst prüfen, dann reporten)

- uv run weight-atlas --help ok; scan+render auf Fixture laufen fehlerfrei.
- uv run pytest -q grün inkl. Handwerte- und Determinismus-Tests; ruff check + mypy src/ clean.
- Zweitlauf byte-identisch (manifest-sha256-Vergleich im Test).
- Keine neuen Dependencies; keine UI/Blender-Anfänge; neue Ideen stattdessen in docs/BACKLOG.md.
- Commits: conventional, pro Task, CI grün je Commit.

Report-Format am Ende

Bullet-Liste: (1) umgesetzt je Milestone, (2) Abweichungen + Begründung, (3) offene Fragen, (4) Inhalt BACKLOG.md. Danach Stopp – M2 erst nach Freigabe.

----

Review M0+M1

DoD: erfüllt. 44 Tests inkl. Handwerten und Byte-Determinismus, Lint/Typen clean, keine Scope-Drift, Backlog korrekt geführt. Damit ist die wichtigste Architekturregel („Render liest nur Artefakte“) schon in M1 verankert – gut.

Deviations: alle sechs akzeptiert, Kommentar zu dreien:

- quantile_clip normalisiert auf [0,1] → explizit begrüßt; das macht den Tint-Kanal modellvergleichbar, genau das Ziel.
- conftest/Modul-Imports für Registry-Seiteneffekte → pragmatisch korrekt. Bitte in ARCHITECTURE.md als bekannte Einschränkung notieren (Fallback: Entry-Point-Registration, falls Plugin-Zukunft kommt – in den Backlog).
- mypy 3.12 + scipy-stubs → ok, dev-only.

Offene Fragen → Entscheidungen (als Fixup-Batch M1.5, ein Commit-Block, danach Tests grün):

1. Contours: jetzt einbauen, nicht auf M2 schieben – sie sind Teil der Spec und tragen die „Landkarten“-Lesbarkeit. Levels deterministisch und vergleichbar: np.linspace(q02, q98, spec.sheet.contour_levels) über das skalierte Höhenfeld (q = 2%/98%-Quantil des Felds), Linienfarbe fix dunkel, alpha fix. Dokumentieren.
2. PNG-Creation-Time: ja, fix 1970-01-01T00:00:00Z via metadata=-Kwarg setzen. Aktuell ist es zwar schon deterministisch, aber die Spec sagt Fixwert – billig, und verhindert Überraschungen, falls jemand später Default-Metadaten ändert.
3. spec_version in Artefakten: ja, zwingend. fingerprint.json bekommt Top-Level-Block {"spec_version": 1, "tool_version": <importlib>, "loader": "safetensors", "model": {...}, "tensors": [...]}. Das ist die Voraussetzung, dass M4- Vergleiche Cross-Spec-Läufe hart ablehnen können.
4. Render-Discovery: auf Manifest umstellen. Dateinamen bleiben Konvention, aber manifest.json ist Source of Truth für render. Kleiner Refactor, keine Ausreden.

Zusatz-Clarification (kein Code-Change erwartet, nur Doku): Der Sheet bleibt eine reine Höhenkarte – Hillshade + hypsometrische Tönung + Contours alle aus dem Height-Kanal. Tint/Rough sind Felder für Blender (M2) und spätere Sheets. In ARCHITECTURE.md als bewusste Entscheidung festhalten.

Wenn M1.5 landet und grün ist: M2 freigegeben. Report bitte wieder im gleichen Format.

----

ARBEITSAUFTRAG M2 – Blender-Renderer

Scope: Headless-Blender-Renderer als Plugin (renderer_id="blender"), der ausschließlich M1-Artefakte liest. CLI: weight-atlas render OUT_DIR --renderer blender. Keine neuen pip-Deps; Blender bleibt externe Binary.

Festlegungen:

- Binary-Auflösung: Env-Var WEIGHT_ATLAS_BLENDER, Fallback shutil.which("blender"); fehlt beides → klare Fehlermeldung mit Install-Hinweis. Wrapper läuft via subprocess.run, headless (-b … -P … --).
- Engine: BLENDER_WORKBENCH (headless-sicher, deterministisch, kein GPU/EGL-Bedarf). Cycles nur als dokumentierte Beauty-Option im Backlog.
- Datenfluss: Wrapper liest field_height_{raw,smooth}.tif + field_tint_smooth.tif via tifffile → schreibt .npy in tempdir → bpy-Skript baut daraus das Mesh. Kein Image-Texture-Roundtrip.
- Mesh: Grid 1024² (Spec-Wert), Vertex-Z aus Height (skaliert), Vertex-Color aus Tint; Mesh-Erzeugung per foreach_set (nicht from_pydata, zu langsam). Workbench-Shading color_type='VERTEX', Licht STUDIO mit fixer light_rotation aus Spec (az 315°/alt 45°).
- Kamera: orthografisch, exakt von oben, ortho_scale fix; Auflösung 2048², World-Farbe fix (0.05 Grau).
- Spec-Erweiterung: atlas_spec.v1.json bekommt Block "blender": {"grid": 1024, "resolution": 2048, "z_scale": <dokumentierter Wert>}; spec_version bleibt 1 (Pre-Release), Erweiterung in ARCHITECTURE.md notieren.
- OBJ-Export: Wrapper schreibt terrain.obj selbst (plain text, aus numpy, 256²-Downsample des smooth-Fields) – deterministisch ohne bpy-ops. Das Mesh wird damit zum diffbaren Fingerprint-Artefakt; den Backlog-Punkt „Mesh export“ entsprechend abhaken.
- Outputs: terrain_raw.png, terrain_smooth.png, terrain.obj; alle in manifest.json mit sha256.
- Contours in Blender: nicht in M2 – die 2D-Sheets tragen Isolinien; Blender-Contours → Backlog.

Tests/CI:

- CI hat kein Blender → Wrapper-Tests: Dry-Run (Kommando-Konstruktion korrekt), Fehlerpfad ohne Binary, OBJ-Writer-Test (byte-identisch, Handwerte für kleine Grids).
- Lokaler Smoke: scripts/smoke_blender.sh (scan Fixture → render blender → sha-Vergleich Zweitlauf). In README dokumentieren.
- Determinismus-DoD lokal: zwei Renders → byte-identische PNGs (Workbench muss das liefern; falls nicht: Pixel-Max-Diff 0 plus Ursachenanalyse, nicht einfach SSIM akzeptieren).

DoD M2:

- render --renderer blender erzeugt PNGs + OBJ aus M1-Artefakten, ohne Gewichte anzufassen.
- Zweitlauf byte-identisch (lokal verifiziert, im Report belegt).
- CI grün ohne Blender; Fehlerpfad getestet.
- ARCHITECTURE.md: Workbench-Entscheidung, z_scale, OBJ-Konvention (256²) begründet.
- Keine neuen Dependencies; keine M3-Anfänge; Neues → BACKLOG.md.

Report-Format: wie gehabt (Umgesetzt / Abweichungen / Offene Fragen / Backlog-Delta), dann Stopp bis M3-Freigabe.

Kleiner Ausblick, damit du weißt, worauf ich beim M2-Review schaue: Die zwei Stellen, an denen Determinismus typisch bricht, sind Workbench-Lichtrotation und PNG-Encoder-Metadaten – beides muss im Report explizit adressiert sein. Danach kommt M3 (UI), und mit M3 hast du ein Werkzeug, das du Kollegen zeigen kannst.

----

Review M1.5 / M2 / M3 – akzeptiert

M1.5: Alle vier Entscheidungen korrekt umgesetzt. Zur einen bewussten Abweichung (Contour-Levels über raw statt skaliertem Feld): akzeptiert, mit Kommentar. Beide Varianten sind geometrisch konsistent, weil log1p monoton ist – die Isolinien bleiben Isolinien desselben Felds, nur die Staffelung ändert sich (raw betont die Gipfel, log1p staffelt gleichmäßiger). Die Wahl ist dokumentiert, damit ist sie eine Kartografie-Konvention und kein Bug. Beim nächsten Spec-Bump (v2) entscheiden wir bewusst, nicht nebenbei.

M2: Die zwei von mir vorhergesagten Determinismus-Risiken (Lichtrotation, PNG-Encoder) sind adressiert: Rotation fix aus der Spec, Byte-Identität empirisch per Smoke-Test belegt statt behauptet. Wrapper-Testtiefe (Dry-Run, Fehlerpfad, OBJ-Handwerte) ist genau das, was CI ohne Blender braucht. OBJ als diffbares Artefakt ist damit aus dem Backlog und steht für spätere Mesh-Distanzen zur Verfügung.

M3: App-Factory mit injizierbaren Pfaden = testbar ohne Side-Effects; SQLite + In-Prozess-Worker = richtige MVP-Wahl; HTMX-CDN hält das „kein Node“-Versprechen. Die zwei neuen Deps liegen korrekt im web-Extra, Core bleibt sauber.

Lob, explizit: Report-Disziplin über fünf Batches, keine Scope-Drift, Backlog ehrlich geführt, jede Abweichung mit Rationale. So lasse ich mir von einem Coding-Agenten zuarbeiten.

----

Fixup M3.5 (klein, ein Commit-Block)

1. Terrain-Varianten prüfen: Der Report nennt nur „renders to PNG“ – die Spec verlangte terrain_raw.png und terrain_smooth.png, beide im Manifest. Falls nur smooth existiert: raw ergänzen + Test erweitern. Falls beide existieren: im Report nachreichen, kein Code-Change.

2. Doku-Satz: README erhält einen Satz: UI ist ein Lokalttool ohne Auth/Validation-Härtung – bewusst so entschieden. (Falls jemand das UI ins Netz hängt, ist das sein Problem; wir dokumentieren es.)

3. Backlog-Eintrag: „HTMX vendoren (offline-fähige UI)“ – optional, CDN bleibt Default.

----

Nächste Freigabe: M4 – Vergleichs-/Δ-Ebene

Damit landet die Funktion im Tool, die dein Ursprungsziel war (Modell-Signaturen vergleichen, Abliteration sichtbar machen). Scope-Vorschlag:

- CLI: weight-atlas compare DIR_A DIR_B --out DIR – liest nur Artefakte (Regel bleibt), hard-reject bei spec_version/tool_version-Mismatch mit klarer Fehlermeldung.

- Zwei Modi: strict (identische Raster-Shape – der Abliteration-Fall: gleiche Architektur, vor/nach der Operation) und aligned (Layer-Achse auf Tiefe normalisiert, gemeinsames Grid – der Cross-Architektur-Fall 27B vs. 35B-A3B). Die Resampling-Konvention ist eine Projektionsentscheidung → Spec v2 + ARCHITECTURE.md.

- Outputs: Δ-Sheet je Kanal (divergierende Blau/Rot-Tönung, symmetrisch um 0, fixe Quantil-Limits), quantitativer Summary (pro Layer: Δ-L2, Kosinus, max|Δ| + argmax-Region) als JSON, alles im Manifest.

- UI: Compare-Seite (zwei Modelle wählen → Job → Δ-Sheet + Tabelle).

- Tests: Fixture + mutierte Variante mit injiziertem Δ in bekannten Tensoren – compare muss exakt diese (Layer, Slot)-Regionen als Hotspots melden; plus Determinismus.

Empfohlene Backlog-Reihenfolge danach: M5 GGUF-Loader (deine Original-Anforderung *.gguf), M6 MoE-Experten-Panel, M7 UMAP-Embedding-Sheet, M8 Aktivitäts-/„fMRI“-Modus.
Sag „OK“, dann schreibe ich den M4-Arbeitsauftrag in derselben Form wie die bisherigen; der Fixup M3.5 kann parallel sofort laufen.

----

ARBEITSAUFTRAG weight-atlas · M4 – Vergleichs-/Δ-Ebene

Reihenfolge: Zuerst M3.5 abschließen (Terrain raw+smooth im Manifest prüfen/ergänzen; README-Satz „Lokaltool ohne Auth“; Backlog-Eintrag HTMX-Vendoring). Danach M4.

Ziel: Zwei gescannte Modelle quantitativ und kartografisch vergleichen – der Abliteration-Fall (gleiche Architektur, strict) und der Cross-Architektur-Fall (aligned). Regel bleibt: compare liest ausschließlich Artefakte, nie Gewichte.

1. Festlegungen & Projektionsentscheidungen (in ARCHITECTURE.md begründen)

- Δ-Felder werden über skalierten Kanalwerten gerechnet (B − A). Begründung: height/log1p-Differenz ≈ log-Ratio der Spektralnormen (modellvergleichbar); tint ist rang-normalisiert → Δtint = Rangverschiebung in [−1,1]. Dokumentieren.
- Quantitative Summary-Metriken werden über rohen Stats gerechnet (rel. L2, Kosinus), weil dort die absolute Änderungsstärke (Abliteration: „wie stark wurde operiert?“) steckt. Beide Ebenen bewusst getrennt, dokumentieren.
- Hard-Reject bei spec_version-Mismatch (klare Fehlermeldung). tool_version-Mismatch → Warnung, Lauf erlaubt, beide Versionen im Report. Das verfeinert die M1.5-Entscheidung; Begründung: tool_version darf Vergleiche nicht blockieren, solange Scan-Semantik (spec_version) identisch ist.
- aligned-Modus = Projektionskonvention, keine Entdeckung: Layer-Achse wird auf Tiefe t∈[0,1] normalisiert und auf gemeinsames Grid resampelt (linear, NaN-aware über Valid-Masken). Diese Konvention steht maschinenlesbar im Compare-Report.
- Spec-Erweiterung (additiv, spec_version bleibt 1, Scan-Output unverändert): specs/atlas_spec.v1.json bekommt Block "compare": {"grid_rows": 64, "interp": "linear", "diverging_clip": 0.98}. In ARCHITECTURE.md notieren, warum kein Spec-Bump.

2. Module

Neues Paket compare/:

- align.py: align_fields(fieldA, fieldB, mode, spec) -> AlignedPair (je Variante: skaliert + roh, resampelt; strict: Identity mit Shape-Check, Fehler ValueError mit Hinweis „--mode aligned“ bei Mismatch; aligned: grid_rows Zeilen, Row-Labels = t-Werte 0.00…1.00).
- delta.py: compute_delta(pair) -> DeltaField (signed float32, raw + smooth wie bestehende Konvention) und compute_summary(pair, delta) -> dict (Schema unten).
- render/delta_sheet.py: Renderer Registry-ID delta: divergierende Blau→Weiß→Rot-Colormap (fix), Limits symmetrisch ±q(diverging_clip) von |Δ| je Kanal, Colorbar, Achsenlabels, NaN = grau schraffiert; PNG-Metadaten wie bestehende Sheet-Konvention (Creation-Time fix). Zusätzlich delta_profile_<channel>.png: 1×L-Strip der pro-Layer rel. L2 (hot-Colormap) – die „Abliteration-Leiste“ aus dem Ursprungskonzept.
- Compare schreibt eigene Artefakte in --out: delta_<channel>_{raw,smooth}.tif, delta_sheet_<channel>.png, delta_profile_<channel>.png, compare_summary.json, manifest.json (alle sha256).

compare_summary.json-Schema:


{  "mode": "strict|aligned",  "spec_version": 1,  "tool_versions": {"a": "...", "b": "..."},  "models": {"a": {"id": "...", "path": "..."}, "b": {"...": "..."}},  "projection": {"interp": "linear", "grid_rows": 64},  "channels": {    "height": {      "global": {"l2_rel": 0.0, "cosine": 0.0, "max_abs_delta": 0.0,                 "argmax": {"layer": 12, "slot": "mlp_down"}},      "per_layer": [{"row": 0, "label": "0|0.00", "l2_rel": 0.0,                     "cosine": 0.0, "max_abs": 0.0}]    }  },  "hotspots": [{"channel": "height", "layer": 12, "slot": "mlp_down",                "delta_scaled": 0.0, "raw_a": 0.0, "raw_b": 0.0}]}

hotspots = Top-10 nach |Δ_scaled| je Kanal, mit Raw-Werten annotiert.

3. CLI

weight-atlas compare DIR_A DIR_B --out DIR --mode {strict,aligned} (Default strict). Strict-Shape-Mismatch → Fehler mit Mode-Hinweis. Render-Schritt ist in compare integriert (Registry-Renderer delta wird aufgerufen), render-Kommando bleibt unverändert.

4. UI

- GET /compare: Form mit zwei Modell-Selects (aus bestehender Modellliste) + Mode-Select; POST /api/compare → Job (JobQueue-Typ compare, wiederverwenden).
- GET /compares/{id}: Report-Seite – Δ-Sheets, Profile-Strips, Hotspot-Tabelle, Per-Layer-Tabelle, Projektions-/Versionsblock (inkl. tool_version-Warnung, falls vorhanden).
- Templates compare.html, compare_report.html im bestehenden Stil; keine neuen Frontend-Deps.

5. Tests (erwartet ~20 neue)

- Lokalisierungs-Test (Kern): Fixture A + mutiertes B (layers.2.mlp.down_proj ×1.5, Rang-1-Perturbation auf layers.3.self_attn.o_proj); scan beide; compare --mode strict → Hotspots-Rangliste muss exakt (2, mlp_down) und (3, attn_o) als Top-2 des height-Kanals melden; argmax == (2, mlp_down).
- Strict-Mismatch → sauberer Fehler mit Mode-Hinweis.
- Aligned: Fixtures 4 vs. 7 Layer → läuft; Output-Zeilen == grid_rows; Row-Labels = t-Werte; Per-Layer-Metriken NaN-frei wo Valid-Paare.
- spec_version-Mismatch → Hard-Reject; tool_version-Mismatch → Warnung + Lauf (caplog-Assert).
- Δ-Sheet + Profile deterministisch byte-identisch (Zweitlauf).
- API: Compare-Job via Endpoint, Report-Seite rendert Hotspot-Tabelle.

6. Doku

- ARCHITECTURE.md: Compare-Sektion (Δ-auf-skaliert vs. Metriken-auf-roh, aligned als Konvention, spec/tool-version-Regel, NaN-Strategie).
- README: Quickstart „Abliteration prüfen“: scan vorher/nachher → compare --mode strict → Hotspots + Profile-Strip lesen; plus Cross-Architektur-Beispiel --mode aligned.

Definition of Done

- M3.5 abgeschlossen (oder im Report als bereits erledigt bestätigt).
- compare strict + aligned laufen auf Fixtures; Lokalisierungs-Test grün.
- Hard-Reject/Warn-Verhalten getestet; Zweitlauf byte-identisch.
- UI-Flow „zwei Modelle wählen → Job → Report-Seite“ ohne Handarbeit.
- pytest -q, ruff, mypy grün; keine neuen Dependencies; keine M5-Anfänge; Neues → BACKLOG.md.
- Commits conventional, CI grün je Commit.

Report-Format: wie gehabt (Umgesetzt / Abweichungen / Offene Fragen / Backlog-Delta), dann Stopp bis M5-Freigabe (nächster Kandidat: GGUF-Loader).
Bitte schreiben nach docs/reports/2026-08-06_m3.5+m4+m5.md

Sobald der Report kommt, reviewe ich vor allem den Lokalisierungs-Test (steht und fällt mit ihm der analytische Wert des Tools) und die Sauberkeit der aligned-Projektion. Danach wäre das Tool bereit für deine beiden Original-Modellvergleiche – echte Qwen-Scans kannst du parallel schon mal anstoßen, die Artefakte sind spec-v1-stabil.

----

Review M4 – akzeptiert, mit Bestätigungsvorbehalt

Stark: Lokalisierungs-Test grün mit exakt der geforderten Top-2-Rangliste – damit ist der analytische Kern bewiesen. 113 Tests, Lint/Typen clean, Read-only-Regel durchgehalten, Spec-Additivität wie festgelegt.

Abweichungen, akzeptiert (mit Doku-Pflicht, falls noch nicht geschehen):

- compare/render/ statt render/: kohäsionsgetrieben, ok – aber in ARCHITECTURE.md als Platzierungsentscheidung notieren; mit dem Backlog-Punkt „Entry-Point-Registration“ wird das später ohnehin vereinheitlicht.
- Spec-Keys umbenannt (aligned_grid, colormap, default_mode statt meiner Vorschläge): inhaltlich gleichwertig; Schema muss in ARCHITECTURE.md stehen.
- Route /compare/{id} statt /compares/{id}: kosmetisch, geschenkt.

Bestätigungsliste für den nächsten Report (je Punkt: „vorhanden + Verweis“ ODER Fixup-Commit):

1. M3.5 erledigt? (terrain raw+smooth im Manifest; README-Lokaltool-Satz; Backlog HTMX-Vendoring) – der M4-Report erwähnt M3.5 mit keinem Wort, und die Reihenfolge-Regel war explizit.
2. delta_profile_<channel>.png (die 1×L-Abliteration-Leiste) existiert und liegt im Manifest.
3. Divergierende Limits symmetrisch ±q(diverging_clip) je Kanal, nicht global.
4. Strict-Mismatch-Fehlertext enthält den --mode aligned-Hinweis (Test vorhanden).
5. Aligned-Row-Labels = t-Werte; tool_version-Mismatch → Warnung + Lauf (caplog-Test).
6. Δ-Sheet/Profile byte-deterministisch (Zweitlauf-Test); README enthält den Abliteration-Quickstart und das Cross-Architektur-Beispiel.

Falls alles vorhanden: nur abhaken, kein Prosa-Report.

----

Freigabe M5 – GGUF-Loader

Damit kann das Tool beide Original-Formate lesen. Besonderheit, die den Umfang bestimmt: GGUF-Namen sind anders (blk.0.attn_q.weight vs. HF-Namen) und Quantisierung ist Teil der Signatur – beides designen wir bewusst.

ARBEITSAUFTRAG weight-atlas · M5 – GGUF-Loader

Scope: *.gguf lesen, dequantisieren, durch die bestehende Pipeline jagen; Quantisierung als dokumentierte Signatur behandeln; Cross-Loader-Vergleiche erlauben, aber warnen.

Festlegungen:

- Dependency: offizielles gguf-Paket, ausschließlich im Extra gguf (jetzt befüllen); CI installiert das Extra, Tests laufen dort ohne Skip.
- Loader loaders/gguf_loader.py, Registry-ID gguf: GGUFReader (mmap), lazy TensorHandles; Metadaten (general.architecture, block count) in den Fingerprint-Top-Level-Block model.
- Auto-Detect: detect_loader(path) via Magic-Bytes (GGUF vs. safetensors-Headerlänge); scan nutzt Auto-Detect, --loader override optional.
- Dequant loaders/gguf_dequant.py, kanonisch float32: M5-Scope F32, F16, BF16, Q8_0, Q4_0; andere Typen → klare Fehlermeldung mit Typname + Backlog-Eintrag „volle k-quant-Unterstützung“. BF16 via Bit-Shift (uint16→uint32-View), Q8_0/Q4_0 blockweise (32er-Blöcke, f16-Scale).
- name_map: zweite, loader-unabhängige Regelmenge für GGUF-Namen: blk.(\d+).attn_q/k/v → attn_q/k/v, attn_output → attn_o, ffn_gate/up/down → mlp_gate/up/down, attn_norm → norm_attn, ffn_norm → norm_mlp, token_embd → embed, output → lm_head, output_norm → other-bzw. norm-Slot nach Entscheidung (dokumentieren). Gleiche Slot-IDs wie safetensors – das Raster bleibt loader-übergreifend vergleichbar.
- Fingerprint-Erweiterung: pro Tensor ggml_type; Top-Level quantization: {Typ: Anzahl}; loader: "gguf".
- Compare-Erweiterung: compare_summary.json bekommt loaders + quantization beider Seiten; Loader-Mismatch ODER Quantisierungs-Mismatch → Warnung (CLI-Log + UI-Banner auf der Report-Seite), kein Reject. Begründung in ARCHITECTURE.md: Quantisierungsrauschen ist realer Teil der Signatur, aber für Abliteration müssen beide Seiten identisch quantisiert sein – README-Satz.
- UI: Detailseite zeigt ggml_type-Verteilung (kleine Tabelle), sonst nichts Neues.

Tests (~15 neu):

- Dequant-Handwerte: selbstgebaute Q8_0- und Q4_0-Byte-Blöcke → exakt erwartete float32-Werte; BF16-Known-Values.
- Fixture via GGUFWriter: winziges Modell in F32; Konsistenz-Test: dieselben Gewichte als safetensors(F32) und GGUF(F32) gescannt → Stats identisch (≤1e-6); Name-Mapping-Test gguf→Slots.
- Q8_0-Fixture: Scan läuft, ggml_type im Fingerprint, Stats weichen messbar (aber klein) vom F32-Scan ab.
- Compare F32-Scan vs. Q8_0-Scan → Warnung, Lauf erlaubt; caplog-Assert.
- Unbekannter Quant-Typ → Fehlermeldung mit Typname (Fixture: rohe Bytes manipulieren oder Writer-Mock).
- Determinismus: gguf-Scan Zweitlauf byte-identisch.

Doku: ARCHITECTURE.md (Quantisierung als Feature, Dequant-Kanon, Cross-Loader-Warnregel, GGUF-Name-Mapping); README (GGUF-Quickstart + „Quantisierung für Δ-Vergleiche gleich halten“).

DoD:

- Bestätigungsliste aus §1 abgearbeitet (abhaken oder Fixup).
- scan model.gguf → Sheet + Terrain ohne Handarbeit; Auto-Detect getestet.
- Konsistenz-Test safetensors≡gguf(F32) grün; alle neuen Tests grün; 128+ Tests gesamt.
- ruff, mypy clean; keine Deps außerhalb des gguf-Extras.
- Commits conventional, CI grün; Neues → BACKLOG.md; dann Stopp bis M6.

Report-Format: wie gehabt, zusätzlich erster Block: Bestätigungsliste §1.

Nach M5 ist das Tool bereit für deine Ursprungsszenarien mit echten Dateien (safetensors und gguf). M6-Kandidat bleibt das MoE-Experten-Panel – das braucht ihr für die 35B-A3B-Signatur, also voraussichtlich der nächste Auftrag nach M5.

----

Review zuerst, dann M6. Inhaltlich ist M5 sauber – aber der Report lässt erneut angeforderte Blöcke aus, und zwei spezifizierte Sub-Items fehlen offenbar ganz. Das wird als Fixup-Batch M5.5 vorgezogen, bevor M6 reviewed wird.

Review M5 – akzeptiert, Fixup M5.5 erforderlich

Stark: Der Konsistenz-Test safetensors ≡ gguf(F32) ist der Beweis, dass die Loader-übergreifende Signatur steht; Dequant gegen Handwerte, BF16-Bit-Shift, Extra-Disziplin (gguf bleibt Extra), Auto-Detect via Magic-Bytes – alles wie beauftragt. 138 Tests, Lint/Typen clean.

M5.5 – jetzt abschließen (Reihenfolge-Regel, diesmal Gate für das M6-Review):

1. Bestätigungsliste §1 aus dem M4-Review (M3.5-Status, delta_profile-Strips im Manifest, Diverging-Limits je Kanal, Strict-Fehlertext mit Mode-Hinweis, aligned-t-Labels + tool_version-Warn-Test, README-Abliteration-Quickstart + Cross-Architektur-Beispiel). Zweimal vertagt – im M6-Report ist sie der erste Block, je Punkt „vorhanden + Verweis“ oder Fixup-Commit.

2. UI-Banner auf der Compare-Report-Seite bei Loader-/Quantisierungs-Mismatch war spezifiziert (compare_report.html ist in M5 unangetastet) → nachrüsten.

3. Tests nachliefern: Q8_0-Scan (ggml_type im Fingerprint, Stats messbar ≠ F32) und Compare F32 vs. Q8_0 → Warnung via caplog. Falls der „support check“-Test keine Fehlermeldung mit Typname für unsupported quants assertet: ergänzen.

4. Backlog-Eintrag „volle k-quant-Unterstützung (Q2_K…Q6_K, IQ)“ – im Report nicht genannt; prüfen/anlegen.

5. README-Satz „Quantisierung für Δ-Vergleiche gleich halten“ – prüfen/anlegen.

----

Freigabe M6 – MoE-Experten-Panel

Damit wird die 35B-A3B-Signatur endlich first-class: Das granulare „Hochgebirge“ aus deinem Ursprungsbild ist nichts anderes als das Experten-Panel als Terrain.

ARBEITSAUFTRAG weight-atlas · M6 – MoE-Experten-Panel

Scope: Experten-Tensoren erkennen, splitten, als eigene Panel-Felder (Layer × Experte) durch die bestehende Pipeline jagen; Main-Raster unverändert lassen; Compare um Panel-Δ erweitern.

Festlegungen (in ARCHITECTURE.md begründen):

- Main-Raster-Slots bleiben unverändert (kein Spec-Bump, spec_version 1). MoE-Mapping: router-Slot existiert bereits; Shared-Expert-Tensoren belegen die mlp-Slots (dokumentieren: MoE-mlp-Spalte = Shared Expert); Experten fließen nicht ins Main-Raster.
- Experten-Panel = eigene Feldklasse: je mlp-Slot (gate/up/down) eine Matrix Layer × Experte-IDs, gleiche Kanal-Definitionen (height/tint/rough) und Skalierungs-Konventionen wie das Main-Raster. Additiv → kein Cross-Spec-Problem.
- Name-Regeln, Reihenfolge kritisch: HF mlp.gate.weight → router vor mlp.gate_proj → mlp_gate (Regel-Reihenfolge testen!); mlp.experts.{e}.(gate|up|down)_proj → Slot + expert_id; shared_expert.* → mlp-Slots; shared_expert_gate → other. GGUF: ffn_gate_inp → router; ffn_(gate|up|down)_exps (3D-stacked) → per-Expert-Handles; ffn_*_shexp → mlp-Slots.
- Stacked 3D-Tensoren: Loader erzeugt lazy Sub-Handles (…ffn_gate_exps[7]), load() liefert den 2D-Slice; TensorHandle bekommt expert_id: int | None.
- Fingerprint: model-Block bekommt moe: {num_experts, shared_expert} – bei GGUF aus Metadaten (*.expert_count), bei safetensors aus Tensor-Präsenz abgeleitet (beide Wege dokumentieren).
- Artefakte: field_expert_<slot>_{raw,smooth}.tif, Sheets via bestehendem sheet-Renderer (er ist generisch über Field2D – kein neuer Renderer!), Manifest-Einträge.
- Blender: Wrapper bekommt --field <name> (Default height), damit Terrain direkt aus expert_mlp_down renderbar ist – das MoE-Hochgebirge.
- Compare: Panel-Felder werden verglichen, wenn beidseitig vorhanden und shape-gleich (strict); sonst Summary-Eintrag "expert_panels": {"status": "skipped", "reason": "…"} – kein Crash. Δ-Panel-Sheets via bestehendem delta-Renderer.

CLI: keine neuen Kommandos; scan/render/compare verarbeiten Panels automatisch; render --renderer blender --field expert_mlp_down funktioniert.

UI: Detailseite: MoE-Badge (num_experts/shared), Panel-Sheet-Grid; Compare-Report: Δ-Panel-Sektion nur wenn verglichen, sonst Skip-Note sichtbar.

Tests (~18 neu):

- Fixtures: HF-MoE klein (4 Layer, 8 Experts, shared) + GGUF-MoE mit 3D-exps-Tensoren via Writer.
- Regel-Reihenfolge gate.weight vs. gate_proj; Split-Slice-Test (Handle[3] == np-Slice[3], lazy).
- Rasterizer: Main-Raster-mlp-Slots nur aus Shared Expert gefüllt; Panel-Shape (4×8); NaN wo Expert fehlt.
- Sheet + Δ-Panel deterministisch byte-identisch.
- Lokalisierungs-Test MoE: Perturbation in Expert (2, 5, down) → strict-compare-Hotspots melden exakt diese Panel-Zelle.
- Expert-Count-Mismatch (8 vs. 16) → skipped-Reason im Summary, Lauf grün.
- Blender-Wrapper --field Dry-Run-Command-Test.

Doku: ARCHITECTURE.md (MoE-Konventionen, Shared-Expert-Entscheidung, 3D-Split, Panel als eigene Feldklasse); README (MoE-Quickstart mit „35B-A3B vs. 27B“-Szenario: aligned-Main-Raster + Panel-Only-bei-gleicher-E).

DoD:

- M5.5 + Bestätigungsliste §1 vollständig abgearbeitet (erster Report-Block!).
- MoE-Scan → Panels + Sheets + Terrain ohne Handarbeit.
- MoE-Lokalisierungs-Test grün; Mismatch-Pfad grün; 156+ Tests gesamt.
- ruff/mypy clean; keine neuen Dependencies; Commits conventional; CI grün.
- Neues → BACKLOG.md; dann Stopp bis M7.

Report-Format: Block 1: Bestätigungsliste §1 + M5.5; Block 2: M6 wie gehabt.

Nach M6 kannst du deine beiden Ursprungsmodelle (dense 27B vs. MoE 35B-A3B) vollständig kartieren: aligned-Main-Raster für den Architektur-Vergleich, Experten-Panel-Terrain für die MoE-Signatur. M7-Kandidat: UMAP-Embedding-Sheet; M8: der „fMRI“-Modus.

----

Zwei Blöcke, zwei unterschiedliche Urteile: M5.5 ist vorbildlich abgeschlossen – M6 ist nur eine halbe Lieferung. Details und die nächsten Aufträge unten.

Review M5.5 – akzeptiert, ohne Vorbehalt

Exakt das geforderte Format: je Punkt Status + Verweis, zusätzliche Fixes transparent gelistet, §1-Liste vollständig beantwortet. Dass M3.5 seinerzeit in einem eigenen Report-File (2026-08-07_m3.5+m4.md) lief, erklärt die frühere Lücke – akzeptiert, solange der Verweis steht. Damit ist die Prozess-Seite endlich so, wie ich sie will.

----

Review M6 – teilweise akzeptiert; Fixup-Batch M6.5

Gut: ExpertPanel als eigene Feldklasse, Main-Raster unverändert (kein Spec-Bump), Shared-Expert-Entscheidung, Regel-Reihenfolge router-vor-gate_proj, MoE-Badge, --field für Blender, Doku. Die HF-MoE-Seite steht.

Aber der Report bleibt hinter dem Auftrag zurück – drei Lücken:

1. GGUF-3D-Split fehlt. Kein gguf_loader.py in der Modified-Liste: gestackte ffn_*_exps-Tensoren werden nicht in lazy Sub-Handles gesplittet, ein GGUF-MoE-Fixture fehlt. Genau das ist aber die reale Welt – die meisten 35B-A3B-Dateien da draußen sind GGUFs mit 3D-exps-Tensoren.
2. Compare-Panel-Integration fehlt. Die Skip-Regel steht als „Design Decision“ im Report, aber weder compare/* noch compare_report.html sind angetastet. Entscheidungen ohne Code sind Backlog-Einträge, keine Implementierung.
3. Test-Coverage unter Spec: 12 neue Tests statt ~18; DoD „156+“ mit 154 verfehlt. Offenbar fehlen u. a. MoE-Lokalisierungs-Test, Mismatch-Skip-Test, --field-Dry-Run – der Lokalisierungs-Test war aber Kern des Auftrags.

----

ARBEITSAUFTRAG M6.5 (Gate für M7-Review)

1. GGUF-MoE: gguf_loader.py splittet 3D-exps-Tensoren in lazy Sub-Handles (…exps[7], load() = 2D-Slice, expert_id gesetzt); ffn_*_shexp → mlp-Slots, ffn_gate_inp → router (Regeln testen). Fixture: GGUF-MoE via Writer mit 3D-Tensoren; Split-Slice-Test (Handle[3] == np-Slice[3]).
2. Compare-Panels: align/delta verarbeiten ExpertPanel-Felder (shape-gleich → Δ-Panel-Sheets + Summary-Einträge; sonst "expert_panels": {"status": "skipped", "reason": …}); CLI-compare discovert Panels via Manifest; compare_report.html bekommt Δ-Panel-Sektion und sichtbare Skip-Note.
3. Tests nachziehen: MoE-Lokalisierung (Perturbation in Expert (2,5,down) → Panel-Hotspot Top-1), Expert-Count-Mismatch (8 vs. 16 → skipped-Reason, Lauf grün), Blender---field-Dry-Run-Command, Panel-Sheet-Determinismus, Regel-Reihenfolge (falls nicht vorhanden). Ziel: 156+ Tests.
4. Doku: ARCHITECTURE.md um GGUF-3D-Split + Compare-Panel-Konvention ergänzen.

Freigabe M7 – UMAP-/Embedding-Sheet

Die semantische Landkarte aus dem Ursprungskonzept. Wichtigste Design-Regel vorab: Die Projektion wird in der Scan-Phase gerechnet und als Artefakt gespeichert – Render und UI lesen wie immer nur Artefakte.

---

ARBEITSAUFTRAG weight-atlas · M7 – Embedding-Sheet

Scope: Deterministische PCA-Projektion der Token-Embeddings (Core) + optionale UMAP-Projektion (Extra umap), daraus Dichtefelder → bestehende Sheet-/Blender-Pipeline. Kein Cross-Modell-Embedding-Vergleich in M7 (Alignment ist ehrlich schwer) → Backlog.

Festlegungen:

- PCA (Core, dep-frei): Embedding-Tensor je Loader (embed_tokens / token_embd, Slot embed existiert), zentrieren, top-3 Komponenten via randomisierte SVD (bestehende Machinery, Seed aus Spec). Sign-Konvention fix: größte |Ladung| je Komponente positiv – sonst ist PCA vorzeichen-ambig und der Fingerprint nicht stabil. Dokumentieren + Test (Input gespiegelt → identischer Output).
- Artefakte: embedding_pca.npy (V×3 float32) + embedding_meta.json (explained variance, Konventionen, Lib-Versionen); Dichtefeld via 2D-Histogramm über quantil-normalisierte Koordinaten, Grid 256, deterministic → field_embed_density_{raw,smooth}.tif + Manifest-Eintrag, damit Discovery/Render/Detailseite automatisch funktionieren.
- UMAP (Extra umap, jetzt befüllen): spec.embedding.method: "pca"|"umap"; UMAP mit random_state aus Spec, init aus PCA; Ergebnis embedding_umap.npy. CI testet nur PCA-Pfad; UMAP-Tests skipif ohne Extra, lokal dokumentiert. UMAP-Version in embedding_meta.json; Compare/Detailseite zeigt Version (Versions-Unterschiede können Koordinaten ändern – transparent machen).
- Spec-Erweiterung (additiv): "embedding": {"method": "pca", "grid": 256, "components": 3, "subsample_scatter": 5000, "seeds": {"pca": 0, "umap": 0}}.
- Renderer: kein neuer – Dichtefeld läuft durch sheet (log1p) und optional blender --field embed_density („semantisches Gelände“). Scatter-Overlay: subsample 5000 Punkte, fixed Seed, s=0.3, alpha fix, auf dem Sheet.
- UI: Detailseite: Embedding-Sektion (Sheet + explained variance + Methode/Version).
- Backlog-Eintrag: „Embedding-Alignment cross-model (Procrustes, shared-vocab)“.

Tests (~12 neu): Fixture-Embedding mit gepflanzten Gauss-Clustern → PCA-Top-2 trennt Cluster (Dichte-argmax-Bins je Cluster korrekt); Histogramm-Summe == Vocab-Größe; Sign-Konvention-Test; Sheet + Dichtefeld byte-deterministisch; Manifest-Discovery des neuen Felds; Detailseite rendert Embedding-Sektion (API-Test).

Doku: ARCHITECTURE.md (PCA-Sign-Konvention, Dichte-als-Feld-Trick, UMAP-Versions-Transparenz, warum kein Cross-Modell-Vergleich in M7); README (Embedding-Quickstart).

DoD:

- M6.5 vollständig (erster Report-Block, je Punkt Status+Verweis).
- scan → Embedding-Sheet + Terrain ohne Handarbeit; PCA deterministisch über Zweitlauf.
- 168+ Tests; ruff/mypy clean; neue Dep nur im umap-Extra.
- Commits conventional, CI grün; Neues → BACKLOG.md; Stopp bis M8.

Report-Format: Block 1: M6.5-Checkliste; Block 2: M7 wie gehabt.

Stand danach: Von der Ursprungsvision fehlt nur noch M8 (der „fMRI“-Aktivitätsmodus via NNsight) – das größte und forschungsnächste Stück. M8 würde ich anders schneiden als M0–M7 (Protokoll-Design vor Code); dazu beim nächsten Review ein eigener Entwurf.

----

Zwei Reports, zwei saubere Lieferungen mit kleinen Resten – und jetzt das große letzte Stück. Reihenfolge: Review M6.5, Review M7 mit Fixup M7.5, dann der M8-Entwurf, wie versprochen protokoll-zuerst.

Review M6.5 – akzeptiert

Alle vier Punkte mit Verweis, 161 Tests, Lokalisierungs- und Mismatch-Tests da, compare/panel.py als sauberer eigener Schnitt. Eine Mikro-Bestätigung bleibt offen und wandert in die M7.5-Liste: Exists-Verweis für den Blender---field-Dry-Run-Test (war in M6.5 spezifiziert, im Report nicht genannt).

Review M7 – akzeptiert, Fixup M7.5 erforderlich

Stark: Sign-Konvention als Fingerprint-Stabilisator verstanden und umgesetzt, Dichtefeld-Trick (Punktwolke → Raster → bestehende Pipeline) exakt wie beauftragt, embedding_meta.json mit Versions-Transparenz, 175 Tests.

M7.5 (Gate für M8-Review):

1. umap-Extra ist nicht befüllt – pyproject.toml unangetastet; weight-atlas[umap] installiert aktuell nichts. Nachholen (umap-learn).
2. Scatter-Overlay auf dem Embedding-Sheet (subsample 5000, fixed Seed, s=0.3, alpha fix) war spezifiziert, im Report nicht genannt → umsetzen oder im Report als bewussten Drop begründen (ich will es haben).
3. Ruff ist nicht clean („minor warnings“) – die DoD sagte clean. ruff --fix, fertig.
4. Backlog-Eintrag „Embedding-Alignment (Procrustes, shared-vocab)“ – im Report nicht genannt; prüfen/anlegen.
5. Mikro aus §1: --field-Dry-Run-Test-Verweis nachreichen.

----

Freigabe M8 – „fMRI“-Aktivitätsmodus

Anders als M0–M7 beginnt M8 nicht mit Code, sondern mit dem Messprotokoll. Der Grund ist derselbe wie in der echten Bildgebung: Aktivitätsdaten sind nur innerhalb von Protokoll + „Scanner“ vergleichbar. Wir behandeln Device/Dtype/Torch-Version als Scanner-Parameter und das Stimulus-Set als eingefrorenes Protokoll – die fMRI-Analogie, wörtlich genommen.

ARBEITSAUFTRAG weight-atlas · M8 – Aktivitätsmodus („fMRI“)

Scope: Echte Forward-Passes (nichts simuliert) über ein versioniertes Stimulus-Protokoll; Layer×Position-Aktivitätsfelder und (MoE) Layer×Expert-Usage-Felder als Artefakte; Δ zwischen zwei Activity-Runs via bestehendem compare = funktionelle Läsionskarte (z. B. pre/post-Abliteration auf Refusal-Stimuli).

A. Protokoll zuerst (bindendes Artefakt vor Code)

State
Inhalt
max_len
rest
nur BOS
1
induction
fixes repetitives Muster („AB AB …“)
128
de_text / en_text
fixe Absätze (eigene Formulierung, kein Copyright)
128
code
fixes Python-Snippet
128
math
fixe Arithmetik-Sequenz
128
refusal
fixer Refusal-Trigger (benigner Wortlaut)
128
long
fixer Repeat-Text
1024

Jede Änderung an Strings/States = Protokoll-v2, nie still.

B. Capture (dep-arm, hooks statt Framework)

- Plain PyTorch-Forward-Hooks auf model.layers[i]-Output (Residual-Stream) und (MoE) Router-Softmax – keine NNsight/TransformerLens-Dependency; NNsight für Komponenten-Level-Capture → BACKLOG. Begründen (ARCHITECTURE.md).
- Metriken MVP: residual_rms (RMS pro Position → Feld Layer×Position je State) und expert_usage (mittlere Router-Masse pro Expert → Feld Layer×Expert je State).
- Loader-Pfad: HF-Modell via transformers (safetensors-Checkpoints). GGUF-Aktivität → BACKLOG (llama.cpp-Brücke).
- Extra activity (jetzt befüllen): torch, transformers. CI installiert es nicht; CI testet nur Pure-Teile (s. u.).

C. Determinismus = Scanner-Disziplin

torch.set_num_threads(1), use_deterministic_algorithms(True), Dtype fix via --dtype {float32,bfloat16}, Seed fix. activity_meta.json: Protokoll-Hash, Device, Dtype, Torch-/Transformers-Version. Compare zweier Activity-Runs: Meta-Mismatch (Device/Dtype/Protokoll) → Warnung, kein Reject – wie Cross-Scanner-fMRI, dokumentieren. Byte-Identität nur same-machine garantiert und nur dort getestet.

D. Integration in bestehende Architektur (nichts Neues erfinden)

- CLI: weight-atlas activity MODEL_PATH --out DIR --protocol … --device … --dtype …. Output-Dir ist ein normales Artefakt-Verzeichnis: field_activity_<state>_residual_{raw,smooth}.tif, field_activity_<state>_experts_*.tif, Meta, Manifest.
- compare funktioniert damit ohne neuen Code-Pfad: States = „Kanäle“; State-Mismatch → skipped-with-reason (Mechanik aus M6.5-Panels wiederverwenden). Δ-Activity-Sheet = funktionelle Läsionskarte.
- UI: Modellliste bekommt Eintrags-Typ (scan vs. activity), Detailseite zeigt Activity-Sheets + Expert-Usage; Compare-Seite unverändert.

E. Tests (~12, CI-sicher)

- Protokoll: Hash-Assert, unbekannter State → Fehler.
- Hook-Mechanik an Stub-nn.Module: known-value Forward → RMS handgerechnet.
- Field-Assembly aus aufgezeichneten .npy-Fixtures → Manifest/Sheet deterministisch byte-identisch.
- Compare rest-vs-task zweier Activity-Dirs grün; Meta-Mismatch → caplog-Warnung.
- skipif-Tests (lokal, Extra): winziges zufälliges HF-Llama und HF-MoE (echte Forward-Passes!): zwei Captures same-machine byte-identisch bei threads=1; Expert-Usage-Feld nicht überall NaN.

F. Doku

ARCHITECTURE.md: fMRI-Analogie (Protokoll=Messprotokoll, Device=Scanner), Hooks-vs-NNsight, threads=1-Entscheidung, Meta-Warnregel. README: Activity-Quickstart + „funktionelle Abliterations-Karte“ (scan pre/post → activity → compare auf refusal-State).

DoD

- M7.5 vollständig (erster Report-Block).
- Lokaler Smoke: tiny HF dense + tiny HF MoE → Sheets + Compare-Δ ohne Handarbeit.
- 187+ Tests; CI grün ohne activity-Extra; ruff diesmal wirklich clean.
- Keine Deps außerhalb des activity-Extras; Commits conventional; Neues → BACKLOG.md.

Report-Format: Block 1: M7.5; Block 2: M8. Danach Stopp – M8 ist das letzte Stück der Ursprungsvision; im Anschluss schlage ich den v0.1.0-Release-Schnitt vor (Tag, Release-Notes, Backlog als Roadmap).

----

Der letzte Meilenstein ist da – und mit ihm das inzwischen bekannte Muster: Kern steht, Peripherie fehlt teilweise. Erst das Review, dann der finale Fixup- und Release-Auftrag.

Review M7.5 – vollständig

Alle fünf Punkte mit Verweis, 178 Tests, Ruff clean. Zwei Schönheitsnoten ohne Aktion: TestFieldDryRun liegt in test_embedding.py statt bei den Blender-Wrapper-Tests (kosmetisch, nicht anfassen), und der M7.5-Block meldet mypy-Reste wegen umap-learn, der M8-Block „mypy clean“ – dieser Widerspruch wird in M8.5 verbindlich geklärt (s. u.).

----

Review M8 – akzeptiert, Release-Gate M8.5

Stark: Protokoll-zuerst eingehalten (8 States, Hash, „Änderung = v2, nie still“), Hooks ohne NNsight, Scanner-Disziplin (threads=1, deterministic algorithms), 192+1 Tests, CI bleibt ohne activity-Extra grün. Die fMRI-Analogie steht architektonisch genau dort, wo sie stehen soll.

Aber vor einem v0.1.0-Tag muss M8.5 durch – derselbe Fehlertyp wie bei M7 (leeres Extra) wiederholt sich:

1. activity-Extra nicht befüllt – pyproject.toml wieder unangetastet; torch/transformers fehlen im Extra. Das ist jetzt das zweite Mal; ab sofort ist „Extra befüllt?“ ein stehender DoD-Punkt.
2. Protokoll-Source-of-Truth: docs/activity_protocol.v1.md ist Prosa, aber protocol.py hält die Strings – die Doku kann lautlos driften. Entscheidung: specs/activity_protocol.v1.json wird Source of Truth (Muster wie atlas_spec.v1.json), protocol.py lädt sie, Hash-Test läuft gegen die JSON; das MD wird aus der JSON generiert (kleiner Doc-Generator + Drift-Test) oder gestrichen – beides akzeptiert, Drift-Schutz Pflicht.
3. activity_meta.json bestätigen oder bauen: Protokoll-Hash, Device, Dtype, Torch-/Transformers-Version; plus die drei Compare-Tests (rest-vs-task grün, State-Mismatch → skipped-with-reason, Meta-Mismatch → caplog-Warnung) – im Report nicht genannt.
4. UI: Eintrags-Typ (scan vs. activity) in der Modellliste + Activity-Sheets/Expert-Usage auf der Detailseite – Templates unangetastet.
5. README: Activity-Quickstart + „funktionelle Abliterations-Karte“-Beispiel fehlen.
6. Backlog: NNsight (Komponenten-Level) und GGUF-Aktivitätsbrücke (llama.cpp) eintragen.
7. mypy verbindlich clean inkl. umap-Modul (Ignore mit Begründung oder Stub – Entscheidung dokumentieren).

----

ARBEITSAUFTRAG Release v0.1.0 (nach M8.5, Gate)

Scope: Release-Schnitt, kein Feature-Code.

1. Versionierung: version = "0.1.0" im pyproject; weight-atlas --version + Test.

2. CHANGELOG.md aus den Milestones: je M0–M8 ein Eintrag (Feature + wichtigste Konvention), oben die drei Garantien des Tools: (a) Artefakte sind renderer-unabhängig kanonisch, (b) Render/Compare lesen nie Gewichte, (c) Determinismus ist Teil des Messprotokolls.

3. docs/ROADMAP.md aus dem Backlog, priorisiert: 1) Embedding-Alignment (Procrustes), 2) volle k-quants, 3) NNsight/Component-Capture, 4) GGUF-Aktivität, 5) Morph-Animation, 6) Cycles/Shader-Contours, 7) Entry-Point-Plugins, 8) HTMX-Vendoring.

4. README-Finalpass: Feature-Überblick + ehrlicher Abschnitt „Was weight-atlas ist / nicht ist“ (Fingerprint- und Vergleichswerkzeug, kein Benchmark, keine Capability-Aussage; Höhen = Projektionskonvention).

5. scripts/release_check.sh: frische venv-Äquivalenz (Core-Install), Fixture-Scan → Sheet → Compare → Embedding, Blender-Smoke nur wenn Binary, Activity-Smoke nur mit Extra; Exit-Codes sauber; in README verlinkt.

6. Tag: annotiertes v0.1.0 lokal erstellen; kein Push – Push/Release-Dialog ist meine/Auftraggeber-Sache, im Report vermerken.

7. Vollständiger Lauf: pytest (192+), ruff, mypy clean; CI grün.

DoD Release: M8.5-Liste vollständig (Block 1), Release-Artefakte vorhanden (Block 2), release_check.sh lokal grün, Tag lokal gesetzt.

Report-Format: Block 1: M8.5; Block 2: Release; dann Stopp – danach gehört das Repo dir zum Pushen.

----

Und weil nach dem Release die eigentliche Arbeit beginnt, hier schon der Anschluss-Plan für dich: der erste echte Atlas. Ich würde so vorgehen: scan über die safetensors-Shards des dense 27B und des MoE 35B-A3B (je ~10–20 min, CPU-freundlich dank mmap), dann drei Vergleiche, die deine Ursprungsfragen beantworten: (1) compare --mode aligned der beiden → Architektur-Δ (dense-Bänder vs. Experten-Hochland erstmals quantitativ), (2) falls du ein abliteriertes Derivat hast: compare --mode strict → Läsionskarte, (3) activity mit Protokoll v1 auf beiden → funktionelle Karte inkl. Expert-Usage des MoE. Sag Bescheid, wenn der Release-Report da ist – dann terminieren wir die echte Vermessung.
