"""Job queue: SQLite-backed persistence + in-process worker thread."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from typing import Any


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    model_path: str
    out_dir: str
    spec_path: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    created_at: str = ""
    updated_at: str = ""
    error: str = ""
    artefacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_path": self.model_path,
            "out_dir": self.out_dir,
            "spec_path": self.spec_path,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "artefacts": self.artefacts,
        }


class JobQueue:
    """SQLite-backed job queue with an in-process worker thread.

    Jobs are persisted to SQLite; a background thread picks up QUEUED jobs
    and runs them sequentially. Progress is updated via a callback that
    writes to the DB.
    """

    def __init__(self, db_path: Path, on_job: Callable[[Job], None]) -> None:
        self._db_path = db_path
        self._on_job = on_job
        self._queue: Queue[str] = Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    out_dir TEXT NOT NULL,
                    spec_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0.0,
                    message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    artefacts TEXT NOT NULL DEFAULT '[]'
                )
            """)

    def _now(self) -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def _save(self, job: Job) -> None:
        import json
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs
                (job_id, model_path, out_dir, spec_path, status, progress,
                 message, created_at, updated_at, error, artefacts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.model_path,
                    job.out_dir,
                    job.spec_path,
                    job.status.value,
                    job.progress,
                    job.message,
                    job.created_at,
                    job.updated_at,
                    job.error,
                    json.dumps(job.artefacts),
                ),
            )

    def _load(self, job_id: str) -> Job | None:
        import json
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        return Job(
            job_id=row[0],
            model_path=row[1],
            out_dir=row[2],
            spec_path=row[3],
            status=JobStatus(row[4]),
            progress=row[5],
            message=row[6],
            created_at=row[7],
            updated_at=row[8],
            error=row[9],
            artefacts=json.loads(row[10]),
        )

    def start(self) -> None:
        """Start the background worker thread."""
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        """Signal the worker to stop and wait for it."""
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except Empty:
                continue
            job = self._load(job_id)
            if job is None:
                continue
            self._execute(job)

    def _execute(self, job: Job) -> None:
        from weight_atlas.core.types import AtlasSpec

        now = self._now()
        job.status = JobStatus.RUNNING
        job.updated_at = now

        # Determine job type from message field
        job_type = job.message if job.message in ("compare", "scan") else "scan"
        if job_type == "scan":
            job.message = "Starting scan..."
        else:
            job.message = "Starting compare..."

        self._save(job)
        self._on_job(job)

        def progress_cb(pct: float, msg: str) -> None:
            job.progress = pct
            job.message = msg
            job.updated_at = self._now()
            self._save(job)
            self._on_job(job)

        try:
            spec = AtlasSpec.from_json(Path(job.spec_path))
            artefacts: list[str]
            if job_type == "compare":
                progress_cb(0.2, "Comparing models...")
                artefacts = self._run_compare(job, spec, progress_cb)
            else:
                from weight_atlas.scan import scan as run_scan
                progress_cb(0.1, "Loading model...")
                progress_cb(0.3, "Computing statistics...")
                artefacts = [str(a) for a in run_scan(
                    Path(job.model_path), Path(job.out_dir), spec
                )]
                # Auto-render sheets after scan (v0.2.0)
                progress_cb(0.8, "Rendering sheets...")
                try:
                    render_artefacts = self._auto_render_sheets(
                        Path(job.out_dir), spec
                    )
                    artefacts.extend(render_artefacts)
                except Exception as render_err:
                    # Rendering is best-effort; don't fail the job
                    print(f"Warning: auto-render failed: {render_err}",
                          file=__import__('sys').stderr)
            job.artefacts = artefacts
            progress_cb(1.0, "Complete")
            job.status = JobStatus.DONE
            job.updated_at = self._now()
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.updated_at = self._now()
        self._save(job)
        self._on_job(job)

    def _run_compare(
        self,
        job: Job,
        spec: Any,
        progress_cb: Callable[[float, str], None],
    ) -> list[str]:
        """Run comparison between two scanned model directories."""
        import json

        from weight_atlas.compare import compute_compare_summary, hotspot_ranking
        from weight_atlas.fields.tif_io import read_tif, write_tif

        # Parse job.model_path as "dir_a|dir_b"
        parts = job.model_path.split("|")
        dir_a = Path(parts[0])
        dir_b = Path(parts[1])

        out = Path(job.out_dir)

        fp_a_path = dir_a / "fingerprint.json"
        fp_b_path = dir_b / "fingerprint.json"
        fp_a = json.loads(fp_a_path.read_text()) if fp_a_path.exists() else None
        fp_b = json.loads(fp_b_path.read_text()) if fp_b_path.exists() else None

        manifest_path = dir_a / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {dir_a}")

        manifest = json.loads(manifest_path.read_text())
        channels = self._discover_channels_from_manifest(manifest)

        summary_channels = {}
        all_artefacts: list[str] = []

        for channel in channels:
            field_a_path = dir_a / f"field_{channel}_raw.tif"
            field_b_path = dir_b / f"field_{channel}_raw.tif"

            if not field_a_path.exists() or not field_b_path.exists():
                continue

            field_a = read_tif(field_a_path)
            field_b = read_tif(field_b_path)

            summary = compute_compare_summary(
                field_a, field_b, spec,
                mode="strict",  # Could be extended to support mode parameter
                fingerprint_a=fp_a,
                fingerprint_b=fp_b,
            )
            summary_channels[channel] = summary.channels[channel]

            delta_path = out / f"delta_{channel}_raw.tif"
            write_tif(delta_path, summary.channels[channel].delta)
            all_artefacts.append(str(delta_path))

        compare_summary = {
            "mode": "strict",
            "spec_version": spec.spec_version,
            "model_a": summary.model_a,
            "model_b": summary.model_b,
            "warnings": summary.warnings,
            "channels": {},
        }
        for ch_name, ch_delta in summary_channels.items():
            ranking = hotspot_ranking(ch_delta, col_labels=summary.aligned_col_labels, top_k=5)
            compare_summary["channels"][ch_name] = {
                "rel_l2": ch_delta.rel_l2,
                "cosine_sim": ch_delta.cosine_sim,
                "hotspot_layer": ch_delta.hotspot_layer,
                "hotspot_slot": ch_delta.hotspot_slot,
                "hotspot_value": ch_delta.hotspot_value,
                "argmax": list(ch_delta.argmax),
                "hotspot_ranking": [
                    {"layer": r[0], "slot": r[1], "abs_delta": r[2]} for r in ranking
                ],
            }

        summary_path = out / "compare_summary.json"
        with open(summary_path, "w") as f:
            json.dump(compare_summary, f, indent=2, sort_keys=True)
            f.write("\n")
        all_artefacts.append(str(summary_path))

        return all_artefacts

    def _discover_channels_from_manifest(self, manifest: dict[str, str]) -> list[str]:
        """Discover channel names from manifest keys."""
        channels: set[str] = set()
        for key in manifest:
            if not key.startswith("field_") or not key.endswith(".tif"):
                continue
            core = key[len("field_"):-len(".tif")]
            if core.endswith("_raw"):
                channels.add(core[:-len("_raw")])
            elif core.endswith("_smooth"):
                channels.add(core[:-len("_smooth")])
        return sorted(channels)

    def submit(
        self,
        model_path: Path,
        out_dir: Path,
        spec_path: Path,
    ) -> Job:
        """Create a new job and enqueue it."""
        now = self._now()
        job = Job(
            job_id=str(uuid.uuid4()),
            model_path=str(model_path),
            out_dir=str(out_dir),
            spec_path=str(spec_path),
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            message="Queued",
        )
        self._save(job)
        self._queue.put(job.job_id)
        return job

    def submit_compare(
        self,
        dir_a: Path,
        dir_b: Path,
        out_dir: Path,
        spec_path: Path,
    ) -> Job:
        """Create a new compare job and enqueue it."""
        now = self._now()
        # Encode both paths into model_path field with a delimiter
        path_key = f"{dir_a}|{dir_b}"
        job = Job(
            job_id=str(uuid.uuid4()),
            model_path=path_key,
            out_dir=str(out_dir),
            spec_path=str(spec_path),
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            message="compare",
        )
        self._save(job)
        self._queue.put(job.job_id)
        return job

    def import_scan(
        self,
        scan_dir: Path,
        model_path: str = "",
        auto_render: bool = True,
    ) -> Job:
        """Import an existing scan directory into the job database.

        Args:
            scan_dir: Path to the directory containing scan artefacts
            model_path: Optional model path to display (default: scan_dir name)
            auto_render: If True, render sheets after import

        Returns:
            Job: The created job entry
        """
        import json
        # shutil imported where needed
        now = self._now()

        # Load fingerprint if available
        fp_path = scan_dir / "fingerprint.json"
        if fp_path.exists():
            with open(fp_path) as f:
                fp = json.load(f)
            if not model_path:
                model_path = fp.get("model", {}).get("path", str(scan_dir))
        else:
            if not model_path:
                model_path = str(scan_dir)

        # Auto-render sheets if requested
        if auto_render:
            try:
                from weight_atlas.core.registry import get_renderer
                from weight_atlas.core.types import AtlasSpec, Field2D
                from weight_atlas.fields.tif_io import read_tif

                spec = AtlasSpec.from_json(Path("specs/atlas_spec.v2.json"))
                renderer = get_renderer("sheet")()

                # Discover channels from scan directory
                channels = set()
                for tif in scan_dir.glob("field_*.tif"):
                    core = tif.name[len("field_"):-len(".tif")]
                    if core.endswith("_raw"):
                        channels.add(core[:-len("_raw")])
                    elif core.endswith("_smooth"):
                        channels.add(core[:-len("_smooth")])

                render_dir = scan_dir / "render"
                for channel in channels:
                    # Render smooth version if available, else raw
                    smooth_path = scan_dir / f"field_{channel}_smooth.tif"
                    raw_path = scan_dir / f"field_{channel}_raw.tif"
                    tif = smooth_path if smooth_path.exists() else raw_path
                    if not tif.exists():
                        continue

                    data = read_tif(tif)
                    n_rows, n_cols = data.shape
                    row_labels = [str(i) for i in range(n_rows)]
                    col_labels = list(spec.slots)

                    field = Field2D(
                        channel=channel,
                        data=data,
                        row_labels=row_labels,
                        col_labels=col_labels,
                        spec_version=spec.spec_version,
                    )
                    renderer.render(field, spec, render_dir)
            except Exception:
                pass  # Rendering is best-effort

        # Discover artefacts (including any rendered PNGs)
        artefacts = []
        if (scan_dir / "manifest.json").exists():
            with open(scan_dir / "manifest.json") as f:
                manifest = json.load(f)
            artefacts = list(manifest.keys())
        # Add rendered PNGs
        render_dir = scan_dir / "render"
        if render_dir.exists():
            for png in render_dir.glob("*.png"):
                artefacts.append(f"render/{png.name}")

        job = Job(
            job_id=str(uuid.uuid4()),
            model_path=model_path,
            out_dir=str(scan_dir),
            spec_path="",
            status=JobStatus.DONE,
            progress=1.0,
            message="Imported",
            created_at=now,
            updated_at=now,
            artefacts=artefacts,
        )
        self._save(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._load(job_id)

    def list_jobs(self, limit: int = 50) -> list[Job]:
        import json
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            Job(
                job_id=r[0],
                model_path=r[1],
                out_dir=r[2],
                spec_path=r[3],
                status=JobStatus(r[4]),
                progress=r[5],
                message=r[6],
                created_at=r[7],
                updated_at=r[8],
                error=r[9],
                artefacts=json.loads(r[10]),
            )
            for r in rows
        ]


    def _auto_render_sheets(self, out_dir: Path, spec: Any) -> list[str]:
        """Auto-render sheet PNGs from scan artefacts (best-effort)."""
        from weight_atlas.core.registry import get_renderer
        from weight_atlas.core.types import Field2D
        from weight_atlas.fields.tif_io import read_tif

        renderer = get_renderer("sheet")()
        render_dir = out_dir / "render"
        render_dir.mkdir(exist_ok=True)

        # Discover channels from tif files
        channels: set[str] = set()
        for tif in out_dir.glob("field_*.tif"):
            core = tif.name[len("field_"):-len(".tif")]
            if core.endswith("_raw"):
                channels.add(core[:-len("_raw")])
            elif core.endswith("_smooth"):
                channels.add(core[:-len("_smooth")])

        rendered: list[str] = []
        for channel in channels:
            smooth_path = out_dir / f"field_{channel}_smooth.tif"
            raw_path = out_dir / f"field_{channel}_raw.tif"
            tif = smooth_path if smooth_path.exists() else raw_path
            if not tif.exists():
                continue

            data = read_tif(tif)
            n_rows, n_cols = data.shape
            field = Field2D(
                channel=channel,
                data=data,
                row_labels=[str(i) for i in range(n_rows)],
                col_labels=list(spec.slots)[:n_cols] if n_cols <= len(spec.slots) else [str(i) for i in range(n_cols)],
                spec_version=spec.spec_version,
            )
            renderer.render(field, spec, render_dir)

        # Collect rendered PNGs
        for png in render_dir.glob("*.png"):
            rendered.append(f"render/{png.name}")

        return rendered
