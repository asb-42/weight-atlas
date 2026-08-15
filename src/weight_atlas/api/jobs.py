"""Job queue: SQLite-backed persistence + in-process worker thread."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
    # Job type is persisted separately from ``message`` so a restart can
    # recover the original kind (scan/render/compare) even after progress
    # text overwrote ``message``. ``renderer``/``compare_mode``/
    # ``compare_interp`` carry the per-type params.
    job_type: str = "scan"
    renderer: str = ""
    compare_mode: str = "strict"
    compare_interp: str = "linear"

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
            "job_type": self.job_type,
            "renderer": self.renderer,
            "compare_mode": self.compare_mode,
            "compare_interp": self.compare_interp,
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
        # Job ids already placed on this instance's in-memory queue (via submit/
        # submit_compare/submit_rescan/submit_render). start()'s restart
        # recovery must not re-enqueue these, or a job submitted before start()
        # would run twice.
        self._enqueued: set[str] = set()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL + a busy timeout.

        Concurrent worker writes and request reads otherwise hit
        ``database is locked`` under load.
        """
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a WAL connection that is guaranteed to be closed afterwards.

        ``with sqlite3.Connection`` commits/rolls back but does NOT close the
        connection, so ``with self._connect() as conn:`` leaked one or two file
        descriptors per call (db + WAL) until the cyclic GC happened to run.
        The UI polls the DB every 2 s and the worker saves progress frequently,
        so over a long scan the leaks exhausted the process's fd limit
        (EMFILE → ``unable to open database file``). Always close explicitly.
        """
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
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
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add the job_type columns on pre-existing databases and backfill.

        Older schema encoded the job type in ``message`` (``scan``,
        ``compare[:mode[:interp]]``, ``render:<renderer>``) and recovery
        overwrote it with ``re-queued after restart``, turning render/compare
        jobs into scans on restart. New columns persist the type explicitly;
        legacy rows are backfilled from their still-intact message where
        possible (running/complete rows no longer carry the marker and fall
        back to ``scan``).
        """
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
        for name, ddl in (
            ("job_type", "TEXT NOT NULL DEFAULT 'scan'"),
            ("renderer", "TEXT NOT NULL DEFAULT ''"),
            ("compare_mode", "TEXT NOT NULL DEFAULT 'strict'"),
            ("compare_interp", "TEXT NOT NULL DEFAULT 'linear'"),
        ):
            if name not in cols:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {ddl}")
        rows = conn.execute(
            "SELECT job_id, message, out_dir FROM jobs WHERE job_type = 'scan'"
        ).fetchall()
        for job_id, message, out_dir in rows:
            if message == "compare" or message.startswith("compare:"):
                parts = message.split(":", 2)
                conn.execute(
                    "UPDATE jobs SET job_type='compare', compare_mode=?, "
                    "compare_interp=? WHERE job_id=?",
                    (parts[1] if len(parts) > 1 else "strict",
                     parts[2] if len(parts) > 2 else "linear", job_id),
                )
            elif message.startswith("render:"):
                conn.execute(
                    "UPDATE jobs SET job_type='render', renderer=? WHERE job_id=?",
                    (message.split(":", 1)[1], job_id),
                )
            elif Path(out_dir).name.startswith("compare_"):
                # Completed legacy compare jobs have message="Complete" (the
                # marker was overwritten by progress text), so recover the type
                # from the compare_* out_dir naming instead.
                conn.execute(
                    "UPDATE jobs SET job_type='compare' WHERE job_id=?",
                    (job_id,),
                )

    def _now(self) -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def _save(self, job: Job) -> None:
        import json
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs
                (job_id, model_path, out_dir, spec_path, status, progress,
                 message, created_at, updated_at, error, artefacts,
                 job_type, renderer, compare_mode, compare_interp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    job.job_type,
                    job.renderer,
                    job.compare_mode,
                    job.compare_interp,
                ),
            )

    def _load(self, job_id: str) -> Job | None:
        import json
        with self._connection() as conn:
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
            job_type=row[11],
            renderer=row[12],
            compare_mode=row[13],
            compare_interp=row[14],
        )

    def start(self) -> None:
        """Start the background worker thread, recovering interrupted jobs.

        Jobs persisted as ``queued`` (never picked up) are re-enqueued, and
        jobs left as ``running`` by a crash/restart are reset to ``queued`` so
        they re-run idempotently (scan/compare overwrite their own outputs).
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT job_id, status FROM jobs WHERE status IN (?, ?)",
                (JobStatus.QUEUED.value, JobStatus.RUNNING.value),
            ).fetchall()
            now = self._now()
            for job_id, status in rows:
                if job_id in self._enqueued:
                    continue  # already on this instance's queue — don't double-enqueue
                if status == JobStatus.RUNNING.value:
                    conn.execute(
                        "UPDATE jobs SET status = ?, message = ?, updated_at = ? WHERE job_id = ?",
                        (JobStatus.QUEUED.value, "re-queued after restart", now, job_id),
                    )
                self._queue.put(job_id)
                self._enqueued.add(job_id)

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
            if job.status == JobStatus.DONE:
                # Idempotent recovery: never re-execute a job already marked
                # done (e.g. one completed by a manual/out-of-band render), even
                # if its id is still sitting in the in-memory queue.
                continue
            self._execute(job)

    def _execute(self, job: Job) -> None:
        from weight_atlas.core.types import AtlasSpec

        now = self._now()
        job.status = JobStatus.RUNNING
        job.updated_at = now

        # Job type is persisted explicitly (``job.job_type``), never parsed
        # back from ``message`` — recovery rewrites message with progress text.
        job_type = job.job_type
        compare_mode = job.compare_mode
        compare_interp = job.compare_interp
        renderer_id = job.renderer
        if job_type == "scan":
            job.message = "Starting scan..."
        elif job_type == "compare":
            job.message = "Starting compare..."
        else:
            job.message = "Starting render..."

        self._save(job)
        self._on_job(job)

        def progress_cb(pct: float, msg: str) -> None:
            job.progress = pct
            job.message = msg
            job.updated_at = self._now()
            self._save(job)
            self._on_job(job)

        try:
            from weight_atlas.core.types import load_default_spec
            spec_path = Path(job.spec_path) if job.spec_path else None
            spec = AtlasSpec.from_json(spec_path) if spec_path and spec_path.exists() else load_default_spec()
            artefacts: list[str]
            if job_type == "compare":
                progress_cb(0.2, "Comparing models...")
                artefacts = self._run_compare(job, spec, progress_cb, mode=compare_mode, interp=compare_interp)
            elif job_type == "render":
                progress_cb(0.2, "Rendering sheets...")
                artefacts = self._render_job(job, spec, renderer_id or "sheet", progress_cb)
            else:
                from weight_atlas.scan import scan as run_scan
                # scan() reports granular phase progress; map its [0,1] into
                # the job's [0.05, 0.85] (rendering follows in [0.85, 1.0]).
                def scan_progress(pct: float, msg: str) -> None:
                    progress_cb(0.05 + 0.8 * pct, msg)
                artefacts = [str(a) for a in run_scan(
                    Path(job.model_path), Path(job.out_dir), spec,
                    progress=scan_progress,
                )]
                # Auto-render sheets after scan (v0.2.0) → [0.85, 1.0]
                progress_cb(0.85, "Rendering sheets...")
                try:
                    render_artefacts = self._auto_render_sheets(
                        Path(job.out_dir), spec,
                        progress=lambda pct, msg: progress_cb(0.85 + 0.15 * pct, msg),
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
        mode: str = "strict",
        interp: str = "linear",
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
        # Only compare channels the alignment/compare infrastructure supports —
        # the main spec.channels. Vision (``vision_*``) and MoE expert-panel
        # (``expert_*``) fields use their own taxonomies and are not comparable
        # here; including them would KeyError on ``summary.channels[channel]``.
        channels = [c for c in channels if c in spec.channels]

        summary_channels = {}
        all_artefacts: list[Path] = []
        total_channels = max(1, len(channels))

        for i, channel in enumerate(channels):
            progress_cb(0.1 + 0.8 * (i / total_channels), f"Comparing {channel} field...")
            field_a_path = dir_a / f"field_{channel}_raw.tif"
            field_b_path = dir_b / f"field_{channel}_raw.tif"

            if not field_a_path.exists() or not field_b_path.exists():
                continue

            field_a = read_tif(field_a_path)
            field_b = read_tif(field_b_path)

            summary = compute_compare_summary(
                field_a, field_b, spec,
                mode=mode,
                interp=interp,
                fingerprint_a=fp_a,
                fingerprint_b=fp_b,
            )
            summary_channels[channel] = summary.channels[channel]

            delta_path = out / f"delta_{channel}_raw.tif"
            write_tif(delta_path, summary.channels[channel].delta)
            all_artefacts.append(delta_path)

            # Render delta sheet PNGs so the compare report has visuals.
            try:
                import weight_atlas.compare.render  # noqa: F401 — registers "delta"
                from weight_atlas.core.registry import get_renderer

                renderer = get_renderer("delta")()
                rendered = renderer.render(
                    summary.channels[channel].delta,
                    spec,
                    out / "render",
                    channel=channel,
                    row_labels=summary.aligned_row_labels,
                    col_labels=summary.aligned_col_labels,
                    mode=mode,
                    model_a=dir_a.name,
                    model_b=dir_b.name,
                )
                all_artefacts.extend(rendered)
            except KeyError:
                pass  # delta renderer not registered

        if not summary_channels:
            # Nothing to compare (e.g. activity-only or partial scans) — avoid
            # the NameError from referencing loop-scoped `summary`.
            return []

        compare_summary = {
            "mode": mode,
            "spec_version": spec.spec_version,
            "model_a": summary.model_a,
            "model_b": summary.model_b,
            "loaders": {
                "a": summary.model_a.get("loader"),
                "b": summary.model_b.get("loader"),
            },
            "warnings": summary.warnings,
            "alignment": summary.alignment,
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
        all_artefacts.append(summary_path)

        return [str(p) for p in all_artefacts]

    def _render_job(
        self,
        job: Job,
        spec: Any,
        renderer_id: str,
        progress_cb: Callable[[float, str], None],
    ) -> list[str]:
        """Render all channels of a completed scan (runs on the worker thread)."""
        from weight_atlas.core.registry import get_renderer
        from weight_atlas.fields.rasterizer import load_channel_field

        out_dir = Path(job.out_dir)
        render_dir = out_dir / "render"
        render_dir.mkdir(exist_ok=True)

        model_name = Path(job.model_path).name or out_dir.name
        renderer_obj = get_renderer(renderer_id)()

        channels: set[str] = set()
        for tif in out_dir.glob("field_*.tif"):
            core = tif.name[len("field_"):-len(".tif")]
            if core.endswith("_raw"):
                channels.add(core[:-len("_raw")])
            elif core.endswith("_smooth"):
                channels.add(core[:-len("_smooth")])

        produced: list[str] = []
        total = max(1, len(channels))
        for i, channel in enumerate(sorted(channels)):
            progress_cb(0.2 + 0.6 * i / total, f"Rendering {channel}...")
            field = load_channel_field(out_dir, channel, spec, model_name=model_name)
            if field is None:
                continue
            try:
                paths = renderer_obj.render(field, spec, render_dir)
                produced.extend(str(p.name) for p in paths)
            except Exception as e:  # noqa: BLE001 — per-channel render is best-effort
                produced.append(f"Error rendering {channel}: {e}")

        if renderer_id == "sheet":
            from weight_atlas.render.preview import PreviewRenderer
            preview_renderer = PreviewRenderer()
            for channel in sorted(channels):
                field = load_channel_field(out_dir, channel, spec, model_name=model_name)
                if field is None:
                    continue
                try:
                    paths = preview_renderer.render(field, spec, render_dir)
                    produced.extend(str(p.name) for p in paths)
                except Exception as e:  # noqa: BLE001
                    produced.append(f"Error rendering preview {channel}: {e}")

        return produced

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
            job_type="scan",
        )
        self._save(job)
        self._enqueued.add(job.job_id)
        self._queue.put(job.job_id)
        return job

    def submit_compare(
        self,
        dir_a: Path,
        dir_b: Path,
        out_dir: Path,
        spec_path: Path,
        mode: str = "strict",
        interp: str = "linear",
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
            message="Queued",
            job_type="compare",
            compare_mode=mode,
            compare_interp=interp,
        )
        self._save(job)
        self._enqueued.add(job.job_id)
        self._queue.put(job.job_id)
        return job

    def submit_rescan(self, job_id: str) -> Job:
        """Enqueue a re-scan of an existing job into the same output directory.

        Offloads the (potentially very slow) full scan off the event loop onto
        the single worker thread, like a normal scan job.
        """
        original = self._load(job_id)
        if original is None:
            raise KeyError(f"job not found: {job_id}")
        now = self._now()
        job = Job(
            job_id=str(uuid.uuid4()),
            model_path=original.model_path,
            out_dir=original.out_dir,
            spec_path=original.spec_path,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            message="Queued",
            job_type="scan",
        )
        self._save(job)
        self._enqueued.add(job.job_id)
        self._queue.put(job.job_id)
        return job

    def submit_render(self, job_id: str, renderer: str) -> Job:
        """Enqueue a render of a completed scan (worker thread, not event loop)."""
        original = self._load(job_id)
        if original is None:
            raise KeyError(f"job not found: {job_id}")
        now = self._now()
        job = Job(
            job_id=str(uuid.uuid4()),
            model_path=original.model_path,
            out_dir=original.out_dir,
            spec_path=original.spec_path,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            message="Queued",
            job_type="render",
            renderer=renderer,
        )
        self._save(job)
        self._enqueued.add(job.job_id)
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
                from weight_atlas.core.types import load_default_spec
                from weight_atlas.fields.rasterizer import load_channel_field

                spec = load_default_spec()
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
                    field = load_channel_field(scan_dir, channel, spec, model_name=model_path or scan_dir.name)
                    if field is None:
                        continue
                    renderer.render(field, spec, render_dir)
            except Exception as render_err:  # noqa: BLE001 — best-effort, but log it
                print(
                    f"Warning: import auto-render failed: {render_err}",
                    file=__import__('sys').stderr,
                )

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

    def update_model_path(self, job_id: str, model_path: str) -> Job:
        """Update the model path recorded on a job (e.g. for re-scanning an import)."""
        job = self._load(job_id)
        if job is None:
            raise KeyError(f"job not found: {job_id}")
        job.model_path = str(model_path)
        job.updated_at = self._now()
        self._save(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._load(job_id)

    def list_jobs(self, limit: int = 50) -> list[Job]:
        import json
        with self._connection() as conn:
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
                job_type=r[11],
                renderer=r[12],
                compare_mode=r[13],
                compare_interp=r[14],
            )
            for r in rows
        ]


    def _auto_render_sheets(
        self,
        out_dir: Path,
        spec: Any,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[str]:
        """Auto-render sheet PNGs from scan artefacts (best-effort)."""
        from weight_atlas.core.registry import get_renderer
        from weight_atlas.fields.rasterizer import load_channel_field

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
        sorted_channels = sorted(channels)
        total = max(1, len(sorted_channels))
        for i, channel in enumerate(sorted_channels):
            if progress is not None:
                progress(i / total, f"Rendering {channel} sheet...")
            field = load_channel_field(out_dir, channel, spec, model_name=out_dir.name)
            if field is None:
                continue
            renderer.render(field, spec, render_dir)

        # Collect rendered PNGs
        for png in render_dir.glob("*.png"):
            rendered.append(f"render/{png.name}")

        return rendered
