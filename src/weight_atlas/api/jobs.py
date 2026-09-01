"""Job queue: SQLite-backed persistence + in-process worker thread."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from typing import Any

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# A ``running`` row whose last progress update is older than this is presumed
# to be a zombie left behind by a crashed process (start()'s startup recovery
# can't see it if it was marked running after start ran). The single worker
# thread can only run one job at a time, so any other running row this stale
# is not executing. Scans report per-tensor progress; renders report per
# channel; the Blender subprocess reports when it returns. 5 minutes is far
# longer than any legitimate quiet gap between progress callbacks.
_STALE_RUNNING_SECONDS = 300.0
# How often the sweeper thread checks for stale running rows.
_SWEEP_INTERVAL_SECONDS = 30.0


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
    # ``compare_interp`` carry the per-type params. ``sheet_knobs`` holds
    # optional per-render sheet overrides (e.g. ``{"normalized_depth": True}``)
    # applied on top of the recorded spec when a render job runs.
    job_type: str = "scan"
    renderer: str = ""
    compare_mode: str = "strict"
    compare_interp: str = "linear"
    sheet_knobs: dict[str, Any] = field(default_factory=dict)
    # Scan-only: run the measured RTN-SQNR probe (scan --quant-probe) during
    # the job. Persisted so restart recovery re-runs the SAME probe settings.
    quant_probe: bool = False

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
            "sheet_knobs": self.sheet_knobs,
            "quant_probe": self.quant_probe,
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
        self._sweeper: threading.Thread | None = None
        self._current_job_id: str | None = None
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
            ("sheet_knobs", "TEXT NOT NULL DEFAULT '{}'"),
            ("quant_probe", "INTEGER NOT NULL DEFAULT 0"),
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

    def _apply_sheet_knobs(self, spec: Any, knobs: dict[str, Any]) -> Any:
        """Return a copy of ``spec`` with per-render overrides applied.

        The ``sheet`` block is overlaid with its display knobs
        (``normalized_depth``, ``drop_empty_cols``) and the ``fractal`` block
        with the fractal-mode knob (``fractal_mode`` → ``fractal.mode``); the
        scan's recorded spec stays unchanged.
        """
        sheet = dict(getattr(spec, "sheet", {}) or {})
        for key in ("normalized_depth", "drop_empty_cols"):
            if key in knobs:
                sheet[key] = knobs[key]
        new_spec = replace(spec, sheet=sheet)

        fractal_mode = knobs.get("fractal_mode")
        if fractal_mode:
            fractal = dict(getattr(spec, "fractal", {}) or {})
            fractal["mode"] = fractal_mode
            new_spec = replace(new_spec, fractal=fractal)
        return new_spec

    def _save(self, job: Job) -> None:
        import json
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs
                (job_id, model_path, out_dir, spec_path, status, progress,
                 message, created_at, updated_at, error, artefacts,
                 job_type, renderer, compare_mode, compare_interp, sheet_knobs,
                 quant_probe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(job.sheet_knobs),
                    1 if job.quant_probe else 0,
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
            sheet_knobs=json.loads(row[15]) if len(row) > 15 else {},
            quant_probe=bool(row[16]) if len(row) > 16 else False,
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
        self._sweeper = threading.Thread(target=self._sweep_loop, daemon=True)
        self._sweeper.start()

    def stop(self) -> None:
        """Signal the worker to stop and wait for it."""
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
        if self._sweeper is not None:
            self._sweeper.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except Empty:
                continue
            self._current_job_id = job_id
            try:
                job = self._load(job_id)
                if job is None:
                    continue
                if job.status == JobStatus.DONE:
                    # Idempotent recovery: never re-execute a job already marked
                    # done (e.g. one completed by a manual/out-of-band render), even
                    # if its id is still sitting in the in-memory queue.
                    continue
                self._execute(job)
            finally:
                self._current_job_id = None

    def _sweep_loop(self) -> None:
        """Periodically reset stale ``running`` rows via _recover_stale_running."""
        while not self._stop.is_set():
            time.sleep(_SWEEP_INTERVAL_SECONDS)
            self._recover_stale_running()

    def _recover_stale_running(self) -> None:
        """Reset ``running`` rows left behind by a crashed process.

        ``start()`` resets running jobs present at startup, but a job marked
        ``running`` *after* start ran (by a worker that then died) would stay
        stuck as running forever. The worker is single-threaded: exactly one
        job runs at a time, so any other ``running`` row whose last update is
        older than ``_STALE_RUNNING_SECONDS`` is a zombie. Reset it to queued
        and re-enqueue so it re-runs idempotently.
        """
        cutoff = datetime.now(UTC).timestamp() - _STALE_RUNNING_SECONDS
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT job_id, updated_at FROM jobs "
                "WHERE status = ?",
                (JobStatus.RUNNING.value,),
            ).fetchall()
        now = self._now()
        for job_id, updated_at in rows:
            try:
                if datetime.fromisoformat(updated_at).timestamp() > cutoff:
                    continue  # updated recently — genuinely executing
            except (TypeError, ValueError):
                continue  # unparseable timestamp — treat as not stale
            if job_id == self._current_job_id:
                continue  # the job this worker is executing right now
            with self._connection() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, message = ?, updated_at = ? "
                    "WHERE job_id = ?",
                    (JobStatus.QUEUED.value, "re-queued after stale running", now, job_id),
                )
            self._queue.put(job_id)
            self._enqueued.add(job_id)

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
            if job.sheet_knobs:
                # Apply per-render sheet overrides (e.g. normalized_depth)
                # on top of the recorded spec; the scan's spec stays untouched.
                spec = self._apply_sheet_knobs(spec, job.sheet_knobs)
            artefacts: list[str]
            if job_type == "compare":
                progress_cb(0.2, "Comparing models...")
                artefacts = self._run_compare(job, spec, progress_cb, mode=compare_mode, interp=compare_interp)
            elif job_type == "render":
                progress_cb(0.2, "Rendering sheets...")
                produced, failed = self._render_job(job, spec, renderer_id or "sheet", progress_cb)
                artefacts = produced
                render_note = (
                    f" ({len(failed)} render failure(s): {', '.join(failed)})"
                    if failed else ""
                )
            else:
                from weight_atlas.scan import scan as run_scan
                # scan() reports granular phase progress; map its [0,1] into
                # the job's [0.05, 0.85] (rendering follows in [0.85, 1.0]).
                def scan_progress(pct: float, msg: str) -> None:
                    progress_cb(0.05 + 0.8 * pct, msg)
                artefacts = [str(a) for a in run_scan(
                    Path(job.model_path), Path(job.out_dir), spec,
                    progress=scan_progress,
                    quant_probe=job.quant_probe,
                )]
                # Auto-render sheets after scan (v0.2.0) → [0.85, 1.0]
                progress_cb(0.85, "Rendering sheets...")
                try:
                    render_artefacts = self._auto_render_sheets(
                        Path(job.out_dir), spec,
                        progress=lambda pct, msg: progress_cb(0.85 + 0.15 * pct, msg),
                    )
                    artefacts.extend(render_artefacts)
                except Exception:
                    # Rendering is best-effort; don't fail the job
                    logger.warning(
                        "auto-render after scan %s failed", job.job_id, exc_info=True
                    )
            job.artefacts = artefacts
            progress_cb(1.0, f"Complete{render_note if job_type == 'render' else ''}")
            job.status = JobStatus.DONE
            job.updated_at = self._now()
        except Exception as e:
            logger.exception("job %s (%s) failed", job.job_id, job_type)
            job.status = JobStatus.FAILED
            # Persist the traceback (truncated) so failures reported from the
            # UI are debuggable without server log access.
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            job.error = f"{type(e).__name__}: {e}\n{tb}"[-4000:]
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
        from weight_atlas.compare.pipeline import run_compare

        # Parse job.model_path as "dir_a|dir_b"
        parts = job.model_path.split("|")
        dir_a = Path(parts[0])
        dir_b = Path(parts[1])

        artefacts = run_compare(
            dir_a, dir_b, Path(job.out_dir), spec,
            mode=mode,
            interp=interp,
            progress=progress_cb,
        )
        return [str(p) for p in artefacts]

    def _render_job(
        self,
        job: Job,
        spec: Any,
        renderer_id: str,
        progress_cb: Callable[[float, str], None],
    ) -> tuple[list[str], list[str]]:
        """Render all channels of a completed scan (runs on the worker thread).

        Returns ``(produced_artefact_names, failed_channels)``.
        """
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
        failed: list[str] = []
        total = max(1, len(channels))
        for i, channel in enumerate(sorted(channels)):
            progress_cb(0.2 + 0.6 * i / total, f"Rendering {channel}...")
            field = load_channel_field(out_dir, channel, spec, model_name=model_name)
            if field is None:
                continue
            try:
                paths = renderer_obj.render(field, spec, render_dir)
                produced.extend(str(p.name) for p in paths)
            except Exception:  # noqa: BLE001 — per-channel render is best-effort
                logger.warning(
                    "render of %s/%s with %s failed",
                    out_dir.name, channel, renderer_id, exc_info=True,
                )
                failed.append(channel)

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
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "preview render of %s/%s failed", out_dir.name, channel,
                        exc_info=True,
                    )
                    failed.append(f"preview:{channel}")

        # Failures are logged above; the artefacts list stays a pure list of
        # file names. The caller surfaces failures via the completion message.
        return produced, failed

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
        quant_probe: bool = False,
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
            quant_probe=quant_probe,
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

    def submit_render(self, job_id: str, renderer: str, sheet_knobs: dict[str, Any] | None = None) -> Job:
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
            sheet_knobs=dict(sheet_knobs or {}),
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
            except Exception:  # noqa: BLE001 — best-effort, but log it
                logger.warning(
                    "auto-render during import of %s failed", scan_dir, exc_info=True
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
