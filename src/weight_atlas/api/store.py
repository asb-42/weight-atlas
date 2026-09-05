"""Job persistence backends: SQLite (local/dev) and MariaDB (server).

Phase 1 M1: the ``JobQueue`` (``api/jobs.py``) talks to storage only
through the ``JobStore`` protocol below, exchanging plain row dicts
keyed by column name. Both backends share one ordered column definition
(``COLUMNS``) so ``SELECT *`` positional mapping can never drift between
dialects, and both implement the same schema including the legacy
migrations.

- ``SQLiteJobStore``: the historical behavior, verbatim (WAL + busy
  timeout, fd-leak-safe close, legacy message-marker backfill).
- ``MariaDBJobStore``: same contract in MySQL dialect (``REPLACE INTO``
  for upsert, ``%s`` placeholders, ``SHOW COLUMNS`` introspection).
  MariaDB has no TEXT defaults, so its DDL uses sized types and no
  column defaults except numerics; INSERT always provides every column,
  and fresh deployments CREATE the full table up front (the ADD COLUMN
  path only ever fires when upgrading a MariaDB DB created by an older
  tool version — any future added column must therefore be a
  DEFAULT-able type, or the migration needs a two-step backfill).

The ``pymysql`` driver is an optional ``mysql`` extra, imported lazily
so the base install never requires it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

# Ordered (name, sqlite_type, mariadb_type). The order IS the SELECT *
# positional contract — both DDLs render from this list, so the mapping
# can never drift between backends. MariaDB types avoid TEXT DEFAULT
# (unsupported): sized VARCHAR/MEDIUMTEXT, no string defaults.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("job_id", "TEXT PRIMARY KEY", "VARCHAR(64) PRIMARY KEY"),
    ("model_path", "TEXT NOT NULL", "TEXT NOT NULL"),
    ("out_dir", "TEXT NOT NULL", "TEXT NOT NULL"),
    ("spec_path", "TEXT NOT NULL", "TEXT NOT NULL"),
    ("status", "TEXT NOT NULL", "VARCHAR(32) NOT NULL"),
    ("progress", "REAL NOT NULL DEFAULT 0.0", "DOUBLE NOT NULL DEFAULT 0.0"),
    ("message", "TEXT NOT NULL DEFAULT ''", "MEDIUMTEXT NOT NULL"),
    ("created_at", "TEXT NOT NULL", "VARCHAR(64) NOT NULL"),
    ("updated_at", "TEXT NOT NULL", "VARCHAR(64) NOT NULL"),
    ("error", "TEXT NOT NULL DEFAULT ''", "MEDIUMTEXT NOT NULL"),
    ("artefacts", "TEXT NOT NULL DEFAULT '[]'", "MEDIUMTEXT NOT NULL"),
    ("job_type", "TEXT NOT NULL DEFAULT 'scan'", "VARCHAR(32) NOT NULL"),
    ("renderer", "TEXT NOT NULL DEFAULT ''", "VARCHAR(128) NOT NULL"),
    ("compare_mode", "TEXT NOT NULL DEFAULT 'strict'", "VARCHAR(32) NOT NULL"),
    ("compare_interp", "TEXT NOT NULL DEFAULT 'linear'", "VARCHAR(32) NOT NULL"),
    ("sheet_knobs", "TEXT NOT NULL DEFAULT '{}'", "MEDIUMTEXT NOT NULL"),
    ("quant_probe", "INTEGER NOT NULL DEFAULT 0", "TINYINT(1) NOT NULL DEFAULT 0"),
)

COLUMN_NAMES: tuple[str, ...] = tuple(name for name, _, _ in COLUMNS)

# sqlite3 module is stdlib; pymysql arrives via the ``mysql`` extra and
# is imported lazily in _require_pymysql so the base install never
# requires it. All driver handles are typed Any.


def _require_pymysql() -> Any:
    import importlib

    try:
        return importlib.import_module("pymysql")
    except ImportError:
        raise ImportError(
            "MariaDB backend requires the 'mysql' extra: pip install weight-atlas[mysql]"
        ) from None


class JobStore(Protocol):
    """Persistence contract for the job queue (row dicts, column names)."""

    def init_schema(self) -> None:
        """Create the table and apply pending migrations. Idempotent."""
        ...

    def save_row(self, row: dict[str, Any]) -> None:
        """Upsert one job row (all COLUMNS keys expected)."""
        ...

    def load_row(self, job_id: str) -> dict[str, Any] | None:
        """One job row by id, or None."""
        ...

    def list_rows(self, limit: int) -> list[dict[str, Any]]:
        """Newest-first rows, at most ``limit``."""
        ...

    def recoverable_rows(self) -> list[tuple[str, str]]:
        """(job_id, status) for rows stuck queued/running (restart recovery)."""
        ...

    def running_rows(self) -> list[tuple[str, str]]:
        """(job_id, updated_at) for running rows (stale sweeper)."""
        ...

    def reset_row(self, job_id: str, status: str, message: str, updated_at: str) -> None:
        """Mark one row back to queued with a recovery message."""
        ...


def _placeholders(n: int, style: str) -> str:
    return ", ".join(["?"] * n) if style == "qmark" else ", ".join(["%s"] * n)


class SQLiteJobStore:
    """The historical SQLite behavior, moved verbatim behind the protocol."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

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

    def _column_names(self, conn: sqlite3.Connection) -> set[str]:
        return {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}

    def _row_to_dict(self, row: tuple[Any, ...]) -> dict[str, Any]:
        return dict(zip(COLUMN_NAMES, row, strict=True))

    def init_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            cols = ", ".join(f"{name} {stype}" for name, stype, _ in COLUMNS)
            conn.execute(f"CREATE TABLE IF NOT EXISTS jobs ({cols})")
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
        cols = self._column_names(conn)
        for name, stype, _ in COLUMNS:
            if name not in cols and name != "job_id":
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {stype}")
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

    def save_row(self, row: dict[str, Any]) -> None:
        with self._connection() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO jobs ({', '.join(COLUMN_NAMES)}) "
                f"VALUES ({_placeholders(len(COLUMN_NAMES), 'qmark')})",
                tuple(row[name] for name in COLUMN_NAMES),
            )

    def load_row(self, job_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_rows(self, limit: int) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def recoverable_rows(self) -> list[tuple[str, str]]:
        with self._connection() as conn:
            return conn.execute(
                "SELECT job_id, status FROM jobs WHERE status IN (?, ?)",
                ("queued", "running"),
            ).fetchall()

    def running_rows(self) -> list[tuple[str, str]]:
        with self._connection() as conn:
            return conn.execute(
                "SELECT job_id, updated_at FROM jobs WHERE status = ?",
                ("running",),
            ).fetchall()

    def reset_row(self, job_id: str, status: str, message: str, updated_at: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, message = ?, updated_at = ? "
                "WHERE job_id = ?",
                (status, message, updated_at, job_id),
            )


def parse_db_url(url: str) -> dict[str, Any]:
    """Parse ``mysql://user:pass@host:port/dbname`` into connect kwargs.

    Raises ValueError on anything else (only mysql:// and mariadb://
    schemes are accepted).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("mysql", "mariadb"):
        raise ValueError(
            f"unsupported database URL scheme {parsed.scheme!r} "
            "(want mysql://user:pass@host:port/dbname)"
        )
    if not parsed.hostname or not parsed.path or parsed.path == "/":
        raise ValueError(
            "database URL must include a host and database name: "
            "mysql://user:pass@host:port/dbname"
        )
    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": parsed.path.lstrip("/"),
    }


class MariaDBJobStore:
    """Same contract in MySQL dialect (server deployments, Phase 1 M1).

    One connection per operation (matching the SQLite pattern — PyMySQL
    connections are not thread-sharable), committed on clean exit.
    Fresh deployments CREATE the full table up front, so the ADD COLUMN
    path only fires when upgrading a MariaDB DB created by an older tool
    version (never for legacy SQLite rows — those transfer through
    ``save_row`` with all columns present).
    """

    def __init__(
        self,
        host: str,
        database: str,
        user: str = "",
        password: str = "",
        port: int = 3306,
        connect_timeout: int = 10,
    ) -> None:
        self._params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": "utf8mb4",
            "connect_timeout": connect_timeout,
            "read_timeout": 60,
            "write_timeout": 60,
        }

    @classmethod
    def from_url(cls, url: str) -> MariaDBJobStore:
        """Build from ``mysql://user:pass@host:port/dbname``."""
        return cls(**parse_db_url(url))

    def _connect(self) -> Any:
        pymysql = _require_pymysql()
        return pymysql.connect(**self._params)

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _column_names(self, conn: Any) -> set[str]:
        with conn.cursor() as cur:
            cur.execute("SHOW COLUMNS FROM jobs")
            return {r[0] for r in cur.fetchall()}

    def _row_to_dict(self, row: tuple[Any, ...]) -> dict[str, Any]:
        return dict(zip(COLUMN_NAMES, row, strict=True))

    def _execute(
        self, conn: Any, sql: str, params: tuple[Any, ...] = ()
    ) -> list[tuple[Any, ...]]:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows: list[tuple[Any, ...]] = cur.fetchall()
            return rows

    def init_schema(self) -> None:
        with self._connection() as conn:
            cols = ", ".join(f"`{name}` {mtype}" for name, _, mtype in COLUMNS)
            self._execute(conn, f"CREATE TABLE IF NOT EXISTS jobs ({cols})")
            existing = self._column_names(conn)
            for name, _, mtype in COLUMNS:
                if name not in existing and name != "job_id":
                    self._execute(conn, f"ALTER TABLE jobs ADD COLUMN `{name}` {mtype}")

    def save_row(self, row: dict[str, Any]) -> None:
        cols = ", ".join(f"`{n}`" for n in COLUMN_NAMES)
        with self._connection() as conn:
            self._execute(
                conn,
                f"REPLACE INTO jobs ({cols}) VALUES ({_placeholders(len(COLUMN_NAMES), 'percent')})",
                tuple(row[name] for name in COLUMN_NAMES),
            )

    def load_row(self, job_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            rows = self._execute(
                conn, "SELECT * FROM jobs WHERE job_id = %s", (job_id,)
            )
        if not rows:
            return None
        return self._row_to_dict(rows[0])

    def list_rows(self, limit: int) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = self._execute(
                conn,
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
        return [self._row_to_dict(r) for r in rows]

    def recoverable_rows(self) -> list[tuple[str, str]]:
        with self._connection() as conn:
            return self._execute(
                conn,
                "SELECT job_id, status FROM jobs WHERE status IN (%s, %s)",
                ("queued", "running"),
            )

    def running_rows(self) -> list[tuple[str, str]]:
        with self._connection() as conn:
            return self._execute(
                conn,
                "SELECT job_id, updated_at FROM jobs WHERE status = %s",
                ("running",),
            )

    def reset_row(self, job_id: str, status: str, message: str, updated_at: str) -> None:
        with self._connection() as conn:
            self._execute(
                conn,
                "UPDATE jobs SET status = %s, message = %s, updated_at = %s "
                "WHERE job_id = %s",
                (status, message, updated_at, job_id),
            )


def transfer_jobs(source: JobStore, dest: JobStore, limit: int = 100000) -> int:
    """Copy all job rows from one backend to another (history migration).

    Reads newest-first pages from ``source`` and upserts into ``dest``;
    returns the number transferred. Both sides must already be
    initialized (dest gets ``init_schema`` here for convenience).
    Values transfer verbatim — no re-interpretation, so legacy rows
    upgraded by the SQLite backend arrive complete.
    """
    dest.init_schema()
    moved = 0
    batch = 500
    offset_rows: list[dict[str, Any]] = source.list_rows(limit)
    for i in range(0, len(offset_rows), batch):
        for row in offset_rows[i : i + batch]:
            dest.save_row(dict(row))
            moved += 1
    return moved
