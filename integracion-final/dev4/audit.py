"""Bounded background audit queue with local SQLite persistence.

Only metadata is recorded, never audio, transcripts, request bodies or secrets.
Queued events can be lost on abrupt process death; counters expose failures.
"""
import json
from contextlib import contextmanager
import os
import sqlite3
import time
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread

COLUMNS = ("request_id", "created_at", "status_code", "is_synthetic", "p_synthetic",
           "latency_ms", "duration_s", "model_variant", "model_sha256")
SCHEMA = """CREATE TABLE IF NOT EXISTS calls (
request_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status_code INTEGER NOT NULL,
is_synthetic INTEGER, p_synthetic REAL, latency_ms REAL NOT NULL, duration_s REAL,
model_variant TEXT, model_sha256 TEXT, exported INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS calls_created ON calls(created_at DESC);"""


class AuditStore:
    def __init__(self, path=None, capacity=1024):
        self.path = Path(path or os.environ.get("AUDIT_DB_PATH", "data/audit.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
        self.queue = Queue(maxsize=capacity)
        self.lock = Lock()
        self.counts = {"queued": 0, "written": 0, "duplicate_ignored": 0, "dropped": 0, "write_errors": 0}
        self.last_error = None
        self.closed = False
        self.worker = Thread(target=self._write_loop, daemon=True, name="altur-audit")
        self.worker.start()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=2)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, event):
        clean = {key: event.get(key) for key in COLUMNS}
        with self.lock:
            if self.closed:
                self.counts["dropped"] += 1
                return False
            try:
                self.queue.put_nowait(clean)
                self.counts["queued"] += 1
                return True
            except Full:
                self.counts["dropped"] += 1
                return False

    def _write_loop(self):
        while True:
            event = self.queue.get()
            try:
                if event is None:
                    return
                with self.connect() as db:
                    cursor = db.execute("INSERT OR IGNORE INTO calls (" + ",".join(COLUMNS) + ") VALUES (" + ",".join("?" for _ in COLUMNS) + ")", [event[key] for key in COLUMNS])
                    inserted = cursor.rowcount == 1
                with self.lock:
                    self.counts["written" if inserted else "duplicate_ignored"] += 1
            except sqlite3.Error:
                with self.lock:
                    self.counts["write_errors"] += 1
                    self.last_error = "sqlite_write_failed"
            finally:
                self.queue.task_done()

    def wait_idle(self, timeout=5):
        deadline = time.monotonic() + timeout
        while self.queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(.01)
        return self.queue.unfinished_tasks == 0

    def close(self):
        with self.lock:
            self.closed = True
        if self.wait_idle():
            self.queue.put(None)
            self.worker.join(timeout=3)

    def recent(self, limit=50):
        limit = min(max(int(limit), 1), 200)
        with self.connect() as db:
            rows = db.execute("SELECT * FROM calls ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def stats(self):
        with self.connect() as db:
            row = db.execute("SELECT count(*) total, sum(status_code=200) successful, sum(is_synthetic=1) synthetic, sum(is_synthetic=0) human, avg(latency_ms) average_ms FROM calls").fetchone()
        with self.lock:
            queue_stats = dict(self.counts)
        return {"storage": "sqlite", "persisted": dict(row), "process_queue": queue_stats,
                "pending": self.queue.qsize(), "last_error": self.last_error,
                "score_definition": "p_synthetic is model probability of synthetic, not proof of fraud or calibrated confidence"}
