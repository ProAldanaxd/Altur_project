"""Explicit idempotent export of local audit events to PostgreSQL/Tiger Data.

Requires DATABASE_URL and requirements-postgres.txt. Never runs from /detect.
"""
import argparse
import os
import sqlite3

from dev4.audit import COLUMNS

DDL = """CREATE TABLE IF NOT EXISTS altur_calls (
request_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, status_code INTEGER NOT NULL,
is_synthetic BOOLEAN, p_synthetic DOUBLE PRECISION, latency_ms DOUBLE PRECISION NOT NULL,
duration_s DOUBLE PRECISION, model_variant TEXT, model_sha256 TEXT)"""


def sync_batch(path, connect=None, limit=500):
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return {"status": "unavailable", "reason": "missing_database_url"}
    if connect is None:
        try:
            import psycopg
        except ImportError:
            return {"status": "unavailable", "reason": "install_requirements_postgres"}
        connect = psycopg.connect
    local = sqlite3.connect(path, timeout=2)
    try:
        local.row_factory = sqlite3.Row
        try:
            with local:
                rows = local.execute("SELECT * FROM calls WHERE exported=0 ORDER BY created_at LIMIT ?", (min(max(limit, 1), 1000),)).fetchall()
        except sqlite3.OperationalError:
            # Fresh or uninitialized local database: nothing has been audited yet.
            return {"status": "unavailable", "reason": "audit_db_not_initialized"}
        if not rows:
            return {"status": "ok", "exported": 0}
        try:
            with connect(dsn, connect_timeout=5, options="-c statement_timeout=5000") as remote:
                remote.execute(DDL)
                for row in rows:
                    values = [row[key] for key in COLUMNS]
                    values[3] = bool(values[3]) if values[3] is not None else None
                    remote.execute("INSERT INTO altur_calls (" + ",".join(COLUMNS) + ") VALUES (" + ",".join("%s" for _ in COLUMNS) + ") ON CONFLICT (request_id) DO NOTHING", values)
        except Exception:
            # Remote transaction did not confirm success; keep events pending.
            return {"status": "unavailable", "reason": "postgres_sync_failed", "retryable": True}
        with local:
            local.executemany("UPDATE calls SET exported=1 WHERE request_id=?", [(row["request_id"],) for row in rows])
    finally:
        local.close()
    return {"status": "ok", "exported": len(rows)}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/audit.sqlite3")
    args = parser.parse_args()
    print(json.dumps(sync_batch(args.db_path)))
