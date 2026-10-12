"""SQLite engine for the coordinator's own database (users, epsilon ledger, audit log).

Every transaction starts with BEGIN IMMEDIATE: the writer takes the database lock up front, so
"read the last audit entry, then append the next one" can never interleave with another writer - no race, no
forked hash chain. (SQLAlchemy recipe: switch pysqlite to autocommit and emit our own BEGIN.)
"""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine


def make_sqlite_engine(path: str) -> Engine:
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):  # noqa: ANN001
        dbapi_conn.isolation_level = None            # we issue BEGIN ourselves
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        dbapi_conn.execute("PRAGMA synchronous=NORMAL")

    @event.listens_for(engine, "begin")
    def _on_begin(conn):  # noqa: ANN001
        conn.exec_driver_sql("BEGIN IMMEDIATE")

    return engine
