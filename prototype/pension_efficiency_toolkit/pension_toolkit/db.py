"""SQLite persistence layer — stores fund data with full upload history."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

_DB_PATH = Path(__file__).parent.parent / "pension_data.db"


@dataclass
class UploadRecord:
    id: int
    filename: str
    uploaded_at: str
    n_rows: int
    n_funds: int


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create tables if missing; migrate old fund_data schema if needed."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS upload_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT    NOT NULL,
            uploaded_at TEXT    NOT NULL,
            n_rows      INTEGER NOT NULL,
            n_funds     INTEGER NOT NULL
        )
    """)

    # If fund_data exists but has no upload_id column, drop it (old schema migration).
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "fund_data" in tables:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(fund_data)").fetchall()}
        if "upload_id" not in cols:
            conn.execute("DROP TABLE fund_data")

    conn.commit()


def save_fund_data(df: pd.DataFrame, filename: str) -> int:
    """Persist a new upload. Returns the new upload_id."""
    with _connect() as conn:
        _init_tables(conn)
        n_rows = len(df)
        n_funds = int(df["fund_id"].nunique())
        uploaded_at = datetime.now().isoformat(timespec="seconds")

        cur = conn.execute(
            "INSERT INTO upload_history (filename, uploaded_at, n_rows, n_funds) VALUES (?, ?, ?, ?)",
            (filename, uploaded_at, n_rows, n_funds),
        )
        upload_id = cur.lastrowid

        df_to_save = df.copy()
        df_to_save["upload_id"] = upload_id
        df_to_save.to_sql("fund_data", conn, if_exists="append", index=False)

        return upload_id


def load_fund_data(upload_id: int) -> pd.DataFrame | None:
    """Load fund data for a specific upload. Returns None if not found."""
    try:
        with _connect() as conn:
            _init_tables(conn)
            df = pd.read_sql(
                "SELECT * FROM fund_data WHERE upload_id = ?",
                conn,
                params=(upload_id,),
            )
        if df.empty:
            return None
        return df.drop(columns=["upload_id"], errors="ignore")
    except Exception:
        return None


def get_upload_history() -> list[UploadRecord]:
    """Return all upload records, newest first."""
    try:
        with _connect() as conn:
            _init_tables(conn)
            rows = conn.execute(
                "SELECT id, filename, uploaded_at, n_rows, n_funds "
                "FROM upload_history ORDER BY id DESC"
            ).fetchall()
        return [
            UploadRecord(id=r[0], filename=r[1], uploaded_at=r[2], n_rows=r[3], n_funds=r[4])
            for r in rows
        ]
    except Exception:
        return []


def delete_upload(upload_id: int) -> None:
    """Delete one upload and all its fund data rows."""
    with _connect() as conn:
        _init_tables(conn)
        conn.execute("DELETE FROM fund_data WHERE upload_id = ?", (upload_id,))
        conn.execute("DELETE FROM upload_history WHERE id = ?", (upload_id,))


def has_fund_data() -> bool:
    """Return True if any uploads exist."""
    try:
        with _connect() as conn:
            _init_tables(conn)
            return conn.execute("SELECT COUNT(*) FROM upload_history").fetchone()[0] > 0
    except Exception:
        return False
