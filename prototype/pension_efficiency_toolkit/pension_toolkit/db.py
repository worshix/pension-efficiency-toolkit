"""SQLite persistence layer — fund data scoped per user, with user management."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

_DB_PATH = Path(__file__).parent.parent / "pension_data.db"

# user_id=0 is the admin sentinel (never stored in the users table)
ADMIN_USER_ID = 0


@dataclass
class UploadRecord:
    id: int
    filename: str
    uploaded_at: str
    n_rows: int
    n_funds: int


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name  TEXT NOT NULL,
            email      TEXT NOT NULL UNIQUE,
            password   TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS upload_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL DEFAULT 0,
            filename    TEXT    NOT NULL,
            uploaded_at TEXT    NOT NULL,
            n_rows      INTEGER NOT NULL,
            n_funds     INTEGER NOT NULL
        )
    """)

    # Migrate upload_history: add user_id column if it was created before this change.
    upload_cols = {r[1] for r in conn.execute("PRAGMA table_info(upload_history)").fetchall()}
    if "user_id" not in upload_cols:
        conn.execute("ALTER TABLE upload_history ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0")

    # Migrate fund_data: drop if it has no upload_id (old schema).
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "fund_data" in tables:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(fund_data)").fetchall()}
        if "upload_id" not in cols:
            conn.execute("DROP TABLE fund_data")

    conn.commit()


# ── Fund data ─────────────────────────────────────────────────────────────────

def save_fund_data(df: pd.DataFrame, filename: str, user_id: int) -> int:
    """Persist a new upload for the given user. Returns the new upload_id."""
    with _connect() as conn:
        _init_tables(conn)
        n_rows = len(df)
        n_funds = int(df["fund_id"].nunique())
        uploaded_at = datetime.now().isoformat(timespec="seconds")

        cur = conn.execute(
            "INSERT INTO upload_history (user_id, filename, uploaded_at, n_rows, n_funds) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, filename, uploaded_at, n_rows, n_funds),
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


def get_upload_history(user_id: int) -> list[UploadRecord]:
    """Return upload records for the given user, newest first."""
    try:
        with _connect() as conn:
            _init_tables(conn)
            rows = conn.execute(
                "SELECT id, filename, uploaded_at, n_rows, n_funds "
                "FROM upload_history WHERE user_id = ? ORDER BY id DESC",
                (user_id,),
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


def has_fund_data(user_id: int) -> bool:
    """Return True if the given user has any uploads."""
    try:
        with _connect() as conn:
            _init_tables(conn)
            return conn.execute(
                "SELECT COUNT(*) FROM upload_history WHERE user_id = ?", (user_id,)
            ).fetchone()[0] > 0
    except Exception:
        return False


# ── User management ───────────────────────────────────────────────────────────

def create_user(full_name: str, email: str, password: str) -> None:
    """Insert a new regular user. Raises ValueError if email already exists."""
    with _connect() as conn:
        _init_tables(conn)
        existing = conn.execute(
            "SELECT id FROM users WHERE LOWER(email) = LOWER(?)", (email,)
        ).fetchone()
        if existing:
            raise ValueError("An account with that email already exists.")
        conn.execute(
            "INSERT INTO users (full_name, email, password, created_at) VALUES (?, ?, ?, ?)",
            (full_name.strip(), email.strip().lower(), password,
             datetime.now().isoformat(timespec="seconds")),
        )


def get_user_by_email(email: str) -> dict | None:
    """Return user row as dict, or None if not found."""
    try:
        with _connect() as conn:
            _init_tables(conn)
            row = conn.execute(
                "SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),)
            ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def get_all_users() -> list[dict]:
    """Return all regular users, newest first."""
    try:
        with _connect() as conn:
            _init_tables(conn)
            rows = conn.execute(
                "SELECT id, full_name, email, created_at FROM users ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def delete_user(user_id: int) -> None:
    """Remove a user and all their uploads from the database."""
    with _connect() as conn:
        _init_tables(conn)
        # Remove their fund data rows first
        upload_ids = [
            r[0] for r in conn.execute(
                "SELECT id FROM upload_history WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
        for uid in upload_ids:
            conn.execute("DELETE FROM fund_data WHERE upload_id = ?", (uid,))
        conn.execute("DELETE FROM upload_history WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
