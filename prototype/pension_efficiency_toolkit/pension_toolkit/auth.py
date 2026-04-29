"""Authentication: admin from env vars, regular users from SQLite."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).parent.parent / ".env"


def _load_env() -> None:
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env()


def _admin_email() -> str:
    return os.getenv("MANAGER_EMAIL", "h220021y@hit.ac.zw").strip().lower()


def _admin_password() -> str:
    return os.getenv("MANAGER_PASSWORD", "lorraine")


def _admin_name() -> str:
    return os.getenv("MANAGER_NAME", "Lorraine Mujuru")


def is_admin_email(email: str) -> bool:
    return email.strip().lower() == _admin_email()


def login(email: str, password: str) -> tuple[bool, str, bool]:
    """Attempt login. Returns (success, display_name, is_admin)."""
    email = email.strip()

    # Admin check first (env vars, never in DB)
    if is_admin_email(email):
        if password == _admin_password():
            return True, _admin_name(), True
        return False, "", False

    # Regular user check (SQLite)
    from pension_toolkit.db import get_user_by_email
    user = get_user_by_email(email)
    if user and user["password"] == password:
        return True, user["full_name"], False
    return False, "", False


# Keep backward-compatible helper used elsewhere in the codebase
def check_credentials(email: str, password: str) -> bool:
    success, _, _ = login(email, password)
    return success


def get_manager_name() -> str:
    return _admin_name()
