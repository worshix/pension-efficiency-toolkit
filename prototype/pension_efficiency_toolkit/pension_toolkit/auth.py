"""Single-user authentication backed by environment variables."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).parent.parent / ".env"


def _load_env() -> None:
    """Load .env file into os.environ (no external dependency required)."""
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env()


def check_credentials(email: str, password: str) -> bool:
    expected_email = os.getenv("MANAGER_EMAIL", "h220021hit.ac.zw")
    expected_password = os.getenv("MANAGER_PASSWORD", "lorraine")
    return email.strip().lower() == expected_email.lower() and password == expected_password


def get_manager_name() -> str:
    return os.getenv("MANAGER_NAME", "Lorraine Mujuru")
