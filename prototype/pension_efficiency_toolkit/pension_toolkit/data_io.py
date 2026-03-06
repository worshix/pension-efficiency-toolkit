"""CSV ingestion and validation for pension fund data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS: list[str] = [
    "fund_id",
    "year",
    "fund_name",
    "fund_type",
    "total_assets_usd",
    "operating_expenses_usd",
    "equity_debt_usd",
    "net_investment_income_usd",
    "member_contributions_usd",
    "inflation",
    "exchange_volatility",
    "fund_age",
]

NUMERIC_COLUMNS: list[str] = [
    "total_assets_usd",
    "operating_expenses_usd",
    "equity_debt_usd",
    "net_investment_income_usd",
    "member_contributions_usd",
    "inflation",
    "exchange_volatility",
    "fund_age",
    "year",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a pension fund CSV file.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with correct column types.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If schema validation fails or data issues are found.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    logger.info("Loading CSV from %s", p)
    df = pd.read_csv(p)

    _validate_schema(df)
    df = _coerce_numerics(df)
    _check_missing(df)
    _check_positivity(df)

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to float, raise on failure."""
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Column '{col}' contains non-numeric values: {exc}") from exc
    return df


def _check_missing(df: pd.DataFrame) -> None:
    """Raise ValueError if any numeric column contains NaN."""
    for col in NUMERIC_COLUMNS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            raise ValueError(f"Column '{col}' has {n_missing} missing values.")


def _check_positivity(df: pd.DataFrame) -> None:
    """Warn if financial columns contain non-positive values (DEA requires >0)."""
    financial_cols = [
        "total_assets_usd",
        "operating_expenses_usd",
        "equity_debt_usd",
        "net_investment_income_usd",
        "member_contributions_usd",
    ]
    for col in financial_cols:
        if (df[col] <= 0).any():
            logger.warning("Column '%s' contains non-positive values; DEA requires positive inputs/outputs.", col)


def get_dea_matrices(
    df: pd.DataFrame,
    input_cols: list[str] | None = None,
    output_cols: list[str] | None = None,
) -> tuple[Any, Any, list[str]]:
    """Extract DEA input/output matrices from the dataframe.

    Parameters
    ----------
    df:
        Validated DataFrame from :func:`load_csv`.
    input_cols:
        Column names to use as DEA inputs. Defaults to operating_expenses_usd.
    output_cols:
        Column names to use as DEA outputs. Defaults to net_investment_income_usd
        and member_contributions_usd.

    Returns
    -------
    tuple of (X_in, Y_out, fund_ids)
        X_in shape (n, m), Y_out shape (n, s), fund_ids list of length n.
    """
    import numpy as np

    if input_cols is None:
        input_cols = ["operating_expenses_usd", "total_assets_usd", "equity_debt_usd"]
    if output_cols is None:
        output_cols = ["net_investment_income_usd", "member_contributions_usd"]

    X_in = df[input_cols].to_numpy(dtype=float)
    Y_out = df[output_cols].to_numpy(dtype=float)
    fund_ids = df["fund_id"].tolist()
    return X_in, Y_out, fund_ids
