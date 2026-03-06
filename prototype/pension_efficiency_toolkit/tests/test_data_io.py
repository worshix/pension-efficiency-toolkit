"""Tests for data_io module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pension_toolkit.data_io import load_csv, get_dea_matrices, REQUIRED_COLUMNS


SAMPLE_CSV = Path(__file__).parent / "sample_data.csv"


class TestLoadCsv:
    def test_loads_sample_data(self):
        df = load_csv(SAMPLE_CSV)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_all_required_columns_present(self):
        df = load_csv(SAMPLE_CSV)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_numeric_columns_are_float(self):
        df = load_csv(SAMPLE_CSV)
        assert df["total_assets_usd"].dtype.kind == "f"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/data.csv")

    def test_missing_column_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("fund_id,year\nF001,2022\n")
            tmp_path = f.name
        with pytest.raises(ValueError, match="Missing required columns"):
            load_csv(tmp_path)

    def test_non_numeric_value_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            header = ",".join(REQUIRED_COLUMNS)
            # Put 'abc' in total_assets_usd
            row = "F001,2022,TestFund,private,abc,1000,2000,3000,4000,5.0,0.1,10"
            f.write(f"{header}\n{row}\n")
            tmp_path = f.name
        with pytest.raises(ValueError):
            load_csv(tmp_path)

    def test_missing_value_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            header = ",".join(REQUIRED_COLUMNS)
            row = "F001,2022,TestFund,private,,1000,2000,3000,4000,5.0,0.1,10"
            f.write(f"{header}\n{row}\n")
            tmp_path = f.name
        with pytest.raises(ValueError):
            load_csv(tmp_path)


class TestGetDeaMatrices:
    def test_returns_correct_shapes(self):
        df = load_csv(SAMPLE_CSV)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        df_agg = df.groupby("fund_id")[numeric_cols].mean().reset_index()
        X, Y, ids = get_dea_matrices(df_agg)
        assert X.ndim == 2
        assert Y.ndim == 2
        assert X.shape[0] == Y.shape[0] == len(ids)

    def test_fund_ids_are_strings(self):
        df = load_csv(SAMPLE_CSV)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        df_agg = df.groupby("fund_id")[numeric_cols].mean().reset_index()
        _, _, ids = get_dea_matrices(df_agg)
        assert all(isinstance(i, str) for i in ids)
