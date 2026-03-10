"""Tests for bootstrap module."""

from __future__ import annotations

import numpy as np
import pytest

from pension_toolkit.dea_core import dea_ccr_input_oriented
from pension_toolkit.bootstrap import simar_wilson, BootstrapResult, bootstrap_to_dataframe


def _sample_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(99)
    n = 8
    X = rng.uniform(1, 5, (n, 2))
    Y = rng.uniform(2, 6, (n, 2))
    ids = [f"F{i:02d}" for i in range(n)]
    return X, Y, ids


def _ccr_func(X, Y):
    return dea_ccr_input_oriented(X, Y)


class TestSimarWilson:
    def test_returns_bootstrap_result(self):
        X, Y, ids = _sample_data()
        result = simar_wilson(_ccr_func, X, Y, fund_ids=ids, B=50, seed=0)
        assert isinstance(result, BootstrapResult)

    def test_shapes(self):
        X, Y, ids = _sample_data()
        n = X.shape[0]
        result = simar_wilson(_ccr_func, X, Y, fund_ids=ids, B=50, seed=0)
        assert result.theta_raw.shape == (n,)
        assert result.theta_bias_corrected.shape == (n,)
        assert result.ci_lower.shape == (n,)
        assert result.ci_upper.shape == (n,)

    def test_bias_corrected_in_range(self):
        X, Y, ids = _sample_data()
        result = simar_wilson(_ccr_func, X, Y, fund_ids=ids, B=50, seed=0)
        assert (result.theta_bias_corrected >= 0).all()
        assert (result.theta_bias_corrected <= 1.0 + 1e-6).all()

    def test_ci_lower_leq_upper(self):
        X, Y, ids = _sample_data()
        result = simar_wilson(_ccr_func, X, Y, fund_ids=ids, B=50, seed=0)
        assert (result.ci_lower <= result.ci_upper + 1e-9).all()

    def test_reproducibility(self):
        X, Y, ids = _sample_data()
        r1 = simar_wilson(_ccr_func, X, Y, fund_ids=ids, B=50, seed=42)
        r2 = simar_wilson(_ccr_func, X, Y, fund_ids=ids, B=50, seed=42)
        np.testing.assert_array_almost_equal(r1.theta_bias_corrected, r2.theta_bias_corrected)

    def test_to_dataframe(self):
        X, Y, ids = _sample_data()
        result = simar_wilson(_ccr_func, X, Y, fund_ids=ids, B=50, seed=0)
        df = bootstrap_to_dataframe(result)
        assert "fund_id" in df.columns
        assert "theta_bias_corrected" in df.columns
        assert "ci_lower" in df.columns
        assert len(df) == X.shape[0]
