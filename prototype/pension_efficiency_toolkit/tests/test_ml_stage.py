"""Tests for ml_stage module — Random Forest second-stage analysis."""

from __future__ import annotations

import numpy as np
import pytest

from pension_toolkit.ml_stage import fit_rf, ENV_FEATURE_COLS, RFResult


def _sample_rf_data(n: int = 12, seed: int = 7) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic efficiency scores and 6 environmental features."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.5, 1.0, n)
    # 6 features matching ENV_FEATURE_COLS order:
    # inflation, exchange_volatility, fund_age, fund_size_log, expense_ratio, rts_encoded
    features = np.column_stack([
        rng.uniform(5, 300, n),      # inflation
        rng.uniform(0.1, 0.9, n),    # exchange_volatility
        rng.integers(5, 70, n),      # fund_age
        rng.uniform(15, 21, n),      # fund_size_log (log of assets)
        rng.uniform(0.01, 0.05, n),  # expense_ratio
        rng.choice([-1, 0, 1], n),   # rts_encoded
    ])
    feature_names = list(ENV_FEATURE_COLS)
    return scores, features, feature_names


class TestFitRF:
    def test_returns_rf_result(self):
        scores, features, names = _sample_rf_data()
        result = fit_rf(scores, features, feature_names=names, seed=0)
        assert isinstance(result, RFResult)

    def test_feature_importance_has_all_features(self):
        scores, features, names = _sample_rf_data()
        result = fit_rf(scores, features, feature_names=names, seed=0)
        assert set(result.feature_importance["feature"]) == set(names)

    def test_feature_importance_sums_to_one(self):
        scores, features, names = _sample_rf_data()
        result = fit_rf(scores, features, feature_names=names, seed=0)
        total = result.feature_importance["importance"].sum()
        assert abs(total - 1.0) < 1e-6

    def test_permutation_importance_shape(self):
        scores, features, names = _sample_rf_data()
        result = fit_rf(scores, features, feature_names=names, seed=0)
        assert len(result.permutation_importance) == len(names)

    def test_top3_features_are_subset(self):
        scores, features, names = _sample_rf_data()
        result = fit_rf(scores, features, feature_names=names, seed=0)
        assert all(f in names for f in result.top3_features)
        assert len(result.top3_features) == 3

    def test_cv_r2_scores_shape(self):
        scores, features, names = _sample_rf_data()
        result = fit_rf(scores, features, feature_names=names, cv_folds=3, seed=0)
        assert result.cv_r2_scores.shape == (3,)

    def test_reproducibility(self):
        scores, features, names = _sample_rf_data()
        r1 = fit_rf(scores, features, feature_names=names, seed=99)
        r2 = fit_rf(scores, features, feature_names=names, seed=99)
        np.testing.assert_array_almost_equal(
            r1.feature_importance["importance"].values,
            r2.feature_importance["importance"].values,
        )

    def test_env_feature_cols_constant(self):
        """ENV_FEATURE_COLS must include all 6 determinant variables."""
        expected = {
            "inflation", "exchange_volatility", "fund_age",
            "fund_size_log", "expense_ratio", "rts_encoded",
        }
        assert set(ENV_FEATURE_COLS) == expected
