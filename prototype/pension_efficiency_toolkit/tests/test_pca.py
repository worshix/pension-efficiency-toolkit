"""Tests for pca_utils module."""

from __future__ import annotations

import numpy as np
import pytest

from pension_toolkit.pca_utils import run_pca, build_composite_input


class TestRunPca:
    def test_returns_pca_result(self):
        X = np.random.default_rng(0).random((10, 4))
        result = run_pca(X, n_components=3)
        assert result.components.shape == (10, 3)

    def test_explained_variance_sums_to_leq_one(self):
        X = np.random.default_rng(1).random((15, 5))
        result = run_pca(X, n_components=3)
        assert result.cumulative_variance[-1] <= 1.0 + 1e-9

    def test_explained_variance_is_descending(self):
        X = np.random.default_rng(2).random((20, 6))
        result = run_pca(X, n_components=4)
        evr = result.explained_variance_ratio
        assert all(evr[i] >= evr[i + 1] - 1e-9 for i in range(len(evr) - 1))

    def test_n_components_capped_at_features(self):
        X = np.random.default_rng(3).random((10, 2))
        result = run_pca(X, n_components=5)  # request more than available
        assert result.components.shape[1] <= 2

    def test_invalid_ndim_raises(self):
        X = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="2-D"):
            run_pca(X)

    def test_feature_names_stored(self):
        X = np.random.default_rng(4).random((8, 3))
        names = ["a", "b", "c"]
        result = run_pca(X, n_components=2, feature_names=names)
        assert result.feature_names == names


class TestBuildCompositeInput:
    def test_positive_values(self):
        X = np.random.default_rng(5).random((10, 4))
        result = run_pca(X, n_components=3)
        composite = build_composite_input(result, n_keep=2)
        assert (composite > 0).all()

    def test_shape(self):
        X = np.random.default_rng(6).random((12, 4))
        result = run_pca(X, n_components=3)
        composite = build_composite_input(result, n_keep=2)
        assert composite.shape == (12, 2)
