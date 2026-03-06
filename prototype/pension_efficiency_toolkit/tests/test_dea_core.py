"""Tests for dea_core module."""

from __future__ import annotations

import numpy as np
import pytest

from pension_toolkit.dea_core import dea_ccr_input_oriented, dea_bcc_input_oriented, DEAResult


def _simple_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """5-DMU toy dataset where DMU 0 is frontier-efficient."""
    X = np.array([
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 2.0],
        [5.0, 5.0],
        [3.5, 3.5],
    ])
    Y = np.array([
        [4.0, 2.0],
        [4.0, 2.0],
        [4.0, 3.0],
        [5.0, 3.0],
        [3.0, 1.5],
    ])
    ids = ["A", "B", "C", "D", "E"]
    return X, Y, ids


class TestCCR:
    def test_returns_dea_result(self):
        X, Y, ids = _simple_data()
        result = dea_ccr_input_oriented(X, Y, ids)
        assert isinstance(result, DEAResult)
        assert result.model == "CCR"

    def test_theta_shape(self):
        X, Y, ids = _simple_data()
        result = dea_ccr_input_oriented(X, Y, ids)
        assert result.theta.shape == (5,)

    def test_theta_in_range(self):
        X, Y, ids = _simple_data()
        result = dea_ccr_input_oriented(X, Y, ids)
        assert (result.theta >= 0).all()
        assert (result.theta <= 1.0 + 1e-6).all()

    def test_frontier_dmu_has_score_one(self):
        """An efficient frontier DMU should score 1.0."""
        # Construct a case where DMU 0 is clearly on the frontier
        X = np.array([[1.0], [2.0], [3.0]])
        Y = np.array([[2.0], [2.0], [2.0]])
        result = dea_ccr_input_oriented(X, Y)
        assert result.theta[0] == pytest.approx(1.0, abs=1e-4)

    def test_lambda_shape(self):
        X, Y, ids = _simple_data()
        result = dea_ccr_input_oriented(X, Y, ids)
        assert result.lambdas.shape == (5, 5)

    def test_slack_shapes(self):
        X, Y, ids = _simple_data()
        result = dea_ccr_input_oriented(X, Y, ids)
        assert result.slacks_in.shape == (5, 2)
        assert result.slacks_out.shape == (5, 2)

    def test_peer_ids_list(self):
        X, Y, ids = _simple_data()
        result = dea_ccr_input_oriented(X, Y, ids)
        assert len(result.peer_ids) == 5
        for peers in result.peer_ids:
            assert isinstance(peers, list)

    def test_too_few_dmus_raises(self):
        X = np.array([[1.0, 2.0]])
        Y = np.array([[3.0]])
        with pytest.raises(ValueError, match="at least 2"):
            dea_ccr_input_oriented(X, Y)

    def test_default_fund_ids(self):
        X, Y, _ = _simple_data()
        result = dea_ccr_input_oriented(X, Y)
        assert result.fund_ids[0].startswith("DMU")


class TestBCC:
    def test_bcc_theta_geq_ccr(self):
        """BCC scores should be >= CCR scores (VRS is more lenient)."""
        X, Y, ids = _simple_data()
        ccr = dea_ccr_input_oriented(X, Y, ids)
        bcc = dea_bcc_input_oriented(X, Y, ids)
        assert (bcc.theta >= ccr.theta - 1e-6).all()

    def test_model_label(self):
        X, Y, ids = _simple_data()
        result = dea_bcc_input_oriented(X, Y, ids)
        assert result.model == "BCC"

    def test_identical_dmus_handled(self):
        """Identical DMUs should not cause solver errors."""
        X = np.array([[2.0], [2.0], [2.0], [3.0], [4.0]])
        Y = np.array([[3.0], [3.0], [3.0], [4.0], [5.0]])
        result = dea_bcc_input_oriented(X, Y)
        assert result.theta is not None
