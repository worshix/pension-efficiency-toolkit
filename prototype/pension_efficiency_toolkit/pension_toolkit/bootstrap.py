"""Simar-Wilson bootstrap bias correction for DEA efficiency scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .utils import get_logger, make_rng

logger = get_logger(__name__)


@dataclass
class BootstrapResult:
    """Container for Simar-Wilson bootstrap results."""

    fund_ids: list[str]
    theta_raw: np.ndarray
    theta_bias_corrected: np.ndarray
    bias: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    B: int


def _bootstrap_iteration(
    b: int,
    X: np.ndarray,
    Y: np.ndarray,
    theta_hat: np.ndarray,
    dea_func: Callable,
    seed: int,
) -> np.ndarray:
    """Run a single bootstrap replication.

    Implements the Simar-Wilson (1998) Algorithm 1:
    1. Resample theta* from reflected kernel density of theta_hat
    2. Construct pseudo-dataset X* = theta* / theta_hat * X  (input-oriented)
    3. Compute DEA on (X*, Y) to get theta_hat_b*
    """
    rng = make_rng(seed + b)
    n = len(theta_hat)

    # --- Step 1: draw theta* from reflected kernel density ---
    h = _silverman_bandwidth(theta_hat)

    # Draw indices
    idx = rng.integers(0, n, size=n)
    theta_sampled = theta_hat[idx]

    # Add Gaussian noise and reflect at 1 (scores <= 1)
    noise = rng.normal(0, h, size=n)
    theta_star = theta_sampled + noise

    # Reflection: values > 1 are folded back
    theta_star = np.where(theta_star > 1, 2 - theta_star, theta_star)
    # Clip to valid range (0, 1]
    theta_star = np.clip(theta_star, 1e-6, 1.0)

    # --- Step 2: construct pseudo-dataset ---
    # Rescale inputs so that pseudo-DMU has efficiency theta_star
    scale_factors = theta_star / theta_hat  # shape (n,)
    X_star = X * scale_factors[:, np.newaxis]

    # --- Step 3: DEA on pseudo-dataset ---
    try:
        result_b = dea_func(X_star, Y)
        return result_b.theta
    except Exception:
        return theta_hat.copy()


def _silverman_bandwidth(theta: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth for 1-D KDE."""
    n = len(theta)
    std = theta.std(ddof=1)
    iqr = np.percentile(theta, 75) - np.percentile(theta, 25)
    s = min(std, iqr / 1.34)
    if s == 0:
        s = std if std > 0 else 0.01
    return 0.9 * s * n ** (-0.2)


def simar_wilson(
    dea_func: Callable,
    X: np.ndarray,
    Y: np.ndarray,
    fund_ids: list[str] | None = None,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    n_jobs: int = -1,
) -> BootstrapResult:
    """Simar-Wilson Algorithm 1 bootstrap bias correction.

    Parameters
    ----------
    dea_func:
        Callable(X, Y) -> DEAResult.  Must be a CCR or BCC solver.
    X:
        Input matrix, shape (n, m).
    Y:
        Output matrix, shape (n, s).
    fund_ids:
        Optional DMU identifiers.
    B:
        Number of bootstrap replications (default 2000; use 200 for CI/testing).
    alpha:
        Significance level for confidence intervals (default 0.05 → 95% CI).
    seed:
        Base RNG seed for reproducibility.
    n_jobs:
        Joblib parallelism (-1 = all cores).

    Returns
    -------
    BootstrapResult
    """
    n = X.shape[0]
    if fund_ids is None:
        fund_ids = [f"DMU{i}" for i in range(n)]

    logger.info("Running Simar-Wilson bootstrap: B=%d, alpha=%.2f", B, alpha)

    # Original DEA scores
    orig_result = dea_func(X, Y)
    theta_hat = orig_result.theta.copy()

    # Bootstrap replications (parallelised with joblib)
    boot_scores = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_bootstrap_iteration)(b, X, Y, theta_hat, dea_func, seed)
        for b in range(B)
    )
    boot_matrix = np.vstack(boot_scores)  # shape (B, n)

    # Bias = mean(theta_hat_b*) - theta_hat
    bias = boot_matrix.mean(axis=0) - theta_hat

    # Bias-corrected scores
    theta_bc = theta_hat - bias
    theta_bc = np.clip(theta_bc, 0.0, 1.0)

    # Confidence intervals from bootstrap distribution
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    ci_lower = np.percentile(boot_matrix, lo, axis=0)
    ci_upper = np.percentile(boot_matrix, hi, axis=0)

    logger.info(
        "Bootstrap complete. Mean bias=%.4f, Mean bias-corrected=%.4f",
        bias.mean(),
        theta_bc.mean(),
    )

    return BootstrapResult(
        fund_ids=fund_ids,
        theta_raw=theta_hat,
        theta_bias_corrected=theta_bc,
        bias=bias,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        B=B,
    )


def bootstrap_to_dataframe(result: BootstrapResult) -> pd.DataFrame:
    """Convert BootstrapResult to a DataFrame for export."""
    return pd.DataFrame(
        {
            "fund_id": result.fund_ids,
            "theta_raw": result.theta_raw,
            "theta_bias_corrected": result.theta_bias_corrected,
            "bias": result.bias,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
        }
    )
