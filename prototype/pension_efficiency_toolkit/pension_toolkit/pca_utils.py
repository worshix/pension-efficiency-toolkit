"""PCA input validation utilities for DEA preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class PCAResult:
    """Container for PCA results."""

    pca: PCA
    scaler: StandardScaler
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    components: np.ndarray  # transformed data, shape (n_samples, n_components)
    feature_names: list[str]


def run_pca(
    X: np.ndarray,
    n_components: int = 3,
    feature_names: list[str] | None = None,
) -> PCAResult:
    """Fit PCA on input matrix X and return a PCAResult.

    The data is standardised (zero mean, unit variance) before PCA so that
    columns on different scales do not dominate.

    Parameters
    ----------
    X:
        Input matrix of shape (n_samples, n_features).
    n_components:
        Number of principal components to retain.
    feature_names:
        Optional list of column names for reporting.

    Returns
    -------
    PCAResult
        Fitted PCA object, scaler, explained variance, and transformed components.

    Raises
    ------
    ValueError
        If n_components exceeds the number of features.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    n_samples, n_features = X.shape
    n_components = min(n_components, n_features, n_samples)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)

    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    logger.info("PCA: %d components explain %.1f%% of variance", n_components, cumvar[-1] * 100)
    for i, (ev, cv) in enumerate(zip(evr, cumvar)):
        logger.info("  PC%d: %.2f%% (cumulative %.2f%%)", i + 1, ev * 100, cv * 100)

    return PCAResult(
        pca=pca,
        scaler=scaler,
        explained_variance_ratio=evr,
        cumulative_variance=cumvar,
        components=components,
        feature_names=feature_names,
    )


def build_composite_input(result: PCAResult, n_keep: int = 2) -> np.ndarray:
    """Return the first *n_keep* principal components as a composite input matrix.

    Parameters
    ----------
    result:
        Output of :func:`run_pca`.
    n_keep:
        Number of components to keep (must be <= result.components.shape[1]).

    Returns
    -------
    np.ndarray of shape (n_samples, n_keep).
    """
    n_keep = min(n_keep, result.components.shape[1])
    composite = result.components[:, :n_keep]
    # Shift so all values are positive (DEA requirement)
    col_mins = composite.min(axis=0)
    for i, col_min in enumerate(col_mins):
        if col_min <= 0:
            composite[:, i] = composite[:, i] - col_min + 1e-6
    return composite
