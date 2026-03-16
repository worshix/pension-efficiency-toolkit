"""Random Forest second-stage determinant analysis of DEA efficiency scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import cross_val_score

from .utils import get_logger, ensure_dir, make_rng

logger = get_logger(__name__)

ENV_FEATURE_COLS = [
    "inflation",
    "exchange_volatility",
    "fund_age",
    "fund_size_log",
    "expense_ratio",
    "rts_encoded",
]


@dataclass
class RFResult:
    """Container for Random Forest second-stage results."""

    model: RandomForestRegressor
    feature_names: list[str]
    feature_importance: pd.DataFrame        # name, importance columns
    permutation_importance: pd.DataFrame    # name, mean, std columns
    cv_r2_scores: np.ndarray
    top3_features: list[str] = field(default_factory=list)


def fit_rf(
    target_scores: np.ndarray,
    env_features: np.ndarray,
    feature_names: list[str] | None = None,
    n_estimators: int = 500,
    max_features: str = "sqrt",
    seed: int = 42,
    cv_folds: int = 5,
) -> RFResult:
    """Fit a Random Forest regressor to explain DEA efficiency scores.

    Parameters
    ----------
    target_scores:
        Array of DEA efficiency (or bias-corrected) scores, shape (n,).
    env_features:
        Environmental/contextual variables, shape (n, p).
    feature_names:
        Optional list of feature column names.
    n_estimators:
        Number of trees (default 500).
    max_features:
        Number of features to consider at each split (default "sqrt").
    seed:
        RNG seed for reproducibility.
    cv_folds:
        Number of cross-validation folds.

    Returns
    -------
    RFResult
    """
    rng = make_rng(seed)
    n_samples = env_features.shape[0]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(env_features.shape[1])]

    logger.info(
        "Fitting RandomForest: n=%d, n_estimators=%d, max_features=%s",
        n_samples,
        n_estimators,
        max_features,
    )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=int(rng.integers(0, 2**31)),
        n_jobs=-1,
    )
    rf.fit(env_features, target_scores)

    # Feature importance (MDI)
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Permutation importance
    perm = permutation_importance(
        rf,
        env_features,
        target_scores,
        n_repeats=30,
        random_state=seed,
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_importance": perm.importances_mean,
            "std_importance": perm.importances_std,
        }
    ).sort_values("mean_importance", ascending=False)

    # Cross-validation R² (small dataset: use LOO or 5-fold)
    n_cv = min(cv_folds, n_samples)
    cv_scores = cross_val_score(rf, env_features, target_scores, cv=n_cv, scoring="r2")
    logger.info("CV R²: mean=%.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    top3 = fi_df["feature"].head(3).tolist()

    return RFResult(
        model=rf,
        feature_names=feature_names,
        feature_importance=fi_df,
        permutation_importance=perm_df,
        cv_r2_scores=cv_scores,
        top3_features=top3,
    )


def plot_pdp(
    result: RFResult,
    env_features: np.ndarray,
    out_dir: str | Path = "out",
) -> Path:
    """Generate partial dependence plots for the top 3 features.

    Parameters
    ----------
    result:
        Output of :func:`fit_rf`.
    env_features:
        The same feature matrix used for fitting.
    out_dir:
        Directory to save the PNG.

    Returns
    -------
    Path to saved PNG file.
    """
    import matplotlib.pyplot as plt

    out_dir = ensure_dir(out_dir)
    out_path = out_dir / "pdp_top3.png"

    top3_idx = [result.feature_names.index(f) for f in result.top3_features]
    n_plots = len(top3_idx)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    disp = PartialDependenceDisplay.from_estimator(
        result.model,
        env_features,
        features=top3_idx,
        feature_names=result.feature_names,
        ax=axes,
        kind="average",
    )
    fig.suptitle("Partial Dependence Plots — Top 3 Efficiency Drivers", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PDP plot to %s", out_path)
    return out_path
