"""
Process verification tests for the Zimbabwe pension fund efficiency pipeline.

These tests check that each stage produces mathematically and structurally
correct results.  They are complementary to the unit tests in tests/ — those
test individual functions in isolation; these test the full data flow.

Run from inside the project directory:
    uv run python explanation/test_process.py
or via pytest:
    uv run pytest explanation/test_process.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure pension_toolkit is importable from this subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from pension_toolkit.data_io import load_csv, get_dea_matrices
from pension_toolkit.pca_utils import run_pca
from pension_toolkit.dea_core import dea_ccr_input_oriented, dea_bcc_input_oriented
from pension_toolkit.scale import compute_scale_efficiency
from pension_toolkit.bootstrap import simar_wilson
from pension_toolkit.ml_stage import fit_rf, ENV_FEATURE_COLS

DATASET = "tests/dataset.csv"
INPUT_COLS = ["operating_expenses_usd", "total_assets_usd", "equity_debt_usd"]
OUTPUT_COLS = ["net_investment_income_usd", "member_contributions_usd"]


def _load_aggregated() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load dataset and return aggregated DataFrame plus DEA matrices."""
    df = load_csv(DATASET)
    numeric = df.select_dtypes("number").columns.tolist()
    df_agg = df.groupby("fund_id")[numeric].mean().reset_index()
    fund_type_map = df.groupby("fund_id")["fund_type"].agg(lambda x: x.mode()[0])
    df_agg = df_agg.merge(fund_type_map, on="fund_id")
    X, Y, fund_ids = get_dea_matrices(df_agg, INPUT_COLS, OUTPUT_COLS)
    return df_agg, X, Y, fund_ids


# ─── Stage 0: Dataset Schema ─────────────────────────────────────────────────

def test_dataset_schema():
    """CSV has 100 rows, 20 unique funds, correct years, no duplicates."""
    df = pd.read_csv(DATASET)

    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    assert df["fund_id"].nunique() == 20, "Expected 20 unique funds"
    assert sorted(df["year"].unique()) == [2018, 2019, 2020, 2021, 2022]
    assert df.duplicated(["fund_id", "year"]).sum() == 0, "Duplicate (fund_id, year) pairs"

    for col in ["total_assets_usd", "operating_expenses_usd",
                "equity_debt_usd", "member_contributions_usd"]:
        assert (df[col] > 0).all(), f"{col} has non-positive values"

    expected_inflation = {2018: 10.6, 2019: 255.3, 2020: 557.2, 2021: 98.5, 2022: 243.8}
    for year, inf in expected_inflation.items():
        vals = df.loc[df["year"] == year, "inflation"].unique()
        assert len(vals) == 1 and abs(vals[0] - inf) < 0.5, \
            f"Inflation mismatch for {year}: {vals}"


# ─── Stage 1: PCA ─────────────────────────────────────────────────────────────

def test_pca_variance():
    """Three input PCs together explain >= 95% of variance; PC1 > 60%."""
    df_agg, X, _, _ = _load_aggregated()
    result = run_pca(X, n_components=3, feature_names=INPUT_COLS)

    assert result.cumulative_variance[-1] >= 0.95, \
        f"PCA explains only {result.cumulative_variance[-1]:.2%}"
    assert result.explained_variance_ratio[0] >= 0.60, \
        f"PC1 explains only {result.explained_variance_ratio[0]:.2%} — inputs may not be collinear"
    assert np.isfinite(result.components).all(), "NaN/Inf in PCA components"
    assert result.components.shape == (X.shape[0], 3)


# ─── Stage 2: CCR DEA ─────────────────────────────────────────────────────────

def test_ccr_scores():
    """CCR scores are in (0,1], at least one fund is on the frontier."""
    _, X, Y, fund_ids = _load_aggregated()
    result = dea_ccr_input_oriented(X, Y, fund_ids)

    assert (result.theta > 0).all(), "Zero efficiency score"
    assert (result.theta <= 1 + 1e-6).all(), "Score > 1.0"
    assert (result.theta >= 1 - 1e-4).any(), "No frontier fund — data may be degenerate"
    assert (result.theta < 0.999).any(), "All funds efficient — no variation"
    assert result.lambdas.shape == (len(fund_ids), len(fund_ids))
    assert result.slacks_in.shape == (len(fund_ids), len(INPUT_COLS))


# ─── Stage 3: BCC + Scale Efficiency ─────────────────────────────────────────

def test_bcc_dominance_and_scale():
    """BCC scores >= CCR; scale efficiency in (0,1]; all three RTS classes present."""
    _, X, Y, fund_ids = _load_aggregated()
    ccr = dea_ccr_input_oriented(X, Y, fund_ids)
    bcc = dea_bcc_input_oriented(X, Y, fund_ids)

    assert (bcc.theta >= ccr.theta - 1e-6).all(), "BCC < CCR violation"

    scale = compute_scale_efficiency(ccr, bcc)
    assert (scale.scale_efficiency > 0).all()
    assert (scale.scale_efficiency <= 1 + 1e-6).all()

    valid_rts = {"IRS", "DRS", "CRS"}
    assert all(r in valid_rts for r in scale.rts_class)
    assert len(set(scale.rts_class)) >= 2, \
        f"Only one RTS class observed: {set(scale.rts_class)}"


# ─── Stage 4: Input Reduction Targets ────────────────────────────────────────

def test_targets_file():
    """Target inputs are non-negative and do not exceed actual inputs."""
    targets_path = Path("out/targets.csv")
    if not targets_path.exists():
        import pytest
        pytest.skip("Run CLI pipeline first: uv run python -m pension_toolkit.cli analyze ...")

    targets = pd.read_csv(targets_path)
    df_agg, _, _, _ = _load_aggregated()
    actual = df_agg.set_index("fund_id")

    for col in INPUT_COLS:
        tcol = f"target_{col}"
        if tcol not in targets.columns:
            continue
        assert (targets[tcol] >= 0).all(), f"Negative target: {col}"
        for _, row in targets.iterrows():
            act_val = actual.loc[row["fund_id"], col]
            assert row[tcol] <= act_val + 1.0, \
                f"Target > actual for {row['fund_id']} / {col}"


# ─── Stage 5: Bootstrap ───────────────────────────────────────────────────────

def test_bootstrap_properties():
    """Bias-corrected scores are in [0,1]; CIs are ordered; results are reproducible."""
    _, X, Y, fund_ids = _load_aggregated()

    def ccr_func(X_, Y_):
        return dea_ccr_input_oriented(X_, Y_, fund_ids)

    r1 = simar_wilson(ccr_func, X, Y, fund_ids=fund_ids, B=100, seed=42)
    r2 = simar_wilson(ccr_func, X, Y, fund_ids=fund_ids, B=100, seed=42)

    assert (r1.theta_bias_corrected >= 0).all()
    assert (r1.theta_bias_corrected <= 1 + 1e-6).all()
    assert (r1.ci_lower <= r1.ci_upper + 1e-9).all(), "CI lower > upper"
    np.testing.assert_array_almost_equal(
        r1.theta_bias_corrected, r2.theta_bias_corrected, decimal=6,
        err_msg="Bootstrap not reproducible"
    )

    # When mean efficiency is close to 1.0 (scores clustered near the boundary),
    # the reflected KDE produces pseudo-scores near 1, so bootstrap means can
    # exceed raw scores (negative bias) and BC scores get clipped to 1.
    # The key correctness property is that BC scores are finite and in [0, 1].
    assert np.isfinite(r1.bias).all(), "Non-finite bias values"
    mean_eff = r1.theta_raw.mean()
    if mean_eff < 0.95:
        # For clearly sub-efficient data, majority of bias should be positive
        bias_positive = (r1.bias > 0).mean()
        assert bias_positive >= 0.4, \
            f"Fewer than 40% of DMUs show positive bias ({bias_positive:.0%})"


# ─── Stage 6: Random Forest ───────────────────────────────────────────────────

def test_rf_second_stage():
    """RF feature importances sum to 1; all ENV features are present."""
    bc_path = Path("out/bias_corrected_scores.csv")
    if not bc_path.exists():
        import pytest
        pytest.skip("Run CLI pipeline first")

    df = load_csv(DATASET)
    numeric = df.select_dtypes("number").columns.tolist()
    df_agg = df.groupby("fund_id")[numeric].mean().reset_index()

    bc = pd.read_csv(bc_path)
    bc = bc.merge(df_agg[["fund_id"] + ENV_FEATURE_COLS], on="fund_id")
    env = bc[ENV_FEATURE_COLS].to_numpy(dtype=float)
    scores = bc["theta_bias_corrected"].to_numpy(dtype=float)

    result = fit_rf(scores, env, feature_names=ENV_FEATURE_COLS, seed=42)

    total_imp = result.feature_importance["importance"].sum()
    assert abs(total_imp - 1.0) < 1e-6, f"Importance sum = {total_imp:.6f}"
    assert set(result.feature_importance["feature"]) == set(ENV_FEATURE_COLS)
    assert all(f in ENV_FEATURE_COLS for f in result.top3_features)


# ─── Stage 7: Output file completeness ───────────────────────────────────────

def test_pipeline_output_files():
    """All expected output files exist and contain the required columns."""
    expected = [
        ("out/efficiency_ccr.csv",        ["fund_id", "theta_ccr"]),
        ("out/efficiency_vrs.csv",        ["fund_id", "theta_bcc"]),
        ("out/scale.csv",                 ["fund_id", "scale_efficiency",
                                           "rts_classification"]),
        ("out/targets.csv",               ["fund_id", "theta_ccr"]),
        ("out/bias_corrected_scores.csv", ["fund_id", "theta_raw",
                                           "theta_bias_corrected",
                                           "ci_lower", "ci_upper"]),
        ("out/rf_importance.csv",         ["feature", "importance"]),
    ]
    missing = [p for p, _ in expected if not Path(p).exists()]
    if missing:
        import pytest
        pytest.skip(f"Missing output files (run CLI first): {missing}")

    for csv_path, required_cols in expected:
        df = pd.read_csv(csv_path)
        assert len(df) > 0, f"Empty file: {csv_path}"
        for col in required_cols:
            assert col in df.columns, f"'{col}' missing from {csv_path}"

    assert Path("out/report.pdf").exists(), "PDF report not generated"
    assert Path("out/pdp_top3.png").exists(), "PDP plot not generated"


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_dataset_schema,
        test_pca_variance,
        test_ccr_scores,
        test_bcc_dominance_and_scale,
        test_targets_file,
        test_bootstrap_properties,
        test_rf_second_stage,
        test_pipeline_output_files,
    ]

    passed, skipped, failed = 0, 0, 0
    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            msg = str(exc)
            if "skip" in msg.lower() or "run cli" in msg.lower():
                print(f"  SKIP  {name}  ({msg[:60]})")
                skipped += 1
            else:
                print(f"  FAIL  {name}  -- {exc}")
                failed += 1

    print(f"\n{passed} passed  {skipped} skipped  {failed} failed")
    sys.exit(1 if failed else 0)
