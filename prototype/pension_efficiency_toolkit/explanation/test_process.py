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
from pension_toolkit.dea_core import dea_ccr_input_oriented, dea_bcc_input_oriented
from pension_toolkit.scale import compute_scale_efficiency
from pension_toolkit.bootstrap import simar_wilson
from pension_toolkit.ml_stage import fit_rf, ENV_FEATURE_COLS

DATASET = "tests/sample_data.csv"
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
    """CSV has 20 rows, 10 unique funds, correct years, no duplicates, all self_administered."""
    df = pd.read_csv(DATASET)

    assert len(df) == 20, f"Expected 20 rows, got {len(df)}"
    assert df["fund_id"].nunique() == 10, "Expected 10 unique funds"
    assert sorted(df["year"].unique()) == [2021, 2022]
    assert df.duplicated(["fund_id", "year"]).sum() == 0, "Duplicate (fund_id, year) pairs"

    assert (df["fund_type"] == "self_administered").all(), \
        "All funds must be self_administered"

    for col in ["total_assets_usd", "operating_expenses_usd",
                "equity_debt_usd", "member_contributions_usd"]:
        assert (df[col] > 0).all(), f"{col} has non-positive values"


# ─── Stage 1: CCR DEA ─────────────────────────────────────────────────────────

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


# ─── Stage 2: BCC + Scale Efficiency ─────────────────────────────────────────

def test_bcc_dominance_and_scale():
    """BCC scores >= CCR; scale efficiency in (0,1]; RTS classes are valid."""
    _, X, Y, fund_ids = _load_aggregated()
    ccr = dea_ccr_input_oriented(X, Y, fund_ids)
    bcc = dea_bcc_input_oriented(X, Y, fund_ids)

    assert (bcc.theta >= ccr.theta - 1e-6).all(), "BCC < CCR violation"

    scale = compute_scale_efficiency(ccr, bcc)
    assert (scale.scale_efficiency > 0).all()
    assert (scale.scale_efficiency <= 1 + 1e-6).all()

    valid_rts = {"IRS", "DRS", "CRS"}
    assert all(r in valid_rts for r in scale.rts_class)


# ─── Stage 3: Input Reduction Targets ────────────────────────────────────────

def test_targets_file():
    """Target inputs are non-negative and do not exceed actual inputs."""
    targets_path = Path("out/targets.csv")
    if not targets_path.exists():
        import pytest
        pytest.skip("Run CLI pipeline first: uv run python -m pension_toolkit.cli analyze ...")

    targets = pd.read_csv(targets_path)
    df_agg, _, _, _ = _load_aggregated()

    # Skip if output was generated from a different dataset
    current_ids = set(df_agg["fund_id"].tolist())
    output_ids = set(targets["fund_id"].tolist())
    if not output_ids.issubset(current_ids):
        import pytest
        pytest.skip("Output files are from a different dataset — re-run CLI pipeline")

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


# ─── Stage 4: Bootstrap ───────────────────────────────────────────────────────

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
    assert np.isfinite(r1.bias).all(), "Non-finite bias values"


# ─── Stage 5: Derived Features + Random Forest ───────────────────────────────

def test_rf_second_stage():
    """Derived features compute correctly; RF importances sum to 1."""
    df_agg, X, Y, fund_ids = _load_aggregated()

    # Build bias-corrected scores with small B for speed
    def ccr_func(X_, Y_):
        return dea_ccr_input_oriented(X_, Y_, fund_ids)

    boot = simar_wilson(ccr_func, X, Y, fund_ids=fund_ids, B=50, seed=42)

    # Derive features (mirrors cli.py)
    ccr = dea_ccr_input_oriented(X, Y, fund_ids)
    bcc = dea_bcc_input_oriented(X, Y, fund_ids)
    scale = compute_scale_efficiency(ccr, bcc)
    rts_map = {"IRS": 1, "CRS": 0, "DRS": -1}

    df_agg["fund_size_log"] = np.log(df_agg["total_assets_usd"])
    df_agg["expense_ratio"] = (df_agg["operating_expenses_usd"]
                                / df_agg["total_assets_usd"])
    df_agg["rts_encoded"] = [rts_map.get(r, 0) for r in scale.rts_class]

    env_cols = [c for c in ENV_FEATURE_COLS if c in df_agg.columns]
    assert set(env_cols) == set(ENV_FEATURE_COLS), \
        f"Missing env cols: {set(ENV_FEATURE_COLS) - set(env_cols)}"

    env_features = df_agg[env_cols].values.astype(float)
    assert np.isfinite(env_features).all(), "Non-finite values in env features"

    result = fit_rf(boot.theta_bias_corrected, env_features,
                    feature_names=env_cols, seed=42)

    # Feature names are always present regardless of variance in target
    assert set(result.feature_importance["feature"]) == set(env_cols)
    assert all(f in env_cols for f in result.top3_features)

    # MDI importance sums to 1 only when the target has variance (degenerate
    # datasets with all-equal scores produce zero splits and zero importances)
    scores_var = np.var(boot.theta_bias_corrected)
    if scores_var > 1e-8:
        total_imp = result.feature_importance["importance"].sum()
        assert abs(total_imp - 1.0) < 1e-6, f"Importance sum = {total_imp:.6f}"


# ─── Stage 6: Output file completeness ───────────────────────────────────────

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
        except BaseException as exc:
            msg = str(exc)
            is_skip = (
                "skip" in msg.lower()
                or "run cli" in msg.lower()
                or type(exc).__name__ == "Skipped"
            )
            if is_skip:
                print(f"  SKIP  {name}  ({msg[:60]})")
                skipped += 1
            else:
                print(f"  FAIL  {name}  -- {exc}")
                failed += 1

    print(f"\n{passed} passed  {skipped} skipped  {failed} failed")
    sys.exit(1 if failed else 0)
