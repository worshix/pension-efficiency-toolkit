"""Command-line interface for the Pension Efficiency Toolkit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import load_csv, get_dea_matrices
from .dea_core import dea_ccr_input_oriented, dea_bcc_input_oriented
from .scale import compute_scale_efficiency, scale_to_dataframe
from .bootstrap import simar_wilson, bootstrap_to_dataframe
from .ml_stage import fit_rf, plot_pdp, ENV_FEATURE_COLS
from .reporting import generate_pdf_report
from .utils import ensure_dir, get_logger

logger = get_logger(__name__)


def _build_ccr_dataframe(result, input_cols: list[str]) -> pd.DataFrame:
    """Build the CCR output DataFrame."""
    data: dict = {
        "fund_id": result.fund_ids,
        "theta_ccr": result.theta,
    }
    for i, col in enumerate(input_cols):
        data[f"slack_{col}"] = result.slacks_in[:, i]
    data["peer_ids"] = ["|".join(p) for p in result.peer_ids]
    return pd.DataFrame(data)


def _build_bcc_dataframe(result, input_cols: list[str]) -> pd.DataFrame:
    """Build the BCC output DataFrame."""
    data: dict = {
        "fund_id": result.fund_ids,
        "theta_bcc": result.theta,
    }
    for i, col in enumerate(input_cols):
        data[f"slack_{col}"] = result.slacks_in[:, i]
    data["peer_ids"] = ["|".join(p) for p in result.peer_ids]
    return pd.DataFrame(data)


def _build_targets_dataframe(
    df: pd.DataFrame,
    ccr_result,
    input_cols: list[str],
) -> pd.DataFrame:
    """Build peer target / input reduction DataFrame."""
    rows = []
    for k, fid in enumerate(ccr_result.fund_ids):
        theta = ccr_result.theta[k]
        row: dict = {"fund_id": fid, "theta_ccr": theta}
        for i, col in enumerate(input_cols):
            actual = df.iloc[k][col]
            target = theta * actual - ccr_result.slacks_in[k, i]
            reduction_abs = actual - target
            reduction_pct = reduction_abs / actual * 100 if actual > 0 else 0.0
            row[f"target_{col}"] = max(target, 0.0)
            row[f"reduction_abs_{col}"] = max(reduction_abs, 0.0)
            row[f"reduction_pct_{col}"] = max(reduction_pct, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run the full DEA + ML pipeline."""
    out_dir = ensure_dir(args.out)
    seed: int = args.seed
    B: int = args.bootstrap_b

    # 1. Load data
    df = load_csv(args.input)

    # Aggregate by fund_id (mean across years) for single-period DEA
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df_agg = df.groupby("fund_id")[numeric_cols].mean().reset_index()
    # Re-attach fund_type (take mode)
    fund_type_map = df.groupby("fund_id")["fund_type"].agg(lambda x: x.mode()[0])
    df_agg = df_agg.merge(fund_type_map, on="fund_id")

    input_cols = ["operating_expenses_usd", "total_assets_usd", "equity_debt_usd"]
    output_cols = ["net_investment_income_usd", "member_contributions_usd"]

    X_in, Y_out, fund_ids = get_dea_matrices(df_agg, input_cols, output_cols)

    # 2. CCR DEA
    ccr_result = dea_ccr_input_oriented(X_in, Y_out, fund_ids)
    ccr_df = _build_ccr_dataframe(ccr_result, input_cols)
    ccr_path = out_dir / "efficiency_ccr.csv"
    ccr_df.to_csv(ccr_path, index=False)
    print(f"\nCCR efficiency saved to {ccr_path}")
    print(ccr_df[["fund_id", "theta_ccr"]].to_string(index=False))

    # 3. BCC DEA
    bcc_result = dea_bcc_input_oriented(X_in, Y_out, fund_ids)
    bcc_df = _build_bcc_dataframe(bcc_result, input_cols)
    bcc_path = out_dir / "efficiency_vrs.csv"
    bcc_df.to_csv(bcc_path, index=False)
    print(f"\nBCC (VRS) efficiency saved to {bcc_path}")

    # 4. Scale efficiency
    scale_result = compute_scale_efficiency(ccr_result, bcc_result)
    scale_df = scale_to_dataframe(scale_result)
    scale_path = out_dir / "scale.csv"
    scale_df.to_csv(scale_path, index=False)
    print(f"\nScale efficiency saved to {scale_path}")
    print(scale_df.to_string(index=False))

    # 5. Peer targets
    targets_df = _build_targets_dataframe(df_agg, ccr_result, input_cols)
    targets_path = out_dir / "targets.csv"
    targets_df.to_csv(targets_path, index=False)
    print(f"\nInput reduction targets saved to {targets_path}")

    # 6. Simar-Wilson bootstrap bias correction
    def _ccr_func(X: np.ndarray, Y: np.ndarray):
        return dea_ccr_input_oriented(X, Y, fund_ids)

    boot_result = simar_wilson(
        _ccr_func, X_in, Y_out, fund_ids=fund_ids, B=B, seed=seed
    )
    boot_df = bootstrap_to_dataframe(boot_result)
    boot_path = out_dir / "bias_corrected_scores.csv"
    boot_df.to_csv(boot_path, index=False)
    print(f"\nBias-corrected scores saved to {boot_path}")
    print(boot_df[["fund_id", "theta_raw", "theta_bias_corrected", "ci_lower", "ci_upper"]].to_string(index=False))

    # 7. Prepare derived features for RF second stage
    df_agg["fund_size_log"] = np.log(df_agg["total_assets_usd"])
    df_agg["expense_ratio"] = df_agg["operating_expenses_usd"] / df_agg["total_assets_usd"]
    rts_map = {"IRS": 1, "CRS": 0, "DRS": -1}
    df_agg["rts_encoded"] = [rts_map.get(r, 0) for r in scale_result.rts_class]

    env_cols = [c for c in ENV_FEATURE_COLS if c in df_agg.columns]
    if env_cols and len(df_agg) >= 3:
        env_features = df_agg[env_cols].values.astype(float)
        rf_result = fit_rf(
            boot_result.theta_bias_corrected,
            env_features,
            feature_names=env_cols,
            seed=seed,
        )
        rf_imp_path = out_dir / "rf_importance.csv"
        rf_result.feature_importance.to_csv(rf_imp_path, index=False)
        print(f"\nRF feature importance saved to {rf_imp_path}")
        print(rf_result.feature_importance.to_string(index=False))

        pdp_path = plot_pdp(rf_result, env_features, out_dir=out_dir)
    else:
        logger.warning("Skipping RF stage: insufficient data or missing env columns.")
        rf_result = None
        pdp_path = None

    # 8. PDF report
    generate_pdf_report(
        out_dir=out_dir,
        ccr_df=ccr_df,
        bcc_df=bcc_df,
        scale_df=scale_df,
        bootstrap_df=boot_df,
        rf_importance_df=rf_result.feature_importance if rf_result else None,
        targets_df=targets_df,
        pdp_path=pdp_path,
    )
    print(f"\nPDF report saved to {out_dir / 'report.pdf'}")
    print("\nAnalysis complete.")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pension-toolkit",
        description="Pension Fund Efficiency Toolkit — DEA + ML pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Run the full DEA + ML pipeline")
    analyze.add_argument("--input", required=True, help="Path to input CSV file")
    analyze.add_argument("--out", default="out", help="Output directory (default: out)")
    analyze.add_argument(
        "--bootstrap-B",
        dest="bootstrap_b",
        type=int,
        default=2000,
        help="Number of bootstrap replications (default: 2000)",
    )
    analyze.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)",
    )

    return parser


def app() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        try:
            cmd_analyze(args)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Error: %s", exc)
            sys.exit(1)


if __name__ == "__main__":
    app()
