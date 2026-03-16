"""Streamlit dashboard for the Pension Efficiency Toolkit."""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

# Ensure the project root is on sys.path so pension_toolkit is importable
# when Streamlit runs this file as a top-level script.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Pension Efficiency Toolkit",
    page_icon="📊",
    layout="wide",
)

# Lazy imports (avoid loading heavy modules until needed)
from pension_toolkit.data_io import load_csv, get_dea_matrices
from pension_toolkit.dea_core import dea_ccr_input_oriented, dea_bcc_input_oriented
from pension_toolkit.scale import compute_scale_efficiency, scale_to_dataframe
from pension_toolkit.bootstrap import simar_wilson, bootstrap_to_dataframe
from pension_toolkit.ml_stage import fit_rf, plot_pdp, ENV_FEATURE_COLS
from pension_toolkit.reporting import generate_pdf_report
from pension_toolkit.utils import ensure_dir


INPUT_COLS = ["operating_expenses_usd", "total_assets_usd", "equity_debt_usd"]
OUTPUT_COLS = ["net_investment_income_usd", "member_contributions_usd"]


def run_pipeline(df: pd.DataFrame, B: int, seed: int) -> dict:
    """Execute the full analysis pipeline and return all results as a dict."""
    # Aggregate across years
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df_agg = df.groupby("fund_id")[numeric_cols].mean().reset_index()
    fund_type_map = df.groupby("fund_id")["fund_type"].agg(lambda x: x.mode()[0])
    df_agg = df_agg.merge(fund_type_map, on="fund_id")

    X_in, Y_out, fund_ids = get_dea_matrices(df_agg, INPUT_COLS, OUTPUT_COLS)

    # CCR + BCC
    ccr = dea_ccr_input_oriented(X_in, Y_out, fund_ids)
    bcc = dea_bcc_input_oriented(X_in, Y_out, fund_ids)
    scale = compute_scale_efficiency(ccr, bcc)
    scale_df = scale_to_dataframe(scale)

    # CCR DataFrame
    ccr_df = pd.DataFrame(
        {"fund_id": ccr.fund_ids, "theta_ccr": ccr.theta, "peer_ids": ["|".join(p) for p in ccr.peer_ids]}
    )
    bcc_df = pd.DataFrame(
        {"fund_id": bcc.fund_ids, "theta_bcc": bcc.theta, "peer_ids": ["|".join(p) for p in bcc.peer_ids]}
    )

    # Targets
    rows = []
    for k, fid in enumerate(ccr.fund_ids):
        theta = ccr.theta[k]
        row = {"fund_id": fid, "theta_ccr": theta}
        for i, col in enumerate(INPUT_COLS):
            actual = df_agg.iloc[k][col]
            target = theta * actual - ccr.slacks_in[k, i]
            row[f"target_{col}"] = max(target, 0.0)
            row[f"reduction_pct_{col}"] = max((actual - target) / actual * 100, 0.0) if actual > 0 else 0.0
        rows.append(row)
    targets_df = pd.DataFrame(rows)

    # Bootstrap
    def _ccr_func(X, Y):
        return dea_ccr_input_oriented(X, Y, fund_ids)

    boot = simar_wilson(_ccr_func, X_in, Y_out, fund_ids=fund_ids, B=B, seed=seed)
    boot_df = bootstrap_to_dataframe(boot)

    # Derived RF features
    df_agg["fund_size_log"] = np.log(df_agg["total_assets_usd"])
    df_agg["expense_ratio"] = df_agg["operating_expenses_usd"] / df_agg["total_assets_usd"]
    rts_map = {"IRS": 1, "CRS": 0, "DRS": -1}
    df_agg["rts_encoded"] = [rts_map.get(r, 0) for r in scale.rts_class]

    # RF
    env_cols = [c for c in ENV_FEATURE_COLS if c in df_agg.columns]
    rf_result = None
    pdp_path = None
    if env_cols and len(df_agg) >= 3:
        env_features = df_agg[env_cols].to_numpy(dtype=float)
        rf_result = fit_rf(boot.theta_bias_corrected, env_features, feature_names=env_cols, seed=seed)
        with tempfile.TemporaryDirectory() as tmp:
            pdp_path = plot_pdp(rf_result, env_features, out_dir=tmp)
            pdp_bytes = Path(pdp_path).read_bytes()
    else:
        pdp_bytes = None

    return dict(
        df_agg=df_agg,
        ccr_df=ccr_df,
        bcc_df=bcc_df,
        scale_df=scale_df,
        targets_df=targets_df,
        boot_df=boot_df,
        rf_result=rf_result,
        pdp_bytes=pdp_bytes,
        X_in=X_in,
        fund_ids=fund_ids,
    )


def main() -> None:
    st.title("Pension Fund Efficiency Toolkit")
    st.caption("DEA + ML frontier analysis for pension funds")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        uploaded = st.file_uploader("Upload pension fund CSV", type="csv")
        bootstrap_B = st.slider("Bootstrap replications (B)", 50, 2000, 200, step=50)
        seed = st.number_input("RNG seed", value=42, step=1)
        run_btn = st.button("Run Analysis", type="primary")

    if uploaded is None:
        st.info("Upload a CSV file in the sidebar to begin. See README for the required schema.")
        return

    # Load and cache
    @st.cache_data
    def _load(data_bytes: bytes) -> pd.DataFrame:
        import io, tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(data_bytes)
            tmp_path = f.name
        return load_csv(tmp_path)

    try:
        df = _load(uploaded.read())
    except (ValueError, FileNotFoundError) as e:
        st.error(f"Data validation error: {e}")
        return

    st.success(f"Loaded {len(df)} rows from '{uploaded.name}'")

    if not run_btn and "results" not in st.session_state:
        st.info("Click **Run Analysis** in the sidebar to start.")
        return

    if run_btn:
        with st.spinner("Running DEA + ML pipeline..."):
            try:
                results = run_pipeline(df, B=int(bootstrap_B), seed=int(seed))
                st.session_state["results"] = results
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                return

    results = st.session_state.get("results")
    if results is None:
        return

    tabs = st.tabs(["Overview", "Efficiency", "Drivers", "Recommendations"])

    # --- Overview tab ---
    with tabs[0]:
        st.subheader("Dataset Overview")
        st.dataframe(results["df_agg"], use_container_width=True)

        st.subheader("Pipeline Summary")
        ccr_mean = results["ccr_df"]["theta_ccr"].mean()
        bcc_mean = results["bcc_df"]["theta_bcc"].mean()
        n_efficient = (results["ccr_df"]["theta_ccr"] >= 1 - 1e-4).sum()
        n_funds = len(results["ccr_df"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Funds analysed", n_funds)
        col2.metric("Mean CCR efficiency", f"{ccr_mean:.3f}")
        col3.metric("Mean BCC efficiency", f"{bcc_mean:.3f}")
        col4.metric("Frontier funds (CCR)", int(n_efficient))

    # --- Efficiency tab ---
    with tabs[1]:
        st.subheader("CCR (CRS) Efficiency Scores")
        st.dataframe(results["ccr_df"].round(4), use_container_width=True)

        st.subheader("BCC (VRS) Efficiency Scores")
        st.dataframe(results["bcc_df"].round(4), use_container_width=True)

        st.subheader("Scale Efficiency")
        st.dataframe(results["scale_df"].round(4), use_container_width=True)

        st.subheader("Simar-Wilson Bias-Corrected Scores")
        st.dataframe(results["boot_df"].round(4), use_container_width=True)

        # Efficiency bar chart
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(results["ccr_df"]))
        ax.bar(x, results["ccr_df"]["theta_ccr"], alpha=0.7, label="CCR", color="#2980B9")
        ax.bar(x, results["bcc_df"]["theta_bcc"], alpha=0.5, label="BCC", color="#E74C3C")
        ax.set_xticks(list(x))
        ax.set_xticklabels(results["ccr_df"]["fund_id"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Efficiency Score")
        ax.set_title("CCR vs BCC Efficiency Scores")
        ax.axhline(1.0, color="black", linestyle="--", lw=0.8)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # --- Drivers tab ---
    with tabs[2]:
        if results["rf_result"] is not None:
            st.subheader("Random Forest Feature Importance")
            st.dataframe(results["rf_result"].feature_importance.round(4), use_container_width=True)

            st.subheader("Permutation Importance")
            st.dataframe(results["rf_result"].permutation_importance.round(4), use_container_width=True)

            cv_mean = results["rf_result"].cv_r2_scores.mean()
            st.metric("Cross-Validation R²", f"{cv_mean:.4f}")

            if results["pdp_bytes"] is not None:
                st.subheader("Partial Dependence Plots")
                st.image(results["pdp_bytes"], use_container_width=True)
        else:
            st.info("Insufficient data for RF second-stage analysis (need >= 3 DMUs with environmental variables).")

    # --- Recommendations tab ---
    with tabs[3]:
        st.subheader("Input Reduction Targets")
        st.markdown(
            "Recommended input reductions to move inefficient funds to the efficiency frontier."
        )
        st.dataframe(results["targets_df"].round(4), use_container_width=True)

        st.subheader("Export PDF Report")
        if st.button("Generate & Download PDF"):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)

                # Save PDP image if available
                pdp_path = None
                if results["pdp_bytes"]:
                    pdp_path = tmp_path / "pdp_top3.png"
                    pdp_path.write_bytes(results["pdp_bytes"])

                pdf_path = generate_pdf_report(
                    out_dir=tmp_path,
                    ccr_df=results["ccr_df"],
                    bcc_df=results["bcc_df"],
                    scale_df=results["scale_df"],
                    bootstrap_df=results["boot_df"],
                    rf_importance_df=results["rf_result"].feature_importance if results["rf_result"] else None,
                    targets_df=results["targets_df"],
                    pdp_path=pdp_path,
                )
                pdf_bytes = pdf_path.read_bytes()

            st.download_button(
                label="Download report.pdf",
                data=pdf_bytes,
                file_name="pension_efficiency_report.pdf",
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
