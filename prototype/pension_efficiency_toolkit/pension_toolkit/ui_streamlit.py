"""Streamlit dashboard — manager-facing UI for the Pension Efficiency Toolkit."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Pension Efficiency Toolkit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pension_toolkit.auth import check_credentials, get_manager_name
from pension_toolkit.data_io import load_csv, get_dea_matrices, NON_POSITIVE_WARNINGS
from pension_toolkit.db import save_fund_data, load_fund_data, has_fund_data, get_upload_history, delete_upload
from pension_toolkit.dea_core import dea_ccr_input_oriented, dea_bcc_input_oriented
from pension_toolkit.scale import compute_scale_efficiency, scale_to_dataframe
from pension_toolkit.bootstrap import simar_wilson, bootstrap_to_dataframe
from pension_toolkit.ml_stage import fit_rf, plot_pdp, ENV_FEATURE_COLS
from pension_toolkit.reporting import generate_pdf_report
from pension_toolkit.utils import ensure_dir

INPUT_COLS = ["operating_expenses_usd", "total_assets_usd", "equity_debt_usd"]
OUTPUT_COLS = ["net_investment_income_usd", "member_contributions_usd"]

INPUT_LABELS = {
    "operating_expenses_usd": "Operating Expenses",
    "total_assets_usd": "Total Assets",
    "equity_debt_usd": "Equity & Debt",
}

FEATURE_LABELS = {
    "exchange_volatility": "Exchange Rate Stability",
    "fund_age": "Fund Maturity (Years)",
    "fund_size_log": "Fund Size",
    "expense_ratio": "Cost Efficiency Ratio",
    "rts_encoded": "Operating Scale",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def fmt_usd(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def efficiency_status(score: float) -> tuple[str, str]:
    if score >= 0.90:
        return "Excellent", "#27AE60"
    elif score >= 0.75:
        return "Good", "#2980B9"
    elif score >= 0.60:
        return "Fair", "#E67E22"
    return "Needs Attention", "#E74C3C"


def status_badge(label: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;color:{color};'
        f'padding:4px 12px;border-radius:20px;font-size:0.82rem;'
        f'font-weight:600;border:1px solid {color}55">{label}</span>'
    )


def _fmt_upload_date(iso_str: str) -> str:
    from datetime import datetime as _dt
    try:
        return _dt.fromisoformat(iso_str).strftime("%d %b %Y, %H:%M")
    except Exception:
        return iso_str


def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Remove top padding */
        .block-container { padding-top: 1.5rem; }

        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
            border-left: 4px solid #1E6B8A;
            height: 100%;
        }
        .metric-card.alert { border-left-color: #E74C3C; }
        .metric-card.success { border-left-color: #27AE60; }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1E6B8A;
            line-height: 1.1;
        }
        .metric-value.alert { color: #E74C3C; }
        .metric-value.success { color: #27AE60; }
        .metric-label {
            font-size: 0.82rem;
            color: #64748B;
            margin-top: 6px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-sub {
            font-size: 0.88rem;
            color: #334155;
            margin-top: 4px;
            font-weight: 500;
        }

        .rec-card {
            background: white;
            border-radius: 10px;
            padding: 1rem 1.4rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-left: 4px solid #E67E22;
        }
        .rec-title { font-weight: 600; font-size: 1rem; color: #1A2B3C; }
        .rec-detail { font-size: 0.9rem; color: #475569; margin-top: 5px; line-height: 1.5; }

        .peer-card {
            background: #EEF7EC;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border: 1px solid #C3E6CB;
        }
        .peer-name { font-weight: 600; color: #155724; font-size: 0.95rem; }
        .peer-score { color: #27AE60; font-size: 0.88rem; margin-top: 2px; }

        .insight-box {
            background: linear-gradient(135deg, #EBF5FB 0%, #E8F8F5 100%);
            border-radius: 10px;
            padding: 1rem 1.4rem;
            border-left: 4px solid #1E6B8A;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Pipeline ────────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame, B: int, seed: int) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df_agg = df.groupby("fund_id")[numeric_cols].mean().reset_index()
    fund_name_map = df.groupby("fund_id")["fund_name"].agg(lambda x: x.mode()[0])
    fund_type_map = df.groupby("fund_id")["fund_type"].agg(lambda x: x.mode()[0])
    df_agg = df_agg.merge(fund_name_map, on="fund_id")
    df_agg = df_agg.merge(fund_type_map, on="fund_id")

    X_in, Y_out, fund_ids = get_dea_matrices(df_agg, INPUT_COLS, OUTPUT_COLS)

    ccr = dea_ccr_input_oriented(X_in, Y_out, fund_ids)
    bcc = dea_bcc_input_oriented(X_in, Y_out, fund_ids)
    scale = compute_scale_efficiency(ccr, bcc)
    scale_df = scale_to_dataframe(scale)

    id_to_name = dict(zip(df_agg["fund_id"], df_agg["fund_name"]))

    # Build targets with plain-language dollar amounts
    rows = []
    for k, fid in enumerate(ccr.fund_ids):
        theta = ccr.theta[k]
        row = {"fund_id": fid, "fund_name": id_to_name.get(fid, fid), "theta_ccr": theta}
        for i, col in enumerate(INPUT_COLS):
            actual = float(df_agg.iloc[k][col])
            target = theta * actual - float(ccr.slacks_in[k, i])
            target = max(target, 0.0)
            reduction_usd = max(actual - target, 0.0)
            reduction_pct = (reduction_usd / actual * 100) if actual > 0 else 0.0
            row[f"target_{col}"] = target
            row[f"actual_{col}"] = actual
            row[f"reduction_usd_{col}"] = reduction_usd
            row[f"reduction_pct_{col}"] = reduction_pct
        # Exclude self from peer list
        row["peer_names"] = [id_to_name.get(p, p) for p in ccr.peer_ids[k] if p != fid]
        rows.append(row)
    targets_df = pd.DataFrame(rows)

    # Bootstrap bias correction
    def _ccr_func(X, Y):
        return dea_ccr_input_oriented(X, Y, fund_ids)

    boot = simar_wilson(_ccr_func, X_in, Y_out, fund_ids=fund_ids, B=B, seed=seed)
    boot_df = bootstrap_to_dataframe(boot)

    # RF second stage
    df_agg["fund_size_log"] = np.log(df_agg["total_assets_usd"])
    df_agg["expense_ratio"] = df_agg["operating_expenses_usd"] / df_agg["total_assets_usd"]
    rts_map = {"IRS": 1, "CRS": 0, "DRS": -1}
    df_agg["rts_encoded"] = [rts_map.get(r, 0) for r in scale.rts_class]

    env_cols = [c for c in ENV_FEATURE_COLS if c in df_agg.columns]
    rf_result = None
    pdp_bytes = None
    if env_cols and len(df_agg) >= 3:
        env_features = df_agg[env_cols].to_numpy(dtype=float)
        rf_result = fit_rf(boot.theta_bias_corrected, env_features, feature_names=env_cols, seed=seed)
        with tempfile.TemporaryDirectory() as tmp:
            pdp_path = plot_pdp(rf_result, env_features, out_dir=tmp)
            pdp_bytes = Path(pdp_path).read_bytes()

    return dict(
        df_agg=df_agg,
        targets_df=targets_df,
        scale_df=scale_df,
        boot_df=boot_df,
        rf_result=rf_result,
        pdp_bytes=pdp_bytes,
        ccr=ccr,
        bcc=bcc,
        id_to_name=id_to_name,
    )


# ─── Login Page ──────────────────────────────────────────────────────────────

def render_login() -> None:
    inject_css()
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown(
            '<div style="text-align:center;margin-bottom:0.5rem">'
            '<span style="font-size:3rem">🏦</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h2 style="text-align:center;color:#1E6B8A;margin-bottom:0">Pension Efficiency Toolkit</h2>'
            '<p style="text-align:center;color:#64748B;margin-top:4px">Zimbabwe Pension Fund Analysis System</p>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Sign In", use_container_width=True, type="primary"
            )
            if submitted:
                if check_credentials(email, password):
                    st.session_state["authenticated"] = True
                    st.session_state["manager_name"] = get_manager_name()
                    st.rerun()
                else:
                    st.error("Incorrect email or password. Please try again.")


# ─── Dashboard Tab ───────────────────────────────────────────────────────────

def render_dashboard(results: dict) -> None:
    if NON_POSITIVE_WARNINGS:
        col_labels = [INPUT_LABELS.get(c, c) for c in NON_POSITIVE_WARNINGS]
        st.warning(
            f"**Data notice:** The following column(s) contained zero or negative values in your dataset, "
            f"which is not valid for efficiency analysis: **{', '.join(col_labels)}**. "
            f"Those entries were substituted with a minimum floor value ($1) so the analysis could proceed. "
            f"Please review your source data to correct these figures for the most accurate results.",
            icon="⚠️",
        )

    targets = results["targets_df"]
    scores = targets["theta_ccr"].values
    avg_score = scores.mean()
    best_row = targets.loc[targets["theta_ccr"].idxmax()]
    n_attention = int((scores < 0.75).sum())
    n_funds = len(targets)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{n_funds}</div>'
            f'<div class="metric-label">Funds Analysed</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{avg_score * 100:.0f}%</div>'
            f'<div class="metric-label">Sector Average Efficiency</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card success">'
            f'<div class="metric-value success">🏆</div>'
            f'<div class="metric-label">Top Performer</div>'
            f'<div class="metric-sub">{best_row["fund_name"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c4:
        cls = "alert" if n_attention > 0 else "success"
        val_color = "alert" if n_attention > 0 else "success"
        label = "Funds Needing Attention" if n_attention > 0 else "All Funds Performing Well"
        st.markdown(
            f'<div class="metric-card {cls}">'
            f'<div class="metric-value {val_color}">{n_attention}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Ranked horizontal bar chart
    df_chart = targets[["fund_name", "theta_ccr"]].copy()
    df_chart["Efficiency (%)"] = (df_chart["theta_ccr"] * 100).round(1)
    df_chart = df_chart.sort_values("Efficiency (%)", ascending=True)

    fig = px.bar(
        df_chart,
        x="Efficiency (%)",
        y="fund_name",
        orientation="h",
        color="Efficiency (%)",
        color_continuous_scale=["#E74C3C", "#E67E22", "#3498DB", "#27AE60"],
        range_color=[50, 100],
        title="Fund Efficiency Rankings",
        labels={"fund_name": ""},
        text="Efficiency (%)",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height=440,
        coloraxis_showscale=False,
        xaxis=dict(range=[0, 112], ticksuffix="%", showgrid=True, gridcolor="#F0F0F0"),
        yaxis=dict(automargin=True),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_font_size=16,
        margin=dict(l=10, r=60, t=50, b=20),
    )
    fig.add_vline(
        x=avg_score * 100,
        line_dash="dash",
        line_color="#94A3B8",
        annotation_text=f" Sector avg: {avg_score*100:.0f}%",
        annotation_position="top right",
        annotation_font_color="#64748B",
    )
    st.plotly_chart(fig, width="stretch")

    # Insight summary
    n_excellent = int((scores >= 0.90).sum())
    n_good = int(((scores >= 0.75) & (scores < 0.90)).sum())
    total_savings = sum(
        sum(row.get(f"reduction_usd_{col}", 0) for col in INPUT_COLS)
        for _, row in targets.iterrows()
    )
    st.markdown(
        f'<div class="insight-box">'
        f'<strong>Sector Insight:</strong> '
        f'{n_excellent} fund(s) are at excellent efficiency (≥ 90%). '
        f'{n_good} fund(s) are performing well (75–89%). '
        f'If all underperforming funds reached the frontier, the sector could save an estimated '
        f'<strong>{fmt_usd(total_savings)}</strong> per year in operating costs.'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── Rankings Tab ────────────────────────────────────────────────────────────

def render_rankings(results: dict) -> None:
    targets = results["targets_df"]
    ccr = results["ccr"]
    bcc = results["bcc"]
    id_to_name = results["id_to_name"]

    st.subheader("Fund Performance Rankings")
    st.markdown(
        "Every fund is scored on how efficiently it converts its resources into returns for members. "
        "A score of **100%** means the fund is operating at the frontier — "
        "the best possible performance given its size and circumstances."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Build ranked table
    df_ranked = targets[["fund_name", "theta_ccr"]].copy()
    df_ranked["score_pct"] = (df_ranked["theta_ccr"] * 100).round(1)
    df_ranked = df_ranked.sort_values("score_pct", ascending=False).reset_index(drop=True)
    df_ranked.index += 1

    savings_map: dict[str, float] = {}
    for _, row in targets.iterrows():
        savings_map[row["fund_name"]] = sum(row.get(f"reduction_usd_{c}", 0) for c in INPUT_COLS)

    display_rows = []
    for rank, row in df_ranked.iterrows():
        label, color = efficiency_status(row["theta_ccr"])
        savings = savings_map.get(row["fund_name"], 0)
        display_rows.append({
            "Rank": rank,
            "Fund Name": row["fund_name"],
            "Efficiency Score": f"{row['score_pct']}%",
            "Status": label,
            "Potential Annual Savings": fmt_usd(savings) if savings > 10_000 else "—",
        })

    st.dataframe(
        pd.DataFrame(display_rows).set_index("Rank"),
        use_container_width=True,
        height=380,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Overall vs Internal efficiency scatter
    scatter_df = pd.DataFrame({
        "Fund": [id_to_name.get(fid, fid) for fid in ccr.fund_ids],
        "Overall Efficiency": (ccr.theta * 100).round(1),
        "Internal Efficiency": (bcc.theta * 100).round(1),
    })
    scatter_df["Status"] = scatter_df["Overall Efficiency"].apply(
        lambda x: efficiency_status(x / 100)[0]
    )

    colour_map = {"Excellent": "#27AE60", "Good": "#2980B9", "Fair": "#E67E22", "Needs Attention": "#E74C3C"}
    fig = px.scatter(
        scatter_df,
        x="Overall Efficiency",
        y="Internal Efficiency",
        color="Status",
        color_discrete_map=colour_map,
        text="Fund",
        title="Overall Efficiency vs Internal Efficiency",
        labels={"Overall Efficiency": "Overall Efficiency (%)", "Internal Efficiency": "Internal Efficiency (%)"},
    )
    fig.update_traces(textposition="top center", marker_size=12)
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        xaxis=dict(range=[40, 112], ticksuffix="%", showgrid=True, gridcolor="#F0F0F0"),
        yaxis=dict(range=[40, 112], ticksuffix="%", showgrid=True, gridcolor="#F0F0F0"),
        title_font_size=15,
    )
    fig.add_shape(
        type="line", x0=40, y0=40, x1=110, y1=110,
        line=dict(color="#CBD5E1", dash="dot"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Funds sitting on the dotted line have matching overall and internal efficiency. "
        "Funds above the line manage their internal operations well but are operating at the wrong scale."
    )


# ─── Fund Details Tab ────────────────────────────────────────────────────────

def render_fund_details(results: dict) -> None:
    targets = results["targets_df"]
    rf_result = results.get("rf_result")

    fund_names = (
        targets.sort_values("theta_ccr", ascending=False)["fund_name"].tolist()
    )
    selected = st.selectbox("Select a Fund to View", fund_names)

    row = targets[targets["fund_name"] == selected].iloc[0]
    score = float(row["theta_ccr"])
    pct = round(score * 100, 1)
    status_label, status_color = efficiency_status(score)
    avg_pct = round(targets["theta_ccr"].mean() * 100, 1)

    targets_sorted = targets.sort_values("theta_ccr", ascending=False).reset_index(drop=True)
    rank = int(targets_sorted[targets_sorted["fund_name"] == selected].index[0]) + 1
    n_funds = len(targets)

    col_gauge, col_info = st.columns([1, 1.8], gap="large")

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 46, "color": status_color}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickwidth": 1},
                "bar": {"color": status_color, "thickness": 0.25},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 60], "color": "#FDECEC"},
                    {"range": [60, 75], "color": "#FEF9EC"},
                    {"range": [75, 90], "color": "#EEF7EC"},
                    {"range": [90, 100], "color": "#D5F5E3"},
                ],
                "threshold": {
                    "line": {"color": "#94A3B8", "width": 2},
                    "thickness": 0.75,
                    "value": avg_pct,
                },
            },
        ))
        fig.update_layout(
            height=260,
            margin=dict(t=20, b=10, l=10, r=10),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            f'<div style="text-align:center;margin-top:-10px">{status_badge(status_label, status_color)}</div>',
            unsafe_allow_html=True,
        )

    with col_info:
        st.markdown(f"### {selected}")
        st.markdown(f"**Ranked {rank} of {n_funds} funds**")
        st.markdown(
            f"| | |\n|---|---|\n"
            f"| **Efficiency Score** | {pct}% |\n"
            f"| **Sector Average** | {avg_pct}% |\n"
            f"| **Gap to Average** | {pct - avg_pct:+.1f}% |"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if pct >= 90:
            st.success(
                "This fund is a top performer. It is one of the benchmark funds "
                "that other funds in the sector should look to for best practices."
            )
        elif pct >= 75:
            st.info(
                f"This fund performs above the sector average but has a "
                f"**{100 - pct:.0f}% efficiency gap** remaining. Targeted improvements "
                f"to its cost structure could bring it to full efficiency."
            )
        else:
            st.warning(
                f"This fund is underperforming relative to its peers. "
                f"For every $1.00 of resources used, a top fund achieves the same "
                f"results with only **${score:.2f}**. The recommendations below show "
                f"where to focus improvement efforts."
            )

    st.markdown("---")

    # Recommendations
    has_reductions = any(row.get(f"reduction_usd_{c}", 0) > 5_000 for c in INPUT_COLS)

    if has_reductions:
        st.subheader("Recommended Actions")
        st.markdown(
            "These are the specific changes this fund can make to reach the efficiency frontier:"
        )
        for col in INPUT_COLS:
            reduction_usd = float(row.get(f"reduction_usd_{col}", 0))
            reduction_pct = float(row.get(f"reduction_pct_{col}", 0))
            actual = float(row.get(f"actual_{col}", 0))
            target = float(row.get(f"target_{col}", 0))

            if reduction_usd > 5_000 and reduction_pct > 0.5:
                label = INPUT_LABELS.get(col, col)
                st.markdown(
                    f'<div class="rec-card">'
                    f'<div class="rec-title">📉 Reduce {label}</div>'
                    f'<div class="rec-detail">'
                    f'Current level: <strong>{fmt_usd(actual)}</strong> &nbsp;→&nbsp; '
                    f'Recommended target: <strong>{fmt_usd(target)}</strong><br>'
                    f'Potential annual saving: <strong>{fmt_usd(reduction_usd)}</strong> '
                    f'({reduction_pct:.1f}% reduction)'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.success(
            "This fund is already at or very near the efficiency frontier. "
            "No significant resource reductions are recommended."
        )

    # Peer benchmarks
    peer_names: list[str] = row.get("peer_names", [])
    if peer_names:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Learn From These Funds")
        st.markdown(
            "The following high-performing funds are the closest benchmarks for "
            f"**{selected}**. Study how they structure their operations:"
        )
        for pn in peer_names[:3]:
            peer_rows = targets[targets["fund_name"] == pn]
            if not peer_rows.empty:
                peer_score = float(peer_rows.iloc[0]["theta_ccr"])
                peer_rank = int(
                    targets_sorted[targets_sorted["fund_name"] == pn].index[0]
                ) + 1
                st.markdown(
                    f'<div class="peer-card">'
                    f'<div class="peer-name">{pn}</div>'
                    f'<div class="peer-score">Ranked #{peer_rank} &nbsp;·&nbsp; {peer_score*100:.1f}% efficient</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # What drives efficiency
    if rf_result is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("What Drives Efficiency Across All Funds")
        st.markdown(
            "Based on the data, these factors most strongly predict whether a fund "
            "will be efficient or inefficient:"
        )
        fi = rf_result.feature_importance.copy()
        fi["Factor"] = fi["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
        fi["Impact (%)"] = (fi["importance"] * 100).round(1)
        fi = fi.sort_values("Impact (%)", ascending=True)

        fig = px.bar(
            fi,
            x="Impact (%)",
            y="Factor",
            orientation="h",
            color="Impact (%)",
            color_continuous_scale=["#BDC3C7", "#1E6B8A"],
            text="Impact (%)",
            title="Factors That Influence Fund Efficiency",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            height=320,
            coloraxis_showscale=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(ticksuffix="%", showgrid=True, gridcolor="#F0F0F0"),
            margin=dict(l=10, r=50, t=50, b=20),
            title_font_size=14,
        )
        st.plotly_chart(fig, width="stretch")


# ─── Reports Tab ─────────────────────────────────────────────────────────────

def render_reports(results: dict) -> None:
    st.subheader("Download Full Analysis Report")
    st.markdown(
        "The PDF report includes efficiency scores for all funds, input reduction "
        "targets, the fund ranking table, and key findings. "
        "You can share this report with stakeholders or use it for regulatory submissions."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating report — please wait..."):
                out_dir = ensure_dir(_project_root / "out")

                pdp_path = None
                if results.get("pdp_bytes"):
                    pdp_path = out_dir / "pdp_top3.png"
                    pdp_path.write_bytes(results["pdp_bytes"])

                ccr = results["ccr"]
                bcc = results["bcc"]
                id_to_name = results["id_to_name"]

                ccr_df = pd.DataFrame({
                    "Fund Name": [id_to_name.get(fid, fid) for fid in ccr.fund_ids],
                    "Efficiency Score (%)": (ccr.theta * 100).round(2),
                }).sort_values("Efficiency Score (%)", ascending=False)

                bcc_df = pd.DataFrame({
                    "Fund Name": [id_to_name.get(fid, fid) for fid in bcc.fund_ids],
                    "Internal Efficiency (%)": (bcc.theta * 100).round(2),
                })

                pdf_path = generate_pdf_report(
                    out_dir=out_dir,
                    ccr_df=ccr_df,
                    bcc_df=bcc_df,
                    scale_df=results["scale_df"],
                    bootstrap_df=results["boot_df"],
                    rf_importance_df=(
                        results["rf_result"].feature_importance
                        if results.get("rf_result") else None
                    ),
                    targets_df=results["targets_df"],
                    pdp_path=pdp_path,
                )
                st.session_state["pdf_bytes"] = pdf_path.read_bytes()

    if st.session_state.get("pdf_bytes"):
        st.download_button(
            label="📥 Download Report (PDF)",
            data=st.session_state["pdf_bytes"],
            file_name="pension_efficiency_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()

    if not st.session_state.get("authenticated"):
        render_login()
        return

    manager_name = st.session_state.get("manager_name", get_manager_name())
    first_name = manager_name.split()[0]

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {first_name}")
        st.markdown(f"<small style='color:#64748B'>{manager_name}</small>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("#### 📁 Fund Data")

        history = get_upload_history()

        # Auto-load the most recent upload on first visit
        if history and st.session_state.get("active_upload_id") is None:
            latest = history[0]
            df_auto = load_fund_data(latest.id)
            if df_auto is not None:
                st.session_state["df"] = df_auto
                st.session_state["active_upload_id"] = latest.id

        if history:
            st.markdown("**Previous uploads**")
            active_id = st.session_state.get("active_upload_id")

            for r in history:
                is_active = r.id == active_id
                col_main, col_del = st.columns([5, 1])

                with col_main:
                    prefix = "✓ " if is_active else ""
                    btn_label = (
                        f"{prefix}{r.filename}\n"
                        f"{_fmt_upload_date(r.uploaded_at)} · {r.n_funds} fund(s)"
                    )
                    if st.button(
                        btn_label,
                        key=f"sel_{r.id}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        if not is_active:
                            df_sel = load_fund_data(r.id)
                            if df_sel is not None:
                                st.session_state["df"] = df_sel
                                st.session_state["active_upload_id"] = r.id
                                st.session_state.pop("results", None)
                                st.rerun()

                with col_del:
                    if st.button("🗑", key=f"del_{r.id}", help=f"Delete {r.filename}"):
                        delete_upload(r.id)
                        if is_active:
                            st.session_state.pop("active_upload_id", None)
                            st.session_state.pop("df", None)
                            st.session_state.pop("results", None)
                        st.rerun()

            # Show summary of active dataset
            if "df" in st.session_state:
                df = st.session_state["df"]
                years = sorted(df["year"].unique())
                year_range = f"{min(years)}" if len(years) == 1 else f"{min(years)}–{max(years)}"
                st.success(f"✅ {df['fund_id'].nunique()} funds · {year_range}")
        else:
            st.info("No uploads yet. Upload a CSV below to begin.")

        st.markdown("**Upload new dataset**")
        uploaded = st.file_uploader(
            "Upload CSV", type="csv", label_visibility="collapsed"
        )
        if uploaded:
            import hashlib, tempfile as _tmp
            file_bytes = uploaded.read()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            if file_hash != st.session_state.get("last_upload_hash"):
                try:
                    with _tmp.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                        f.write(file_bytes)
                        df_new = load_csv(f.name)
                    new_id = save_fund_data(df_new, filename=uploaded.name)
                    st.session_state["df"] = df_new
                    st.session_state["active_upload_id"] = new_id
                    st.session_state["last_upload_hash"] = file_hash
                    st.session_state.pop("results", None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload error: {e}")

        st.markdown("---")
        st.markdown("#### ⚙️ Settings")
        bootstrap_B = st.slider(
            "Statistical accuracy",
            min_value=50, max_value=500, value=100, step=50,
            help="Higher values give more accurate results but take longer to compute.",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button(
            "▶  Run Analysis",
            type="primary",
            use_container_width=True,
            disabled="df" not in st.session_state,
        )

        st.markdown("---")
        if st.button("🔒 Sign Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Main area ─────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 style="color:#1E6B8A;margin-bottom:0">🏦 Pension Fund Efficiency Dashboard</h1>'
        '<p style="color:#64748B;margin-top:4px;font-size:1rem">'
        'Helping Zimbabwe\'s pension funds operate at their best</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if "df" not in st.session_state:
        st.markdown(
            "### Getting Started\n\n"
            "Upload your fund data CSV using the **sidebar** to begin. "
            "Your data will be saved automatically — you won't need to upload it again "
            "on your next visit.\n\n"
            "The CSV must include columns for fund financials across one or more years. "
            "See the README for the required format."
        )
        return

    if run_btn:
        with st.spinner("Running efficiency analysis — this may take up to a minute..."):
            try:
                results = run_pipeline(
                    st.session_state["df"], B=int(bootstrap_B), seed=42
                )
                st.session_state["results"] = results
                st.success("Analysis complete.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    if "results" not in st.session_state:
        st.info("Click **▶ Run Analysis** in the sidebar to generate the efficiency report.")
        return

    results = st.session_state["results"]

    tab_dash, tab_rank, tab_detail, tab_report = st.tabs(
        ["📊 Dashboard", "🏆 Fund Rankings", "🔍 Fund Details", "📄 Download Report"]
    )

    with tab_dash:
        render_dashboard(results)

    with tab_rank:
        render_rankings(results)

    with tab_detail:
        render_fund_details(results)

    with tab_report:
        render_reports(results)


if __name__ == "__main__":
    main()
