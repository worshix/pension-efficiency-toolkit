"""PDF report generation using ReportLab."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

from .utils import get_logger, ensure_dir

logger = get_logger(__name__)

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm


_TABLE_STYLE = TableStyle(
    [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ECF0F1")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
)

_USABLE_WIDTH = PAGE_W - 2 * MARGIN

_CELL_STYLE = ParagraphStyle(
    "TableCell",
    fontSize=7,
    leading=9,
    wordWrap="LTR",
)
_HEADER_CELL_STYLE = ParagraphStyle(
    "TableHeader",
    fontSize=7,
    leading=9,
    textColor=colors.white,
    fontName="Helvetica-Bold",
    wordWrap="LTR",
)


def _col_widths(n_cols: int) -> list[float]:
    if n_cols == 1:
        return [_USABLE_WIDTH]
    if n_cols == 2:
        # Give the first column (usually names) 45% so text has room to breathe
        return [_USABLE_WIDTH * 0.45, _USABLE_WIDTH * 0.55]
    # 3+ columns: first column gets 25%, rest share equally
    first = _USABLE_WIDTH * 0.25
    rest = (_USABLE_WIDTH - first) / (n_cols - 1)
    return [first] + [rest] * (n_cols - 1)


def _df_to_table(df: pd.DataFrame, max_rows: int = 20) -> Table:
    """Convert a DataFrame to a ReportLab Table that fits the page width."""
    df = df.head(max_rows).round(4)

    header = [[Paragraph(str(col), _HEADER_CELL_STYLE) for col in df.columns]]
    data_rows = [
        [Paragraph(str(val), _CELL_STYLE) for val in row]
        for row in df.values.tolist()
    ]
    table_data = header + data_rows

    col_w = _col_widths(len(df.columns))
    t = Table(table_data, colWidths=col_w, repeatRows=1)
    t.setStyle(_TABLE_STYLE)
    return t


def generate_pdf_report(
    out_dir: str | Path,
    ccr_df: pd.DataFrame | None = None,
    bcc_df: pd.DataFrame | None = None,
    scale_df: pd.DataFrame | None = None,
    bootstrap_df: pd.DataFrame | None = None,
    rf_importance_df: pd.DataFrame | None = None,
    targets_df: pd.DataFrame | None = None,
    pdp_path: str | Path | None = None,
    title: str = "Pension Fund Efficiency Analysis Report",
) -> Path:
    """Generate a PDF report from analysis results.

    Parameters
    ----------
    out_dir:
        Directory to save the PDF.
    ccr_df, bcc_df, scale_df, bootstrap_df, rf_importance_df, targets_df:
        Optional DataFrames for each section.
    pdp_path:
        Optional path to the PDP PNG image.
    title:
        Report title.

    Returns
    -------
    Path to the generated PDF.
    """
    out_dir = ensure_dir(out_dir)
    out_path = out_dir / "report.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=18,
        textColor=colors.HexColor("#2C3E50"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "CustomH1",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=colors.HexColor("#2980B9"),
        spaceBefore=16,
        spaceAfter=6,
    )
    body_style = styles["BodyText"]
    body_style.fontSize = 9

    story: list[Any] = []

    # --- Title page ---
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Executive Summary", h1_style))
    story.append(
        Paragraph(
            "This report presents frontier-based efficiency analysis of pension funds using "
            "Data Envelopment Analysis (DEA). The analysis covers CCR (CRS) and BCC (VRS) "
            "models, scale efficiency classification, Simar-Wilson bootstrap bias correction, "
            "and a Random Forest second-stage analysis of efficiency determinants.",
            body_style,
        )
    )

    # --- CCR results ---
    if ccr_df is not None:
        story.append(Paragraph("CCR (CRS) Efficiency Scores", h1_style))
        story.append(_df_to_table(ccr_df))
        story.append(Spacer(1, 0.3 * cm))

    # --- BCC results ---
    if bcc_df is not None:
        story.append(Paragraph("BCC (VRS) Efficiency Scores", h1_style))
        story.append(_df_to_table(bcc_df))
        story.append(Spacer(1, 0.3 * cm))

    # --- Scale efficiency ---
    if scale_df is not None:
        story.append(Paragraph("Scale Efficiency & Returns to Scale", h1_style))
        story.append(_df_to_table(scale_df))
        story.append(Spacer(1, 0.3 * cm))

    # --- Bootstrap ---
    if bootstrap_df is not None:
        story.append(PageBreak())
        story.append(Paragraph("Simar-Wilson Bias-Corrected Efficiency Scores", h1_style))
        story.append(_df_to_table(bootstrap_df))
        story.append(Spacer(1, 0.3 * cm))

    # --- Targets (split into two sub-tables to avoid overflow) ---
    if targets_df is not None:
        story.append(Paragraph("Input Reduction Targets", h1_style))
        target_cols = ["fund_id", "theta_ccr"] + [c for c in targets_df.columns if c.startswith("target_")]
        pct_cols = ["fund_id"] + [c for c in targets_df.columns if c.startswith("reduction_pct_")]
        story.append(Paragraph("Target input levels (USD):", body_style))
        story.append(Spacer(1, 0.15 * cm))
        story.append(_df_to_table(targets_df[[c for c in target_cols if c in targets_df.columns]]))
        story.append(Spacer(1, 0.2 * cm))
        if any(c.startswith("reduction_pct_") for c in targets_df.columns):
            story.append(Paragraph("Required reductions (%):", body_style))
            story.append(Spacer(1, 0.15 * cm))
            story.append(_df_to_table(targets_df[[c for c in pct_cols if c in targets_df.columns]]))
        story.append(Spacer(1, 0.3 * cm))

    # --- RF importance ---
    if rf_importance_df is not None:
        story.append(PageBreak())
        story.append(Paragraph("Random Forest Feature Importance", h1_style))
        story.append(_df_to_table(rf_importance_df))
        story.append(Spacer(1, 0.3 * cm))

    # --- PDP image ---
    if pdp_path is not None and Path(pdp_path).exists():
        story.append(Paragraph("Partial Dependence Plots — Top 3 Efficiency Drivers", h1_style))
        img = Image(str(pdp_path), width=PAGE_W - 2 * MARGIN, height=8 * cm)
        story.append(img)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )
    doc.build(story)
    logger.info("PDF report saved to %s", out_path)
    return out_path
