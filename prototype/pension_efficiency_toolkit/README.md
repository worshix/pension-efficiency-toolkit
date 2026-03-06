# Pension Efficiency Toolkit

**Frontier-based optimization analysis of pension fund efficiency using Data Envelopment Analysis (DEA) + Machine Learning.**

This toolkit implements a full DEA + ML pipeline:
- Input-oriented CCR (CRS) and BCC (VRS) DEA models
- Scale efficiency computation with IRS/DRS/CRS classification
- PCA input validation
- Simar-Wilson bootstrap bias correction
- Random Forest second-stage determinant analysis
- Streamlit dashboard with PDF export

---

## Setup

### Prerequisites

Install `uv` (if not already installed):

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync
```

---

## Usage

### Run the full analysis pipeline

```bash
uv run python -m pension_toolkit.cli analyze \
  --input tests/sample_data.csv \
  --out out/
```

With all options:

```bash
uv run python -m pension_toolkit.cli analyze \
  --input tests/sample_data.csv \
  --out out/ \
  --bootstrap-B 200 \
  --seed 42 \
  --use-pca-composite
```

**CLI flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--input` | Path to input CSV file | (required) |
| `--out` | Output directory | `out/` |
| `--bootstrap-B` | Number of bootstrap replications | `2000` |
| `--use-pca-composite` | Use PCA composite inputs for DEA | `False` |
| `--seed` | RNG seed for reproducibility | `42` |

### Launch Streamlit dashboard

```bash
uv run streamlit run pension_toolkit/ui_streamlit.py
```

### Run tests

```bash
uv run pytest -q
```

### Run everything locally

```bash
bash scripts/run_local.sh
```

---

## CSV Schema

The input CSV must have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `fund_id` | str | Unique fund identifier |
| `year` | int | Year of observation |
| `fund_name` | str | Fund name |
| `fund_type` | str | `public` or `private` |
| `total_assets_usd` | float | Total assets (USD) |
| `operating_expenses_usd` | float | Operating expenses (USD) |
| `equity_debt_usd` | float | Equity + debt (USD) |
| `net_investment_income_usd` | float | Net investment income (USD) |
| `member_contributions_usd` | float | Member contributions (USD) |
| `inflation` | float | Annual inflation rate (%) |
| `exchange_volatility` | float | Exchange rate volatility |
| `fund_age` | int | Years since fund inception |

---

## Output Files

| File | Description |
|------|-------------|
| `out/efficiency_ccr.csv` | CCR efficiency scores and slacks |
| `out/efficiency_vrs.csv` | BCC (VRS) efficiency scores |
| `out/scale.csv` | Scale efficiency and RTS classification |
| `out/targets.csv` | Input reduction recommendations |
| `out/bias_corrected_scores.csv` | Simar-Wilson bias-corrected scores + 95% CI |
| `out/rf_importance.csv` | Random Forest feature importance |
| `out/pdp_top3.png` | Partial dependence plots (top 3 drivers) |
| `out/report.pdf` | Full PDF report |

---

## Docker

```bash
docker build -t pension-toolkit .
docker run -p 8501:8501 pension-toolkit
```

Then open http://localhost:8501

---

## Project Structure

```
pension_efficiency_toolkit/
├── pension_toolkit/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── data_io.py          # CSV loading + validation
│   ├── pca_utils.py        # PCA input validation
│   ├── dea_core.py         # CCR + BCC DEA models
│   ├── scale.py            # Scale efficiency
│   ├── bootstrap.py        # Simar-Wilson bootstrap
│   ├── ml_stage.py         # Random Forest second stage
│   ├── reporting.py        # PDF report generation
│   ├── ui_streamlit.py     # Streamlit dashboard
│   └── utils.py            # Shared utilities
├── tests/
│   ├── sample_data.csv     # 10-fund synthetic dataset
│   ├── test_data_io.py
│   ├── test_dea_core.py
│   ├── test_pca.py
│   └── test_bootstrap.py
├── scripts/
│   └── run_local.sh
├── Dockerfile
├── .github/workflows/ci.yml
├── pyproject.toml
└── README.md
```

---

## Technical Notes

- **DEA formulation**: Input-oriented LP solved with PuLP/CBC
- **Bootstrap**: Simar-Wilson Algorithm 1 (1998) with reflected kernel density
- **Parallelism**: Bootstrap replications parallelised with joblib
- **Reproducibility**: All RNG uses `np.random.default_rng(seed)`
- **Python**: 3.11+, fully type-annotated
