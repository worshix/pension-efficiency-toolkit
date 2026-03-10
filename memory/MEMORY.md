# Project Memory

## Pension Efficiency Toolkit — Prototype

**Location**: `/home/worshix/school-projects/pension-efficiency-toolkit/prototype/pension_efficiency_toolkit/`

**Status**: All 9 stages (0–8) complete and tested. 35/35 tests pass.

### Key architecture decisions
- DEA uses **two-phase PuLP LP**: Phase I minimises theta (bounded [0,1]), Phase II maximises slacks with theta fixed. Single-phase with epsilon penalty caused unbounded LP with large financial values (hundreds of millions USD).
- `uv run` must use `--project <path>` flag since shell starts in parent directory
- Data is aggregated by `fund_id` (mean across years) before DEA

### Running
```bash
uv run --project prototype/pension_efficiency_toolkit pytest -q
uv run --project prototype/pension_efficiency_toolkit python -m pension_toolkit.cli analyze --input tests/sample_data.csv --out out/ --bootstrap-B 200
uv run --project prototype/pension_efficiency_toolkit streamlit run pension_toolkit/ui_streamlit.py
```

### Module map
- `data_io.py` — CSV load + validate (schema, float coercion, missing check)
- `pca_utils.py` — PCA with StandardScaler, composite input builder
- `dea_core.py` — CCR + BCC two-phase LP via PuLP
- `scale.py` — Scale efficiency + IRS/DRS/CRS classification
- `bootstrap.py` — Simar-Wilson Algorithm 1 + joblib parallelism
- `ml_stage.py` — RF(n_estimators=500) + MDI + permutation + PDP
- `reporting.py` — ReportLab PDF
- `ui_streamlit.py` — 4-tab dashboard + PDF download
- `cli.py` — argparse CLI with all flags
