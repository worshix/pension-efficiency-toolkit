# PR Description — All Stages

## Stage 0: Project Scaffold
- Initialized uv project with Python 3.11+
- Configured pyproject.toml with all dependencies (pandas, numpy, scikit-learn, pulp, joblib, streamlit, matplotlib, reportlab, pytest) and dev deps (black, ruff)
- Created full directory structure: pension_toolkit/, tests/, scripts/, .github/workflows/
- Added 20-row synthetic dataset (10 funds × 2 years) at tests/sample_data.csv
- Written README with setup, usage, schema, and output documentation

## Stage 1: Data IO + PCA
- `data_io.load_csv()` — schema validation, numeric coercion (to float64), missing value detection, positivity warnings
- `data_io.get_dea_matrices()` — extracts DEA input/output numpy arrays
- `pca_utils.run_pca()` — StandardScaler + PCA, returns PCAResult with explained variance, cumulative variance, components
- `pca_utils.build_composite_input()` — builds positive PCA composite for DEA use
- Tests: test_data_io.py (9 tests), test_pca.py (6 tests)

## Stage 2: DEA Core CCR
- `dea_core.dea_ccr_input_oriented()` — input-oriented CCR (CRS) DEA using PuLP/CBC
- Two-phase LP: Phase I minimises theta (radial score, bounded [0,1]), Phase II maximises slacks with theta fixed
- Returns DEAResult with theta scores, lambda weights, input/output slacks, peer IDs
- CLI outputs `out/efficiency_ccr.csv`
- Tests: test_dea_core.py (8 CCR tests)

## Stage 3: BCC + Scale Efficiency
- `dea_core.dea_bcc_input_oriented()` — BCC (VRS) DEA with convexity constraint sum(lambda)=1
- `scale.compute_scale_efficiency()` — SE = theta_CCR / theta_BCC, classifies IRS/DRS/CRS
- CLI outputs `out/efficiency_vrs.csv`, `out/scale.csv`
- Tests: 4 BCC tests verifying BCC >= CCR, model label, identical DMU edge case

## Stage 4: Peer Analysis
- `cli._build_targets_dataframe()` — computes target inputs, absolute and percentage reductions
- Formula: target = theta* × actual - slack; reduction = actual - target
- CLI outputs `out/targets.csv` with per-fund, per-input recommendations

## Stage 5: Simar-Wilson Bootstrap
- `bootstrap.simar_wilson()` — Algorithm 1 (Simar & Wilson, 1998)
  - Silverman bandwidth KDE with reflection at boundary (scores ≤ 1)
  - Parallelised with joblib (loky backend)
  - Returns bias, bias-corrected scores, 95% CI
- CLI outputs `out/bias_corrected_scores.csv`
- Tests: test_bootstrap.py (6 tests: shapes, ranges, CI ordering, reproducibility, DataFrame export)

## Stage 6: Random Forest Second Stage
- `ml_stage.fit_rf()` — RandomForest(n_estimators=500, max_features="sqrt")
- MDI feature importance + permutation importance (30 repeats)
- Cross-validation R² with k-fold
- `ml_stage.plot_pdp()` — partial dependence plots for top-3 features
- CLI outputs `out/rf_importance.csv`, `out/pdp_top3.png`

## Stage 7: Streamlit Dashboard + PDF Export
- 4-tab dashboard: Overview, Efficiency, Drivers, Recommendations
- CSV upload → cached pipeline execution → live tables and charts
- Overview: dataset table + PCA scree plot
- Efficiency: CCR/BCC/scale/bootstrap tables + comparative bar chart
- Drivers: RF importance + permutation + CV R² + PDP image
- Recommendations: targets table + "Export PDF Report" download button
- PDF generated via ReportLab with styled tables and embedded PDP image

## Stage 8: CLI Polish + Docker + CI
- CLI flags: --input, --out, --bootstrap-B, --use-pca-composite, --seed
- Dockerfile: python:3.11-slim + uv install + uv sync + Streamlit CMD
- GitHub Actions CI: install uv → uv sync → uv run pytest → ruff lint → black check
- scripts/run_local.sh: one-command local run

---

STAGE COMPLETE: 0–8
