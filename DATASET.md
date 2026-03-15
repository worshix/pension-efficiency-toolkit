# Dataset Modification Prompt

## Context

This dataset is used in an academic research study titled:
**"Frontier-Based Optimization Analysis of Pension Fund Efficiency in Zimbabwe Using Data Envelopment Analysis (DEA)"**

The study applies a DEA framework to evaluate the technical efficiency of Zimbabwean pension funds over the period 2018–2022. The study uses 3 input variables and 2 output variables:

- **Inputs:** `total_assets_usd`, `operating_expenses_usd`, `equity_debt_usd`
- **Outputs:** `net_investment_income_usd`, `member_contributions_usd`

The macroeconomic control variables (`inflation`, `exchange_volatility`) and `fund_age` are used in a second-stage Random Forest determinant analysis, not in the DEA model itself.

---

## Required Changes

### 1. Fund Type: Self-Administered Only

The study has been revised to focus **exclusively on self-administered standalone pension funds** in Zimbabwe. This means:

- The `fund_type` column must be updated. Replace all values ("public", "private") with **`self_administered`** for all funds.
- All fund names must be **realistic Zimbabwean self-administered standalone pension funds** — these are large employer-sponsored funds managed by their own board of trustees, not underwritten by an insurance company and not occupational/pillar-II schemes.
- **Exclude** the National Social Security Authority (NSSA) — it operates under a distinct statutory mandate and is explicitly out of scope.

### 2. Fund Count and DEA Convention

The DEA model uses 5 variables (3 inputs + 2 outputs). By the standard DEA rule of thumb, the minimum number of DMUs (funds) should be at least **3 × 5 = 15**. To ensure robust results, the dataset should contain **between 16 and 20 self-administered funds**, each with data for all 5 years (2018–2022).

If the existing dataset has fewer than 16 suitable funds after the type change, **add new self-administered fund rows** using realistic Zimbabwean fund names and plausible financial data consistent with the data patterns already in the dataset.

### 3. Fund Naming

Rename the funds (where necessary) to reflect actual or highly plausible Zimbabwean self-administered standalone pension funds. Examples of real/realistic funds include:

- Mining Industry Pension Fund (MIPF)
- ZESA Staff Pension Fund (Zimbabwe Electricity Supply Authority)
- CBZ Bank Staff Pension Fund
- Delta Corporation Pension Fund
- Econet Wireless Staff Pension Fund
- Zimplats Mineworkers Pension Fund
- NMB Bank Staff Pension Scheme
- Bindura Nickel Corporation Pension Fund
- First Capital Bank Staff Pension Fund
- Innscor Africa Staff Pension Fund
- Old Mutual Zimbabwe Staff Pension Fund
- NetOne Cellular Staff Pension Fund
- Air Zimbabwe Staff Pension Fund
- POSB Staff Pension Fund
- Telecel Zimbabwe Staff Pension Fund
- Cottco Staff Pension Fund (Cotton Company of Zimbabwe)
- ZISCO Steel Workers Pension Fund
- Agricultural Development Bank Staff Pension Fund
- Grain Marketing Board Staff Pension Fund
- NRZ Staff Pension Fund (National Railways of Zimbabwe)

Use fund IDs in the format ZW001, ZW002, etc.

### 4. Macroeconomic Variables (Keep Consistent)

The `inflation` and `exchange_volatility` values are **country-level annual figures** — they must be **identical for all funds in the same year**. Keep the existing year-level values:

| Year | Inflation | Exchange Volatility |
|------|-----------|---------------------|
| 2018 | 10.6      | 0.18                |
| 2019 | 255.3     | 0.72                |
| 2020 | 557.2     | 0.85                |
| 2021 | 98.5      | 0.45                |
| 2022 | 243.8     | 0.61                |

### 5. Financial Data Realism

Maintain the following realistic patterns in the financial data:

- **2020** should generally show the worst performance: lowest or negative `net_investment_income_usd`, lowest `total_assets_usd` relative to adjacent years — reflecting the height of hyperinflation and currency crisis.
- **2019** may also show significantly depressed or negative `net_investment_income_usd` for most funds.
- Fund sizes should vary significantly to capture the scale heterogeneity of the Zimbabwean pension sector (some funds with total assets >$500M USD, others <$20M USD).
- `equity_debt_usd` should always be less than `total_assets_usd` for each record.
- `operating_expenses_usd` should be small relative to `total_assets_usd` (roughly 1–3% of total assets).
- `member_contributions_usd` should be positive for all records (even in difficult years).
- `fund_age` should increment by 1 each year for each fund (consistent with the existing pattern).

### 6. Output Format

Return the complete modified dataset as a CSV with the **exact same column order** as the input:

```
fund_id,year,fund_name,fund_type,total_assets_usd,operating_expenses_usd,equity_debt_usd,net_investment_income_usd,member_contributions_usd,inflation,exchange_volatility,fund_age
```

All monetary values in USD. No additional columns. No header changes.
