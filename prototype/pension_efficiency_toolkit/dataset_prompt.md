# Dataset Generation Prompt — Zimbabwe Pension Fund Efficiency Study

Use this prompt with DeepSeek (or any LLM) to generate a realistic dataset for the pension efficiency toolkit.

---

## Prompt

You are an expert in Zimbabwean financial markets, pension fund regulation, and actuarial science. Generate a realistic CSV dataset of **20 Zimbabwean pension funds** observed over **5 years (2018–2022)** for use in a Data Envelopment Analysis (DEA) efficiency study.

### Context

Zimbabwe's pension industry operates under the Insurance and Pensions Commission (IPEC). It includes large public funds (NSSA, civil service schemes) and smaller private occupational funds (mining, banking, agriculture, manufacturing sectors). The sector has been severely affected by:
- Hyperinflation in 2019–2020 (ZIMSTAT reported annual inflation peaking above 800% in July 2020)
- Currency reforms (RTGS dollar introduced Feb 2019, ZWL reintroduced, US dollar re-dollarisation by 2022)
- High exchange rate volatility throughout
- COVID-19 economic shock in 2020

All monetary values must be expressed in **USD equivalent** at the time of observation (use official or mid-market USD conversion rates for the ZWL).

---

### Output Format

Produce a single CSV with exactly these columns in this order:

```
fund_id,year,fund_name,fund_type,total_assets_usd,operating_expenses_usd,equity_debt_usd,net_investment_income_usd,member_contributions_usd,inflation,exchange_volatility,fund_age
```

**Column definitions:**

| Column | Type | Description |
|--------|------|-------------|
| `fund_id` | str | Stable fund identifier, e.g. `ZW001` through `ZW020` |
| `year` | int | Observation year: 2018, 2019, 2020, 2021, or 2022 |
| `fund_name` | str | Realistic Zimbabwean fund name (see examples below) |
| `fund_type` | str | `public` or `private` |
| `total_assets_usd` | float | Total fund assets under management in USD. Range: $2M–$600M depending on fund size |
| `operating_expenses_usd` | float | Total admin + management expenses in USD. Typically 1–4% of AUM |
| `equity_debt_usd` | float | Value of equity + fixed income portfolio in USD. Typically 40–70% of AUM |
| `net_investment_income_usd` | float | Investment returns net of costs in USD. Can be negative in crisis years |
| `member_contributions_usd` | float | Total member + employer contributions received in USD |
| `inflation` | float | Zimbabwe annual CPI inflation rate (%) for that year |
| `exchange_volatility` | float | Annualised standard deviation of the ZWL/USD exchange rate (0.0–1.0 scale). Higher = more volatile |
| `fund_age` | int | Years since fund inception as of the observation year |

---

### Realism Requirements

**Macroeconomic variables (use these exact figures per year):**

| Year | Inflation (%) | Exchange Volatility |
|------|--------------|---------------------|
| 2018 | 10.6         | 0.18                |
| 2019 | 255.3        | 0.72                |
| 2020 | 557.2        | 0.85                |
| 2021 | 98.5         | 0.45                |
| 2022 | 243.8        | 0.61                |

**Fund size tiers (assign each fund to one tier and keep it consistent across years):**

- Large (4 funds): total_assets_usd $200M–$600M — public sector, national schemes
- Medium (8 funds): total_assets_usd $30M–$200M — banking, mining, manufacturing, telecoms
- Small (8 funds): total_assets_usd $2M–$30M — agriculture, retail, NGO, university schemes

**Fund names to use (assign one per fund_id, keep constant):**
ZW001 NationalSocialSecurityAuthority, ZW002 CivilServicePensionFund, ZW003 ZimstateRailwaysPension, ZW004 DefenceForcesPensionScheme, ZW005 CBZBankStaffFund, ZW006 FirstCapitalBankPension, ZW007 ZimplatsMineworkers, ZW008 BinduraNickelPension, ZW009 EconnetWirelessStaff, ZW010 TelecelEmployeesFund, ZW011 AgriculturalBankPension, ZW012 CottonCompanyRetirement, ZW013 ZiscoSteelWorkers, ZW014 DeltaCorporationPension, ZW015 NMBBankStaffScheme, ZW016 HarareCityCouncilFund, ZW017 UniversityOfZimbabwePension, ZW018 MidlandsStateUniversityFund, ZW019 ZimRefPensionScheme, ZW020 GrainMarketingBoardFund

**Fund types:**
- public: ZW001, ZW002, ZW003, ZW004, ZW016, ZW017, ZW018, ZW019, ZW020
- private: ZW005–ZW015

**Efficiency variation (critical for DEA to produce meaningful results):**
- 4–5 funds should be consistently efficient across years (theta ≈ 1.0 in DEA terms): high output per unit of input
- 6–8 funds should be moderately inefficient (theta ≈ 0.70–0.90)
- 5–6 funds should be clearly inefficient (theta ≈ 0.50–0.70): high expenses relative to investment income
- 1–2 funds may be severely inefficient (theta < 0.50): very high operating costs, very low returns

Operationally, efficiency is implied by:
- **Efficient** funds: low `operating_expenses_usd` relative to `total_assets_usd` (expense ratio < 1.5%), high `net_investment_income_usd` relative to `total_assets_usd` (return > 6%)
- **Inefficient** funds: expense ratio > 3%, net investment return < 3%

**Year-over-year dynamics:**
- AUM should grow 5–15% per year in USD terms for healthy funds; shrink 10–30% in 2019–2020 for weaker funds (crisis impact)
- `net_investment_income_usd` may be negative in 2019 and 2020 for some funds (acceptable — the DEA pipeline handles this)
- `member_contributions_usd` should be 5–15% of AUM
- `fund_age` increments by 1 each year

**Scale returns guidance:**
- Large funds should show DRS (decreasing returns to scale)
- Small funds should show IRS (increasing returns to scale)
- Medium efficient funds should show CRS

---

### Output Instructions

1. Output only the CSV — no explanation, no markdown, no headers other than the column row
2. 100 rows total (20 funds × 5 years)
3. Preserve fund characteristics consistently (same fund_name, fund_type, approximate size tier across years)
4. Round monetary values to the nearest 1000
5. Round inflation to 1 decimal place, exchange_volatility to 2 decimal places
6. Do not produce duplicate (fund_id, year) pairs

---

### Example rows (for format reference only — do not copy these values):

```
fund_id,year,fund_name,fund_type,total_assets_usd,operating_expenses_usd,equity_debt_usd,net_investment_income_usd,member_contributions_usd,inflation,exchange_volatility,fund_age
ZW001,2018,NationalSocialSecurityAuthority,public,580000000,8200000,320000000,42000000,38000000,10.6,0.18,42
ZW005,2020,CBZBankStaffFund,private,45000000,1800000,26000000,-1200000,4100000,557.2,0.85,18
ZW012,2019,CottonCompanyRetirement,private,8500000,420000,4200000,180000,720000,255.3,0.72,9
```
