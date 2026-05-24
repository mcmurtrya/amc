# Phase 2 — Events and Local Projections

## Goal
Bring in scheduled macro events and use Jordà local projections to estimate per-scenario impulse responses. This is the first of the three scenario-discovery methods.

## Prerequisites
- Phase 1 complete
- ALFRED access (the archival FRED API — same key)

## Steps

### 2.1 FOMC calendar
Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (parse HTML) or the community-maintained CSV at github.com/federalreserve/fomc-dates. Cover 2007–present.

Store rows into `events` with `event_type = 'FOMC'` and metadata: scheduled vs unscheduled, statement timestamp, minutes release timestamp, press-conference Y/N.

### 2.2 Macro release calendar
For CPI, PPI, NFP, retail sales, ISM, GDP, ECB, BoE meetings: capture release date, release time, and headline value. Sources:
- BLS schedules at https://www.bls.gov/schedule/
- ECB/BoE on their respective sites
- ALFRED gives first-print plus revision history — critical for distinguishing what was known on the day

Store in `events` with appropriate `event_type`.

### 2.3 Surprise measures
Surprise = actual − consensus. Consensus history is the binding constraint here:
- Free: ALFRED for first-print (proxy), MarketWatch / Investing.com archives via scraping
- Better: Bloomberg ECO function via UChicago access
- Best for FOMC: shadow-rate surprises (Kuttner, or Bauer–Swanson updates) — these decompose target-rate surprise from path-rate surprise and are far cleaner

Document your consensus source in `journal.md`. Cleanliness of this measure is the binding constraint on Phase 2's signal-to-noise ratio.

### 2.4 CFTC Commitments of Traders
Source: https://www.cftc.gov/MarketReports/CommitmentsofTraders, "Disaggregated" report for metals.

Critical lag handling: the report is dated to Tuesday positioning but released Friday after market close. The as-of timestamp in your `positioning` table must be **Friday close**, not Tuesday. Using Tuesday as the as-of is the most common data-leakage bug in this field.

Store columns: timestamp_utc, metal, commercial_long, commercial_short, managed_money_long, managed_money_short, other_reportables, non_reportables.

### 2.5 Event-feature pipeline
`src/metals/features/events.py`:
- Indicators: `is_fomc_day`, `days_since_fomc`, `days_until_fomc`, same for CPI/NFP/etc.
- Surprises: standardized (z-scored over rolling 3-year window of past surprises)
- Positioning: net managed-money position, 4-week change, 1-year percentile rank

### 2.6 Local projections framework
`src/metals/models/lp.py`:
```python
def local_projection(returns, treatment, controls, horizons=[1,3,5,10,20,60]):
    # for each h:
    #   r_{t+h} = alpha_h + beta_h * treatment_t + gamma_h' * controls_t + e_{t+h}
    # estimate by OLS with HAC (Newey-West) standard errors
    # return DataFrame: horizon, beta, se, ci_low, ci_high, n_obs
```
Implement with `statsmodels.OLS(...).fit(cov_type='HAC', cov_kwds={'maxlags': h})`. Write a pytest test on simulated data with a known IRF to confirm correctness.

### 2.7 FOMC IRFs
For each metal:
- Treatment 1: hawkish surprise indicator (top tercile of policy-rate surprise)
- Treatment 2: dovish surprise (bottom tercile)
- Controls: lagged 5-day return, lagged 20-day vol, DXY 5-day change, VIX level, real-yield level
- Horizons: 1, 3, 5, 10, 20 days
- Plot IRF with 95% bands; save to `results/phase2/fomc_<metal>_<treatment>.png`

Expected pattern for gold: negative IRF on hawkish surprises, positive on dovish, peak around h=5–10.

### 2.8 CPI surprise IRFs
Same template. Historical pattern (pre-2022): gold rises on upside CPI surprises. Post-2022 the relationship has been muddier — note any regime difference.

### 2.9 Geopolitical and DXY shocks
- GPR daily change > 95th percentile → "GPR spike"
- DXY 5-day change > 2σ → "DXY shock"
- Estimate IRFs for both, all four metals

### 2.10 Cross-metal comparison
For each scenario, produce a single panel of four IRFs (Au, Ag, Pt, Pd) side by side. Use consistent axes. Gold and silver should track on macro/monetary scenarios; Pt/Pd diverge on industrial and supply scenarios.

### 2.11 Robustness
- Subsample stability: split at 2015. Re-estimate. Persist split-specific IRFs alongside full-sample ones.
- Alternative treatment thresholds: top quartile vs top tercile vs continuous treatment with z-scored magnitude
- Placebo: randomly shift each treatment date by ±30 days and re-estimate. The placebo IRF should be flat. Compute a placebo p-value (fraction of placebo IRFs at h=5 that exceed the real estimate).

### 2.12 Document scenarios
`results/phase2_scenarios.md`: for every scenario type produced, include IRF plot, point estimates with CIs at h=5 and h=20, placebo p-value, subsample stability comment, and a paragraph of economic interpretation. This document feeds directly into the Phase 5 triangulation table.

## Deliverables
- Events table populated with FOMC, CPI, NFP, ECB, BoE
- Surprise measures with documented source
- Positioning table with correct Friday-close as-of timestamps
- `metals.models.lp` module with tests
- IRF plot library: ≥ 5 scenarios × 4 metals × multiple horizons
- `results/phase2_scenarios.md` write-up

## Common pitfalls
- Announcement-time vs daily timestamp mismatch. FOMC moves happen in a 2-hour window around 2pm ET; daily returns dilute the signal substantially. If you have intraday data, use it for FOMC; otherwise note the dilution in your write-up.
- Using post-revision values instead of first-print. The surprise is what was unknown at the moment, not what was eventually true.
- Treating COT as a Tuesday feature. It isn't. Friday close, always.
- Conflating "scheduled-event surprise" with "scheduled-event indicator." The indicator alone is mostly priced in and produces weak IRFs.
- Stacking heavily overlapping treatments (e.g., FOMC + CPI in the same week) and then claiming independent effects.
