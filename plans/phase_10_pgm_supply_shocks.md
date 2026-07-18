# Phase 10: A PGM supply-shock event ledger and the supply-channel event study

Added 2026-07-18. **Status: scoped, not started.** A design briefing for review — no
code written yet, now wired into `00_roadmap.md` as Phase 10 (2026-07-18).

This phase specifies the **right treatment for the metals the monetary channel gets
wrong.** In Phase 5 palladium's hawkish-FOMC effect flipped sign post-2020 and it was the
only metal to fail placebo (`results/phase5_triangulation.md`): PGM prices trade on
*supply*, not the monetary channel, so a monetary treatment is simply mis-specified for
them. The correct treatment is a clean, dated ledger of genuine PGM **supply disruptions**,
tested with the Phase-2/5 event-study machinery. This is also AMC's fattest tail — `PL=F`/
`PA=F` (and rhodium in catalytic scrap) carry the largest inventory risk — so even a
directional, honestly-underpowered result is decision-relevant.

Inherits Phase 5/7 discipline: causal-not-predictive, placebo + triangulation +
pre-registration, walk-forward CV, `features/leakage.py`, harness logging, UTC, a
`journal.md` entry per session. The deliverable is an **event ledger + an event study**,
not a forecaster.

---

## 0. The recommendation in one paragraph

Build a **PGM supply-event ledger** as the phase's primary artifact — a dated, typed table
of genuine supply disruptions (mine outages, export/sanctions restrictions, labour actions,
logistics/refining failures, demand-substitution announcements) — then run the existing
local-projection + DoubleML event study of `PL=F`/`PA=F` returns around those dates. Source
the ledger from three legs that cross-check each other: **(1)** the Phase-8.1 LLM annotator
(`src/metals/annotate/`, schema-v2 `event{}` object with `type`/`entity`/
`supply_demand_side`/`framing`), run **date-blind** over the title-era GDELT corpus; **(2)**
hand-verified anchors for the known large shocks (2018–22 Russian palladium/sanctions
timeline, South African load-shedding and strikes, the 2021 Nornickel mine flood,
gasoline-catalyst Pd→Pt substitution announcements); **(3)** trade-press confirmation for
dating (Factiva via university access, Northern Miner/mining.com, per
`results/amc_paid_data_review.md`). The ledger is valuable independent of any forecast — it
also supplies clean event-dating to Phase-2 local projections and feeds the rhodium /
catalytic-scrap pricing work — which is exactly why it survives the Phase-6 prior: its worth
does not depend on beating a vol baseline.

---

## 1. Honest framing — the treatment, and the power ceiling

- **Why supply, not monetary.** Palladium's Phase-5 sign flip and placebo failure are the
  evidence: its price is set by autocatalyst demand against a concentrated, fragile supply
  (Russia + South Africa dominate mine output). The monetary IRF that works for gold is the
  wrong model for it.
- **N is small — this is the binding constraint.** Genuinely clean, exogenous supply shocks
  over 2015–2026 number perhaps **10–30**, not hundreds. Every effect size, CI, and
  significance claim must be sized against *that*, and the honest expectation is **wide
  confidence intervals and possibly an underpowered null.** A directional,
  block-bootstrap-significant response is a real result; "too few events to say" is also a
  real, decision-relevant result (it tells AMC the tail is not statistically characterizable
  and must be managed by hard limits, not a model).
- **Rhodium has no clean price.** Rhodium dominates catalytic-converter value but has no
  exchange quote — only infrequent Johnson Matthey base prices (`data/jm_pgm.py`, the
  Phase-7 collector; barred-pending-licence, read `quarantine_reason IS NULL`). The event
  study is cleanest on exchange-priced `PL=F`/`PA=F`; rhodium gets only a coarse,
  stale-price treatment, stated as such.
- **The corpus is English-centric and title-only.** Per `plans/phase_8_ssl_probing.md` §5,
  the annotator sees titles only (2019-09-22+ for real titles), ~36% English on the PGM
  channel, and the PGM stratum is sparse — so leg (1) alone under-recovers; legs (2)/(3)
  are load-bearing, not optional.

---

## 2. The ledger — schema and construction

New table `pgm_supply_events` (next free migration id after `013_spread_floor`, i.e. `014`), grain **one row
per distinct supply event per metal**:

- `announcement_utc` — when the market first learned (drives the event study).
- `realization_utc` — when the physical disruption began, if different (sanctions are
  telegraphed; a mine flood is not). Both stored; the study keys on `announcement_utc` and
  checks pre-event drift against `realization_utc`.
- `metals` — `{platinum, palladium, rhodium, ...}` (an event can hit several).
- `event_type` — `mine_outage | export_restriction | sanctions | labour | logistics_refining
  | demand_substitution | policy`.
- `entity` — producer/region/regulator (Nornickel, Anglo/Amplats, Sibanye, Eskom, Russia,
  China-6/Euro-7).
- `direction` — supply-tightening (+) vs -loosening (−).
- `confidence`, `evidence` (verbatim title/quote), `source_leg` (`annotator|anchor|press`),
  `pulled_at`, `is_realtime`.

Construction:

1. **Annotator pass** (`src/metals/annotate/`): run the schema-v2 `event{}` extraction over
   the title-era corpus, **date-blind** (the Phase-8 trap-11 parametric-leakage control — a
   dated title lets the model "know what happened next"), keeping only `supply_demand_side =
   supply` events on the PGM channel. This is exactly the "flagship byproduct" the Phase-8.1
   design named ("auto-drafts the Phase-7.5 PGM supply-event ledger").
2. **Anchor set**: hand-curate the ~10–15 known large shocks with authoritative dates; these
   are the ground truth the annotator's recall is measured against (reuse the
   `annotate/checks.py` known-event-recall harness).
3. **Press verification**: confirm/repair dates for any annotator-surfaced candidate that
   lacks an anchor, under the personal-use license limits.
4. **De-duplication and syndication collapse** (`annotate/titles.py` hardened dedup) so one
   wire story is not counted as several events.

The ledger is the deliverable even if the event study is underpowered.

---

## 3. The event study

Per PGM (`PA=F` primary — the metal that broke Phase 5; `PL=F` for cross-check; rhodium
coarse):

- **Local projections** (`models/lp.py`): cumulative log return h ∈ {1,5,20,60} on the
  supply-event indicator (optionally signed by `direction`/`confidence`) + controls, HAC
  errors. This is the Phase-2 method with a supply treatment.
- **DoubleML** (`models/causal.py`): supply-event treatment, price/macro nuisance, shuffled
  event-date placebos **preserving the event count** (a naive iid shuffle is
  anti-conservative on ~20 events).
- **Robustness sized for small N:** moving/stationary **block bootstrap** CIs (block ≈ 10–20
  trading days); pre-event-drift check (`announcement_utc` vs `realization_utc`) to catch
  anticipation/leakage; leave-one-event-out to expose any single-event dependence;
  event-type subsets reported with explicit power caveats. Log through `eval/harness.py`;
  the honest comparison is against a placebo/random-date null, never "effect > 0".

---

## 4. Identification & leakage traps

1. **Anticipation / leakage.** Sanctions and standards changes are telegraphed; keying on
   `announcement_utc` and checking pre-event drift is mandatory, and "the effect is already
   in the price by the announcement" is a valid, reportable finding.
2. **Annotator parametric leakage** (trap 11). Date-blind prompt; treat labels as
   hindsight-colored; cross-check a date-blinded re-run for drift (`annotate/pilot.py`
   already has the A/B drift check).
3. **Announcement-vs-realization confound.** Store and distinguish both; do not collapse.
4. **Exogeneity of the "shock".** Sanctions co-occur with geopolitical risk-off that also
   moves gold — control for the contemporaneous risk-off state (VIX/GPR) so the PGM supply
   effect is not risk-off in disguise; report the effect net of the gold move.
5. **Rhodium staleness.** JM base prices are infrequent and dealer-quoted; a "return" around
   an event may straddle stale quotes — flag `is_realtime`, and never present rhodium event
   effects with the same confidence as `PA=F`.
6. **Quarantine filter.** Any join to `pgm_prices`/collector tables applies
   `quarantine_reason IS NULL` (CLAUDE.md); today those largely return empty by design.
7. **Multiple testing.** Pre-declare (metals × horizons × event-types) and control FDR;
   with ~20 events, most cells will be underpowered — say so up front.

---

## 5. Success / null

- **Success:** a sign- and magnitude-stable PGM response to supply-tightening events (a Pd
  spike on export/mine shocks) that survives the event-count-preserving placebo and
  block-bootstrap CI net of the risk-off control — enough to support a pre-registered
  **AMC decision rule**: when a supply-tightening event of type X fires, cap PGM scrap intake
  / accelerate offload for N days, with an OOS-estimated tail-reduction number. This is the
  fattest-tail lever.
- **Null / underpowered:** the effect is directionally sensible but ~20 events cannot
  establish it at FDR. Business call: manage the PGM tail with **hard inventory limits**, not
  a model, and treat the ledger as a monitoring/alerting artifact (fire an alert on a
  high-confidence supply event) rather than a calibrated predictor. Also fully acceptable —
  and the ledger still pays for itself as Phase-2 event-dating and rhodium/cat-scrap pricing
  input.

---

## 6. Execution order

1. Anchor set + `pgm_supply_events` migration (`014`) first — the ground truth.
2. Annotator supply-event pass (date-blind), measured against the anchors via
   `annotate/checks.py`; press-verify the residual candidates.
3. Event study on `PA=F`/`PL=F` (LP + DML + block bootstrap, risk-off control); rhodium as a
   coarse, caveated add-on.
4. Pre-register the frozen ledger hash, event grid, primary metric, block length, and the
   candidate AMC decision rule in `journal.md` before scoring. `ruff`/`ruff format`/`mypy`/
   `pytest` before done; session entry appended.

**Grounding files:** `src/metals/annotate/` (`schema.py`, `titles.py`, `sample.py`,
`pilot.py`, `checks.py`), `data/jm_pgm.py` (rhodium/PGM base prices), `models/lp.py`,
`models/causal.py`, `features/leakage.py`, `eval/cv.py`, `eval/harness.py`,
`results/phase5_triangulation.md` (palladium's monetary-channel failure — the motivation),
`plans/phase_8_ssl_probing.md` §5/§8.1 (annotator design, parametric-leakage trap),
`configs/gdelt_themes.yaml` (the PGM theme codes the annotator pre-filters on).
