# Implementation Spec: The Inventory-VaR Spread-Floor Engine

**Prepared 2026-07-18.** The highest-value build identified in
`results/amc_derive_vs_buy_engineering.md` (Part B, "the ledger joins"). This is the
construct that turns everything else in the data program into the number AMC quotes at the
counter every day, and it is the one no competitor can replicate — because its decisive
input is AMC's own transaction ledger. Audience for §1–§3: deep business experience,
non-expert statistics. §4 onward is an engineering spec written to be handed to
implementation.

---

## 1. What it is, in one paragraph

When AMC buys a scrap lot or a coin, it must quote a price below spot wide enough that the
inventory survives an adverse metal move over the days-to-weeks it takes to sell or refine
the metal — but not so wide that the seller walks to a competitor. The engine computes that
floor from first principles, per metal and per product, every day: the wholesale price AMC
can actually *exit* at, minus a cushion sized to how far the metal could plausibly fall over
AMC's *actual* holding period, minus the cost of carrying the position. It then reports the
**dollar Value-at-Risk (VaR) on the current book** — the plausible loss on the inventory
AMC is holding *right now* — which is the number that converts the research into a hedge
size and a spread policy. Generic risk models price risk *per ounce*; this one prices it in
*dollars on AMC's book*, because it is joined to the ledger.

## 2. The governing equation

For metal *m* on date *t*, the maximum defensible buy price per fine troy ounce is:

```
max_buy(m, t) = exit_floor(m, t)  −  cushion(m, t)  −  carry(m, t)
```

where

```
cushion(m, t) = k · tail_vol(m, t) · sqrt(E[float_days | m]) · spot(m, t)
```

- **`exit_floor(m, t)`** — the conservative wholesale price at which AMC can realistically
  liquidate: for coin/specie, a trailing low-side envelope of the Greysheet wholesale *bid*;
  for scrap, `spot · (payable% from AMC's own refining settlements)`. This is the term
  most dealers *guess*.
- **`tail_vol(m, t)`** — a forward-looking lower-tail volatility estimate (see §5.1), in
  daily-return units. Classical, not machine-learned — Phase 6 blessed classical volatility
  over ML for exactly this.
- **`E[float_days | m]`** — the expected holding period for metal *m*, estimated from the
  ledger's realized `purchased_utc → disposed_utc` durations (see §5.3). The `sqrt` is the
  standard scaling of volatility to a multi-day horizon.
- **`k`** — the tail multiplier: how conservative the floor is (e.g. `k = 1.65` for a ~5%
  one-sided quantile under a normal approximation; calibrated empirically against the
  extreme-value tail library, §5.2, which will push it higher for silver and the PGMs).
- **`carry(m, t)`** — the financing/lease cost of holding the float, from the
  calendar-spread-implied forward rate (§5.4).

The **book-level output** is the sum over currently-held lots of the plausible adverse move
on each, at each lot's *own* remaining expected float:

```
book_VaR(t) = Σ_lots  fine_oz(lot) · spot(m, t) · k · tail_vol(m, t) · sqrt(remaining_float_days(lot))
```

reported per metal and in total, with the tail-vol and float inputs traceable to their
sources.

## 3. Why this is the highest-value build

- **It is the daily core decision.** Buy-spread-floors is what AMC does at the counter every
  day; the other three decisions are episodic. A one-basis-point improvement in the floor
  compounds across every lot bought.
- **It is the proprietary edge.** The float distribution and the realized exit levels come
  from AMC's ledger — data no competitor and no vendor has. The same tail-vol number applied
  to AMC's *actual* float and *actual* book is worth more than any purchased dataset applied
  to a generic ounce.
- **It composes the rest of the program.** Its terms *are* the tier-1 derived builds — the
  tail-vol floor, the lease-rate carry, the wholesale exit anchor, the extreme-value tail
  library. Building the engine is the reason to build those, and it turns them into one
  decision number instead of four disconnected features.
- **It survives the research priors.** It is an operational reference object, not a fitted
  regime forecaster, so P1 (sentiment/regime features hurt) and P2 (the independent-sample
  wall) do not bind. Its tail calibration draws on genuine crisis exceedances, the explicit
  P2 carve-out.

---

## 4. Data inputs and availability

| Input | Source | Table / field | Status |
|---|---|---|---|
| Spot & daily OHLCV | Yahoo (`prices.py`) | `prices` | **Available now** |
| Forward-looking vol | FRED — GVZ, and silver/PGM siblings | `fred` series | **GVZ not yet ingested — free add** |
| Realized vol | derived from `prices` | — | Available now |
| Float duration, realized exit | AMC ledger (`amc_ledger.py`) | `amc_scrap_lots` (`purchased_utc`, `disposed_utc`, `price_paid_usd`, `spot_usd_oz`, `proceeds_usd`, `fine_troy_oz`), `amc_coin_trades` | **Pending AMC exports** |
| Carry / lease rate | CME statistics via Databento | per-contract settlements | **Recommended buy, not yet pulled** |
| Deep-tail calibration | Norgate (futures) + Johnson Matthey (rhodium) | — | **Deferred buy** |
| Coin wholesale exit anchor | Greysheet CDN API | `coin_premiums` (quarantined until licensed) | **Recommended subscribe** |

The engine is designed so each term degrades gracefully to a documented fallback when its
preferred source is absent (see §7), so it delivers a usable floor from day one on
already-owned data and sharpens as each source lands. **No term blocks on the pending
ledger except the two that are definitionally about AMC's own book** (the float distribution
and `book_VaR`), which is unavoidable and correct.

## 5. Component specifications

### 5.1 Tail-volatility estimate — `tail_vol(m, t)`

- **Primary:** the forward-looking implied volatility where a clean index exists (GVZ for
  gold; the relaunched silver index; self-computed at-the-money implied vol from CME option
  settlements for platinum/palladium once the options backfill is pulled). Implied vol is
  the only genuinely *forward-looking* input and is contemporaneous, so it carries no
  look-ahead.
- **Fallback (always computed, for the blend and for backtest history before the implied
  series exist):** a classical realized-volatility estimator — an exponentially weighted or
  HAR-style (heterogeneous autoregressive) estimate over trailing daily log returns, matching
  the `models/lgbm_vol.py` baseline conventions. Use the **lower semi-deviation** (downside
  returns only), because the floor protects against a *fall*, not symmetric moves.
- **Output units:** daily return standard deviation, so the `sqrt(float_days)` scaling
  applies directly.
- **Leakage:** realized-vol windows are strictly trailing (≤ t−1); implied vol is
  as-of-close on t−1 for a floor quoted on t. Enforce with `assert_features_have_history`
  for the warmup and the trailing-window convention already used in
  `features/spreads.py::compute_spread_zscores` (`min_periods = window`).

### 5.2 Extreme-value tail library and the multiplier `k`

- Fit a peaks-over-threshold generalized-Pareto model to the lower tail of multi-day
  (float-horizon) returns per metal, drawing on the deepest history available (Norgate
  pre-2010 for exchange metals; Johnson Matthey for rhodium). Produce the q05 / q01 terminal
  loss and the **maximum-adverse-excursion** quantile — the worst point *during* the hold,
  which is the true forced-sale risk and is always at least as severe as the terminal loss.
- Set `k` so the normal-approximation cushion matches the empirical tail quantile at the
  chosen protection level; expect `k` materially above 1.65 for silver and the PGMs.
- **Effective-sample-size governor (mandatory).** Before trusting any tail quantile, run a
  declustering / extremal-index audit that counts how many *independent* exceedances the
  history actually contains. Crises cluster: the 1980 silver collapse is essentially one
  independent draw, not forty daily ones. Report the effective count alongside every
  quantile and widen the confidence band accordingly. This is the discipline that makes the
  deep-history purchase defensible rather than overconfident (P2).
- **Leakage:** tail windows must never span a contract roll; limit-locked settlements are
  flagged non-executable and excluded from the "realizable exit" set.

### 5.3 Float-duration distribution — `E[float_days | m]` (ledger)

- From `amc_scrap_lots`: `float_days = disposed_utc − purchased_utc` for closed lots, per
  metal. From `amc_coin_trades`: pair buys to sells by product (or use inventory-turn as a
  proxy) for the coin book.
- Report the **distribution**, not just the mean — the 75th/90th percentile float is what
  sizes the cushion for slow-moving inventory, and the median for fast. For the book-level
  VaR, each open lot uses its own *remaining* expected float conditioned on age (a lot held
  30 days already has a different expected remaining life than a fresh one).
- **Right-censoring:** open (undisposed) lots are censored, not missing — use a
  survival-style estimator (Kaplan–Meier) so long-held inventory is not silently dropped, or
  the float will read too short and the floor too tight.
- **Fallback before the ledger lands:** a documented assumed float (e.g. 10 business days)
  with the engine flagged "assumed float — not calibrated to AMC's book," so nothing ships
  as if it were ledger-calibrated.

### 5.4 Carry — `carry(m, t)`

- The calendar-spread-implied forward/lease rate from raw per-contract CME settlements: the
  annualized spread between the near and next contract, net of the risk-free carry (30-day
  fed-funds / SOFR), converted to a per-float-horizon cost.
- Doubles as the **backwardation squeeze flag**: a near-over-far inversion beyond a frozen
  threshold widens `exit_floor` (physical is tight — AMC's exit is better than screen) while
  raising alarm state.
- **Leakage (critical):** build from the raw (near, far) per-contract pair *live on each
  date* with a point-in-time roll. A back-adjusted/continuous series fabricates
  backwardation at every roll. Fallback before Databento: set `carry = risk_free_rate ·
  float_days/360` (financing only, no lease premium) and flag it.

### 5.5 Exit floor — `exit_floor(m, t)`

- **Coin/specie:** a trailing low-side envelope (e.g. the rolling minimum or a low quantile
  over a window ≥ the float duration) of the Greysheet wholesale *bid* per product. Trailing
  by construction so it never uses a mark AMC could not have realized.
- **Scrap:** `spot · payable%`, where `payable%` is estimated from AMC's own
  refining-settlement invoices (dollars-out ÷ fine-content-in), per metal and lot-size band.
  This is the derive-your-own-payable-curve build; it needs no external panel.
- **Verify before shipping:** confirm the Greysheet ask moves independently of the bid (if
  the ask is a fixed markup, the spread carries no information — see the write-up's landmine
  note).

## 6. Module, schema, and logging layout

- **New feature/model module:** `src/metals/models/spread_floor.py` (the engine) plus
  `src/metals/features/inventory.py` (the ledger-derived float and payable curves). Match the
  functional, docstring-first style of `features/spreads.py`.
- **Migration `013`** (next free stem per the migrations directory; `010`–`012` are applied):
  a `spread_floor_daily` table keyed `(metal, product, date_utc)` holding the floor and every
  input term with its provenance, plus a `book_var_daily` table keyed `(date_utc, metal)`.
  Append-only, UTC, with `source` / `pulled_at` columns per the collector convention. Never
  put a `;` inside a migration comment (the whole-file-execution truncation trap).
- **Eval harness:** register every engine run through `eval/harness.py`
  (`runs` / `run_predictions`) so the backtest (§8) logs like any other model and Phase-6-style
  lift tables can read it without re-running.
- **Walk-forward only:** any parameter fit (`k`, tail thresholds, float quantiles) uses
  `eval/cv.py` walk-forward folds — never a random split (a non-negotiable codebase
  convention).

## 7. Graceful degradation (so it ships before every source lands)

| Term | Preferred | Fallback | Flag emitted |
|---|---|---|---|
| `tail_vol` | GVZ / self-computed implied vol | classical downside realized vol | `vol=realized` |
| `k` / tail | GPD on deep history | normal-approx `k=1.65` on 2010+ sample | `tail=normal_approx` |
| `float_days` | ledger Kaplan–Meier | assumed 10-day float | `float=assumed` |
| `carry` | calendar-spread lease rate | risk-free financing only | `carry=rf_only` |
| `exit_floor` | Greysheet bid / own payable | spot − fixed haircut | `exit=fixed_haircut` |

Every shipped floor carries its flag string, so a consumer always knows which terms are
calibrated to real data and which are placeholders. **`book_VaR` is emitted only when
`float ≠ assumed`** — a book-level dollar risk with a made-up float would be worse than
none.

## 8. Validation plan

- **Backtest against the ledger's own realized outcomes.** For each historical lot, ask:
  did the metal's actual move over the actual float breach the cushion the engine *would
  have* quoted (using only data available at purchase)? Target a breach rate at or below the
  design quantile (e.g. ≤5% for `k` at the 5% level). A materially higher breach rate means
  the cushion is too tight; a near-zero rate over many lots means it is needlessly wide and
  losing AMC deals.
- **Coverage / calibration curve.** Plot realized breach rate against the design quantile
  across `k` values — the engine is well-calibrated where they track the 45° line.
- **Economic backtest.** Compare margin-per-lot and hypothetical breach losses under the
  engine's floor versus AMC's historical quoted spreads (from the ledger). The win condition
  is *fewer/cheaper breaches at equal-or-tighter average spread*, not just fewer breaches.
- **Frozen-threshold discipline.** All thresholds and tail parameters are frozen on the
  training vintage before the hold-out window — the Phase 6.5 lesson that in-window
  thresholds overstate the mechanism 5–90×.
- **Leakage gate (mandatory before any backtest is believed):** the feature matrix must pass
  `features/leakage.py` — `assert_chronological`, `assert_target_strictly_future` (with
  `min_nan_tail = float_horizon`), and `assert_features_have_history`.

## 9. Failure modes — what would falsify or break it

- **Greysheet ask is a fixed markup** → the exit-floor stress signal is dead; fall back to
  own-payable and trailing bid only.
- **Ledger float is thin or heavily censored** (few closed lots) → the float distribution is
  unreliable; keep the assumed-float flag until enough lots close.
- **Tail library has too few independent exceedances** → the effective-sample governor
  widens the bands so far the tail quantile is uninformative; then `k` stays at the
  normal-approx default and the deep-history purchase is *not yet* justified (exactly the
  deferral rule in the write-up).
- **Backtest breach rate ≈ design rate but margins collapse** → the floor is correct but AMC
  was already pricing well; the engine's value is then the *book-VaR / hedge-sizing* output,
  not a tighter spread. Report that honestly rather than manufacturing a spread win.

## 10. What it explicitly does not do

- It does not forecast the metal price or its direction — it sizes a *cushion* against an
  adverse move, using classical volatility Phase 6 validated, and refuses the ML/regime
  features Phase 6 falsified.
- It does not replace the coin desk's or the owner's judgment on a specific lot — it sets a
  defensible *floor* and a *book risk number*; the quoted price still reflects competition,
  relationship, and product mix.
- It does not model counterparty/settlement risk (AMC's metal sitting unsecured at a refiner
  for the float) — that is the separate "fifth risk" monitor flagged in the paid review, not
  this engine.

## 11. First increment (buildable this week, zero new data)

Ship the market-derived floor on already-owned data: ingest GVZ into FRED, compute the
classical downside realized-vol tail-vol, apply a normal-approximation `k` on the 2010+
sample, an assumed float, risk-free-only carry, and a spot-minus-fixed-haircut exit. Log it
through the harness and stand up the `spread_floor_daily` table. Every term carries its
fallback flag, so this is honestly labelled as the uncalibrated baseline — and it becomes
the scaffold that each subsequent source (rhodium history → real tails; Databento → real
carry; Greysheet → real exit; the ledger → real float and `book_VaR`) sharpens in place,
without a rewrite.
