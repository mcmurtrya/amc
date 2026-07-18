-- Phase 7: the inventory-VaR spread-floor engine (first increment).
-- See results/amc_spread_floor_engine_spec.md.
--
-- spread_floor_daily holds, per metal per day, the maximum defensible buy price
-- for scrap/coin inventory held over AMC's days-to-weeks float:
--   max_buy = exit_floor - cushion - carry,  cushion = k * tail_vol * sqrt(float_days) * spot.
-- Every term degrades to a documented fallback recorded in `flags`
-- (e.g. vol=realized_downside, tail=normal_approx, float=assumed, carry=rf_only,
-- exit=fixed_haircut), so a consumer always knows which terms are calibrated to
-- real data and which are placeholders. `source` versions the engine so a later
-- increment can be written alongside without clobbering this one.

CREATE TABLE IF NOT EXISTS spread_floor_daily (
    date_utc            DATE      NOT NULL,   -- reference (close) date, UTC
    metal               VARCHAR   NOT NULL,   -- gold / silver / platinum / palladium
    spot_usd_oz         DOUBLE    NOT NULL,   -- reference spot (Yahoo close), USD/troy oz
    tail_vol_daily      DOUBLE    NOT NULL,   -- lower-tail daily-return vol (fraction)
    float_days          DOUBLE    NOT NULL,   -- expected holding period used (trading days)
    k                   DOUBLE    NOT NULL,   -- tail multiplier (cushion conservatism)
    cushion_usd_oz      DOUBLE    NOT NULL,   -- k * tail_vol_daily * sqrt(float_days) * spot
    carry_usd_oz        DOUBLE    NOT NULL,   -- financing cost over the float
    exit_floor_usd_oz   DOUBLE    NOT NULL,   -- realizable wholesale exit level
    max_buy_usd_oz      DOUBLE    NOT NULL,   -- exit_floor - cushion - carry
    max_buy_frac        DOUBLE    NOT NULL,   -- max_buy / spot
    flags               VARCHAR   NOT NULL,   -- pipe-joined fallback flags per term
    source              VARCHAR   NOT NULL,   -- engine version tag, e.g. 'spread_floor_v1'
    computed_at         TIMESTAMP NOT NULL,   -- vintage of the compute
    PRIMARY KEY (date_utc, metal, source)
);

CREATE INDEX IF NOT EXISTS idx_spread_floor_metal ON spread_floor_daily(metal);

-- book_var_daily holds the dollar Value-at-Risk on AMC's ACTUAL held inventory
-- (the ledger join). It is intentionally NOT populated by the first increment:
-- a book-level dollar risk built on an assumed float would be worse than none,
-- so rows land only once float_days is calibrated to AMC's ledger. The metal
-- 'TOTAL' carries the whole-book aggregate.

CREATE TABLE IF NOT EXISTS book_var_daily (
    date_utc        DATE      NOT NULL,   -- valuation date, UTC
    metal           VARCHAR   NOT NULL,   -- per metal, or 'TOTAL' for the book
    book_var_usd    DOUBLE    NOT NULL,   -- plausible adverse move on held inventory
    fine_oz_held    DOUBLE,               -- inventory basis (from the ledger)
    flags           VARCHAR   NOT NULL,   -- fallback flags carried from the floor engine
    source          VARCHAR   NOT NULL,   -- engine version tag
    computed_at     TIMESTAMP NOT NULL,   -- vintage of the compute
    PRIMARY KEY (date_utc, metal, source)
);
