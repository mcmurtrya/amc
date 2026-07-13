-- Phase 7.1 collectors 2-6: market-side capture tables.
-- Non-backfillable (or aging-out) series recorded as pulled. Every row
-- carries source + pulled_at provenance and an is_realtime honesty flag:
-- false = captured after the fact (setup-time history, retro pulls) and
-- therefore second-class for model training. All timestamps UTC.

-- Collector 2: retail coin-premium panel (posted dealer ask / buyback bid).
CREATE TABLE IF NOT EXISTS coin_premiums (
    pulled_at       TIMESTAMP   NOT NULL,
    dealer          VARCHAR     NOT NULL,   -- apmex | jmbullion
    product_id      VARCHAR     NOT NULL,   -- basket key from configs/premium_basket.yaml
    metal           VARCHAR     NOT NULL,
    fine_troy_oz    DOUBLE      NOT NULL,   -- melt basis per unit
    ask_usd         DOUBLE,
    bid_usd         DOUBLE,                 -- dealer buyback, where published
    spot_usd_oz     DOUBLE,
    ask_premium_pct DOUBLE,                 -- (ask/melt - 1) * 100
    bid_premium_pct DOUBLE,
    url             VARCHAR,
    source          VARCHAR     NOT NULL,
    is_realtime     BOOLEAN     NOT NULL,
    PRIMARY KEY (pulled_at, dealer, product_id)
);

-- Collector 3: Google Trends as-pulled archive. Trends rescales per request,
-- so the verbatim payload + parameters ARE the dataset.
CREATE TABLE IF NOT EXISTS search_interest (
    pulled_at       TIMESTAMP   NOT NULL,
    geo             VARCHAR     NOT NULL,   -- US, or US-XX state
    term            VARCHAR     NOT NULL,
    period_start    DATE        NOT NULL,   -- interval the value describes
    period_end      DATE        NOT NULL,
    value           INTEGER,                -- 0-100 index as returned
    request_params  JSON        NOT NULL,   -- verbatim query parameters
    source          VARCHAR     NOT NULL,
    is_realtime     BOOLEAN     NOT NULL,
    PRIMARY KEY (pulled_at, geo, term, period_start)
);

-- Collector 4: CME daily settlement volume / open interest (forward leg of
-- the spliced series -- the Databento backfill is the historical leg).
-- is_preliminary distinguishes same-day preliminary figures from official
-- finals -- the splice gate classifies semantics before any model consumes
-- the merged series.
CREATE TABLE IF NOT EXISTS cme_daily (
    trade_date      DATE        NOT NULL,
    product         VARCHAR     NOT NULL,   -- GC | SI | PL | PA
    contract_month  VARCHAR     NOT NULL,   -- e.g. 2026-08 (AGG = all months)
    settle          DOUBLE,
    volume          BIGINT,
    open_interest   BIGINT,
    oi_change       BIGINT,
    is_preliminary  BOOLEAN     NOT NULL,
    source          VARCHAR     NOT NULL,
    pulled_at       TIMESTAMP   NOT NULL,
    is_realtime     BOOLEAN     NOT NULL,
    PRIMARY KEY (trade_date, product, contract_month, is_preliminary)
);

-- Collector 6: Johnson Matthey PGM base prices (Rh/Ir/Ru have no exchange
-- price -- JM Pt/Pd kept for cross-checking quote timing against CME settles).
CREATE TABLE IF NOT EXISTS pgm_prices (
    price_date      DATE        NOT NULL,
    metal           VARCHAR     NOT NULL,   -- rhodium | iridium | ruthenium | platinum | palladium
    quote           VARCHAR     NOT NULL,   -- e.g. ny_am | ny_pm | hk_open | hk_close | london
    price_usd_oz    DOUBLE,
    source          VARCHAR     NOT NULL,
    pulled_at       TIMESTAMP   NOT NULL,
    is_realtime     BOOLEAN     NOT NULL,
    PRIMARY KEY (price_date, metal, quote)
);

-- Collector 5d: macro consensus capture (CPI, Employment Situation).
-- Append-only across pulls: pulled_at is part of the key, so pre-release
-- consensus and post-release actuals are separate rows and is_realtime is
-- decidable per row (pulled_at < release_utc).
CREATE TABLE IF NOT EXISTS macro_consensus (
    release_utc      TIMESTAMP   NOT NULL,  -- scheduled release datetime (UTC)
    release_type     VARCHAR     NOT NULL,  -- CPI | EMPSIT
    field            VARCHAR     NOT NULL,  -- e.g. cpi_mom | core_cpi_mom | nfp_change_k | unemployment_rate
    consensus        DOUBLE,
    previous         DOUBLE,
    actual           DOUBLE,                -- NULL until after release
    consensus_source VARCHAR     NOT NULL,  -- which calendar feed
    pulled_at        TIMESTAMP   NOT NULL,
    is_realtime      BOOLEAN     NOT NULL,
    PRIMARY KEY (release_utc, release_type, field, consensus_source, pulled_at)
);
