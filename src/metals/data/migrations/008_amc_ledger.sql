-- Phase 7.1 collector 1: AMC's own ledger.
-- Local-only business data: these tables hold AMC Company's transaction
-- records and must never leave this machine (no cloud sync, never in git).
-- All timestamps UTC. Rows carry import-batch provenance (source_file,
-- batch_id, imported_at) so every figure traces to a specific export.

CREATE TABLE IF NOT EXISTS amc_scrap_lots (
    lot_id          VARCHAR     NOT NULL,   -- stable id from AMC's books
    purchased_utc   TIMESTAMP   NOT NULL,
    metal           VARCHAR     NOT NULL,   -- gold | silver | platinum | palladium
    gross_weight_g  DOUBLE      NOT NULL,
    fineness        DOUBLE      NOT NULL,   -- assayed fine fraction, 0-1
    fine_troy_oz    DOUBLE      NOT NULL,
    price_paid_usd  DOUBLE      NOT NULL,
    spot_usd_oz     DOUBLE,                 -- spot at purchase, if recorded
    disposed_utc    TIMESTAMP,              -- NULL while the lot is open
    disposition     VARCHAR,                -- refined | sold | melted | other
    proceeds_usd    DOUBLE,
    notes           VARCHAR,
    source_file     VARCHAR     NOT NULL,
    batch_id        VARCHAR     NOT NULL,
    imported_at     TIMESTAMP   NOT NULL,
    PRIMARY KEY (lot_id)
);

CREATE INDEX IF NOT EXISTS idx_amc_scrap_lots_purchased ON amc_scrap_lots(purchased_utc);

CREATE TABLE IF NOT EXISTS amc_coin_trades (
    trade_id            VARCHAR     NOT NULL,
    traded_utc          TIMESTAMP   NOT NULL,
    side                VARCHAR     NOT NULL,   -- buy | sell
    product             VARCHAR     NOT NULL,   -- e.g. american_gold_eagle_1oz
    quantity            INTEGER     NOT NULL,
    unit_price_usd      DOUBLE      NOT NULL,
    spot_usd_oz         DOUBLE,
    metal               VARCHAR     NOT NULL,
    fine_troy_oz_per_unit DOUBLE,               -- melt basis for premium calc
    notes               VARCHAR,
    source_file         VARCHAR     NOT NULL,
    batch_id            VARCHAR     NOT NULL,
    imported_at         TIMESTAMP   NOT NULL,
    PRIMARY KEY (trade_id)
);

CREATE INDEX IF NOT EXISTS idx_amc_coin_trades_traded ON amc_coin_trades(traded_utc);

CREATE TABLE IF NOT EXISTS amc_till_daily (
    date_utc        DATE        NOT NULL,
    walk_ins        INTEGER,
    offers_made     INTEGER,
    offers_accepted INTEGER,
    notes           VARCHAR,
    source_file     VARCHAR     NOT NULL,
    batch_id        VARCHAR     NOT NULL,
    imported_at     TIMESTAMP   NOT NULL,
    PRIMARY KEY (date_utc)
);
