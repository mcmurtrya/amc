-- Phase 0 initial schema.
-- All timestamps are UTC.

CREATE TABLE IF NOT EXISTS prices (
    timestamp_utc   TIMESTAMP   NOT NULL,
    ticker          VARCHAR     NOT NULL,
    open            DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    close           DOUBLE,
    adj_close       DOUBLE,
    volume          BIGINT,
    source          VARCHAR     NOT NULL,
    PRIMARY KEY (timestamp_utc, ticker)
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);

CREATE TABLE IF NOT EXISTS macro (
    timestamp_utc   TIMESTAMP   NOT NULL,
    series_id       VARCHAR     NOT NULL,
    value           DOUBLE,
    source          VARCHAR     NOT NULL,
    PRIMARY KEY (timestamp_utc, series_id)
);

CREATE INDEX IF NOT EXISTS idx_macro_series ON macro(series_id);

CREATE TABLE IF NOT EXISTS events (
    timestamp_utc   TIMESTAMP   NOT NULL,
    event_type      VARCHAR     NOT NULL,
    event_id        VARCHAR     NOT NULL,
    metadata        JSON,
    source          VARCHAR     NOT NULL,
    PRIMARY KEY (timestamp_utc, event_type, event_id)
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

CREATE TABLE IF NOT EXISTS positioning (
    timestamp_utc           TIMESTAMP   NOT NULL,
    metal                   VARCHAR     NOT NULL,
    commercial_long         BIGINT,
    commercial_short        BIGINT,
    managed_money_long      BIGINT,
    managed_money_short     BIGINT,
    other_reportable_long   BIGINT,
    other_reportable_short  BIGINT,
    non_reportable_long     BIGINT,
    non_reportable_short    BIGINT,
    open_interest           BIGINT,
    source                  VARCHAR     NOT NULL,
    PRIMARY KEY (timestamp_utc, metal)
);

CREATE TABLE IF NOT EXISTS headlines (
    timestamp_utc   TIMESTAMP   NOT NULL,
    headline_id     VARCHAR     NOT NULL,
    source          VARCHAR     NOT NULL,
    headline        VARCHAR     NOT NULL,
    themes          JSON,
    article_url     VARCHAR,
    PRIMARY KEY (timestamp_utc, headline_id)
);

CREATE INDEX IF NOT EXISTS idx_headlines_source ON headlines(source);
