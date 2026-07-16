-- Phase 7.2: a same-evening FOMC monetary-surprise proxy — the Hanson-Stein (2015)
-- daily change in the 2-year Treasury yield (DGS2) on FOMC announcement days.
--
-- Derived by joining events(event_type='FOMC') to macro(series_id='DGS2') and
-- differencing announcement-day close vs the prior DGS2 trading-day close. A rise
-- in the 2y = hawkish (positive delta_dgs2_bp). It is the free, same-evening stand-in
-- for the intraday GSS target+path composite, which needs Databento minute bars that
-- are currently out of scope (plan 7.2 / 7.7). Unlike an intraday window it absorbs
-- the whole session's news, so it is a noisier hawkishness signal -- a
-- coverage-extending robustness treatment, cross-checked against Bauer-Swanson MPS.
--
-- Materialized (not a live feature) so the DGS2 vintage used is pinned by pulled_at.
-- Provenance follows the Phase 7.1 convention (source/pulled_at/is_realtime): a
-- backfill from the current FRED vintage is is_realtime=false; a genuine
-- meeting-evening capture (within REALTIME_WINDOW_DAYS of the meeting) is true.

CREATE TABLE IF NOT EXISTS fomc_yield_surprises (
    timestamp_utc    TIMESTAMP NOT NULL,   -- FOMC announcement date (joins events + fomc_surprises)
    is_unscheduled   BOOLEAN,              -- inter-meeting / emergency action (from events metadata)
    dgs2_release     DOUBLE    NOT NULL,   -- 2y close on the announcement day (percentage points)
    dgs2_prev        DOUBLE    NOT NULL,   -- prior DGS2 trading-day close (percentage points)
    prev_trading_day DATE      NOT NULL,   -- the actual prior DGS2 trading day used (not calendar -1)
    delta_dgs2_bp    DOUBLE    NOT NULL,   -- (release - prev) * 100 basis points; positive = hawkish
    source           VARCHAR   NOT NULL,   -- 'fred_dgs2_derived'
    pulled_at        TIMESTAMP NOT NULL,   -- vintage of the DGS2 values used
    is_realtime      BOOLEAN   NOT NULL,   -- true only for genuine meeting-evening captures
    PRIMARY KEY (timestamp_utc)
);
