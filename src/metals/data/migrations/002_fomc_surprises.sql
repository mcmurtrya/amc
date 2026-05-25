-- Phase 2.3: Bauer–Swanson high-frequency FOMC policy surprises.
-- Source: SF Fed updates of Bauer–Swanson (2023). One row per FOMC
-- announcement (scheduled and unscheduled). MPS is the composite policy
-- surprise; MPS_ORTH is the version orthogonalized to remove the "Fed
-- information effect" (the cleanest single-number stance surprise).

CREATE TABLE IF NOT EXISTS fomc_surprises (
    timestamp_utc   TIMESTAMP   NOT NULL,
    is_unscheduled  BOOLEAN,
    ff1             DOUBLE,     -- current-month fed funds futures surprise
    ff2             DOUBLE,     -- next-month fed funds futures surprise
    ed4             DOUBLE,     -- 4-quarter-ahead Eurodollar (path)
    mps             DOUBLE,     -- composite monetary policy surprise
    mps_orth        DOUBLE,     -- orthogonalized MPS (Bauer-Swanson 2023)
    source          VARCHAR     NOT NULL,
    PRIMARY KEY (timestamp_utc)
);

CREATE INDEX IF NOT EXISTS idx_fomc_surprises_orth ON fomc_surprises(mps_orth);
