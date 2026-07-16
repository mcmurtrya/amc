-- Phase 7.1 data-hygiene: a quarantine flag for rows captured from sources whose
-- Terms of Use bar AMC's use (ToU audit 2026-07-16, see journal.md).
--
-- A NULL quarantine_reason means the row is usable. A non-NULL reason means the
-- row was acquired outside its source's licence and must be excluded from model
-- training and from any analysis shipped to AMC until a licence clears it.
-- Downstream loaders filter on quarantine_reason IS NULL.
--
-- The column is additive and nullable, so the collectors' explicit-column
-- INSERTs are unaffected and future licensed rows land un-quarantined (NULL)
-- with no code change. Stamping of the already-captured rows is done separately
-- by scripts/quarantine_barred_sources.py (a data classification, re-runnable,
-- reversible by UPDATE ... SET quarantine_reason = NULL once a licence lands) --
-- kept out of this migration so the schema step stays pure.

ALTER TABLE coin_premiums   ADD COLUMN IF NOT EXISTS quarantine_reason VARCHAR;
ALTER TABLE macro_consensus ADD COLUMN IF NOT EXISTS quarantine_reason VARCHAR;
ALTER TABLE search_interest ADD COLUMN IF NOT EXISTS quarantine_reason VARCHAR;
ALTER TABLE pgm_prices      ADD COLUMN IF NOT EXISTS quarantine_reason VARCHAR;
