-- Phase 3 step 3.5 / 3.7: extend the headlines table with GDELT V2Tone
-- fields. Per-article tone metrics are much more useful as typed columns
-- than as JSON for the daily aggregation in step 3.7. The themes JSON
-- column already exists from migration 001.

ALTER TABLE headlines ADD COLUMN IF NOT EXISTS tone_overall   DOUBLE;
ALTER TABLE headlines ADD COLUMN IF NOT EXISTS tone_positive  DOUBLE;
ALTER TABLE headlines ADD COLUMN IF NOT EXISTS tone_negative  DOUBLE;
ALTER TABLE headlines ADD COLUMN IF NOT EXISTS tone_polarity  DOUBLE;
-- Activity Reference Density and Self/Group Reference Density are the
-- two remaining V2Tone fields. Mostly used as quality indicators.
ALTER TABLE headlines ADD COLUMN IF NOT EXISTS tone_ard       DOUBLE;
ALTER TABLE headlines ADD COLUMN IF NOT EXISTS tone_sgrd      DOUBLE;
