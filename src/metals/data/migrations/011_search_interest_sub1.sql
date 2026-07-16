-- Google Trends' manual CSV export encodes a nonzero interest value below 1 as
-- the literal token "<1" on the co-scaled 0-100 index -- distinct from a true
-- "0". The scraper (retired 2026-07-16) never saw this: the JSON API returned an
-- integer 0 there. The CSV importer records "<1" as value = 0 with value_lt1 =
-- TRUE, so the "present but sub-1" signal is never silently merged into true
-- zero, and no row is ever dropped (dropping would shift every later date and
-- corrupt the weekly series).
--
-- Nullable, no default: pre-importer rows (all quarantined, migration 010) keep
-- NULL, meaning "not applicable -- captured before this distinction existed".
-- The importer sets it explicitly on every row it writes.

ALTER TABLE search_interest ADD COLUMN IF NOT EXISTS value_lt1 BOOLEAN;
