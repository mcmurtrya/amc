-- Phase 1 housekeeping (2026-06-18): reclaim redundant URL storage.
--
-- The `headline` column was a byte-for-byte copy of `article_url`. GDELT GKG
-- has no headline text, so gdelt.py set headline equal to article_url as a
-- placeholder fallback. With ~14M rows and a ~93-char URL per row, that
-- duplicate column cost well over a gigabyte for zero information.
--
-- Dropping it removes one of three per-row copies of the URL. The other two
-- (article_url itself and the headline_id dedup key, which embeds the URL
-- truncated to 200 chars) are both kept.
--
-- This updates the schema but DuckDB does not shrink the data file in place on
-- DROP COLUMN. To reclaim the bytes on an existing database, run
-- scripts/compact_headlines.py, which rebuilds a fresh, compacted copy with
-- this column already absent.
--
-- DuckDB also refuses ALTER TABLE ... DROP COLUMN while an index exists on the
-- table (even an index on a different column), so the source index is dropped
-- first and recreated afterward.
--
-- IMPORTANT for future migrations: do not put a semicolon character inside any
-- comment in a migration file. A stray one inside a comment truncates DuckDB's
-- whole-file statement execution and silently skips the later statements.
DROP INDEX IF EXISTS idx_headlines_source;
ALTER TABLE headlines DROP COLUMN IF EXISTS headline;
CREATE INDEX IF NOT EXISTS idx_headlines_source ON headlines(source);
