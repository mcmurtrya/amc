-- Wide GKG ingest (Phase 3): real article titles and source language.
--
-- page_title comes from the GKG Extras column's <PAGE_TITLE> tag (present on
-- ~99.6% of themed rows), HTML-entity-decoded at parse time and stored in the
-- article's original language. src_lang comes from TranslationInfo, where an
-- empty upstream value means English-original and is stored as 'eng'.
--
-- NULL in either column means the row was ingested before this migration and
-- has not yet been re-pulled wide -- do not conflate NULL src_lang with
-- English. The upsert in metals.data.gdelt uses COALESCE on conflict so a
-- narrow re-pull never overwrites populated values with NULL.
--
-- (Reminder: never put a semicolon inside a comment in these files.)

ALTER TABLE headlines ADD COLUMN IF NOT EXISTS page_title VARCHAR;

ALTER TABLE headlines ADD COLUMN IF NOT EXISTS src_lang VARCHAR;
