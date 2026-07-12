-- Phase 3 persistence: daily text features, topic prevalences, and cluster
-- assignments. All keyed on (timestamp_utc, ...) so they slot into the same
-- canonical time axis as prices / macro / events.
--
-- Renamed from 005_phase3_artifacts.sql (2026-07-02) to fix the duplicate
-- 005 prefix. DBs that already ran it under the old stem keep the stale
-- 005_phase3_artifacts row in _schema_migrations and re-run this file as a
-- no-op -- every statement below is IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS daily_text_features (
    timestamp_utc     TIMESTAMP   NOT NULL,
    metal             VARCHAR     NOT NULL,
    n_articles        INTEGER     NOT NULL,
    mean_embedding    BLOB,          -- float32 vector packed via numpy.tobytes
    embedding_dim     INTEGER,
    embedding_dispersion DOUBLE,     -- mean cosine distance from the day's centroid
    mean_tone_overall DOUBLE,
    mean_tone_positive DOUBLE,
    mean_tone_negative DOUBLE,
    PRIMARY KEY (timestamp_utc, metal)
);

CREATE INDEX IF NOT EXISTS idx_daily_text_features_metal
    ON daily_text_features(metal);

-- Long-format per-day topic prevalence. One row per (date, topic_id).
-- The wide pivot is reconstructed at use time in metals.features.context.
CREATE TABLE IF NOT EXISTS daily_topic_prevalence (
    timestamp_utc TIMESTAMP NOT NULL,
    topic_id      INTEGER   NOT NULL,
    prevalence    DOUBLE    NOT NULL,
    PRIMARY KEY (timestamp_utc, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_daily_topic_prevalence_topic
    ON daily_topic_prevalence(topic_id);

-- Per-day cluster assignment from the UMAP + HDBSCAN pipeline.
-- ``cluster_id = -1`` means HDBSCAN noise / outlier.
CREATE TABLE IF NOT EXISTS cluster_assignments (
    timestamp_utc TIMESTAMP NOT NULL,
    model_version VARCHAR   NOT NULL,
    cluster_id    INTEGER   NOT NULL,
    confidence    DOUBLE,
    PRIMARY KEY (timestamp_utc, model_version)
);

CREATE INDEX IF NOT EXISTS idx_cluster_assignments_cluster
    ON cluster_assignments(model_version, cluster_id);

-- Cluster centroids in the learned (post-UMAP) space, plus optional
-- human-assigned label. label_source tracks how the label was assigned
-- so we can re-run downstream analysis on auto-labels alone if needed.
CREATE TABLE IF NOT EXISTS cluster_centroids (
    model_version VARCHAR NOT NULL,
    cluster_id    INTEGER NOT NULL,
    n_members     INTEGER NOT NULL,
    centroid      BLOB,          -- packed float32 vector
    centroid_dim  INTEGER,
    label         VARCHAR,
    label_source  VARCHAR,       -- 'manual' | 'llm' | 'auto' | NULL
    description   VARCHAR,
    PRIMARY KEY (model_version, cluster_id)
);
