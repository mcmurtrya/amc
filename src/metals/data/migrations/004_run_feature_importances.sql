-- Phase 1 cleanup: per-split feature importances for tree-based models.
-- Recorded by metals.eval.harness.log_feature_importances. One row per
-- (run, split, feature, importance_type) so we can compare 'gain' vs 'split'
-- and trace how feature ranks shift across walk-forward splits.

CREATE TABLE IF NOT EXISTS run_feature_importances (
    run_id           VARCHAR     NOT NULL,
    split_id         INTEGER     NOT NULL,
    feature_name     VARCHAR     NOT NULL,
    importance       DOUBLE      NOT NULL,
    importance_type  VARCHAR     NOT NULL DEFAULT 'gain',
    PRIMARY KEY (run_id, split_id, feature_name, importance_type)
);

CREATE INDEX IF NOT EXISTS idx_run_feature_importances_run
    ON run_feature_importances(run_id);
