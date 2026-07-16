"""ΔDGS2 same-evening FOMC monetary-surprise proxy (Phase 7.2).

Materializes ``fomc_yield_surprises`` (migration 012): for each FOMC announcement
day, the Hanson-Stein (2015) daily change in the 2-year Treasury yield — the
announcement-day DGS2 close minus the prior DGS2 trading-day close, in basis
points. A rise in the 2y = hawkish (positive). Derived purely from data already
in the DB (``events`` FOMC dates ⋈ ``macro`` DGS2), so run it after
``metals.refresh`` has updated DGS2.

Why it exists: the Bauer-Swanson MPS series (``fomc_surprises``) is a static
academic panel that ends 2023-12, leaving ~21 recent meetings with no surprise
measure. ΔDGS2 is the free, same-evening stand-in that any meeting evening
produces from the routine FRED refresh — extending the series to the present and
back to 2007. It is the daily-close proxy for the intraday GSS target+path
composite (Databento-blocked); a full session absorbs more than the announcement
window, so it is a noisier signal, validated against MPS on the overlap
(``validate_against_mps``).

Prior-trading-day alignment uses the actual DGS2 observation index (a window LAG),
never calendar −1 — so a Monday meeting correctly differences against the prior
Friday. FOMC days with no same-day DGS2 (weekend/holiday emergency actions;
Treasuries didn't trade) are excluded, not silently nulled.

Provenance (Phase 7.1 convention): a backfill from the current FRED vintage is
``is_realtime=False``; a genuine meeting-evening capture (within
``REALTIME_WINDOW_DAYS`` of the meeting) is True, and the upsert never demotes a
realtime row nor overwrites its pinned values.

Run as (after metals.refresh):
    uv run python -m metals.data.fomc_dgs2
    uv run python -m metals.data.fomc_dgs2 --validate   # also print the MPS cross-check
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from metals.data.db import connection

SOURCE_TAG = "fred_dgs2_derived"
REALTIME_WINDOW_DAYS = 7

# ΔDGS2 per FOMC day: LAG over the DGS2 observation index gives the true prior
# trading day (holidays/weekends skipped for free). The join keeps only FOMC days
# that are themselves DGS2 trading days; days with no same-day DGS2 fall out.
_DELTA_SQL = """
WITH dgs AS (
    SELECT CAST(timestamp_utc AS DATE) AS d,
           value AS y,
           LAG(value) OVER (ORDER BY timestamp_utc) AS y_prev,
           LAG(CAST(timestamp_utc AS DATE)) OVER (ORDER BY timestamp_utc) AS d_prev
    FROM macro
    WHERE series_id = 'DGS2'
),
fomc AS (
    SELECT CAST(timestamp_utc AS DATE) AS d,
           COALESCE(
               TRY_CAST(json_extract(metadata, '$.is_scheduled') AS BOOLEAN), TRUE
           ) AS is_scheduled
    FROM events
    WHERE event_type = 'FOMC'
)
SELECT f.d                        AS fomc_day,
       NOT f.is_scheduled         AS is_unscheduled,
       dgs.y                      AS dgs2_release,
       dgs.y_prev                 AS dgs2_prev,
       dgs.d_prev                 AS prev_trading_day,
       (dgs.y - dgs.y_prev) * 100.0 AS delta_dgs2_bp
FROM fomc f
JOIN dgs ON f.d = dgs.d
WHERE dgs.y_prev IS NOT NULL
ORDER BY f.d
"""

# FOMC days with no same-day DGS2 observation (excluded from the proxy).
_EXCLUDED_SQL = """
SELECT CAST(e.timestamp_utc AS DATE) AS fomc_day
FROM events e
WHERE e.event_type = 'FOMC'
  AND CAST(e.timestamp_utc AS DATE) NOT IN (
      SELECT CAST(timestamp_utc AS DATE) FROM macro WHERE series_id = 'DGS2'
  )
ORDER BY fomc_day
"""

# Never demote a realtime row nor overwrite its pinned (as-known) values: for each
# value column, keep the existing value when the stored row is realtime, else take
# the incoming one; is_realtime is OR'd so it can only ever be promoted.
_PIN_COLS = (
    "is_unscheduled",
    "dgs2_release",
    "dgs2_prev",
    "prev_trading_day",
    "delta_dgs2_bp",
    "source",
    "pulled_at",
)
_T = "fomc_yield_surprises"
_SET_CLAUSE = ",\n    ".join(
    f"{c} = CASE WHEN {_T}.is_realtime THEN {_T}.{c} ELSE EXCLUDED.{c} END" for c in _PIN_COLS
)
_INSERT_SQL = f"""
INSERT INTO fomc_yield_surprises
    (timestamp_utc, is_unscheduled, dgs2_release, dgs2_prev, prev_trading_day,
     delta_dgs2_bp, source, pulled_at, is_realtime)
SELECT timestamp_utc, is_unscheduled, dgs2_release, dgs2_prev,
       CAST(prev_trading_day AS DATE), delta_dgs2_bp, source, pulled_at, is_realtime
FROM incoming_fomc_dgs2
ON CONFLICT (timestamp_utc) DO UPDATE SET
    {_SET_CLAUSE},
    is_realtime = {_T}.is_realtime OR EXCLUDED.is_realtime
"""


def build_delta_dgs2(conn: Any) -> tuple[pd.DataFrame, list[Any]]:
    """Compute ΔDGS2 per FOMC day. Returns (rows, excluded_fomc_days)."""
    df = conn.execute(_DELTA_SQL).fetch_df()
    excluded = [r[0] for r in conn.execute(_EXCLUDED_SQL).fetchall()]
    return df, excluded


def refresh(pulled_at: datetime | None = None) -> dict:
    """Materialize the ΔDGS2 surprise series into fomc_yield_surprises. Returns a summary."""
    if pulled_at is None:
        pulled_at = datetime.now(UTC).replace(tzinfo=None)
    with connection() as conn:
        df, excluded = build_delta_dgs2(conn)
        if df.empty:
            return {"rows_written": 0, "excluded_fomc_days": [str(d) for d in excluded]}
        out = df.copy()
        out["timestamp_utc"] = pd.to_datetime(out["fomc_day"])
        out = out.drop(columns=["fomc_day"])
        out["source"] = SOURCE_TAG
        out["pulled_at"] = pulled_at
        lag_days = (
            pd.Timestamp(pulled_at).normalize() - out["timestamp_utc"].dt.normalize()
        ).dt.days
        out["is_realtime"] = lag_days <= REALTIME_WINDOW_DAYS
        conn.register("incoming_fomc_dgs2", out)
        conn.execute(_INSERT_SQL)
        conn.unregister("incoming_fomc_dgs2")
        n_hawkish = int((out["delta_dgs2_bp"] > 0).sum())
        n_dovish = int((out["delta_dgs2_bp"] < 0).sum())
    return {
        "source": SOURCE_TAG,
        "rows_written": int(len(out)),
        "date_range": [
            out["timestamp_utc"].min().date().isoformat(),
            out["timestamp_utc"].max().date().isoformat(),
        ],
        "n_hawkish": n_hawkish,
        "n_dovish": n_dovish,
        "n_excluded": len(excluded),
        "excluded_fomc_days": [str(d) for d in excluded],
        "realtime_rows": int(out["is_realtime"].sum()),
        "pulled_at": pulled_at.isoformat(),
    }


def validate_against_mps(conn: Any, since: str | None = None) -> dict:
    """Cross-check ΔDGS2 against the Bauer-Swanson MPS/MPS_ORTH on the overlap.

    Returns Pearson correlations and sign-agreement rates. ``since`` restricts to a
    date floor (e.g. "2015-01-01" for the modelling window).
    """
    where = "WHERE s.mps IS NOT NULL"
    if since is not None:
        where += f" AND CAST(y.timestamp_utc AS DATE) >= DATE '{since}'"
    query = f"""
        SELECT y.delta_dgs2_bp, s.mps, s.mps_orth
        FROM fomc_yield_surprises y
        JOIN fomc_surprises s
          ON CAST(y.timestamp_utc AS DATE) = CAST(s.timestamp_utc AS DATE)
        {where}
        ORDER BY y.timestamp_utc
    """
    df = conn.execute(query).fetch_df()
    if df.empty:
        return {"n_overlap": 0, "since": since}
    d = df["delta_dgs2_bp"]

    def _sign_agree(other: str) -> float:
        mask = df[other].notna() & (df[other] != 0) & (d != 0)
        if not mask.any():
            return float("nan")
        return float((np.sign(d[mask]) == np.sign(df[other][mask])).mean())

    return {
        "n_overlap": int(len(df)),
        "since": since,
        "corr_mps": float(d.corr(df["mps"])),
        "corr_mps_orth": float(d.corr(df["mps_orth"])),
        "sign_agree_mps": _sign_agree("mps"),
        "sign_agree_mps_orth": _sign_agree("mps_orth"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ΔDGS2 FOMC surprise proxy (Phase 7.2).")
    parser.add_argument(
        "--validate", action="store_true", help="Also print the Bauer-Swanson MPS cross-check."
    )
    args = parser.parse_args()
    summary = refresh()
    print(f"Rows written:  {summary['rows_written']}")
    print(f"Date range:    {summary['date_range']}")
    print(f"Hawkish/dovish: {summary['n_hawkish']} / {summary['n_dovish']}")
    print(
        f"Excluded FOMC days (no same-day DGS2): {summary['n_excluded']} -> "
        f"{summary['excluded_fomc_days']}"
    )
    if args.validate:
        with connection(read_only=True) as conn:
            full = validate_against_mps(conn)
            modern = validate_against_mps(conn, since="2015-01-01")
        print("\nValidation vs Bauer-Swanson MPS (overlap):")
        for label, v in (("full overlap", full), ("2015+", modern)):
            if v["n_overlap"]:
                print(
                    f"  {label:12} n={v['n_overlap']:>3}  corr(MPS)={v['corr_mps']:+.3f}  "
                    f"corr(MPS_ORTH)={v['corr_mps_orth']:+.3f}  "
                    f"sign-agree(ORTH)={v['sign_agree_mps_orth']:.0%}"
                )


if __name__ == "__main__":
    main()
