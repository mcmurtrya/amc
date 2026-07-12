"""Loaders that read price and macro data from DuckDB into wide-format pandas frames."""

from __future__ import annotations

import pandas as pd

from metals.data.db import connection


def load_prices(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    column: str = "adj_close",
) -> pd.DataFrame:
    """Load price data wide: index=timestamp_utc, columns=ticker, values=`column`."""
    where = ["1=1"]
    params: list = []
    if tickers:
        placeholders = ",".join(["?"] * len(tickers))
        where.append(f"ticker IN ({placeholders})")
        params.extend(tickers)
    if start:
        where.append("timestamp_utc >= ?")
        params.append(start)
    if end:
        where.append("timestamp_utc <= ?")
        params.append(end)
    sql = (
        f"SELECT timestamp_utc, ticker, {column} AS value "
        f"FROM prices WHERE {' AND '.join(where)} "
        f"ORDER BY timestamp_utc, ticker"
    )
    with connection() as conn:
        long = conn.execute(sql, params).fetchdf()
    if long.empty:
        return pd.DataFrame()
    wide = long.pivot(index="timestamp_utc", columns="ticker", values="value")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    return wide


def load_macro(
    series_ids: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load macro data wide: index=timestamp_utc, columns=series_id, values=value."""
    where = ["1=1"]
    params: list = []
    if series_ids:
        placeholders = ",".join(["?"] * len(series_ids))
        where.append(f"series_id IN ({placeholders})")
        params.extend(series_ids)
    if start:
        where.append("timestamp_utc >= ?")
        params.append(start)
    if end:
        where.append("timestamp_utc <= ?")
        params.append(end)
    sql = (
        f"SELECT timestamp_utc, series_id, value "
        f"FROM macro WHERE {' AND '.join(where)} "
        f"ORDER BY timestamp_utc, series_id"
    )
    with connection() as conn:
        long = conn.execute(sql, params).fetchdf()
    if long.empty:
        return pd.DataFrame()
    wide = long.pivot(index="timestamp_utc", columns="series_id", values="value")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    return wide


def load_fomc_surprises(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load Bauer-Swanson FOMC surprises, indexed by ``timestamp_utc``.

    One row per FOMC announcement. Columns: is_unscheduled, ff1, ff2, ed4, mps,
    mps_orth. ``mps_orth`` is the orthogonalised monetary-policy surprise
    (positive = hawkish) — the canonical single-column stance measure used as
    the FOMC treatment driver. Returns an empty frame when no rows match.
    """
    where = ["1=1"]
    params: list = []
    if start:
        where.append("timestamp_utc >= ?")
        params.append(start)
    if end:
        where.append("timestamp_utc <= ?")
        params.append(end)
    sql = (
        "SELECT timestamp_utc, is_unscheduled, ff1, ff2, ed4, mps, mps_orth "
        f"FROM fomc_surprises WHERE {' AND '.join(where)} "
        "ORDER BY timestamp_utc"
    )
    with connection() as conn:
        df = conn.execute(sql, params).fetchdf()
    if df.empty:
        return pd.DataFrame()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df.set_index("timestamp_utc").sort_index()


def load_events(
    event_type: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load scheduled events in long form.

    Columns: timestamp_utc, event_type, event_id, metadata (JSON string). Only
    ``event_type='FOMC'`` is currently populated. ``timestamp_utc`` is a column
    (not the index) because it is not unique across event types. Returns an
    empty frame when no rows match.
    """
    where = ["1=1"]
    params: list = []
    if event_type:
        where.append("event_type = ?")
        params.append(event_type)
    if start:
        where.append("timestamp_utc >= ?")
        params.append(start)
    if end:
        where.append("timestamp_utc <= ?")
        params.append(end)
    sql = (
        "SELECT timestamp_utc, event_type, event_id, metadata "
        f"FROM events WHERE {' AND '.join(where)} "
        "ORDER BY timestamp_utc, event_type, event_id"
    )
    with connection() as conn:
        df = conn.execute(sql, params).fetchdf()
    if df.empty:
        return pd.DataFrame()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def load_positioning(
    metal: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load CFTC COT positioning.

    ``timestamp_utc`` is the **Friday release date** — already lagged for
    point-in-time availability, so do NOT shift it again (dating it to the
    Tuesday positioning date is the classic COT leak). ``metal`` is a lowercase
    name in {gold, silver, platinum, palladium}.

    With a single ``metal`` the frame is indexed by ``timestamp_utc`` (the
    ``metal`` column dropped); otherwise a ``metal`` column is retained and the
    frame is long. Returns an empty frame when no rows match.
    """
    where = ["1=1"]
    params: list = []
    if metal:
        where.append("metal = ?")
        params.append(metal)
    if start:
        where.append("timestamp_utc >= ?")
        params.append(start)
    if end:
        where.append("timestamp_utc <= ?")
        params.append(end)
    sql = (
        "SELECT timestamp_utc, metal, commercial_long, commercial_short, "
        "managed_money_long, managed_money_short, other_reportable_long, "
        "other_reportable_short, non_reportable_long, non_reportable_short, "
        "open_interest "
        f"FROM positioning WHERE {' AND '.join(where)} "
        "ORDER BY timestamp_utc, metal"
    )
    with connection() as conn:
        df = conn.execute(sql, params).fetchdf()
    if df.empty:
        return pd.DataFrame()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    if metal:
        return df.drop(columns="metal").set_index("timestamp_utc").sort_index()
    return df.sort_values(["timestamp_utc", "metal"]).reset_index(drop=True)
