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
