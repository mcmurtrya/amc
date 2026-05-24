"""Daily OHLCV ingestion from Yahoo Finance.

Run as:
    uv run python -m metals.data.prices --refresh

The module fetches all tickers listed in ``configs/universe.yaml`` and upserts
into the ``prices`` table of the canonical DuckDB store.

Coverage validation (≥95% of expected trading days per ticker per year) is
reported but does not abort ingestion — yfinance often has gaps for futures
contracts that we want to record and inspect, not silently retry.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from metals.data.config import universe
from metals.data.db import connection

SOURCE_TAG = "yfinance"


def _flatten_tickers(cfg: dict) -> list[dict]:
    """Return the flat list of {ticker, name} from universe.yaml."""
    out: list[dict] = []
    for group in ("metals", "etfs", "benchmarks"):
        out.extend(cfg.get(group, []) or [])
    return out


def fetch_yfinance(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Pull OHLCV from yfinance.

    Returns a long DataFrame with columns:
      timestamp_utc, ticker, open, high, low, close, adj_close, volume.

    Imports ``yfinance`` lazily so importing this module never requires
    network or the yfinance dependency at test time.
    """
    import yfinance as yf

    tickers = list(tickers)
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    frames: list[pd.DataFrame] = []
    col_rename = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    }
    # Multi-ticker downloads return a column MultiIndex; single returns flat.
    if isinstance(raw.columns, pd.MultiIndex):
        for tk in tickers:
            if tk not in raw.columns.get_level_values(0):
                continue
            sub = raw[tk].reset_index(names="timestamp_utc").rename(columns=col_rename)
            sub["ticker"] = tk
            frames.append(sub)
    else:
        sub = raw.reset_index(names="timestamp_utc").rename(columns=col_rename)
        sub["ticker"] = tickers[0]
        frames.append(sub)

    if not frames:
        return pd.DataFrame(
            columns=["timestamp_utc", "ticker", "open", "high", "low",
                     "close", "adj_close", "volume"]
        )

    df = pd.concat(frames, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["close"])
    return df[["timestamp_utc", "ticker", "open", "high", "low",
               "close", "adj_close", "volume"]]


def upsert_prices(df: pd.DataFrame) -> int:
    """Idempotent upsert into the prices table. Returns row count written."""
    if df.empty:
        return 0
    insert_df = df.copy()
    insert_df["source"] = SOURCE_TAG
    with connection() as conn:
        conn.register("incoming_prices", insert_df)
        conn.execute(
            """
            INSERT INTO prices
                (timestamp_utc, ticker, open, high, low, close, adj_close, volume, source)
            SELECT timestamp_utc, ticker, open, high, low, close, adj_close, volume, source
            FROM incoming_prices
            ON CONFLICT (timestamp_utc, ticker) DO UPDATE SET
                open      = EXCLUDED.open,
                high      = EXCLUDED.high,
                low       = EXCLUDED.low,
                close     = EXCLUDED.close,
                adj_close = EXCLUDED.adj_close,
                volume    = EXCLUDED.volume,
                source    = EXCLUDED.source
            """
        )
        conn.unregister("incoming_prices")
    return len(insert_df)


def coverage_report(df: pd.DataFrame, expected_per_year: int = 250) -> pd.DataFrame:
    """Per-ticker, per-year coverage as a fraction of expected trading days."""
    if df.empty:
        return pd.DataFrame(columns=["ticker", "year", "rows", "coverage"])
    df = df.copy()
    df["year"] = df["timestamp_utc"].dt.year
    g = df.groupby(["ticker", "year"]).size().reset_index(name="rows")
    g["coverage"] = g["rows"] / expected_per_year
    return g.sort_values(["ticker", "year"]).reset_index(drop=True)


def refresh(start: str | None = None, end: str | None = None) -> dict:
    """Refresh all configured tickers, upsert, and return a summary dict."""
    cfg = universe()
    tickers = [row["ticker"] for row in _flatten_tickers(cfg)]
    dr = cfg.get("date_range", {})
    start = start or dr.get("start") or "2007-01-01"
    end = end or dr.get("end") or datetime.now(timezone.utc).date().isoformat()

    df = fetch_yfinance(tickers, start=start, end=end)
    n = upsert_prices(df)
    cov = coverage_report(df)
    low_cov = cov[cov["coverage"] < 0.95]
    return {
        "tickers": tickers,
        "rows_written": n,
        "date_range": [start, end],
        "low_coverage_rows": low_cov.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh price data from yfinance.")
    parser.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD)")
    args = parser.parse_args()
    summary = refresh(start=args.start, end=args.end)
    print(f"Tickers refreshed: {len(summary['tickers'])}")
    print(f"Rows written:      {summary['rows_written']}")
    print(f"Date range:        {summary['date_range']}")
    if summary["low_coverage_rows"]:
        print(f"WARNING: {len(summary['low_coverage_rows'])} (ticker, year) "
              f"buckets under 95% expected coverage.")


if __name__ == "__main__":
    main()
