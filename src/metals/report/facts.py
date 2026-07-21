"""Live project state, read from DuckDB at report time.

Every number a report prints about *current* state comes from here rather than
being typed into prose, so a stale figure is impossible by construction: if the
spread-floor engine is re-run, the next report moves with it.

Each getter degrades to an explicit "unavailable" rather than a plausible
default. A report that cannot find a number must say so — silently substituting
a stand-in is exactly the failure this project's flag discipline exists to
prevent.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from metals.data.db import connection


@dataclass(frozen=True)
class LedgerStatus:
    """Row counts for AMC's own books — the gate on dollar-denominated work."""

    scrap_lots: int
    coin_trades: int
    till_days: int

    @property
    def populated(self) -> bool:
        return (self.scrap_lots + self.coin_trades + self.till_days) > 0


def _scalar(sql: str, default: Any = None) -> Any:
    try:
        with connection(read_only=True) as conn:
            row = conn.execute(sql).fetchone()
    except Exception:
        return default
    if row is None or row[0] is None:
        return default
    return row[0]


def _frame(sql: str) -> pd.DataFrame:
    try:
        with connection(read_only=True) as conn:
            return conn.execute(sql).fetchdf()
    except Exception:
        return pd.DataFrame()


def ledger_status() -> LedgerStatus:
    return LedgerStatus(
        scrap_lots=int(_scalar("SELECT count(*) FROM amc_scrap_lots", 0) or 0),
        coin_trades=int(_scalar("SELECT count(*) FROM amc_coin_trades", 0) or 0),
        till_days=int(_scalar("SELECT count(*) FROM amc_till_daily", 0) or 0),
    )


def latest_spread_floors() -> pd.DataFrame:
    """Most recent spread-floor row per metal, newest computation only.

    Empty frame when the engine has not been run.
    """
    return _frame(
        """
        SELECT metal, date_utc, spot_usd_oz, exit_floor_usd_oz, cushion_usd_oz,
               carry_usd_oz, max_buy_usd_oz, max_buy_frac, float_days,
               tail_vol_daily, flags
        FROM spread_floor_daily
        WHERE (metal, date_utc) IN (
            SELECT metal, max(date_utc) FROM spread_floor_daily GROUP BY metal
        )
        ORDER BY metal
        """
    )


def book_var_rows() -> int:
    return int(_scalar("SELECT count(*) FROM book_var_daily", 0) or 0)


def price_coverage() -> tuple[str, str, int]:
    """(first date, last date, row count) across the price panel."""
    df = _frame("SELECT min(timestamp_utc) a, max(timestamp_utc) b, count(*) n FROM prices")
    if df.empty or pd.isna(df.iloc[0]["a"]):
        return ("n/a", "n/a", 0)
    return (
        pd.Timestamp(df.iloc[0]["a"]).strftime("%d %b %Y"),
        pd.Timestamp(df.iloc[0]["b"]).strftime("%d %b %Y"),
        int(df.iloc[0]["n"]),
    )


def headline_count() -> int:
    return int(_scalar("SELECT count(*) FROM headlines", 0) or 0)


def model_run_count() -> int:
    return int(_scalar("SELECT count(*) FROM runs", 0) or 0)


def quarantined_rows() -> int:
    """Collector rows held back pending a licence (the 2026-07-16 ToU audit)."""
    total = 0
    for table in ("coin_premiums", "macro_consensus", "search_interest", "pgm_prices"):
        total += int(
            _scalar(f"SELECT count(*) FROM {table} WHERE quarantine_reason IS NOT NULL", 0) or 0
        )
    return total


def git_commit(repo: Path | None = None) -> str:
    """Short commit of the working tree, or "" outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo or Path(__file__).resolve().parents[3]),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""
