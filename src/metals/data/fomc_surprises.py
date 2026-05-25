"""Bauer–Swanson FOMC monetary-policy surprises ingestion.

Phase 2 step 2.3. Pulls the SF Fed's update of the Bauer–Swanson (2023)
high-frequency FOMC surprise series and writes to ``fomc_surprises``.

Conventions:

* ``MPS`` — composite policy surprise, basis points. First principal
  component of the response of FF/ED/Treasury futures over the 30-minute
  window around the FOMC announcement. Positive = hawkish.
* ``MPS_ORTH`` — orthogonalized MPS, the recommended single-number stance
  measure in Bauer–Swanson (2023). Cleaned of the "Fed information effect"
  by projecting out NFP_SURP and other pre-announcement state variables.
* ``FF1``, ``FF2`` — fed-funds-futures surprises (target rate component).
* ``ED4`` — 4-quarter-ahead Eurodollar surprise (path component).

Run as:
    uv run python -m metals.data.fomc_surprises
"""

from __future__ import annotations

import argparse
import io

import pandas as pd
import requests

from metals.data.db import connection

SOURCE_TAG = "frbsf_bauer_swanson_2023"
XLSX_URL = "https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx"
SHEET = "FOMC (update 2023)"


def _download_xlsx(url: str = XLSX_URL, timeout: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


REQUIRED_COLUMNS: tuple[str, ...] = ("Date", "Unscheduled", "FF1", "FF2",
                                    "ED4", "MPS", "MPS_ORTH")
SURPRISE_COLUMNS: tuple[str, ...] = ("ff1", "ff2", "ed4", "mps", "mps_orth")


def clean_surprises_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename + coerce a Bauer-Swanson FOMC sheet to our schema.

    Pure-function over a DataFrame — testable without touching disk or net.
    Raises if any expected column is missing.
    """
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise RuntimeError(
            f"Bauer-Swanson sheet missing expected columns: {sorted(missing)}"
        )
    out = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(raw["Date"]).dt.tz_localize(None),
        "is_unscheduled": raw["Unscheduled"].fillna(0).astype(bool),
        "ff1": pd.to_numeric(raw["FF1"], errors="coerce"),
        "ff2": pd.to_numeric(raw["FF2"], errors="coerce"),
        "ed4": pd.to_numeric(raw["ED4"], errors="coerce"),
        "mps": pd.to_numeric(raw["MPS"], errors="coerce"),
        "mps_orth": pd.to_numeric(raw["MPS_ORTH"], errors="coerce"),
    })
    out["source"] = SOURCE_TAG
    # Keep only rows with at least one non-null surprise. The earliest 1988-89
    # rows have NaN MPS because the underlying futures series weren't liquid
    # enough to compute a surprise.
    out = out[out[list(SURPRISE_COLUMNS)].notna().any(axis=1)].reset_index(drop=True)
    return out.sort_values("timestamp_utc").reset_index(drop=True)


def parse_surprises_excel(content: bytes, sheet: str = SHEET) -> pd.DataFrame:
    """Read the FOMC sheet from raw XLSX bytes and return a clean DataFrame."""
    raw = pd.read_excel(io.BytesIO(content), sheet_name=sheet)
    return clean_surprises_dataframe(raw)


def upsert_surprises(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    with connection() as conn:
        conn.register("incoming_surprises", df)
        conn.execute(
            """
            INSERT INTO fomc_surprises
                (timestamp_utc, is_unscheduled, ff1, ff2, ed4,
                 mps, mps_orth, source)
            SELECT
                timestamp_utc, is_unscheduled, ff1, ff2, ed4,
                mps, mps_orth, source
            FROM incoming_surprises
            ON CONFLICT (timestamp_utc) DO UPDATE SET
                is_unscheduled = EXCLUDED.is_unscheduled,
                ff1            = EXCLUDED.ff1,
                ff2            = EXCLUDED.ff2,
                ed4            = EXCLUDED.ed4,
                mps            = EXCLUDED.mps,
                mps_orth       = EXCLUDED.mps_orth,
                source         = EXCLUDED.source
            """
        )
        conn.unregister("incoming_surprises")
    return int(len(df))


def refresh() -> dict:
    """Download + parse + upsert. Return summary dict."""
    content = _download_xlsx()
    df = parse_surprises_excel(content)
    n = upsert_surprises(df)
    return {
        "rows_written": n,
        "min_date": df["timestamp_utc"].min().date().isoformat() if n else None,
        "max_date": df["timestamp_utc"].max().date().isoformat() if n else None,
        "n_unscheduled": int(df["is_unscheduled"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Bauer-Swanson FOMC surprises.")
    parser.parse_args()
    s = refresh()
    print(f"FOMC surprise rows written: {s['rows_written']}")
    print(f"Date range:                 [{s['min_date']}, {s['max_date']}]")
    print(f"Of which unscheduled:       {s['n_unscheduled']}")


if __name__ == "__main__":
    main()
