"""CFTC Commitments of Traders ingestion.

Phase 2 step 2.4. Pulls the *Disaggregated Futures-Only* report for the four
metals (Gold, Silver, Platinum, Palladium) and writes to the ``positioning``
table.

CRITICAL — Friday-close timestamp convention. The COT report is dated to
*Tuesday* positioning but the file is released the following *Friday* after
market close (~3:30 PM ET). The "as-of" timestamp on every row in the
``positioning`` table must therefore be the **Friday release date**, not the
Tuesday positioning date — using Tuesday is the canonical data-leakage bug
in this field. We map the Tuesday report date to its release date with
``release_date`` below, which also accounts for federal-holiday delays.

Source historical files:
    https://www.cftc.gov/files/dea/history/fut_disagg_txt_{YEAR}.zip

Run as:
    uv run python -m metals.data.cot --start-year 2007
"""

from __future__ import annotations

import argparse
import io
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from metals.data.db import connection

SOURCE_TAG = "cftc_disagg"
ZIP_URL_TMPL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"

# Map our internal metal label to the canonical CFTC
# "Market_and_Exchange_Names" prefix for the *standard* contract. We anchor
# the match to the start of the (stripped) market name so we don't pick up
# E-MINI or MICRO variants, whose names also contain the underlying metal.
METAL_NAME_PATTERNS: dict[str, str] = {
    "gold": "GOLD - COMMODITY EXCHANGE INC.",
    "silver": "SILVER - COMMODITY EXCHANGE INC.",
    "platinum": "PLATINUM - NEW YORK MERCANTILE EXCHANGE",
    "palladium": "PALLADIUM - NEW YORK MERCANTILE EXCHANGE",
}

# Tuesday positioning date -> Friday release. Three calendar days forward.
TUE_TO_FRI_OFFSET = timedelta(days=3)

# US federal-holiday-aware business-day machinery for the release-date calc.
_US_CALENDAR = USFederalHolidayCalendar()
_US_BDAY = CustomBusinessDay(calendar=_US_CALENDAR)


def release_date(report_date: pd.Timestamp) -> pd.Timestamp:
    """Map a COT *report* date (Tuesday positioning) to its *release* date.

    The report is normally published the following Friday at 3:30 PM ET
    (``report_date + 3``). When a US federal holiday falls in the report week,
    the CFTC delays publication — Thanksgiving and July-4 weeks typically
    release the following Monday. We approximate that by pushing the nominal
    Friday one business day later per in-week federal holiday, then snapping to
    the next business day. Erring *later* is the leakage-safe choice: we never
    claim the data was available earlier than it actually was.

    This is a heuristic, not the exact CFTC release calendar; for the rare
    holiday weeks it errs at most a day late, which is conservative for any
    downstream as-of join.
    """
    report_date = pd.Timestamp(report_date).normalize()
    nominal_friday = report_date + TUE_TO_FRI_OFFSET
    in_week_holidays = _US_CALENDAR.holidays(report_date + pd.Timedelta(days=1), nominal_friday)
    delayed = nominal_friday + len(in_week_holidays) * _US_BDAY
    return _US_BDAY.rollforward(delayed)


def _download_year(year: int, timeout: int = 60) -> pd.DataFrame:
    """Download and extract one year of disaggregated COT data."""
    url = ZIP_URL_TMPL.format(year=year)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        # The ZIP normally contains a single .txt file (CSV-formatted).
        csv_name = next((n for n in names if n.lower().endswith(".txt")), names[0])
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)
    return df


def parse_cot_dataframe(raw: pd.DataFrame, year_label: int | str = "?") -> pd.DataFrame:
    """Filter a raw CFTC disaggregated frame to the four metals and normalise.

    Pure function over a DataFrame — no network — so it can be unit-tested
    with synthetic inputs.
    """
    market_col = next(
        (c for c in raw.columns if "Market_and_Exchange_Names" in c),
        None,
    )
    if market_col is None:
        raise RuntimeError(
            f"COT {year_label}: could not find Market_and_Exchange_Names column. "
            f"Got: {list(raw.columns)[:5]}..."
        )

    # Anchor to the start of the stripped market name to exclude E-MINI /
    # MICRO variants whose names also contain the underlying metal.
    market_names = raw[market_col].astype(str).str.strip().str.upper()
    frames: list[pd.DataFrame] = []
    for metal, pattern in METAL_NAME_PATTERNS.items():
        mask = market_names.str.startswith(pattern.upper())
        sub = raw.loc[mask].copy()
        if sub.empty:
            continue
        sub["metal"] = metal
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Locate the report-date column (CFTC has used several names).
    date_col = next(
        (c for c in df.columns if "Report_Date_as_YYYY-MM-DD" in c),
        next((c for c in df.columns if "Report_Date_as_MM_DD_YYYY" in c), None),
    )
    if date_col is None:
        raise RuntimeError(f"COT {year_label}: could not find Report_Date_as_YYYY-MM-DD column.")
    report_date = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    # Holiday-aware release date. Compute once per unique report date (there are
    # only ~weekly distinct dates) then map back, to avoid per-row calendar work.
    unique_dates = report_date.dropna().drop_duplicates()
    release_lookup = {d: release_date(d) for d in unique_dates}
    df["timestamp_utc"] = report_date.map(release_lookup)

    rename = {
        "Prod_Merc_Positions_Long_All": "producer_long",
        "Prod_Merc_Positions_Short_All": "producer_short",
        "Swap_Positions_Long_All": "swap_long",
        "Swap__Positions_Short_All": "swap_short",
        "M_Money_Positions_Long_All": "managed_money_long",
        "M_Money_Positions_Short_All": "managed_money_short",
        "Other_Rept_Positions_Long_All": "other_reportable_long",
        "Other_Rept_Positions_Short_All": "other_reportable_short",
        "NonRept_Positions_Long_All": "non_reportable_long",
        "NonRept_Positions_Short_All": "non_reportable_short",
        "Open_Interest_All": "open_interest",
    }
    df = df.rename(columns=rename)

    # "Commercial" in the disaggregated report = Producer + Swap Dealer.
    # Missing components (older years sometimes lack the swap split) are
    # treated as zero.
    for col in ("producer_long", "producer_short", "swap_long", "swap_short"):
        if col not in df.columns:
            df[col] = 0
    df["commercial_long"] = df["producer_long"] + df["swap_long"]
    df["commercial_short"] = df["producer_short"] + df["swap_short"]

    needed = [
        "timestamp_utc",
        "metal",
        "commercial_long",
        "commercial_short",
        "managed_money_long",
        "managed_money_short",
        "other_reportable_long",
        "other_reportable_short",
        "non_reportable_long",
        "non_reportable_short",
        "open_interest",
    ]
    out = df[needed].copy()
    int_cols = [c for c in needed if c not in ("timestamp_utc", "metal")]
    out[int_cols] = out[int_cols].fillna(0).astype("int64")
    out["source"] = SOURCE_TAG
    return out.sort_values(["metal", "timestamp_utc"]).reset_index(drop=True)


def fetch_cot_year(year: int) -> pd.DataFrame:
    """Download + parse one year's disaggregated COT for the four metals."""
    return parse_cot_dataframe(_download_year(year), year_label=year)


def upsert_positioning(df: pd.DataFrame) -> int:
    """Idempotent upsert into the positioning table."""
    if df.empty:
        return 0
    with connection() as conn:
        conn.register("incoming_pos", df)
        conn.execute(
            """
            INSERT INTO positioning
                (timestamp_utc, metal,
                 commercial_long, commercial_short,
                 managed_money_long, managed_money_short,
                 other_reportable_long, other_reportable_short,
                 non_reportable_long, non_reportable_short,
                 open_interest, source)
            SELECT
                timestamp_utc, metal,
                commercial_long, commercial_short,
                managed_money_long, managed_money_short,
                other_reportable_long, other_reportable_short,
                non_reportable_long, non_reportable_short,
                open_interest, source
            FROM incoming_pos
            ON CONFLICT (timestamp_utc, metal) DO UPDATE SET
                commercial_long        = EXCLUDED.commercial_long,
                commercial_short       = EXCLUDED.commercial_short,
                managed_money_long     = EXCLUDED.managed_money_long,
                managed_money_short    = EXCLUDED.managed_money_short,
                other_reportable_long  = EXCLUDED.other_reportable_long,
                other_reportable_short = EXCLUDED.other_reportable_short,
                non_reportable_long    = EXCLUDED.non_reportable_long,
                non_reportable_short   = EXCLUDED.non_reportable_short,
                open_interest          = EXCLUDED.open_interest,
                source                 = EXCLUDED.source
            """
        )
        conn.unregister("incoming_pos")
    return int(len(df))


def refresh(start_year: int = 2007, end_year: int | None = None) -> dict:
    """Pull every year from ``start_year`` to ``end_year`` (default: current)
    and upsert. Returns a summary dict."""
    if end_year is None:
        end_year = datetime.now(timezone.utc).year
    total = 0
    per_year: dict[int, int] = {}
    skipped: list[int] = []
    for y in range(start_year, end_year + 1):
        try:
            df = fetch_cot_year(y)
        except requests.HTTPError as exc:
            print(f"WARNING: skipping COT {y}: {exc}")
            skipped.append(y)
            continue
        n = upsert_positioning(df)
        total += n
        per_year[y] = n
        print(f"  {y}: {n} rows")
    return {
        "rows_written": total,
        "per_year": per_year,
        "skipped": skipped,
        "start_year": start_year,
        "end_year": end_year,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh CFTC COT positioning.")
    parser.add_argument("--start-year", type=int, default=2007)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()
    summary = refresh(start_year=args.start_year, end_year=args.end_year)
    print(f"\nTotal rows: {summary['rows_written']}")
    print(f"Years:      {args.start_year}-{summary['end_year']} (skipped {summary['skipped']})")


if __name__ == "__main__":
    main()
