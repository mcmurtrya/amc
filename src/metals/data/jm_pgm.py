"""Johnson Matthey PGM base prices (Phase 7.1, collector 6).

Johnson Matthey publishes free daily base prices for platinum, palladium,
rhodium, iridium and ruthenium (all USD/troy oz) on matthey.com's
PGM-management page. Rhodium/iridium/ruthenium have no exchange price — JM's
quote is the reference for catalytic-converter scrap; JM Pt/Pd ride along to
cross-check quote timing against CME settles.

Recipe (verified live 2026-07-12): the page's Liferay price portlet answers a
form POST (metals, DD-MM-YYYY date range, interval, region) with JSON
``{"url": ...}`` pointing at a generated CSV — first line a title, second the
header ``Date,Platinum,Palladium,Rhodium,Iridium,Ruthenium``, then one row
per quote date (DD/MM/YYYY). On failure it returns HTTP 200 with
``{"status": "Error"}``. Four regional quotes map to ``pgm_prices.quote``:

    Asia   -> hk_open   Hong Kong opening (08:30 local)
    Asiacl -> hk_close  Hong Kong closing (14:00 local)
    Europe -> london    London (09:00 local)
    USA    -> ny_am     New York (09:30 local; JM publishes one NY quote/day)

History depth (verified): London reaches back to 1992-07-01, Hong Kong to
1992-10-06; the site's own "All prices" range option caps at 30 years back.
Regions skip their own holidays (e.g. Hong Kong is empty on 2026-07-01), so
per-quote coverage differs by a few days per year. A region CSV with zero
data rows raises — so a narrow ``--start/--end`` window that predates a
region's first quote fails loudly; the default full-history window is fine
(each region returns its own available range).

Run as:
    uv run python -m metals.data.jm_pgm                # forward ~14 days
    uv run python -m metals.data.jm_pgm --historical   # one-time full-history backfill

The forward job is weekly cadence; the 14-day window overlaps itself so a
missed week self-heals. ``--historical`` (and ``--start``/``--end``) only
select the pull window: ``is_realtime`` is decided per row from capture lag
(pulled within ``REALTIME_MAX_LAG_DAYS`` of the quote date), so backfills and
gap-fills over old dates are permanently second-class no matter which flags
they ran with.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import requests

from metals.data.db import connection

SOURCE_TAG = "matthey.com"
USER_AGENT = "AMCResearchCollector/0.1 (internal research)"

PAGE_URL = "https://matthey.com/products-and-markets/pgms-and-circularity/pgm-management"
# Liferay resource-serving URL embedded in the page's hidden #getUrl element.
CSV_POST_URL = (
    PAGE_URL
    + "?p_p_id=jm_metal_price_portlet_JmMetalPricePortlet"
    + "&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_cacheability=cacheLevelPage"
)
PORTLET_PREFIX = "_jm_metal_price_portlet_JmMetalPricePortlet_"

# JM's request metal codes (checkbox values) and its CSV column names, mapped
# to the lowercase metal names the pgm_prices table uses.
METAL_CODES = ("Pt", "Pd", "Rh", "Ir", "Ru")
JM_COLUMN_TO_METAL = {
    "Platinum": "platinum",
    "Palladium": "palladium",
    "Rhodium": "rhodium",
    "Iridium": "iridium",
    "Ruthenium": "ruthenium",
}
EXPECTED_HEADER = ["Date", "Platinum", "Palladium", "Rhodium", "Iridium", "Ruthenium"]

# JM region parameter -> pgm_prices.quote code (JM's own labels in comments).
REGION_TO_QUOTE = {
    "Asia": "hk_open",  # "Hong Kong (opening)", 08:30 local
    "Asiacl": "hk_close",  # "Hong Kong (closing)", 14:00 local
    "Europe": "london",  # "London", 09:00 local
    "USA": "ny_am",  # "New York", 09:30 local — JM's single daily NY quote
}

HISTORY_START = date(1992, 7, 1)  # earliest published row (London); HK starts 1992-10-06
FORWARD_WINDOW_DAYS = 14
# A row counts as captured in real time only when pulled within this many days
# of its quote date (matches the forward window; mirrors cme_daily's rule).
REALTIME_MAX_LAG_DAYS = 14
STALE_FLAG_DAYS = 10  # trailing unchanged-price run worth flagging (Rh/Ir/Ru plateau)
REQUEST_SLEEP_S = 2.0
TIMEOUT_S = 120


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    return sess


def _extract_csv_url(payload: str) -> str:
    """Pull the generated-CSV URL out of the portlet's JSON response.

    Raises on non-JSON payloads, the portlet's ``{"status": "Error"}``
    refusal, or a missing/odd ``url`` field — never returns silently.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JM portlet returned non-JSON payload: {payload[:200]!r}") from exc
    if not isinstance(data, dict) or data.get("status") == "Error":
        raise RuntimeError(f"JM portlet refused the CSV request: {payload[:200]!r}")
    url = data.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError(f"JM portlet response lacks a usable CSV url: {payload[:200]!r}")
    return url


def fetch_region_csv(
    region: str,
    start: date,
    end: date,
    session: requests.Session | None = None,
) -> bytes:
    """POST the export form for one region and download the generated CSV.

    Two HTTP requests (form POST -> JSON url, then GET the CSV), each
    preceded by a polite sleep. Returns the raw CSV bytes.
    """
    if region not in REGION_TO_QUOTE:
        raise ValueError(f"Unknown JM region {region!r}; expected one of {list(REGION_TO_QUOTE)}")
    sess = session or _session()
    form = {f"{PORTLET_PREFIX}selectedMetal{i}": code for i, code in enumerate(METAL_CODES)}
    form[f"{PORTLET_PREFIX}start_Date"] = start.strftime("%d-%m-%Y")
    form[f"{PORTLET_PREFIX}end_Date"] = end.strftime("%d-%m-%Y")
    form[f"{PORTLET_PREFIX}IntervalType"] = "DAILY"
    form[f"{PORTLET_PREFIX}selectedRegion"] = region

    time.sleep(REQUEST_SLEEP_S)
    resp = sess.post(CSV_POST_URL, data=form, timeout=TIMEOUT_S)
    resp.raise_for_status()
    csv_url = _extract_csv_url(resp.text)

    time.sleep(REQUEST_SLEEP_S)
    doc = sess.get(csv_url, timeout=TIMEOUT_S)
    doc.raise_for_status()
    if not doc.content.strip():
        raise RuntimeError(f"JM returned an empty CSV body from {csv_url}")
    return doc.content


# A row with the wrong field count cannot be repaired locally: JM's writer
# drops EMPTY fields entirely (verified: 13/06/2014 HK-open lacks platinum and
# every value shifts left), so column assignment for a short row is ambiguous
# and guessing would corrupt the series. Sporadic vendor defects (1 row in
# ~34k over 1992-2026) are skipped with a printed warning; above this fraction
# per file it is schema drift and the parse fails loudly.
MALFORMED_ROW_TOLERANCE = 0.01


def parse_daily_csv(raw: bytes, quote: str) -> pd.DataFrame:
    """Parse one region's daily CSV into long rows.

    Returns columns: price_date (datetime.date), metal, quote, price_usd_oz,
    with ``df.attrs["n_malformed"]`` counting skipped defective rows.
    Raises ``ValueError`` on header drift, duplicate dates, an empty body, or
    a malformed-row fraction above ``MALFORMED_ROW_TOLERANCE`` — schema drift
    must fail loudly, not thin the data.
    """
    if quote not in REGION_TO_QUOTE.values():
        raise ValueError(f"Unknown quote code {quote!r}")
    lines = [ln for ln in raw.decode("utf-8-sig").splitlines() if ln.strip()]
    header_idx: int | None = None
    for i, line in enumerate(lines[:2]):  # title line first, header expected by line 2
        fields = [f.strip() for f in line.split(",")]
        while fields and fields[-1] == "":
            fields.pop()  # the all-region title line carries trailing commas
        if fields == EXPECTED_HEADER:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            f"JM CSV header drift: expected {EXPECTED_HEADER} in the first two lines, "
            f"got {lines[:2]!r}"
        )

    records: list[dict[str, object]] = []
    body = lines[header_idx + 1 :]
    malformed: list[str] = []
    for line in body:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != len(EXPECTED_HEADER):
            malformed.append(line)
            print(
                f"WARNING: JM CSV ({quote}): skipping malformed row "
                f"({len(fields)} of {len(EXPECTED_HEADER)} fields — column assignment "
                f"would be ambiguous): {line!r}"
            )
            continue
        try:
            price_date = datetime.strptime(fields[0], "%d/%m/%Y").date()
        except ValueError as exc:
            raise ValueError(f"JM CSV date not DD/MM/YYYY: {fields[0]!r}") from exc
        for column, value in zip(EXPECTED_HEADER[1:], fields[1:], strict=True):
            if value == "":
                price: float | None = None
            else:
                try:
                    price = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"JM CSV price not numeric on {fields[0]}: {column}={value!r}"
                    ) from exc
            records.append(
                {
                    "price_date": price_date,
                    "metal": JM_COLUMN_TO_METAL[column],
                    "quote": quote,
                    "price_usd_oz": price,
                }
            )
    if not records:
        raise ValueError(f"JM CSV for quote {quote!r} contains no data rows")
    if len(malformed) / len(body) > MALFORMED_ROW_TOLERANCE:
        raise ValueError(
            f"JM CSV for quote {quote!r}: {len(malformed)} of {len(body)} rows malformed "
            f"(> {MALFORMED_ROW_TOLERANCE:.0%}) — schema drift, not a sporadic defect"
        )
    df = pd.DataFrame(records)
    if df.duplicated(subset=["price_date", "metal", "quote"]).any():
        raise ValueError(f"JM CSV for quote {quote!r} repeats (date, metal) rows")
    df.attrs["n_malformed"] = len(malformed)
    return df


def fetch_daily_csv(
    start: date,
    end: date,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch all five metals across the four regional quotes, long format."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    sess = session or _session()
    frames = [
        parse_daily_csv(fetch_region_csv(region, start, end, session=sess), quote)
        for region, quote in REGION_TO_QUOTE.items()
    ]
    n_malformed = sum(f.attrs.get("n_malformed", 0) for f in frames)
    out = pd.concat(frames, ignore_index=True)
    out.attrs["n_malformed"] = n_malformed  # concat drops attrs; re-attach the sum
    return out


def stale_run_report(df: pd.DataFrame) -> pd.DataFrame:
    """Longest and trailing runs of unchanged prices per (metal, quote).

    Rh/Ir/Ru quotes plateau for days-to-weeks; a long trailing run is a
    staleness flag downstream consumers must see before trusting the level.
    Columns: metal, quote, n_days, longest_run, tail_run, last_price.
    """
    cols = ["metal", "quote", "n_days", "longest_run", "tail_run", "last_price"]
    rows: list[dict[str, object]] = []
    priced = df.dropna(subset=["price_usd_oz"])
    for (metal, quote), group in priced.groupby(["metal", "quote"]):
        prices = group.sort_values("price_date")["price_usd_oz"].tolist()
        longest = run = 1
        for prev, cur in zip(prices, prices[1:], strict=False):
            run = run + 1 if cur == prev else 1
            longest = max(longest, run)
        tail = 1
        for i in range(len(prices) - 1, 0, -1):
            if prices[i] != prices[i - 1]:
                break
            tail += 1
        rows.append(
            {
                "metal": metal,
                "quote": quote,
                "n_days": len(prices),
                "longest_run": longest,
                "tail_run": tail,
                "last_price": prices[-1],
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["tail_run", "metal"], ascending=[False, True]).reset_index(drop=True)


def _realtime_flags(price_dates: pd.Series, pulled_at: datetime) -> pd.Series:
    """Per-row real-time flags: pulled within ``REALTIME_MAX_LAG_DAYS`` of the quote date."""
    lag_days = (pd.Timestamp(pulled_at.date()) - pd.to_datetime(price_dates)).dt.days
    return lag_days <= REALTIME_MAX_LAG_DAYS


def upsert_pgm_prices(df: pd.DataFrame, *, pulled_at: datetime | None = None) -> int:
    """Idempotent upsert into ``pgm_prices``. Returns rows written.

    ``is_realtime`` is decided per row from capture lag — True only when
    ``pulled_at`` falls within ``REALTIME_MAX_LAG_DAYS`` of the quote date —
    so a gap-fill over an old window can never claim real-time capture. The
    flag never demotes, and a real-time row keeps its first-capture
    ``pulled_at`` forever; only retro rows take a re-pull's timestamp.
    ``pulled_at`` is naive UTC and defaults to now (kwarg exists for tests).
    """
    if df.empty:
        return 0
    if df.duplicated(subset=["price_date", "metal", "quote"]).any():
        raise ValueError("Duplicate (price_date, metal, quote) rows in upsert input")
    pulled = pulled_at or datetime.now(UTC).replace(tzinfo=None)
    insert_df = df.copy()
    insert_df["price_date"] = pd.to_datetime(insert_df["price_date"])
    insert_df["source"] = SOURCE_TAG
    insert_df["pulled_at"] = pulled
    insert_df["is_realtime"] = _realtime_flags(insert_df["price_date"], pulled)
    with connection() as conn:
        conn.register("incoming_pgm", insert_df)
        conn.execute(
            """
            INSERT INTO pgm_prices
                (price_date, metal, quote, price_usd_oz, source, pulled_at, is_realtime)
            SELECT CAST(price_date AS DATE), metal, quote, price_usd_oz,
                   source, pulled_at, is_realtime
            FROM incoming_pgm
            ON CONFLICT (price_date, metal, quote) DO UPDATE SET
                price_usd_oz = EXCLUDED.price_usd_oz,
                source       = EXCLUDED.source,
                pulled_at    = CASE WHEN pgm_prices.is_realtime
                                    THEN pgm_prices.pulled_at
                                    ELSE EXCLUDED.pulled_at END,
                is_realtime  = pgm_prices.is_realtime OR EXCLUDED.is_realtime
            """
        )
        conn.unregister("incoming_pgm")
    return len(insert_df)


def refresh(
    start: date | None = None,
    end: date | None = None,
    *,
    historical: bool = False,
) -> dict:
    """Pull JM daily PGM prices and upsert. Returns a summary dict.

    Forward mode (default, weekly cadence) pulls the trailing 14 days ending
    today (UTC); ``historical=True`` pulls the full published history (from
    1992-07-01) — the one-time setup backfill. Both flags only select the
    window: ``is_realtime`` is derived per row from capture lag, so a
    gap-fill like ``--start 2026-05-01`` cannot stamp old rows real-time.
    """
    today = datetime.now(UTC).date()
    if historical:
        start = start or HISTORY_START
    else:
        start = start or today - timedelta(days=FORWARD_WINDOW_DAYS)
    end = end or today

    df = fetch_daily_csv(start, end)
    pulled_at = datetime.now(UTC).replace(tzinfo=None)
    n = upsert_pgm_prices(df, pulled_at=pulled_at)
    per_quote = df.groupby("quote").size().to_dict()
    return {
        "rows_written": n,
        "rows_realtime": int(_realtime_flags(df["price_date"], pulled_at).sum()),
        "rows_malformed_skipped": int(df.attrs.get("n_malformed", 0)),
        "date_range": [start.isoformat(), end.isoformat()],
        "max_price_date": df["price_date"].max().isoformat(),
        "rows_per_quote": per_quote,
        "stale_runs": stale_run_report(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Johnson Matthey PGM base prices.")
    parser.add_argument(
        "--historical",
        action="store_true",
        help="One-time full-history backfill window (1992-07-01 onward); "
        "is_realtime is derived per row from capture lag either way.",
    )
    parser.add_argument("--start", default=None, help="YYYY-MM-DD; overrides the default window.")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD; defaults to today (UTC).")
    args = parser.parse_args()
    summary = refresh(
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
        historical=args.historical,
    )
    print(f"Rows written:      {summary['rows_written']}")
    print(f"Real-time rows:    {summary['rows_realtime']}")
    if summary["rows_malformed_skipped"]:
        print(f"Malformed skipped: {summary['rows_malformed_skipped']} (see WARNINGs above)")
    print(f"Date range:        {summary['date_range']}")
    print(f"Max price date:    {summary['max_price_date']}")
    print(f"Rows per quote:    {summary['rows_per_quote']}")
    stale = summary["stale_runs"]
    if not stale.empty:
        print("\nStale-quote runs (unchanged consecutive prices):")
        print(f"  {'metal':10s} {'quote':9s} {'days':>5s} {'longest':>8s} {'tail':>5s} last")
        for _, r in stale.iterrows():
            flag = "  <-- STALE" if r["tail_run"] >= STALE_FLAG_DAYS else ""
            print(
                f"  {r['metal']:10s} {r['quote']:9s} {r['n_days']:>5d} "
                f"{r['longest_run']:>8d} {r['tail_run']:>5d} {r['last_price']:.2f}{flag}"
            )


if __name__ == "__main__":
    main()
