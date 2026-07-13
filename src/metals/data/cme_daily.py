"""CME Group daily settlement volume / open-interest collector (Phase 7.1, collector 4).

Forward-capture leg of the spliced volume/OI series: the funded Databento
backfill supplies official history, this collector records each day's public
figures as they appear. Products: GC, SI (COMEX) and PL, PA (NYMEX). Rows land
in ``cme_daily`` keyed by (trade_date, product, contract_month, is_preliminary),
one row per contract month plus one ``AGG`` row per product/day (sums across
contract months — the roll-neutral series 7.6 consumes).

Endpoints (public JSON web services behind cmegroup.com; verified 2026-07-12
via live-archive snapshots through 2026-06 — CME blocks datacenter IPs, so the
collector must run from AMC's own machine):

- Volume/OI by month (P = preliminary, same evening; F = final, next business
  morning; both remain served once published, so a run captures both)::

      /CmeWS/mvc/Volume/Details/F/{product_id}/{YYYYMMDD}/{P|F}?tradeDate=...&pageSize=500

  Per-month fields used: ``totalVolume``, ``atClose`` (open interest),
  ``change`` (OI change); the ``totals`` block feeds the AGG row.
- Settlements (carries ``reportType`` "Preliminary" | "Final")::

      /CmeWS/mvc/Settlements/Futures/Settlements/{product_id}/FUT?strategy=DEFAULT&tradeDate=MM/DD/YYYY&pageSize=500

  CAUTION: its per-month ``openInterest`` is the *prior* day's final OI
  (verified against the volume endpoint on 2026-02-24/25), so volume/OI are
  taken only from the Volume endpoint; only ``settle`` is taken from here.
- Available trade dates (newest first)::

      /CmeWS/mvc/Settlements/Futures/TradeDate/{product_id}  ->  [["MM/DD/YYYY", "Final"], ...]

Row semantics: ``is_preliminary`` describes the **volume/OI report** (P|F).
Final settles publish the same evening while final volume/OI arrive the next
morning, so a final settle may ride on a preliminary row; a preliminary settle
is never attached to a final row. ``is_realtime`` is True when the pull
happened within ``REALTIME_MAX_LAG_DAYS`` of the trade date — anything older
is retro capture and stays second-class, permanently.

Run as:
    uv run python -m metals.data.cme_daily                       # latest trade date
    uv run python -m metals.data.cme_daily --trade-date 2026-07-10 --products GC SI
    uv run python -m metals.data.cme_daily --check-gaps --gap-days 10
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import requests
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

from metals.data.db import connection

SOURCE_TAG = "cmegroup.com"
USER_AGENT = "AMCResearchCollector/0.1 (internal research)"

# CME web-service numeric product ids (verified from the products' own
# volume pages: GC/SI via archived XHR captures, PL/PA embedded in the pages).
PRODUCTS: dict[str, int] = {"GC": 437, "SI": 458, "PL": 446, "PA": 445}

AGG_MONTH = "AGG"
REALTIME_MAX_LAG_DAYS = 4  # pull within this many days of trade date => real-time
FETCH_DELAY_S = 2.0
REQUEST_TIMEOUT_S = 30

_BASE = "https://www.cmegroup.com/CmeWS/mvc"
VOLUME_URL_TMPL = _BASE + "/Volume/Details/F/{product_id}/{yyyymmdd}/{report}"
SETTLEMENTS_URL_TMPL = _BASE + "/Settlements/Futures/Settlements/{product_id}/FUT"
TRADE_DATES_URL_TMPL = _BASE + "/Settlements/Futures/TradeDate/{product_id}"

# CME month labels; the daily bulletin family uses JLY for July in places.
_MONTH_NUM: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "JLY": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}  # fmt: skip

_COLUMNS = [
    "trade_date",
    "product",
    "contract_month",
    "settle",
    "volume",
    "open_interest",
    "oi_change",
    "is_preliminary",
    "source",
    "pulled_at",
    "is_realtime",
]


class NyseHolidayCalendar(AbstractHolidayCalendar):
    """Approximate NYSE full-closure calendar, good enough for gap alarms.

    The one systematic approximation: ``nearest_workday`` observes
    Saturday holidays on Friday, which NYSE does for July 4 / Christmas but
    not for New Year's Day — the cost is a rare unchecked trading day, never
    a false "missing day" alarm.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


# --------------------------------------------------------------------------
# HTTP (polite: identified UA, >= FETCH_DELAY_S between requests)
# --------------------------------------------------------------------------

_last_fetch_monotonic: float | None = None


def _polite_pause() -> None:
    global _last_fetch_monotonic
    if _last_fetch_monotonic is not None:
        wait = FETCH_DELAY_S - (time.monotonic() - _last_fetch_monotonic)
        if wait > 0:
            time.sleep(wait)
    _last_fetch_monotonic = time.monotonic()


def _get_json(url: str, params: dict[str, str] | None = None) -> Any:
    """GET a CME web-service URL and decode JSON, failing loudly."""
    _polite_pause()
    resp = requests.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError as exc:
        raise RuntimeError(f"CME returned non-JSON from {url}: {resp.text[:200]!r}") from exc


def _product_id(product: str) -> int:
    if product not in PRODUCTS:
        raise ValueError(f"Unknown product {product!r}; expected one of {sorted(PRODUCTS)}")
    return PRODUCTS[product]


def fetch_trade_dates(product: str) -> list[tuple[date, str]]:
    """Available (trade_date, report_type) pairs for a product, newest first."""
    url = TRADE_DATES_URL_TMPL.format(product_id=_product_id(product))
    return parse_trade_dates(_get_json(url), label=product)


def fetch_settlements(product: str, trade_date: date) -> dict[str, Any]:
    url = SETTLEMENTS_URL_TMPL.format(product_id=_product_id(product))
    params = {
        "strategy": "DEFAULT",
        "tradeDate": trade_date.strftime("%m/%d/%Y"),
        "pageSize": "500",
    }
    payload = _get_json(url, params=params)
    if not isinstance(payload, dict):
        raise ValueError(f"{product}: settlements payload is not a JSON object: {payload!r}")
    return payload


def fetch_volume_details(product: str, trade_date: date, preliminary: bool) -> dict[str, Any]:
    url = VOLUME_URL_TMPL.format(
        product_id=_product_id(product),
        yyyymmdd=trade_date.strftime("%Y%m%d"),
        report="P" if preliminary else "F",
    )
    payload = _get_json(url, params={"tradeDate": trade_date.strftime("%Y%m%d")})
    if not isinstance(payload, dict):
        raise ValueError(f"{product}: volume payload is not a JSON object: {payload!r}")
    return payload


# --------------------------------------------------------------------------
# Parsing (pure functions over decoded payloads; every anomaly raises)
# --------------------------------------------------------------------------


def _to_int(value: Any, field: str) -> int | None:
    """Parse a CME integer string ('140,639', '-328', '-' => None)."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("-", ""):
        return None
    try:
        return int(text.replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"Unparseable integer for {field}: {value!r}") from exc


def _to_float(value: Any, field: str) -> float | None:
    """Parse a CME decimal string ('2,942.10', '90.939', '-' => None)."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("-", ""):
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"Unparseable decimal for {field}: {value!r}") from exc


def normalize_contract_month(label: str) -> str:
    """Normalize a CME month label ('FEB 26', 'FEB 2026') to 'YYYY-MM'."""
    parts = str(label).strip().upper().split()
    if len(parts) != 2 or parts[0] not in _MONTH_NUM:
        raise ValueError(f"Unrecognized CME contract month label: {label!r}")
    mon, year_text = parts
    if not year_text.isdigit() or len(year_text) not in (2, 4):
        raise ValueError(f"Unrecognized CME contract month label: {label!r}")
    year = int(year_text) if len(year_text) == 4 else 2000 + int(year_text)
    return f"{year:04d}-{_MONTH_NUM[mon]:02d}"


def _report_is_preliminary(report_type: str, label: str) -> bool:
    text = str(report_type).strip().lower()
    if text.startswith("prelim"):
        return True
    if text.startswith("final"):
        return False
    raise ValueError(f"{label}: unrecognized reportType {report_type!r}")


def payload_is_empty(payload: dict[str, Any], label: str) -> bool:
    """The web services flag not-yet-published reports with 'empty': true."""
    if "empty" not in payload:
        raise ValueError(f"{label}: payload has no 'empty' field — schema drift? {list(payload)}")
    return bool(payload["empty"])


def parse_trade_dates(payload: Any, label: str) -> list[tuple[date, str]]:
    """Parse the TradeDate listing: [["MM/DD/YYYY", "Final"], ...] newest first.

    Report types are normalized to 'preliminary' | 'final'.
    """
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{label}: empty or non-list TradeDate payload: {payload!r}")
    out: list[tuple[date, str]] = []
    for entry in payload:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"{label}: malformed TradeDate entry: {entry!r}")
        day = datetime.strptime(str(entry[0]), "%m/%d/%Y").date()
        rtype = "preliminary" if _report_is_preliminary(str(entry[1]), label) else "final"
        out.append((day, rtype))
    return sorted(out, key=lambda item: item[0], reverse=True)


def parse_settlements(
    payload: dict[str, Any], label: str
) -> tuple[date, bool, dict[str, float | None]]:
    """Parse a Settlements payload into (trade_date, is_preliminary, settle map).

    The settle map is keyed by normalized contract month. Only ``settle`` is
    consumed here: the payload's per-month openInterest is the *prior* day's
    final OI (see module docstring) and its volume semantics are undocumented.
    """
    for key in ("settlements", "reportType", "tradeDate"):
        if key not in payload:
            raise ValueError(f"{label}: settlements payload missing {key!r} — schema drift?")
    if payload_is_empty(payload, label):
        raise ValueError(f"{label}: settlements payload is flagged empty (report not available)")
    trade_date = datetime.strptime(str(payload["tradeDate"]), "%m/%d/%Y").date()
    preliminary = _report_is_preliminary(str(payload["reportType"]), label)

    rows = payload["settlements"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label}: settlements payload has no rows")
    settle_map: dict[str, float | None] = {}
    n_total_rows = 0
    for row in rows:
        if not isinstance(row, dict) or "month" not in row or "settle" not in row:
            raise ValueError(f"{label}: malformed settlements row: {row!r}")
        if str(row["month"]).strip() == "Total":
            n_total_rows += 1
            continue
        month = normalize_contract_month(str(row["month"]))
        if month in settle_map:
            raise ValueError(f"{label}: duplicate settlements month {month}")
        settle_map[month] = _to_float(row["settle"], f"settle[{month}]")
    if n_total_rows != 1:
        raise ValueError(f"{label}: expected exactly one 'Total' settlements row, {n_total_rows=}")
    if not settle_map:
        raise ValueError(f"{label}: settlements payload has no contract-month rows")
    return trade_date, preliminary, settle_map


def parse_volume_details(payload: dict[str, Any], label: str) -> tuple[date, pd.DataFrame]:
    """Parse a Volume Details payload into (trade_date, per-month frame).

    The frame has one row per contract month plus one AGG row built from the
    payload's own ``totals`` block, with columns contract_month, volume,
    open_interest, oi_change (nullable integers).
    """
    for key in ("tradeDate", "totals", "monthData"):
        if key not in payload:
            raise ValueError(f"{label}: volume payload missing {key!r} — schema drift?")
    if payload_is_empty(payload, label):
        raise ValueError(f"{label}: volume payload is flagged empty (report not available)")
    trade_date = datetime.strptime(str(payload["tradeDate"]), "%Y%m%d").date()

    month_rows = payload["monthData"]
    if not isinstance(month_rows, list) or not month_rows:
        raise ValueError(f"{label}: volume payload has no monthData rows")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in month_rows:
        if not isinstance(row, dict) or not {"month", "totalVolume", "atClose", "change"} <= set(
            row
        ):
            raise ValueError(f"{label}: malformed volume monthData row: {row!r}")
        month = normalize_contract_month(str(row["month"]))
        if month in seen:
            raise ValueError(f"{label}: duplicate volume month {month}")
        seen.add(month)
        records.append(
            {
                "contract_month": month,
                "volume": _to_int(row["totalVolume"], f"totalVolume[{month}]"),
                "open_interest": _to_int(row["atClose"], f"atClose[{month}]"),
                "oi_change": _to_int(row["change"], f"change[{month}]"),
            }
        )

    totals = payload["totals"]
    if not isinstance(totals, dict) or not {"totalVolume", "atClose", "change"} <= set(totals):
        raise ValueError(f"{label}: malformed volume totals block: {totals!r}")
    records.append(
        {
            "contract_month": AGG_MONTH,
            "volume": _to_int(totals["totalVolume"], "totals.totalVolume"),
            "open_interest": _to_int(totals["atClose"], "totals.atClose"),
            "oi_change": _to_int(totals["change"], "totals.change"),
        }
    )
    df = pd.DataFrame.from_records(records)
    for col in ("volume", "open_interest", "oi_change"):
        df[col] = df[col].astype("Int64")
    return trade_date, df


def assemble_product_frame(
    product: str,
    volume_payload: dict[str, Any],
    settlements_payload: dict[str, Any],
    preliminary: bool,
    pulled_at: datetime,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge one volume report with the settlements report into upsert-ready rows.

    ``preliminary`` is the report type the volume payload was fetched under
    (P|F in the URL — the payload itself does not repeat it) and becomes the
    row's ``is_preliminary``. Settles attach by contract month unless the
    settlements report is preliminary while the volume report is final.
    """
    _product_id(product)
    if pulled_at.tzinfo is None:
        raise ValueError("pulled_at must be timezone-aware (UTC); got a naive datetime")
    pulled_naive = pulled_at.astimezone(UTC).replace(tzinfo=None)

    vol_date, vol_df = parse_volume_details(volume_payload, f"{product}/volume")
    settle_date, settle_prelim, settle_map = parse_settlements(
        settlements_payload, f"{product}/settlements"
    )
    if vol_date != settle_date:
        raise ValueError(
            f"{product}: volume trade date {vol_date} != settlements trade date {settle_date}"
        )
    lag_days = (pulled_naive.date() - vol_date).days
    if lag_days < 0:
        raise ValueError(f"{product}: trade date {vol_date} is after pull time {pulled_naive}")

    settle_attached = not (settle_prelim and not preliminary)
    attached_map = settle_map if settle_attached else {}

    df = vol_df.copy()
    df["settle"] = pd.array([attached_map.get(m) for m in df["contract_month"]], dtype="Float64")
    df["trade_date"] = pd.Timestamp(vol_date)
    df["product"] = product
    df["is_preliminary"] = preliminary
    df["source"] = SOURCE_TAG
    df["pulled_at"] = pd.Timestamp(pulled_naive)
    df["is_realtime"] = lag_days <= REALTIME_MAX_LAG_DAYS
    df = df[_COLUMNS]

    vol_months = set(vol_df["contract_month"]) - {AGG_MONTH}
    meta = {
        "trade_date": vol_date.isoformat(),
        "report": "P" if preliminary else "F",
        "settle_report": "preliminary" if settle_prelim else "final",
        "settle_attached": settle_attached,
        "settle_months_unmatched": len(set(settle_map) - vol_months),
        "is_realtime": lag_days <= REALTIME_MAX_LAG_DAYS,
    }
    return df, meta


# --------------------------------------------------------------------------
# DB
# --------------------------------------------------------------------------


def upsert_cme_daily(df: pd.DataFrame) -> int:
    """Idempotent upsert into ``cme_daily``. Returns row count written.

    Re-pulls update figures in place for the same (date, product, month,
    report) key, but provenance is first-capture: ``is_realtime`` can never
    be demoted, and a real-time row keeps its original ``pulled_at`` even if
    re-pulled retroactively later (retro rows may refresh theirs).
    """
    if df.empty:
        return 0
    with connection() as conn:
        conn.register("incoming_cme_daily", df)
        conn.execute(
            """
            INSERT INTO cme_daily
                (trade_date, product, contract_month, settle, volume,
                 open_interest, oi_change, is_preliminary, source, pulled_at, is_realtime)
            SELECT
                CAST(trade_date AS DATE), product, contract_month, settle, volume,
                open_interest, oi_change, is_preliminary, source, pulled_at, is_realtime
            FROM incoming_cme_daily
            ON CONFLICT (trade_date, product, contract_month, is_preliminary) DO UPDATE SET
                settle        = EXCLUDED.settle,
                volume        = EXCLUDED.volume,
                open_interest = EXCLUDED.open_interest,
                oi_change     = EXCLUDED.oi_change,
                source        = EXCLUDED.source,
                pulled_at     = CASE WHEN cme_daily.is_realtime THEN cme_daily.pulled_at
                                     ELSE EXCLUDED.pulled_at END,
                is_realtime   = cme_daily.is_realtime OR EXCLUDED.is_realtime
            """
        )
        conn.unregister("incoming_cme_daily")
    return int(len(df))


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def collect_product(
    product: str,
    trade_date: date | None = None,
    pulled_at: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch and assemble all available reports (P and F) for one product/date.

    Defaults to the latest trade date the settlements service advertises.
    Raises if neither a preliminary nor a final volume report exists.
    """
    pulled_at = pulled_at or datetime.now(UTC)
    available = fetch_trade_dates(product)
    if trade_date is None:
        trade_date = available[0][0]
    elif trade_date not in {d for d, _ in available}:
        raise ValueError(
            f"{product}: trade date {trade_date} not offered by CME; "
            f"available: {[d.isoformat() for d, _ in available]}"
        )

    settlements_payload = fetch_settlements(product, trade_date)
    frames: list[pd.DataFrame] = []
    meta: dict[str, Any] = {"trade_date": trade_date.isoformat(), "reports_captured": []}
    for preliminary in (False, True):
        volume_payload = fetch_volume_details(product, trade_date, preliminary)
        report = "P" if preliminary else "F"
        if payload_is_empty(volume_payload, f"{product}/volume/{report}"):
            continue  # this report type is not published yet — a fact, not a failure
        df, report_meta = assemble_product_frame(
            product, volume_payload, settlements_payload, preliminary, pulled_at
        )
        frames.append(df)
        meta["reports_captured"].append(report)
        meta["settle_report"] = report_meta["settle_report"]
        meta["settle_months_unmatched"] = report_meta["settle_months_unmatched"]
        meta["is_realtime"] = report_meta["is_realtime"]
    if not frames:
        raise RuntimeError(
            f"{product}: CME published neither preliminary nor final volume/OI for {trade_date}"
        )
    return pd.concat(frames, ignore_index=True), meta


def refresh(
    products: list[str] | None = None,
    trade_date: date | str | None = None,
) -> dict[str, Any]:
    """Pull the latest (or a given) trade date for every product and upsert.

    Returns a summary dict including "rows_written".
    """
    product_list = list(products) if products else sorted(PRODUCTS)
    if isinstance(trade_date, str):
        trade_date = date.fromisoformat(trade_date)
    pulled_at = datetime.now(UTC)

    total = 0
    per_product: dict[str, Any] = {}
    for product in product_list:
        df, meta = collect_product(product, trade_date=trade_date, pulled_at=pulled_at)
        n = upsert_cme_daily(df)
        total += n
        per_product[product] = {**meta, "rows": n}
    return {
        "rows_written": total,
        "products": per_product,
        "pulled_at": pulled_at.isoformat(),
    }


def gap_check(
    n_days: int = 10,
    products: list[str] | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """List NYSE business days with no captured AGG row in the trailing window.

    The window is the ``n_days`` calendar days before ``as_of`` (exclusive:
    ``as_of`` itself is not yet due). A day counts as covered if any report
    (preliminary or final) wrote the product's AGG row.
    """
    product_list = list(products) if products else sorted(PRODUCTS)
    as_of = as_of or datetime.now(UTC).date()
    start = as_of - timedelta(days=n_days)
    end = as_of - timedelta(days=1)
    holidays = NyseHolidayCalendar().holidays(pd.Timestamp(start), pd.Timestamp(end))
    expected = [ts.date() for ts in pd.bdate_range(start, end, freq="C", holidays=list(holidays))]

    missing: dict[str, list[date]] = {}
    with connection() as conn:
        for product in product_list:
            rows = conn.execute(
                "SELECT DISTINCT trade_date FROM cme_daily "
                "WHERE product = ? AND contract_month = ?",
                [product, AGG_MONTH],
            ).fetchall()
            have = {row[0] for row in rows}
            missing[product] = [d for d in expected if d not in have]
    return {
        "as_of": as_of.isoformat(),
        "window": [start.isoformat(), end.isoformat()],
        "expected_days": [d.isoformat() for d in expected],
        "missing": {p: [d.isoformat() for d in days] for p, days in missing.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CME daily settlement volume/OI collector.")
    parser.add_argument("--products", nargs="*", choices=sorted(PRODUCTS), default=None)
    parser.add_argument("--trade-date", default=None, help="YYYY-MM-DD; default latest available")
    parser.add_argument("--check-gaps", action="store_true", help="Report missing days, no pull.")
    parser.add_argument("--gap-days", type=int, default=10, help="Trailing window for gap check.")
    args = parser.parse_args()

    if args.check_gaps:
        report = gap_check(n_days=args.gap_days, products=args.products)
        print(f"Gap check window:  {report['window'][0]} .. {report['window'][1]}")
        print(f"Expected days:     {len(report['expected_days'])}")
        for product, days in report["missing"].items():
            status = ", ".join(days) if days else "complete"
            print(f"  {product}: {status}")
        return

    summary = refresh(products=args.products, trade_date=args.trade_date)
    print(f"Rows written: {summary['rows_written']}")
    for product, meta in summary["products"].items():
        print(
            f"  {product}: {meta['trade_date']} reports={meta['reports_captured']} "
            f"settle={meta.get('settle_report', '-')} rows={meta['rows']} "
            f"realtime={meta.get('is_realtime', '-')}"
        )


if __name__ == "__main__":
    main()
