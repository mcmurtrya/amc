"""Google Trends as-pulled search-interest archiver (Phase 7.1, collector 3).

Weekly pull of the frozen term groups in ``configs/trends_terms.yaml`` into the
``search_interest`` table. Google Trends *rescales* its 0-100 index on every
request, so a series downloaded later is not the series a real-time observer
would have seen — the only honest history is an archive of as-pulled snapshots.
Every row therefore stores the verbatim ``request_params`` JSON (term set, geo,
timeframe, tz, endpoints, and the resolved widget request): the parameters are
part of the observation.

is_realtime rule: a row is realtime only if its ``period_end`` falls within
``REALTIME_WINDOW_DAYS`` (14) days of ``pulled_at``. Everything earlier in the
same response is setup-time context — Google served it retrospectively under
today's rescaling — and gets ``is_realtime = False``, permanently.

User-Agent deviation: this collector sends a desktop browser UA instead of the
program's "AMCResearchCollector/0.1" convention. The unofficial Trends API
answers 429 to non-browser agents (verified 2026-07-12); an identified UA here
means no data at all. Politeness is kept via the request rate instead
(>= ``FETCH_SLEEP_S`` seconds between calls, weekly cadence, two API calls per
term group).

Transport (verified live 2026-07-12): a cookie must first be seeded from
google.com (the explore endpoint 429s cookie-less), then
POST ``/trends/api/explore`` yields widget tokens, and
GET ``/trends/api/widgetdata/multiline`` yields the interest-over-time data.
Both responses carry the ``)]}'`` anti-JSON prefix. On HTTP 429 the collector
sleeps and retries once, then raises — a silent gap is the failure mode this
program exists to prevent.

Run as (weekly cadence):
    uv run python -m metals.data.trends
"""

from __future__ import annotations

import argparse
import calendar
import json
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from metals.data.db import connection

SOURCE_TAG = "google-trends"
DEFAULT_TERMS_YAML = Path(__file__).resolve().parents[3] / "configs" / "trends_terms.yaml"

COOKIE_SEED_URL = "https://www.google.com/"
EXPLORE_URL = "https://trends.google.com/trends/api/explore"
MULTILINE_URL = "https://trends.google.com/trends/api/widgetdata/multiline"

# Deviation from the collector UA convention — see module docstring.
DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

ANTI_JSON_PREFIX = ")]}'"
REALTIME_WINDOW_DAYS = 14
FETCH_SLEEP_S = 3.0
BACKOFF_SLEEP_S = 60.0
MAX_TERMS_PER_GROUP = 5  # hard Trends limit on comparison items per request
REQUEST_TIMEOUT_S = 30

SleepFn = Callable[[float], object]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_term_groups(path: Path | str = DEFAULT_TERMS_YAML) -> list[dict[str, Any]]:
    """Load and validate the frozen term groups from the YAML config.

    Each group must carry ``name``, ``geo``, ``timeframe`` and 1-5 ``terms``.
    Raises on any structural problem — a malformed config must never produce a
    silently-partial pull.
    """
    with Path(path).open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("groups"), list) or not cfg["groups"]:
        raise ValueError(f"{path}: expected a top-level 'groups' list with at least one group")
    groups: list[dict[str, Any]] = []
    for i, group in enumerate(cfg["groups"]):
        if not isinstance(group, dict):
            raise ValueError(f"{path}: group #{i} is not a mapping")
        for key in ("name", "geo", "timeframe"):
            if not isinstance(group.get(key), str) or not group[key].strip():
                raise ValueError(f"{path}: group #{i} missing required string field {key!r}")
        terms = group.get("terms")
        if (
            not isinstance(terms, list)
            or not terms
            or not all(isinstance(t, str) and t.strip() for t in terms)
        ):
            raise ValueError(f"{path}: group {group['name']!r} needs a non-empty list of terms")
        if len(terms) > MAX_TERMS_PER_GROUP:
            raise ValueError(
                f"{path}: group {group['name']!r} has {len(terms)} terms; "
                f"Trends allows at most {MAX_TERMS_PER_GROUP} per request"
            )
        groups.append(group)
    return groups


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def build_session() -> requests.Session:
    """Return a session with a browser UA and a seeded google.com cookie.

    The explore endpoint answers 429 to cookie-less clients; a plain GET to
    google.com sets the cookie that makes it answer (verified 2026-07-12).
    """
    session = requests.Session()
    session.headers["User-Agent"] = DESKTOP_UA
    session.headers["Accept-Language"] = "en-US,en;q=0.9"
    resp = session.get(COOKIE_SEED_URL, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return session


def _request_with_backoff(
    session: requests.Session,
    method: str,
    url: str,
    params: dict[str, str],
    sleep: SleepFn = time.sleep,
) -> str:
    """Issue one request; on 429 sleep and retry exactly once, then raise.

    Any other non-200 status raises immediately. Never returns silently-empty
    data: an empty body is an error.
    """
    resp = session.request(method, url, params=params, timeout=REQUEST_TIMEOUT_S)
    if resp.status_code == 429:
        sleep(BACKOFF_SLEEP_S)
        resp = session.request(method, url, params=params, timeout=REQUEST_TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Google Trends {url} returned HTTP {resp.status_code} after backoff; "
            "refusing to record a silent gap — investigate before the next pull"
        )
    if not resp.text:
        raise RuntimeError(f"Google Trends {url} returned an empty body")
    return resp.text


def strip_antijson_prefix(text: str) -> str:
    """Strip the ``)]}'`` anti-JSON prefix line Google prepends to API responses.

    Raises if the prefix is absent — that means the response format drifted.
    """
    if not text.startswith(ANTI_JSON_PREFIX):
        raise ValueError(
            f"response does not start with the {ANTI_JSON_PREFIX!r} anti-JSON prefix "
            "— Trends response format drift?"
        )
    _prefix, sep, rest = text.partition("\n")
    if not sep or not rest.strip():
        raise ValueError("no payload after the anti-JSON prefix line")
    return rest


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------


def fetch_explore(
    session: requests.Session,
    terms: list[str],
    geo: str,
    timeframe: str,
    tz: int = 0,
    sleep: SleepFn = time.sleep,
) -> dict[str, Any]:
    """Step 1 of the unofficial flow: obtain widget tokens from /api/explore."""
    req = {
        "comparisonItem": [{"keyword": t, "geo": geo, "time": timeframe} for t in terms],
        "category": 0,
        "property": "",
    }
    params = {"hl": "en-US", "tz": str(tz), "req": json.dumps(req)}
    text = _request_with_backoff(session, "POST", EXPLORE_URL, params, sleep=sleep)
    payload = json.loads(strip_antijson_prefix(text))
    if not isinstance(payload, dict) or not payload.get("widgets"):
        raise ValueError("explore response carries no widgets — empty or drifted response")
    return payload


def extract_timeseries_widget(explore_payload: dict[str, Any]) -> dict[str, Any]:
    """Pick the TIMESERIES widget (token + request) out of an explore payload."""
    widgets = explore_payload.get("widgets")
    if not isinstance(widgets, list):
        raise ValueError("explore payload has no 'widgets' list")
    for widget in widgets:
        if isinstance(widget, dict) and widget.get("id") == "TIMESERIES":
            if not widget.get("token") or not isinstance(widget.get("request"), dict):
                raise ValueError("TIMESERIES widget is missing its token or request")
            return widget
    raise ValueError("explore payload has no TIMESERIES widget — schema drift")


def widget_terms(widget: dict[str, Any]) -> list[str]:
    """Read the term list back out of the TIMESERIES widget request.

    Used to confirm the widget covers exactly the terms we asked for, in
    order — the order defines which slot of each point's value array belongs
    to which term.
    """
    items = widget.get("request", {}).get("comparisonItem")
    if not isinstance(items, list) or not items:
        raise ValueError("TIMESERIES widget request has no comparisonItem list")
    terms: list[str] = []
    for item in items:
        keywords = item.get("complexKeywordsRestriction", {}).get("keyword")
        if not isinstance(keywords, list) or len(keywords) != 1 or "value" not in keywords[0]:
            raise ValueError("comparisonItem keyword structure drifted")
        terms.append(str(keywords[0]["value"]))
    return terms


def fetch_interest_over_time(
    session: requests.Session,
    widget: dict[str, Any],
    tz: int = 0,
    sleep: SleepFn = time.sleep,
) -> dict[str, Any]:
    """Step 2: pull the interest-over-time data for a TIMESERIES widget token."""
    params = {
        "hl": "en-US",
        "tz": str(tz),
        "req": json.dumps(widget["request"]),
        "token": str(widget["token"]),
    }
    text = _request_with_backoff(session, "GET", MULTILINE_URL, params, sleep=sleep)
    payload = json.loads(strip_antijson_prefix(text))
    if not isinstance(payload, dict):
        raise ValueError("multiline response is not a JSON object")
    return payload


def _period_end(start: date, resolution: str) -> date:
    """Nominal end date of the interval starting at ``start``."""
    if resolution == "DAY":
        return start
    if resolution == "WEEK":
        return start + timedelta(days=6)
    if resolution == "MONTH":
        return date(start.year, start.month, calendar.monthrange(start.year, start.month)[1])
    raise ValueError(f"unsupported Trends resolution {resolution!r} — expected DAY, WEEK or MONTH")


def parse_timeline(
    payload: dict[str, Any],
    *,
    terms: list[str],
    geo: str,
    resolution: str,
    pulled_at: datetime,
    request_params: dict[str, Any],
) -> pd.DataFrame:
    """Turn a multiline payload into long ``search_interest`` rows.

    ``pulled_at`` is UTC (naive, or aware — normalized to naive UTC). Raises on
    any structural drift: missing timelineData, unparseable point times, or a
    value array whose length does not match the term list.
    """
    if pulled_at.tzinfo is not None:
        pulled_at = pulled_at.astimezone(UTC).replace(tzinfo=None)
    default = payload.get("default")
    if not isinstance(default, dict):
        raise ValueError("multiline payload missing 'default' — schema drift")
    timeline = default.get("timelineData")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("multiline payload has no timelineData — empty or drifted response")

    params_json = json.dumps(request_params, sort_keys=True)
    rows: list[dict[str, Any]] = []
    for point in timeline:
        raw_time = point.get("time")
        try:
            start = datetime.fromtimestamp(int(raw_time), tz=UTC).date()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unparseable timelineData point time {raw_time!r}") from exc
        values = point.get("value")
        if not isinstance(values, list) or len(values) != len(terms):
            got = len(values) if isinstance(values, list) else "no"
            raise ValueError(
                f"point starting {start} carries {got} values for {len(terms)} terms — schema drift"
            )
        end = _period_end(start, resolution)
        is_realtime = (pulled_at.date() - end).days <= REALTIME_WINDOW_DAYS
        for term, value in zip(terms, values, strict=True):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"non-integer value {value!r} for term {term!r} at {start}")
            rows.append(
                {
                    "pulled_at": pulled_at,
                    "geo": geo,
                    "term": term,
                    "period_start": start,
                    "period_end": end,
                    "value": value,
                    "request_params": params_json,
                    "source": SOURCE_TAG,
                    "is_realtime": is_realtime,
                }
            )
    df = pd.DataFrame(rows)
    df["pulled_at"] = pd.to_datetime(df["pulled_at"])
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    return df


def collect_group(
    session: requests.Session,
    group: dict[str, Any],
    tz: int = 0,
    pulled_at: datetime | None = None,
    sleep: SleepFn = time.sleep,
) -> pd.DataFrame:
    """Run the two-step flow for one term group and return its long rows."""
    terms = [str(t) for t in group["terms"]]
    geo, timeframe = str(group["geo"]), str(group["timeframe"])

    explore = fetch_explore(session, terms, geo=geo, timeframe=timeframe, tz=tz, sleep=sleep)
    widget = extract_timeseries_widget(explore)
    served_terms = widget_terms(widget)
    if served_terms != terms:
        raise ValueError(
            f"TIMESERIES widget terms {served_terms!r} != requested {terms!r} — "
            "value columns would be misattributed"
        )
    resolution = widget["request"].get("resolution")
    if not isinstance(resolution, str):
        raise ValueError("TIMESERIES widget request has no 'resolution' — schema drift")

    sleep(FETCH_SLEEP_S)
    payload = fetch_interest_over_time(session, widget, tz=tz, sleep=sleep)
    if pulled_at is None:
        pulled_at = datetime.now(UTC).replace(tzinfo=None)

    request_params = {
        "group": group["name"],
        "terms": terms,
        "geo": geo,
        "timeframe": timeframe,
        "tz": tz,
        "explore_endpoint": EXPLORE_URL,
        "widget_endpoint": MULTILINE_URL,
        "widget_request": widget["request"],
    }
    return parse_timeline(
        payload,
        terms=terms,
        geo=geo,
        resolution=resolution,
        pulled_at=pulled_at,
        request_params=request_params,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def upsert_search_interest(df: pd.DataFrame) -> int:
    """Idempotent upsert into ``search_interest``. Returns rows written."""
    if df.empty:
        return 0
    with connection() as conn:
        conn.register("incoming_search_interest", df)
        conn.execute(
            """
            INSERT INTO search_interest
                (pulled_at, geo, term, period_start, period_end, value,
                 request_params, source, is_realtime)
            SELECT pulled_at, geo, term,
                   CAST(period_start AS DATE), CAST(period_end AS DATE), value,
                   request_params, source, is_realtime
            FROM incoming_search_interest
            ON CONFLICT (pulled_at, geo, term, period_start) DO UPDATE SET
                period_end     = EXCLUDED.period_end,
                value          = EXCLUDED.value,
                request_params = EXCLUDED.request_params,
                source         = EXCLUDED.source,
                is_realtime    = EXCLUDED.is_realtime
            """
        )
        conn.unregister("incoming_search_interest")
    return int(len(df))


def refresh(path: Path | str = DEFAULT_TERMS_YAML, tz: int = 0) -> dict:
    """Pull every configured term group and upsert. Return a summary dict.

    Fail-loud by design: any HTTP failure, empty response, or schema drift in
    any group raises and aborts the whole run — the scheduler's alert is the
    fix, not a quiet partial write.
    """
    groups = load_term_groups(path)
    session = build_session()
    frames: list[pd.DataFrame] = []
    per_group: dict[str, int] = {}
    for i, group in enumerate(groups):
        if i:
            time.sleep(FETCH_SLEEP_S)
        df = collect_group(session, group, tz=tz)
        per_group[str(group["name"])] = int(len(df))
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    n = upsert_search_interest(df_all)
    return {
        "rows_written": n,
        "rows_per_group": per_group,
        "period_range": [
            df_all["period_start"].min().date().isoformat(),
            df_all["period_end"].max().date().isoformat(),
        ],
        "realtime_rows": int(df_all["is_realtime"].sum()),
        "pulled_at": df_all["pulled_at"].max().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive Google Trends as-pulled snapshots.")
    parser.add_argument("--config", default=str(DEFAULT_TERMS_YAML), help="Term-groups YAML.")
    parser.add_argument("--tz", type=int, default=0, help="Trends tz offset (minutes; 0 = UTC).")
    args = parser.parse_args()
    summary = refresh(path=args.config, tz=args.tz)
    print(f"Rows written:   {summary['rows_written']}")
    print(f"Per group:      {summary['rows_per_group']}")
    print(f"Period range:   {summary['period_range']}")
    print(
        f"Realtime rows:  {summary['realtime_rows']} (period_end within "
        f"{REALTIME_WINDOW_DAYS}d of pull)"
    )
    print(f"Pulled at:      {summary['pulled_at']}Z")


if __name__ == "__main__":
    main()
