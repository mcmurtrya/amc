"""Google Trends as-pulled search-interest importer (Phase 7.1, collector 3).

**Rewritten 2026-07-16 from a live scraper into a manual-CSV importer.** The old
collector called Google's internal ``/trends/api`` endpoints, which answer 429 to
any non-browser client — so it sent a desktop-browser User-Agent to get served.
That defeats a protective measure, which Google's Terms of Service prohibit (ToU
audit 2026-07-16, journal.md). Google *separately* grants use of the data itself
("You can use any information from Google Trends, subject to the Google Terms of
Service", support.google.com/trends/answer/4365538) and offers a sanctioned CSV
download. So acquisition moves from the wire to that file; the ``search_interest``
schema and every downstream meaning are unchanged. **Attribution obligation:** any
analysis shipped to AMC that reuses this series must cite "Google Trends
(https://www.google.com/trends)".

Google Trends *rescales* its 0-100 index on every request, against the tuple
(term basket, geo, timeframe window). A series exported later is therefore not the
series a real-time observer saw — the only honest history is an archive of
as-pulled snapshots. Each row stores the verbatim ``request_params`` (group, geo,
timeframe, the frozen term set, the CSV's own header line, and an
``acquisition: manual_csv_export`` marker): the parameters are part of the
observation. Note the timeframe window is *not* recoverable from the CSV — it is
asserted from config and flagged as such in ``request_params``.

Operator workflow (weekly — mirrors the manual ``amc_ledger`` importer):
    1. In the Trends UI load the frozen ``sell_side_v1`` comparison (its five
       terms, United States, "Past 5 years"); click Download on the
       Interest-over-time panel to get ``multiTimeline.csv``.
    2. Drop it locally (``data/raw/trends/`` suggested; gitignored).
    3. ``uv run python -m metals.data.trends <multiTimeline.csv>``

Not scheduled: the export is a human action, so this collector is absent from the
``run_collectors`` registry (like ``amc_ledger``). Because Trends rescales per
request, every skipped week permanently loses that week's as-pulled snapshot —
run it weekly.

is_realtime rule (unchanged): realtime iff ``period_end`` is within
``REALTIME_WINDOW_DAYS`` (14) days of ``pulled_at``. ``pulled_at`` defaults to the
import moment, which is leakage-safe: import is always at or after the true
download, so the default can only *demote* freshness, never inflate it. Pass
``--pulled-at`` with the true Trends download timestamp to recover the final one
or two weekly rows and to make re-import idempotent (``pulled_at`` is part of the
primary key, so a fixed value upserts while the default mints a fresh snapshot).

Sub-1 values: the CSV emits the literal ``<1`` for nonzero interest below 1 on the
co-scaled index — distinct from a true ``0``. Stored as ``value = 0`` with
``value_lt1 = True`` (migration 011) so the "present but tiny" signal is never
merged into true zero, and the row is never dropped (dropping shifts every later
date and corrupts the weekly series).
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from metals.data.db import connection

SOURCE_TAG = "google-trends"
DEFAULT_TERMS_YAML = Path(__file__).resolve().parents[3] / "configs" / "trends_terms.yaml"
DEFAULT_GROUP = "sell_side_v1"

REALTIME_WINDOW_DAYS = 14
MAX_TERMS_PER_GROUP = 5  # hard Trends limit on comparison items per request

# The CSV's first column header is a resolution token; each data column header is
# "<term>: (<geo label>)". Both are the file's own ground truth (read, don't assume).
RESOLUTION_BY_TOKEN = {"Day": "DAY", "Week": "WEEK", "Month": "MONTH"}
GEO_LABELS = {"US": "United States"}  # config geo code -> CSV parenthetical label
HEADER_RE = re.compile(r"^(?P<term>.+?):\s*\((?P<geo>.+)\)$")
SUB_ONE_TOKEN = "<1"  # literal cell for nonzero interest below 1 (distinct from "0")
_BOM = "﻿"


class TrendsImportError(ValueError):
    """A Trends CSV export failed validation. Carries every violation; nothing written."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        msg = f"{len(self.violations)} validation error(s); nothing imported:\n" + "\n".join(
            f"  - {v}" for v in self.violations
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Config (unchanged from the scraper — the frozen term set still defines the series)
# ---------------------------------------------------------------------------


def load_term_groups(path: Path | str = DEFAULT_TERMS_YAML) -> list[dict[str, Any]]:
    """Load and validate the frozen term groups from the YAML config.

    Each group must carry ``name``, ``geo``, ``timeframe`` and 1-5 ``terms``.
    Raises on any structural problem — a malformed config must never produce a
    silently-partial import.
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


def get_group(name: str = DEFAULT_GROUP, path: Path | str = DEFAULT_TERMS_YAML) -> dict[str, Any]:
    """Return one frozen group from the config by name (raises if absent)."""
    groups = load_term_groups(path)
    for group in groups:
        if group["name"] == name:
            return group
    known = ", ".join(str(g["name"]) for g in groups)
    raise ValueError(f"term group {name!r} not found in {path} (known: {known})")


# ---------------------------------------------------------------------------
# Period math (unchanged)
# ---------------------------------------------------------------------------


def _period_end(start: date, resolution: str) -> date:
    """Nominal end date of the interval starting at ``start``."""
    if resolution == "DAY":
        return start
    if resolution == "WEEK":
        return start + timedelta(days=6)
    if resolution == "MONTH":
        return date(start.year, start.month, calendar.monthrange(start.year, start.month)[1])
    raise ValueError(f"unsupported Trends resolution {resolution!r} — expected DAY, WEEK or MONTH")


# ---------------------------------------------------------------------------
# CSV parse (the multiTimeline.csv layout)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportFrame:
    """A parsed multiTimeline.csv: header metadata + raw (period, value-cell) rows.

    ``periods`` holds raw string value cells (not yet numeric) aligned position-wise
    to ``terms`` — value tokens are validated later, together with term/geo
    reconciliation, so every problem is reported at once.
    """

    resolution_token: str  # "Week"
    resolution: str  # "WEEK"
    geo_label: str  # "United States"
    terms: list[str]  # parsed term set, in CSV column order
    header_line: str  # verbatim header row
    title_line: str | None  # "Category: All categories"
    periods: list[tuple[date, list[str]]]  # (period_start, raw value cells)


def _parse_period_start(cell: str, resolution: str) -> date:
    """Parse the time-column cell for a given resolution. Raises on malformed dates."""
    text = cell.strip()
    if resolution == "MONTH":
        year, _, month = text.partition("-")
        return date(int(year), int(month), 1)
    return date.fromisoformat(text)  # DAY / WEEK are YYYY-MM-DD (week-start Sunday)


def parse_multitimeline(text: str) -> ExportFrame:
    """Parse the raw text of a Trends interest-over-time CSV export.

    Tolerant of an optional UTF-8 BOM, the localized ``Category:`` title line, a
    blank/comma-only separator line, and Excel-resaved trailing commas. Raises
    ``TrendsImportError`` on structural drift: no resolution header row, a data
    column header that is not ``term: (geo)``, an unknown resolution token, a
    ragged data row, or an unparseable date.
    """
    if text.startswith(_BOM):
        text = text[len(_BOM) :]
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        raise TrendsImportError(["file is empty"])

    header_idx = next(
        (i for i, r in enumerate(rows) if r and r[0].strip() in RESOLUTION_BY_TOKEN),
        None,
    )
    if header_idx is None:
        raise TrendsImportError(
            ["no Day/Week/Month header row found — not a Trends interest-over-time export?"]
        )

    preamble = [r for r in rows[:header_idx] if any(c.strip() for c in r)]
    title_line = preamble[0][0].strip() if preamble and preamble[0] else None

    header = rows[header_idx]
    resolution_token = header[0].strip()
    resolution = RESOLUTION_BY_TOKEN[resolution_token]
    if len(header) < 2:
        raise TrendsImportError([f"header row {header!r} has no data columns"])

    terms: list[str] = []
    geo_labels: list[str] = []
    for col in header[1:]:
        match = HEADER_RE.match(col.strip())
        if not match:
            raise TrendsImportError(
                [f"column header {col!r} is not '<term>: (<geo>)' — Trends export format drift"]
            )
        terms.append(match.group("term").strip())
        geo_labels.append(match.group("geo").strip())
    if len(set(geo_labels)) != 1:
        raise TrendsImportError(
            [f"columns mix geographies {sorted(set(geo_labels))} — one expected"]
        )
    geo_label = geo_labels[0]

    periods: list[tuple[date, list[str]]] = []
    for r in rows[header_idx + 1 :]:
        if not any(c.strip() for c in r):
            continue  # tolerate trailing blank lines
        if len(r) != len(header):
            raise TrendsImportError(
                [f"data row {r!r} has {len(r)} cells; header has {len(header)} — ragged export"]
            )
        try:
            start = _parse_period_start(r[0], resolution)
        except ValueError as exc:
            raise TrendsImportError(
                [f"unparseable {resolution_token} date {r[0]!r} ({exc})"]
            ) from exc
        periods.append((start, [c for c in r[1:]]))
    if not periods:
        raise TrendsImportError(["export has a header but no data rows"])

    return ExportFrame(
        resolution_token=resolution_token,
        resolution=resolution,
        geo_label=geo_label,
        terms=terms,
        header_line=",".join(header),
        title_line=title_line,
        periods=periods,
    )


def read_export_csv(path: Path | str) -> ExportFrame:
    """Read a Trends CSV export file and parse it. Raises on an empty/unreadable file."""
    raw = Path(path).read_text(encoding="utf-8")
    if not raw.strip():
        raise TrendsImportError([f"{path}: file is empty"])
    return parse_multitimeline(raw)


def reconcile_terms(exp: ExportFrame, group: dict[str, Any], violations: list[str]) -> None:
    """Confirm the export covers exactly the frozen group's basket and geography.

    Continuity of the series depends on the request tuple being identical, so a
    different term set or geography must reject — not silently import a different
    series under the same table. Appends to ``violations``; never raises.
    """
    expected_terms = [str(t) for t in group["terms"]]
    csv_terms = list(exp.terms)
    missing = sorted(set(expected_terms) - set(csv_terms))
    unexpected = sorted(set(csv_terms) - set(expected_terms))
    if missing:
        violations.append(f"export is missing frozen term(s): {', '.join(missing)}")
    if unexpected:
        violations.append(
            f"export has term(s) not in group {group['name']!r}: {', '.join(unexpected)}"
        )
    dupes = sorted({t for t in csv_terms if csv_terms.count(t) > 1})
    if dupes:
        violations.append(f"export has duplicate term column(s): {', '.join(dupes)}")

    expected_label = GEO_LABELS.get(str(group["geo"]))
    if expected_label is None:
        violations.append(
            f"no CSV geo label known for config geo {group['geo']!r} — add it to GEO_LABELS"
        )
    elif exp.geo_label != expected_label:
        violations.append(
            f"export geography {exp.geo_label!r} != expected {expected_label!r} "
            f"for config geo {group['geo']!r}"
        )


def _parse_value_cell(
    raw: str, term: str, start: date, violations: list[str]
) -> tuple[int | None, bool]:
    """Parse one value cell. Returns (value, is_sub_one). Appends on drift."""
    s = raw.strip()
    if s == SUB_ONE_TOKEN:
        return 0, True
    if s == "":
        violations.append(f"{start} term {term!r}: empty value cell — Trends export drift")
        return None, False
    try:
        value = int(s)
    except ValueError:
        violations.append(
            f"{start} term {term!r}: value {raw!r} is neither an integer nor {SUB_ONE_TOKEN!r}"
        )
        return None, False
    if not 0 <= value <= 100:
        violations.append(f"{start} term {term!r}: value {value} outside the 0-100 index range")
        return None, False
    return value, False


# ---------------------------------------------------------------------------
# Row assembly (the kept core of the old parse_timeline, fed from the CSV)
# ---------------------------------------------------------------------------


def build_rows(
    exp: ExportFrame,
    *,
    geo: str,
    pulled_at: datetime,
    request_params: dict[str, Any],
) -> pd.DataFrame:
    """Turn a reconciled ExportFrame into long ``search_interest`` rows.

    ``geo`` is the config code (e.g. "US") stored in the table — not the CSV's
    human label — so imported rows match the old scraped rows. ``pulled_at`` is
    normalized to naive UTC. Raises ``TrendsImportError`` if any value cell drifts.
    """
    if pulled_at.tzinfo is not None:
        pulled_at = pulled_at.astimezone(UTC).replace(tzinfo=None)
    params_json = json.dumps(request_params, sort_keys=True)
    violations: list[str] = []
    rows: list[dict[str, Any]] = []
    for start, cells in exp.periods:
        end = _period_end(start, exp.resolution)
        is_realtime = (pulled_at.date() - end).days <= REALTIME_WINDOW_DAYS
        for term, raw in zip(exp.terms, cells, strict=True):
            value, is_sub_one = _parse_value_cell(raw, term, start, violations)
            rows.append(
                {
                    "pulled_at": pulled_at,
                    "geo": geo,
                    "term": term,
                    "period_start": start,
                    "period_end": end,
                    "value": value,
                    "value_lt1": is_sub_one,
                    "request_params": params_json,
                    "source": SOURCE_TAG,
                    "is_realtime": is_realtime,
                }
            )
    if violations:
        raise TrendsImportError(violations)
    df = pd.DataFrame(rows)
    df["pulled_at"] = pd.to_datetime(df["pulled_at"])
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["value"] = df["value"].astype("int64")
    df["value_lt1"] = df["value_lt1"].astype("bool")
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def upsert_search_interest(df: pd.DataFrame) -> int:
    """Idempotent upsert into ``search_interest``. Returns rows written.

    ``quarantine_reason`` is intentionally not written: new imported rows default
    it to NULL (usable), and it is excluded from the conflict update so a licence
    clearance is never silently undone by a re-import.
    """
    if df.empty:
        return 0
    with connection() as conn:
        conn.register("incoming_search_interest", df)
        conn.execute(
            """
            INSERT INTO search_interest
                (pulled_at, geo, term, period_start, period_end, value, value_lt1,
                 request_params, source, is_realtime)
            SELECT pulled_at, geo, term,
                   CAST(period_start AS DATE), CAST(period_end AS DATE), value, value_lt1,
                   request_params, source, is_realtime
            FROM incoming_search_interest
            ON CONFLICT (pulled_at, geo, term, period_start) DO UPDATE SET
                period_end     = EXCLUDED.period_end,
                value          = EXCLUDED.value,
                value_lt1      = EXCLUDED.value_lt1,
                request_params = EXCLUDED.request_params,
                source         = EXCLUDED.source,
                is_realtime    = EXCLUDED.is_realtime
            """
        )
        conn.unregister("incoming_search_interest")
    return int(len(df))


def refresh(
    path: Path | str,
    group: str = DEFAULT_GROUP,
    *,
    pulled_at: datetime | None = None,
    config: Path | str = DEFAULT_TERMS_YAML,
) -> dict:
    """Validate + import one Trends CSV export (all-or-nothing). Returns a summary dict.

    ``path`` is the multiTimeline.csv export; ``group`` selects the frozen config
    group to reconcile against. ``pulled_at`` defaults to the import moment (the
    leakage-safe choice — see the module docstring); pass the true Trends download
    time to recover the tail realtime rows and make re-import idempotent.
    """
    grp = get_group(group, config)
    geo = str(grp["geo"])
    exp = read_export_csv(path)

    violations: list[str] = []
    reconcile_terms(exp, grp, violations)
    if violations:
        raise TrendsImportError(violations)

    if pulled_at is None:
        pulled_at = datetime.now(UTC).replace(tzinfo=None)
    downloaded_at = (
        pulled_at.astimezone(UTC) if pulled_at.tzinfo is not None else pulled_at
    ).replace(tzinfo=None)

    request_params = {
        "acquisition": "manual_csv_export",
        "group": grp["name"],
        "geo": geo,
        "timeframe": str(grp["timeframe"]),
        "timeframe_source": "config",  # the rescaling window is NOT in the CSV
        "resolution": exp.resolution,
        "resolution_source": "csv_header_token",
        "terms": list(exp.terms),
        "source_file": Path(path).name,
        "csv_header": exp.header_line,
        "csv_title_line": exp.title_line,
        "config_path": str(config),
        "downloaded_at": downloaded_at.isoformat() + "Z",
    }
    df = build_rows(exp, geo=geo, pulled_at=pulled_at, request_params=request_params)
    n = upsert_search_interest(df)
    return {
        "source": SOURCE_TAG,
        "group": grp["name"],
        "source_file": Path(path).name,
        "resolution": exp.resolution,
        "rows_written": n,
        "period_range": [
            df["period_start"].min().date().isoformat(),
            df["period_end"].max().date().isoformat(),
        ],
        "realtime_rows": int(df["is_realtime"].sum()),
        "sub_one_rows": int(df["value_lt1"].sum()),
        "pulled_at": df["pulled_at"].max().isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Import a Google Trends multiTimeline.csv export into search_interest.",
    )
    parser.add_argument("file", help="Path to the Trends interest-over-time CSV export.")
    parser.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        help=f"Frozen term group to reconcile against (default {DEFAULT_GROUP}).",
    )
    parser.add_argument(
        "--pulled-at",
        default=None,
        help="True Trends download time (ISO 8601). Default: import moment (leakage-safe).",
    )
    parser.add_argument("--config", default=str(DEFAULT_TERMS_YAML), help="Term-groups YAML.")
    args = parser.parse_args(argv)

    pulled_at = None
    if args.pulled_at is not None:
        pulled_at = datetime.fromisoformat(args.pulled_at)

    try:
        summary = refresh(args.file, group=args.group, pulled_at=pulled_at, config=args.config)
    except TrendsImportError as exc:
        print(f"IMPORT REJECTED: {args.file}", file=sys.stderr)
        for v in exc.violations:
            print(f"  - {v}", file=sys.stderr)
        print(f"{len(exc.violations)} violation(s); nothing was written.", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Group:         {summary['group']} ({summary['resolution']})")
    print(f"Rows written:  {summary['rows_written']}")
    print(f"Period range:  {summary['period_range']}")
    print(
        f"Realtime rows: {summary['realtime_rows']} (period_end within "
        f"{REALTIME_WINDOW_DAYS}d of pull)"
    )
    print(f"Sub-1 rows:    {summary['sub_one_rows']} (value '<1' stored as 0, value_lt1=True)")
    print(f"Pulled at:     {summary['pulled_at']}Z")


if __name__ == "__main__":
    main()
