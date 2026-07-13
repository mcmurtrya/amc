"""AMC ledger importer (Phase 7.1, collector 1).

Validates and imports AMC Company's bookkeeping exports — scrap purchases,
coin/specie trades, and daily till counts — into the local DuckDB tables
``amc_scrap_lots``, ``amc_coin_trades`` and ``amc_till_daily`` (migration
``008_amc_ledger.sql``).

**This data never leaves the local machine.** No cloud service touches AMC's
books and nothing here is ever committed to git (``data/raw`` is gitignored).
Recommended drop location for AMC's weekly exports: ``data/raw/amc_ledger/``.

Export format: one CSV per table whose header row is exactly the target
table's business columns — see the templates in ``configs/templates/``
(rows whose notes start with ``EXAMPLE`` are template placeholders and are
rejected; delete them before importing). Provenance columns (``source_file``,
``batch_id``, ``imported_at``) are stamped by the importer, never supplied by
the export. Naive timestamps are interpreted in ``--tz`` (default
America/Chicago) and stored naive UTC; values carrying an explicit UTC offset
are converted as written.

Import is all-or-nothing: any validation error reports EVERY violation with
its line number, writes nothing, and exits nonzero. Re-importing a corrected
export upserts on the primary key (``lot_id`` / ``trade_id`` / ``date_utc``),
so books can be corrected and dispositions can arrive on a later export.

Run as:
    uv run python -m metals.data.amc_ledger <export.csv> --table scrap
    uv run python -m metals.data.amc_ledger <export.csv> --table coins --tz America/Chicago
    uv run python -m metals.data.amc_ledger <export.csv> --table till
"""

from __future__ import annotations

import argparse
import math
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import pandas as pd

from metals.data.db import connection

SOURCE_TAG = "amc_ledger"
DEFAULT_TZ = "America/Chicago"
GRAMS_PER_TROY_OZ = 31.1034768
METALS = {"gold", "silver", "platinum", "palladium"}
SIDES = {"buy", "sell"}
DISPOSITIONS = {"refined", "sold", "melted", "other"}
EXAMPLE_MARKER = "EXAMPLE"  # template placeholder rows start their notes with this
FINENESS_WARN_REL = 0.01  # fine_troy_oz vs gross*fineness/oz: warn above this
FINENESS_ERROR_REL = 0.05  # ... reject above this

SCRAP_COLUMNS = [
    "lot_id",
    "purchased_utc",
    "metal",
    "gross_weight_g",
    "fineness",
    "fine_troy_oz",
    "price_paid_usd",
    "spot_usd_oz",
    "disposed_utc",
    "disposition",
    "proceeds_usd",
    "notes",
]
COIN_COLUMNS = [
    "trade_id",
    "traded_utc",
    "side",
    "product",
    "quantity",
    "unit_price_usd",
    "spot_usd_oz",
    "metal",
    "fine_troy_oz_per_unit",
    "notes",
]
TILL_COLUMNS = ["date_utc", "walk_ins", "offers_made", "offers_accepted", "notes"]
PROVENANCE_COLUMNS = ["source_file", "batch_id", "imported_at"]


class LedgerValidationError(ValueError):
    """An export failed validation. Carries every violation; nothing was written."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        msg = f"{len(self.violations)} validation error(s); nothing imported:\n" + "\n".join(
            f"  - {v}" for v in self.violations
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Reading and field-level validation
# ---------------------------------------------------------------------------


def _read_export(path: Path | str, expected: list[str]) -> pd.DataFrame:
    """Read an export CSV as raw strings; enforce the exact expected header."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise LedgerValidationError(["file is empty"]) from exc
    df.columns = [str(c).strip() for c in df.columns]
    problems: list[str] = []
    missing = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in expected]
    if missing:
        problems.append(f"missing required column(s): {', '.join(missing)}")
    if unexpected:
        problems.append(f"unexpected column(s): {', '.join(unexpected)} - export format drift")
    if problems:
        raise LedgerValidationError(problems)
    if df.empty:
        raise LedgerValidationError(["file has a header but no data rows"])
    df = df[expected].copy()
    for col in expected:
        df[col] = df[col].fillna("").str.strip()
    return df


def _required_str(value: str, col: str, line: int, violations: list[str]) -> str | None:
    if value == "":
        violations.append(f"line {line}: {col} is required")
        return None
    return value


def _choice(
    value: str, col: str, line: int, allowed: set[str], violations: list[str]
) -> str | None:
    if value == "":
        violations.append(f"line {line}: {col} is required")
        return None
    v = value.lower()
    if v not in allowed:
        violations.append(f"line {line}: {col} {value!r} not one of {sorted(allowed)}")
        return None
    return v


def _optional_choice(
    value: str, col: str, line: int, allowed: set[str], violations: list[str]
) -> str | None:
    if value == "":
        return None
    return _choice(value, col, line, allowed, violations)


def _positive_float(
    value: str, col: str, line: int, violations: list[str], required: bool
) -> float | None:
    if value == "":
        if required:
            violations.append(f"line {line}: {col} is required")
        return None
    try:
        x = float(value)
    except ValueError:
        violations.append(f"line {line}: {col} {value!r} is not a number")
        return None
    if not math.isfinite(x):  # float() accepts 'inf'/'Infinity'/'nan' and 1e999 overflows to inf
        violations.append(f"line {line}: {col} must be finite, got {value}")
        return None
    if x <= 0:
        violations.append(f"line {line}: {col} must be > 0, got {value}")
        return None
    return x


def _positive_int(value: str, col: str, line: int, violations: list[str]) -> int | None:
    if value == "":
        violations.append(f"line {line}: {col} is required")
        return None
    try:
        x = int(value)
    except ValueError:
        violations.append(f"line {line}: {col} {value!r} is not a whole number")
        return None
    if x <= 0:
        violations.append(f"line {line}: {col} must be > 0, got {value}")
        return None
    return x


def _count(value: str, col: str, line: int, violations: list[str]) -> int | None:
    """Optional non-negative integer — till counts may legitimately be zero."""
    if value == "":
        return None
    try:
        x = int(value)
    except ValueError:
        violations.append(f"line {line}: {col} {value!r} is not a whole number")
        return None
    if x < 0:
        violations.append(f"line {line}: {col} must be >= 0, got {value}")
        return None
    return x


def _fineness(value: str, line: int, violations: list[str]) -> float | None:
    if value == "":
        violations.append(f"line {line}: fineness is required")
        return None
    try:
        x = float(value)
    except ValueError:
        violations.append(f"line {line}: fineness {value!r} is not a number")
        return None
    if not 0 < x <= 1:
        violations.append(f"line {line}: fineness must be in (0, 1], got {value}")
        return None
    return x


def _utc_timestamp(
    value: str,
    col: str,
    line: int,
    tz: ZoneInfo,
    now_utc: pd.Timestamp,
    violations: list[str],
    required: bool,
) -> pd.Timestamp | None:
    """Parse a timestamp; localize naive values from ``tz``; return naive UTC."""
    if value == "":
        if required:
            violations.append(f"line {line}: {col} is required")
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise ValueError("not a timestamp")
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)  # raises on ambiguous/nonexistent local times
    except Exception as exc:
        violations.append(f"line {line}: {col} {value!r} is not a valid timestamp ({exc})")
        return None
    ts_utc = ts.tz_convert("UTC").tz_localize(None)
    if ts_utc > now_utc:
        violations.append(f"line {line}: {col} {value!r} is in the future")
        return None
    return ts_utc


def _business_date(
    value: str, col: str, line: int, tz: ZoneInfo, violations: list[str]
) -> pd.Timestamp | None:
    """Parse a plain business date; must not be in the future in the export's tz."""
    if value == "":
        violations.append(f"line {line}: {col} is required")
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise ValueError("not a date")
    except Exception as exc:
        violations.append(f"line {line}: {col} {value!r} is not a valid date ({exc})")
        return None
    if ts.tzinfo is not None or ts != ts.normalize():
        violations.append(f"line {line}: {col} {value!r} must be a plain date (YYYY-MM-DD)")
        return None
    if ts.date() > datetime.now(tz).date():
        violations.append(f"line {line}: {col} {value!r} is in the future")
        return None
    return ts


def _notes(value: str, line: int, violations: list[str]) -> str | None:
    if value.startswith(EXAMPLE_MARKER):
        violations.append(
            f"line {line}: template EXAMPLE row - delete example rows before importing"
        )
        return None
    return value or None


def _reject_duplicate_keys(keys: pd.Series, col: str, violations: list[str]) -> None:
    """Reject rows whose PARSED primary key collides.

    Raw spellings may differ yet parse to the same key (e.g. '2026-01-05' and
    '01/05/2026' are the same date_utc), so uniqueness must be checked on the
    parsed values, never the raw strings — otherwise the later row silently
    clobbers the earlier one through the upsert's ON CONFLICT. Position i in
    ``keys`` corresponds to CSV line i + 2; None keys (blank or unparseable)
    are reported by the field validators instead.
    """
    dupes = keys[keys.duplicated(keep=False) & keys.notna()]
    for value in sorted(dupes.unique()):
        lines = ", ".join(str(int(i) + 2) for i in dupes.index[dupes == value])
        violations.append(f"duplicate {col} {value!r} appears on lines {lines}")


# ---------------------------------------------------------------------------
# Per-table parsers: raw string frame -> typed frame ready for upsert.
# Reject-all-or-import-all: every violation is collected, then raised together.
# ---------------------------------------------------------------------------


def parse_scrap_csv(
    df: pd.DataFrame, tz: ZoneInfo, now_utc: pd.Timestamp
) -> tuple[pd.DataFrame, list[str]]:
    """Validate raw scrap-lot rows. Returns (typed frame, warnings); raises on violations."""
    violations: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    for pos, (_, r) in enumerate(df.iterrows()):
        line = pos + 2
        purchased = _utc_timestamp(
            r["purchased_utc"], "purchased_utc", line, tz, now_utc, violations, required=True
        )
        disposed = _utc_timestamp(
            r["disposed_utc"], "disposed_utc", line, tz, now_utc, violations, required=False
        )
        gross = _positive_float(r["gross_weight_g"], "gross_weight_g", line, violations, True)
        fineness = _fineness(r["fineness"], line, violations)
        fine_oz = _positive_float(r["fine_troy_oz"], "fine_troy_oz", line, violations, True)
        if purchased is not None and disposed is not None and disposed < purchased:
            violations.append(f"line {line}: disposed_utc precedes purchased_utc")
        if gross is not None and fineness is not None and fine_oz is not None:
            computed = gross * fineness / GRAMS_PER_TROY_OZ
            rel = abs(fine_oz - computed) / computed
            if rel > FINENESS_ERROR_REL:
                violations.append(
                    f"line {line}: fine_troy_oz {fine_oz} is {rel:.1%} away from "
                    f"gross_weight_g * fineness / {GRAMS_PER_TROY_OZ} = {computed:.4f} (limit 5%)"
                )
            elif rel > FINENESS_WARN_REL:
                warnings.append(
                    f"line {line}: fine_troy_oz {fine_oz} is {rel:.1%} away from computed "
                    f"{computed:.4f} (>1% - check the assay entry)"
                )
        rows.append(
            {
                "lot_id": _required_str(r["lot_id"], "lot_id", line, violations),
                "purchased_utc": purchased,
                "metal": _choice(r["metal"], "metal", line, METALS, violations),
                "gross_weight_g": gross,
                "fineness": fineness,
                "fine_troy_oz": fine_oz,
                "price_paid_usd": _positive_float(
                    r["price_paid_usd"], "price_paid_usd", line, violations, True
                ),
                "spot_usd_oz": _positive_float(
                    r["spot_usd_oz"], "spot_usd_oz", line, violations, False
                ),
                "disposed_utc": disposed,
                "disposition": _optional_choice(
                    r["disposition"], "disposition", line, DISPOSITIONS, violations
                ),
                "proceeds_usd": _positive_float(
                    r["proceeds_usd"], "proceeds_usd", line, violations, False
                ),
                "notes": _notes(r["notes"], line, violations),
            }
        )
    _reject_duplicate_keys(pd.Series([r["lot_id"] for r in rows]), "lot_id", violations)
    if violations:
        raise LedgerValidationError(violations)
    out = pd.DataFrame(rows, columns=SCRAP_COLUMNS)
    out["purchased_utc"] = pd.to_datetime(out["purchased_utc"])
    out["disposed_utc"] = pd.to_datetime(out["disposed_utc"])
    for col in ("gross_weight_g", "fineness", "fine_troy_oz", "price_paid_usd"):
        out[col] = pd.to_numeric(out[col])
    for col in ("spot_usd_oz", "proceeds_usd"):
        out[col] = pd.to_numeric(out[col]).astype("float64")
    for col in ("lot_id", "metal", "disposition", "notes"):
        out[col] = out[col].astype("string")
    return out, warnings


def parse_coins_csv(
    df: pd.DataFrame, tz: ZoneInfo, now_utc: pd.Timestamp
) -> tuple[pd.DataFrame, list[str]]:
    """Validate raw coin-trade rows. Returns (typed frame, warnings); raises on violations."""
    violations: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    for pos, (_, r) in enumerate(df.iterrows()):
        line = pos + 2
        rows.append(
            {
                "trade_id": _required_str(r["trade_id"], "trade_id", line, violations),
                "traded_utc": _utc_timestamp(
                    r["traded_utc"], "traded_utc", line, tz, now_utc, violations, required=True
                ),
                "side": _choice(r["side"], "side", line, SIDES, violations),
                "product": _required_str(r["product"], "product", line, violations),
                "quantity": _positive_int(r["quantity"], "quantity", line, violations),
                "unit_price_usd": _positive_float(
                    r["unit_price_usd"], "unit_price_usd", line, violations, True
                ),
                "spot_usd_oz": _positive_float(
                    r["spot_usd_oz"], "spot_usd_oz", line, violations, False
                ),
                "metal": _choice(r["metal"], "metal", line, METALS, violations),
                "fine_troy_oz_per_unit": _positive_float(
                    r["fine_troy_oz_per_unit"], "fine_troy_oz_per_unit", line, violations, False
                ),
                "notes": _notes(r["notes"], line, violations),
            }
        )
    _reject_duplicate_keys(pd.Series([r["trade_id"] for r in rows]), "trade_id", violations)
    if violations:
        raise LedgerValidationError(violations)
    out = pd.DataFrame(rows, columns=COIN_COLUMNS)
    out["traded_utc"] = pd.to_datetime(out["traded_utc"])
    out["quantity"] = out["quantity"].astype("int64")
    out["unit_price_usd"] = pd.to_numeric(out["unit_price_usd"])
    for col in ("spot_usd_oz", "fine_troy_oz_per_unit"):
        out[col] = pd.to_numeric(out[col]).astype("float64")
    for col in ("trade_id", "side", "product", "metal", "notes"):
        out[col] = out[col].astype("string")
    return out, warnings


def parse_till_csv(
    df: pd.DataFrame, tz: ZoneInfo, now_utc: pd.Timestamp
) -> tuple[pd.DataFrame, list[str]]:
    """Validate raw till-count rows. Returns (typed frame, warnings); raises on violations."""
    violations: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    for pos, (_, r) in enumerate(df.iterrows()):
        line = pos + 2
        offers_made = _count(r["offers_made"], "offers_made", line, violations)
        offers_accepted = _count(r["offers_accepted"], "offers_accepted", line, violations)
        if (
            offers_made is not None
            and offers_accepted is not None
            and offers_accepted > offers_made
        ):
            violations.append(
                f"line {line}: offers_accepted ({offers_accepted}) exceeds "
                f"offers_made ({offers_made})"
            )
        rows.append(
            {
                "date_utc": _business_date(r["date_utc"], "date_utc", line, tz, violations),
                "walk_ins": _count(r["walk_ins"], "walk_ins", line, violations),
                "offers_made": offers_made,
                "offers_accepted": offers_accepted,
                "notes": _notes(r["notes"], line, violations),
            }
        )
    # Compare PARSED dates: '2026-01-05' and '01/05/2026' are the same PK.
    parsed_dates = [None if r["date_utc"] is None else str(r["date_utc"].date()) for r in rows]
    _reject_duplicate_keys(pd.Series(parsed_dates), "date_utc", violations)
    if violations:
        raise LedgerValidationError(violations)
    out = pd.DataFrame(rows, columns=TILL_COLUMNS)
    # Python date objects register as DATE in DuckDB (the column is a DATE).
    out["date_utc"] = pd.to_datetime(out["date_utc"]).dt.date
    for col in ("walk_ins", "offers_made", "offers_accepted"):
        out[col] = out[col].astype("Int64")
    out["notes"] = out["notes"].astype("string")
    return out, warnings


# ---------------------------------------------------------------------------
# Upserts (ON CONFLICT DO UPDATE: books get corrected on re-import)
# ---------------------------------------------------------------------------


def _upsert(df: pd.DataFrame, table: str, key: str, business_columns: list[str]) -> int:
    """Upsert a parsed + provenance-stamped frame. Returns rows written.

    Guards key uniqueness on the final typed frame: with an in-frame duplicate,
    ON CONFLICT would apply rows in order and silently keep only the last one.
    """
    if df.empty:
        return 0
    dup = df[key].duplicated(keep=False)
    if dup.any():
        lines = ", ".join(str(int(i) + 2) for i in df.index[dup])
        raise LedgerValidationError(
            [f"duplicate {key} in parsed frame (lines {lines}); refusing to upsert"]
        )
    cols = business_columns + PROVENANCE_COLUMNS
    col_list = ", ".join(cols)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != key)
    with connection() as conn:
        conn.register("incoming_ledger", df)
        conn.execute(
            f"""
            INSERT INTO {table} ({col_list})
            SELECT {col_list} FROM incoming_ledger
            ON CONFLICT ({key}) DO UPDATE SET {updates}
            """
        )
        conn.unregister("incoming_ledger")
    return int(len(df))


def upsert_scrap_lots(df: pd.DataFrame) -> int:
    """Idempotent upsert into amc_scrap_lots keyed on lot_id. Returns rows written."""
    return _upsert(df, "amc_scrap_lots", "lot_id", SCRAP_COLUMNS)


def upsert_coin_trades(df: pd.DataFrame) -> int:
    """Idempotent upsert into amc_coin_trades keyed on trade_id. Returns rows written."""
    return _upsert(df, "amc_coin_trades", "trade_id", COIN_COLUMNS)


def upsert_till_daily(df: pd.DataFrame) -> int:
    """Idempotent upsert into amc_till_daily keyed on date_utc. Returns rows written."""
    return _upsert(df, "amc_till_daily", "date_utc", TILL_COLUMNS)


class _TableSpec(NamedTuple):
    table: str
    columns: list[str]
    parse: Callable[[pd.DataFrame, ZoneInfo, pd.Timestamp], tuple[pd.DataFrame, list[str]]]
    upsert: Callable[[pd.DataFrame], int]
    date_col: str


TABLE_SPECS: dict[str, _TableSpec] = {
    "scrap": _TableSpec(
        "amc_scrap_lots", SCRAP_COLUMNS, parse_scrap_csv, upsert_scrap_lots, "purchased_utc"
    ),
    "coins": _TableSpec(
        "amc_coin_trades", COIN_COLUMNS, parse_coins_csv, upsert_coin_trades, "traded_utc"
    ),
    "till": _TableSpec(
        "amc_till_daily", TILL_COLUMNS, parse_till_csv, upsert_till_daily, "date_utc"
    ),
}


def refresh(path: Path | str, table: str, tz: str = DEFAULT_TZ) -> dict:
    """Validate + import one export file (all-or-nothing). Returns a summary dict."""
    if table not in TABLE_SPECS:
        raise ValueError(f"unknown table {table!r}; expected one of {sorted(TABLE_SPECS)}")
    spec = TABLE_SPECS[table]
    zone = ZoneInfo(tz)
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)

    raw = _read_export(path, spec.columns)
    parsed, warnings = spec.parse(raw, zone, now_utc)

    source_file = Path(path).name
    batch_id = str(uuid.uuid4())
    imported_at = datetime.now(UTC).replace(tzinfo=None)
    parsed = parsed.copy()
    parsed["source_file"] = source_file
    parsed["batch_id"] = batch_id
    parsed["imported_at"] = imported_at

    n = spec.upsert(parsed)
    dates = pd.to_datetime(parsed[spec.date_col])
    return {
        "source": SOURCE_TAG,
        "table": spec.table,
        "rows_written": n,
        "source_file": source_file,
        "batch_id": batch_id,
        "imported_at": imported_at.isoformat(),
        "date_range": [str(dates.min().date()), str(dates.max().date())],
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and import an AMC ledger export (all-or-nothing; data stays on this machine)."
        )
    )
    parser.add_argument("file", help="Path to the CSV export (format: configs/templates/).")
    parser.add_argument(
        "--table",
        required=True,
        choices=sorted(TABLE_SPECS),
        help="scrap -> amc_scrap_lots, coins -> amc_coin_trades, till -> amc_till_daily.",
    )
    parser.add_argument(
        "--tz",
        default=DEFAULT_TZ,
        help=f"IANA timezone of naive timestamps in the export (default {DEFAULT_TZ}).",
    )
    args = parser.parse_args()
    try:
        summary = refresh(args.file, table=args.table, tz=args.tz)
    except LedgerValidationError as exc:
        print(f"IMPORT REJECTED: {args.file}", file=sys.stderr)
        for v in exc.violations:
            print(f"  - {v}", file=sys.stderr)
        print(f"{len(exc.violations)} violation(s); nothing was written.", file=sys.stderr)
        raise SystemExit(1) from exc
    for w in summary["warnings"]:
        print(f"WARNING: {w}")
    print(f"Table:         {summary['table']}")
    print(f"Rows written:  {summary['rows_written']}")
    print(f"Date range:    {summary['date_range']}")
    print(f"Source file:   {summary['source_file']}")
    print(f"Batch id:      {summary['batch_id']}")


if __name__ == "__main__":
    main()
