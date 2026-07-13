"""Tests for the AMC ledger importer (Phase 7.1, collector 1).

No network. Each test runs against a per-test temp DuckDB (METALS_DB_PATH)
with the real migrations applied, exercising the validating importer
end-to-end: parsing, UTC handling, atomic reject-all-or-import-all, and
upsert-on-reimport semantics.
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from metals.data.amc_ledger import (
    COIN_COLUMNS,
    SCRAP_COLUMNS,
    TILL_COLUMNS,
    LedgerValidationError,
    refresh,
    upsert_scrap_lots,
)
from metals.data.migrations import runner

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "configs" / "templates"

SCRAP_HEADER = ",".join(SCRAP_COLUMNS)
COIN_HEADER = ",".join(COIN_COLUMNS)
TILL_HEADER = ",".join(TILL_COLUMNS)

SCRAP_OK = (
    f"{SCRAP_HEADER}\n"
    "L-1001,2026-01-05 10:30,gold,31.10,0.583,0.583,1500,2650,,,,\n"
    "L-1002,2026-06-10 12:00,silver,311.00,0.925,9.249,270,30.15,"
    "2026-06-15 09:00,sold,280,walk-in batch\n"
)
COINS_OK = (
    f"{COIN_HEADER}\n"
    "T-1,2026-01-05 11:00,buy,american_gold_eagle_1oz,2,2705,2650,gold,1.0,\n"
    "T-2,2026-01-06 16:20,sell,american_silver_eagle_1oz,20,34.5,30.15,silver,1.0,walk-in\n"
)
TILL_OK = f"{TILL_HEADER}\n2026-01-05,14,9,6,\n2026-01-06,,7,7,quiet day\n"


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """Per-test temp DuckDB with all migrations (008 included) applied."""
    db_file = tmp_path / "ledger.duckdb"
    monkeypatch.setenv("METALS_DB_PATH", str(db_file))
    runner.apply_migrations(verbose=False)
    return db_file


def _write_csv(tmp_path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _rows(db_file, sql: str) -> list[tuple]:
    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _table_count(db_file, table: str) -> int:
    return _rows(db_file, f"SELECT COUNT(*) FROM {table}")[0][0]


# ---------------------------------------------------------------------------
# Happy paths + UTC handling
# ---------------------------------------------------------------------------


def test_scrap_happy_path_writes_rows_and_converts_tz(tmp_db, tmp_path):
    path = _write_csv(tmp_path, "scrap_week01.csv", SCRAP_OK)
    summary = refresh(path, table="scrap")
    assert summary["rows_written"] == 2
    assert summary["table"] == "amc_scrap_lots"
    assert summary["warnings"] == []

    got = dict(_rows(tmp_db, "SELECT lot_id, purchased_utc FROM amc_scrap_lots"))
    # Winter: America/Chicago is CST (UTC-6) -> 10:30 local = 16:30 UTC.
    assert got["L-1001"] == dt.datetime(2026, 1, 5, 16, 30)
    # Summer: CDT (UTC-5) -> 12:00 local = 17:00 UTC.
    assert got["L-1002"] == dt.datetime(2026, 6, 10, 17, 0)
    disposed = dict(_rows(tmp_db, "SELECT lot_id, disposed_utc FROM amc_scrap_lots"))
    assert disposed["L-1001"] is None
    assert disposed["L-1002"] == dt.datetime(2026, 6, 15, 14, 0)


def test_scrap_provenance_stamped(tmp_db, tmp_path):
    path = _write_csv(tmp_path, "scrap_week01.csv", SCRAP_OK)
    summary = refresh(path, table="scrap")
    rows = _rows(tmp_db, "SELECT source_file, batch_id, imported_at FROM amc_scrap_lots")
    assert all(r[0] == "scrap_week01.csv" for r in rows)
    assert all(r[1] == summary["batch_id"] for r in rows)
    uuid.UUID(summary["batch_id"])  # raises if not a valid uuid4-style id
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    assert all(abs((now - r[2]).total_seconds()) < 60 for r in rows)


def test_explicit_utc_offset_respected_over_tz_flag(tmp_db, tmp_path):
    csv = f"{SCRAP_HEADER}\nL-1,2026-01-05T10:30:00-05:00,gold,31.10,0.583,0.583,1500,,,,,\n"
    path = _write_csv(tmp_path, "scrap.csv", csv)
    refresh(path, table="scrap", tz="America/Chicago")
    (row,) = _rows(tmp_db, "SELECT purchased_utc FROM amc_scrap_lots")
    assert row[0] == dt.datetime(2026, 1, 5, 15, 30)


def test_tz_flag_utc_stores_wall_time_verbatim(tmp_db, tmp_path):
    csv = f"{SCRAP_HEADER}\nL-1,2026-01-05 10:30,gold,31.10,0.583,0.583,1500,,,,,\n"
    path = _write_csv(tmp_path, "scrap.csv", csv)
    refresh(path, table="scrap", tz="UTC")
    (row,) = _rows(tmp_db, "SELECT purchased_utc FROM amc_scrap_lots")
    assert row[0] == dt.datetime(2026, 1, 5, 10, 30)


def test_coins_happy_path(tmp_db, tmp_path):
    path = _write_csv(tmp_path, "coins_week01.csv", COINS_OK)
    summary = refresh(path, table="coins")
    assert summary["rows_written"] == 2
    got = dict(_rows(tmp_db, "SELECT trade_id, traded_utc FROM amc_coin_trades"))
    assert got["T-1"] == dt.datetime(2026, 1, 5, 17, 0)
    (row,) = _rows(tmp_db, "SELECT side, quantity, metal FROM amc_coin_trades WHERE trade_id='T-2'")
    assert row == ("sell", 20, "silver")


def test_till_happy_path_with_null_counts(tmp_db, tmp_path):
    path = _write_csv(tmp_path, "till_week01.csv", TILL_OK)
    summary = refresh(path, table="till")
    assert summary["rows_written"] == 2
    got = {
        r[0]: r[1:] for r in _rows(tmp_db, "SELECT date_utc, walk_ins, notes FROM amc_till_daily")
    }
    assert got[dt.date(2026, 1, 5)] == (14, None)
    assert got[dt.date(2026, 1, 6)] == (None, "quiet day")


# ---------------------------------------------------------------------------
# Upsert semantics
# ---------------------------------------------------------------------------


def test_reimport_same_file_is_idempotent(tmp_db, tmp_path):
    path = _write_csv(tmp_path, "scrap.csv", SCRAP_OK)
    refresh(path, table="scrap")
    refresh(path, table="scrap")
    assert _table_count(tmp_db, "amc_scrap_lots") == 2


def test_reimport_upserts_corrections_and_dispositions(tmp_db, tmp_path):
    v1 = f"{SCRAP_HEADER}\nL-1,2026-01-05 10:30,gold,31.10,0.583,0.583,1500,2650,,,,\n"
    v2 = (
        f"{SCRAP_HEADER}\n"
        "L-1,2026-01-05 10:30,gold,31.10,0.583,0.583,1500,2650,"
        "2026-01-12 14:00,refined,1540,assay confirmed\n"
    )
    refresh(_write_csv(tmp_path, "week01.csv", v1), table="scrap")
    summary2 = refresh(_write_csv(tmp_path, "week02.csv", v2), table="scrap")
    assert _table_count(tmp_db, "amc_scrap_lots") == 1
    (row,) = _rows(
        tmp_db,
        "SELECT disposition, proceeds_usd, source_file, batch_id FROM amc_scrap_lots",
    )
    assert row[0] == "refined"
    assert row[1] == 1540.0
    assert row[2] == "week02.csv"
    assert row[3] == summary2["batch_id"]


def test_upsert_guard_refuses_duplicate_keys_in_typed_frame(tmp_db):
    """Defense in depth: even a typed frame carrying a duplicated key must
    never reach ON CONFLICT (which would silently keep only the last row)."""
    df = pd.DataFrame({"lot_id": ["L-1", "L-1"]})
    with pytest.raises(LedgerValidationError, match="duplicate lot_id"):
        upsert_scrap_lots(df)
    assert _table_count(tmp_db, "amc_scrap_lots") == 0


# ---------------------------------------------------------------------------
# Validation failure modes (reject-all-or-import-all)
# ---------------------------------------------------------------------------


def test_missing_column_rejected(tmp_db, tmp_path):
    header = ",".join(c for c in SCRAP_COLUMNS if c != "fineness")
    csv = f"{header}\nL-1,2026-01-05 10:30,gold,31.10,0.583,1500,,,,,\n"
    with pytest.raises(LedgerValidationError, match="missing required column.*fineness"):
        refresh(_write_csv(tmp_path, "scrap.csv", csv), table="scrap")
    assert _table_count(tmp_db, "amc_scrap_lots") == 0


def test_unexpected_column_rejected(tmp_db, tmp_path):
    csv = f"{SCRAP_HEADER},extra\nL-1,2026-01-05 10:30,gold,31.10,0.583,0.583,1500,,,,,,x\n"
    with pytest.raises(LedgerValidationError, match="unexpected column.*extra"):
        refresh(_write_csv(tmp_path, "scrap.csv", csv), table="scrap")


def test_empty_file_rejected(tmp_db, tmp_path):
    with pytest.raises(LedgerValidationError, match="file is empty"):
        refresh(_write_csv(tmp_path, "scrap.csv", ""), table="scrap")


def test_header_only_file_rejected(tmp_db, tmp_path):
    with pytest.raises(LedgerValidationError, match="no data rows"):
        refresh(_write_csv(tmp_path, "scrap.csv", f"{SCRAP_HEADER}\n"), table="scrap")


def test_every_violation_reported_with_line_numbers(tmp_db, tmp_path):
    csv = (
        f"{SCRAP_HEADER}\n"
        "L-1,2026-01-05 10:30,copper,31.10,0.583,0.583,1500,,,,,\n"
        "L-2,2026-01-05 10:30,gold,31.10,1.5,0.583,1500,,,,,\n"
        "L-3,2035-01-01 00:00,gold,31.10,0.583,0.583,not-a-price,,,,,\n"
    )
    with pytest.raises(LedgerValidationError) as ei:
        refresh(_write_csv(tmp_path, "scrap.csv", csv), table="scrap")
    text = str(ei.value)
    assert "line 2" in text and "copper" in text
    assert "line 3" in text and "fineness must be in (0, 1]" in text
    assert "line 4" in text and "in the future" in text
    assert "line 4" in text and "not a number" in text
    assert len(ei.value.violations) == 4
    assert _table_count(tmp_db, "amc_scrap_lots") == 0


def test_atomicity_one_bad_row_writes_nothing(tmp_db, tmp_path):
    csv = SCRAP_OK + "L-9999,2026-01-07 09:00,gold,-5,0.583,0.583,1500,,,,,\n"
    with pytest.raises(LedgerValidationError, match="gross_weight_g must be > 0"):
        refresh(_write_csv(tmp_path, "scrap.csv", csv), table="scrap")
    assert _table_count(tmp_db, "amc_scrap_lots") == 0


def test_duplicate_lot_id_rejected(tmp_db, tmp_path):
    csv = (
        f"{SCRAP_HEADER}\n"
        "L-1,2026-01-05 10:30,gold,31.10,0.583,0.583,1500,,,,,\n"
        "L-1,2026-01-06 10:30,gold,31.10,0.583,0.583,1450,,,,,\n"
    )
    with pytest.raises(LedgerValidationError, match="duplicate lot_id 'L-1'.*lines 2, 3"):
        refresh(_write_csv(tmp_path, "scrap.csv", csv), table="scrap")


def test_duplicate_trade_id_rejected(tmp_db, tmp_path):
    csv = (
        f"{COIN_HEADER}\n"
        "T-1,2026-01-05 11:00,buy,generic_round_1oz,1,2700,,gold,1.0,\n"
        "T-1,2026-01-05 12:00,sell,generic_round_1oz,1,2750,,gold,1.0,\n"
    )
    with pytest.raises(LedgerValidationError, match="duplicate trade_id"):
        refresh(_write_csv(tmp_path, "coins.csv", csv), table="coins")


def test_bad_side_and_nonpositive_values_rejected(tmp_db, tmp_path):
    csv = f"{COIN_HEADER}\nT-1,2026-01-05 11:00,hold,generic_round_1oz,0,-3,,gold,1.0,\n"
    with pytest.raises(LedgerValidationError) as ei:
        refresh(_write_csv(tmp_path, "coins.csv", csv), table="coins")
    text = str(ei.value)
    assert "side 'hold' not one of ['buy', 'sell']" in text
    assert "quantity must be > 0" in text
    assert "unit_price_usd must be > 0" in text


def test_noninteger_quantity_rejected(tmp_db, tmp_path):
    csv = f"{COIN_HEADER}\nT-1,2026-01-05 11:00,buy,generic_round_1oz,2.5,2700,,gold,1.0,\n"
    with pytest.raises(LedgerValidationError, match="quantity '2.5' is not a whole number"):
        refresh(_write_csv(tmp_path, "coins.csv", csv), table="coins")


def test_infinite_numeric_values_rejected(tmp_db, tmp_path):
    """Regression: float() accepts 'inf'/'Infinity' (and 1e999 overflows to inf),
    all > 0 — without an isfinite check they'd import as DOUBLE Infinity."""
    csv = f"{SCRAP_HEADER}\nL-1,2026-01-05 10:30,gold,inf,0.583,0.583,Infinity,1e999,,,,\n"
    with pytest.raises(LedgerValidationError) as ei:
        refresh(_write_csv(tmp_path, "scrap.csv", csv), table="scrap")
    text = str(ei.value)
    assert "gross_weight_g must be finite, got inf" in text
    assert "price_paid_usd must be finite, got Infinity" in text
    assert "spot_usd_oz must be finite, got 1e999" in text
    assert len(ei.value.violations) == 3
    assert _table_count(tmp_db, "amc_scrap_lots") == 0


def test_coins_infinite_price_and_quantity_rejected(tmp_db, tmp_path):
    csv = f"{COIN_HEADER}\nT-1,2026-01-05 11:00,buy,generic_round_1oz,inf,Infinity,,gold,1.0,\n"
    with pytest.raises(LedgerValidationError) as ei:
        refresh(_write_csv(tmp_path, "coins.csv", csv), table="coins")
    text = str(ei.value)
    assert "quantity 'inf' is not a whole number" in text
    assert "unit_price_usd must be finite, got Infinity" in text
    assert _table_count(tmp_db, "amc_coin_trades") == 0


def test_unparseable_timestamp_rejected(tmp_db, tmp_path):
    csv = f"{SCRAP_HEADER}\nL-1,not-a-date,gold,31.10,0.583,0.583,1500,,,,,\n"
    with pytest.raises(LedgerValidationError, match="purchased_utc 'not-a-date'"):
        refresh(_write_csv(tmp_path, "scrap.csv", csv), table="scrap")


def test_invalid_disposition_rejected(tmp_db, tmp_path):
    csv = (
        f"{SCRAP_HEADER}\n"
        "L-1,2026-01-05 10:30,gold,31.10,0.583,0.583,1500,,2026-01-06 10:00,lost,100,\n"
    )
    with pytest.raises(LedgerValidationError, match="disposition 'lost' not one of"):
        refresh(_write_csv(tmp_path, "scrap.csv", csv), table="scrap")


def test_disposed_before_purchased_rejected(tmp_db, tmp_path):
    csv = (
        f"{SCRAP_HEADER}\n"
        "L-1,2026-01-05 10:30,gold,31.10,0.583,0.583,1500,,2026-01-04 10:00,sold,100,\n"
    )
    with pytest.raises(LedgerValidationError, match="disposed_utc precedes purchased_utc"):
        refresh(_write_csv(tmp_path, "scrap.csv", csv), table="scrap")


def test_fineness_consistency_error_over_5pct(tmp_db, tmp_path):
    # 100 g at 0.9 -> 2.8935 fine oz; 3.1 claimed is ~7.1% off -> reject.
    csv = f"{SCRAP_HEADER}\nL-1,2026-01-05 10:30,gold,100,0.9,3.1,1500,,,,,\n"
    with pytest.raises(LedgerValidationError, match="fine_troy_oz 3.1.*limit 5%"):
        refresh(_write_csv(tmp_path, "scrap.csv", csv), table="scrap")
    assert _table_count(tmp_db, "amc_scrap_lots") == 0


def test_fineness_consistency_warns_between_1_and_5pct(tmp_db, tmp_path):
    # 100 g at 0.9 -> 2.8935 fine oz; 2.95 claimed is ~2% off -> import + warn.
    csv = f"{SCRAP_HEADER}\nL-1,2026-01-05 10:30,gold,100,0.9,2.95,1500,,,,,\n"
    summary = refresh(_write_csv(tmp_path, "scrap.csv", csv), table="scrap")
    assert summary["rows_written"] == 1
    assert len(summary["warnings"]) == 1
    assert "check the assay entry" in summary["warnings"][0]
    assert _table_count(tmp_db, "amc_scrap_lots") == 1


def test_till_future_date_rejected(tmp_db, tmp_path):
    csv = f"{TILL_HEADER}\n2035-01-01,3,2,1,\n"
    with pytest.raises(LedgerValidationError, match="date_utc '2035-01-01' is in the future"):
        refresh(_write_csv(tmp_path, "till.csv", csv), table="till")


def test_till_duplicate_date_rejected(tmp_db, tmp_path):
    csv = f"{TILL_HEADER}\n2026-01-05,3,2,1,\n2026-01-05,4,2,2,\n"
    with pytest.raises(LedgerValidationError, match="duplicate date_utc"):
        refresh(_write_csv(tmp_path, "till.csv", csv), table="till")


def test_till_same_date_two_spellings_rejected(tmp_db, tmp_path):
    """Regression: '2026-01-05' and '01/05/2026' are distinct raw strings but
    parse to the same date_utc PK — the second row must not silently clobber
    the first via ON CONFLICT; the whole file is rejected, nothing written."""
    csv = f"{TILL_HEADER}\n2026-01-05,3,2,1,\n01/05/2026,4,2,2,\n"
    with pytest.raises(LedgerValidationError) as ei:
        refresh(_write_csv(tmp_path, "till.csv", csv), table="till")
    assert "duplicate date_utc '2026-01-05' appears on lines 2, 3" in str(ei.value)
    assert _table_count(tmp_db, "amc_till_daily") == 0


def test_till_accepted_exceeding_made_rejected(tmp_db, tmp_path):
    csv = f"{TILL_HEADER}\n2026-01-05,3,2,5,\n"
    with pytest.raises(LedgerValidationError, match=r"offers_accepted \(5\) exceeds"):
        refresh(_write_csv(tmp_path, "till.csv", csv), table="till")


def test_unknown_table_raises_value_error(tmp_db, tmp_path):
    path = _write_csv(tmp_path, "scrap.csv", SCRAP_OK)
    with pytest.raises(ValueError, match="unknown table 'bogus'"):
        refresh(path, table="bogus")


# ---------------------------------------------------------------------------
# Committed templates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "columns"),
    [
        ("amc_scrap_lots.csv", SCRAP_COLUMNS),
        ("amc_coin_trades.csv", COIN_COLUMNS),
        ("amc_till_daily.csv", TILL_COLUMNS),
    ],
)
def test_template_headers_match_business_columns(name, columns):
    header = (TEMPLATES_DIR / name).read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == columns


@pytest.mark.parametrize(
    ("name", "table", "db_table"),
    [
        ("amc_scrap_lots.csv", "scrap", "amc_scrap_lots"),
        ("amc_coin_trades.csv", "coins", "amc_coin_trades"),
        ("amc_till_daily.csv", "till", "amc_till_daily"),
    ],
)
def test_template_example_rows_are_rejected(tmp_db, name, table, db_table):
    """The fake EXAMPLE rows are the templates' ONLY violations — proving the
    templates are otherwise valid exports — and they can never be imported."""
    with pytest.raises(LedgerValidationError) as ei:
        refresh(TEMPLATES_DIR / name, table=table)
    assert len(ei.value.violations) == 2
    assert all("EXAMPLE" in v for v in ei.value.violations)
    assert _table_count(tmp_db, db_table) == 0
