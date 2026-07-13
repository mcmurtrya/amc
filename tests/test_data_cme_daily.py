"""Tests for the CME daily settlement volume/OI collector.

Fixtures in ``tests/fixtures/cme_daily/`` are real payload bytes captured from
live-archive snapshots of the CME web services during endpoint verification
(2026-07-12):

- ``settlements_si_final_20260225.json`` — SI Settlements, Final, 2026-02-25.
- ``volume_gc_prelim_20260213.json`` — GC Volume Details P, 2026-02-13.
- ``volume_si_final_20260224.json`` — SI Volume Details F, 2026-02-24.
- ``volume_empty_20200722.json`` — a real 'empty: true' Volume response.
- ``trade_dates_20250808.json`` — a real TradeDate listing.

The archive holds no same-day settlements+volume pair for one product, so the
merge tests re-date the real SI volume payload from 2026-02-24 to 2026-02-25
in test code (the doctoring is explicit below; the fixture files themselves
are untouched real bytes).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from metals.data import cme_daily
from metals.data.cme_daily import (
    AGG_MONTH,
    SOURCE_TAG,
    assemble_product_frame,
    gap_check,
    normalize_contract_month,
    parse_settlements,
    parse_trade_dates,
    parse_volume_details,
    payload_is_empty,
    upsert_cme_daily,
)

FIXTURES = Path(__file__).parent / "fixtures" / "cme_daily"
PULLED_AT = datetime(2026, 2, 25, 23, 59, tzinfo=UTC)


def _load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def _si_volume_20260225() -> dict[str, Any]:
    """Real SI volume payload, explicitly re-dated to pair with the real
    2026-02-25 settlements payload (see module docstring)."""
    payload = _load("volume_si_final_20260224.json")
    payload["tradeDate"] = "20260225"
    return payload


def _si_frame(preliminary: bool = False, pulled_at: datetime = PULLED_AT) -> pd.DataFrame:
    df, _ = assemble_product_frame(
        "SI",
        _si_volume_20260225(),
        _load("settlements_si_final_20260225.json"),
        preliminary=preliminary,
        pulled_at=pulled_at,
    )
    return df


# --------------------------------------------------------------------------
# Parsing real fixture bytes
# --------------------------------------------------------------------------


def test_parse_settlements_real_fixture():
    trade_date, preliminary, settle_map = parse_settlements(
        _load("settlements_si_final_20260225.json"), "SI"
    )
    assert trade_date == date(2026, 2, 25)
    assert preliminary is False
    # 34 rows in the payload: 33 contract months + one 'Total' row.
    assert len(settle_map) == 33
    assert settle_map["2026-03"] == pytest.approx(90.988)
    assert AGG_MONTH not in settle_map and "Total" not in settle_map


def test_parse_volume_details_real_prelim_fixture():
    trade_date, df = parse_volume_details(_load("volume_gc_prelim_20260213.json"), "GC")
    assert trade_date == date(2026, 2, 13)
    feb = df.set_index("contract_month").loc["2026-02"]
    assert feb["open_interest"] == 4643
    assert feb["oi_change"] == -328  # negative, comma-free
    agg = df.set_index("contract_month").loc[AGG_MONTH]
    assert agg["volume"] == 145433
    assert agg["open_interest"] == 412911
    assert agg["oi_change"] == 6503
    # One AGG row, no duplicate months.
    assert (df["contract_month"] == AGG_MONTH).sum() == 1
    assert df["contract_month"].is_unique


def test_parse_volume_details_real_final_fixture():
    trade_date, df = parse_volume_details(_load("volume_si_final_20260224.json"), "SI")
    assert trade_date == date(2026, 2, 24)
    mar = df.set_index("contract_month").loc["2026-03"]
    assert mar["volume"] == 67446
    assert mar["open_interest"] == 21882
    assert mar["oi_change"] == -15280
    agg = df.set_index("contract_month").loc[AGG_MONTH]
    assert agg["oi_change"] == -9166


def test_parse_trade_dates_real_fixture():
    dates = parse_trade_dates(_load("trade_dates_20250808.json"), "GC")
    assert dates[0] == (date(2025, 8, 8), "final")
    assert len(dates) == 5
    assert [d for d, _ in dates] == sorted((d for d, _ in dates), reverse=True)


# --------------------------------------------------------------------------
# Validation / failure modes
# --------------------------------------------------------------------------


def test_empty_volume_payload_raises():
    payload = _load("volume_empty_20200722.json")  # real 'empty: true' response
    assert payload_is_empty(payload, "SI") is True
    with pytest.raises(ValueError, match="empty"):
        parse_volume_details(payload, "SI")


def test_payload_without_empty_field_is_schema_drift():
    with pytest.raises(ValueError, match="schema drift"):
        payload_is_empty({"tradeDate": "20260213"}, "GC")


def test_settlements_missing_key_raises():
    payload = _load("settlements_si_final_20260225.json")
    del payload["reportType"]
    with pytest.raises(ValueError, match="reportType"):
        parse_settlements(payload, "SI")


def test_settlements_without_total_row_raises():
    payload = _load("settlements_si_final_20260225.json")
    payload["settlements"] = [r for r in payload["settlements"] if r["month"] != "Total"]
    with pytest.raises(ValueError, match="Total"):
        parse_settlements(payload, "SI")


def test_volume_garbage_number_raises():
    payload = _load("volume_gc_prelim_20260213.json")
    payload["monthData"][0]["atClose"] = "N/A"
    with pytest.raises(ValueError, match="atClose"):
        parse_volume_details(payload, "GC")


def test_unrecognized_report_type_raises():
    payload = _load("settlements_si_final_20260225.json")
    payload["reportType"] = "Draft"
    with pytest.raises(ValueError, match="reportType"):
        parse_settlements(payload, "SI")


def test_parse_trade_dates_rejects_empty_and_malformed():
    with pytest.raises(ValueError, match="TradeDate"):
        parse_trade_dates([], "GC")
    with pytest.raises(ValueError, match="TradeDate"):
        parse_trade_dates([["08/08/2025"]], "GC")


def test_normalize_contract_month():
    assert normalize_contract_month("FEB 26") == "2026-02"
    assert normalize_contract_month("FEB 2026") == "2026-02"
    assert normalize_contract_month("JLY 26") == "2026-07"  # bulletin-style July
    with pytest.raises(ValueError, match="month"):
        normalize_contract_month("FEB")
    with pytest.raises(ValueError, match="month"):
        normalize_contract_month("XXX 26")


def test_assemble_trade_date_mismatch_raises():
    """The real, un-doctored pair (volume 02-24 vs settlements 02-25) must fail."""
    with pytest.raises(ValueError, match="trade date"):
        assemble_product_frame(
            "SI",
            _load("volume_si_final_20260224.json"),
            _load("settlements_si_final_20260225.json"),
            preliminary=False,
            pulled_at=PULLED_AT,
        )


def test_assemble_unknown_product_raises():
    with pytest.raises(ValueError, match="product"):
        assemble_product_frame(
            "HG",
            _si_volume_20260225(),
            _load("settlements_si_final_20260225.json"),
            preliminary=False,
            pulled_at=PULLED_AT,
        )


# --------------------------------------------------------------------------
# Assembly: merge semantics, UTC handling, is_realtime / is_preliminary logic
# --------------------------------------------------------------------------


def test_assemble_merges_settle_onto_volume_rows():
    df, meta = assemble_product_frame(
        "SI",
        _si_volume_20260225(),
        _load("settlements_si_final_20260225.json"),
        preliminary=False,
        pulled_at=PULLED_AT,
    )
    assert list(df.columns) == cme_daily._COLUMNS
    mar = df.set_index("contract_month").loc["2026-03"]
    assert mar["settle"] == pytest.approx(90.988)
    assert mar["volume"] == 67446
    assert mar["open_interest"] == 21882
    assert mar["oi_change"] == -15280
    assert (df["product"] == "SI").all()
    assert (df["source"] == SOURCE_TAG).all()
    assert not df["is_preliminary"].any()
    assert meta["settle_report"] == "final"
    assert meta["settle_attached"] is True
    # Settlements list far months the volume report omits; counted, not lost silently.
    assert meta["settle_months_unmatched"] == 33 - (len(df) - 1)


def test_assemble_agg_row_has_null_settle():
    df = _si_frame()
    agg = df.set_index("contract_month").loc[AGG_MONTH]
    assert pd.isna(agg["settle"])
    assert agg["volume"] == 125447
    assert agg["open_interest"] == 125454
    assert agg["oi_change"] == -9166


def test_preliminary_settle_never_attached_to_final_rows():
    settlements = _load("settlements_si_final_20260225.json")
    settlements["reportType"] = "Preliminary"
    df, meta = assemble_product_frame(
        "SI", _si_volume_20260225(), settlements, preliminary=False, pulled_at=PULLED_AT
    )
    assert df["settle"].isna().all()
    assert meta["settle_attached"] is False
    # ... but a preliminary settle does attach to a preliminary row.
    df_p, meta_p = assemble_product_frame(
        "SI", _si_volume_20260225(), settlements, preliminary=True, pulled_at=PULLED_AT
    )
    assert meta_p["settle_attached"] is True
    assert df_p.set_index("contract_month").loc["2026-03", "settle"] == pytest.approx(90.988)
    assert df_p["is_preliminary"].all()


def test_pulled_at_stored_as_naive_utc():
    aware_eastern_style = datetime(2026, 2, 25, 18, 59, tzinfo=UTC)  # already UTC
    df, _ = assemble_product_frame(
        "SI",
        _si_volume_20260225(),
        _load("settlements_si_final_20260225.json"),
        preliminary=False,
        pulled_at=aware_eastern_style,
    )
    assert df["pulled_at"].dt.tz is None
    assert (df["pulled_at"] == pd.Timestamp(2026, 2, 25, 18, 59)).all()
    assert df["trade_date"].dt.tz is None


def test_naive_pulled_at_raises():
    with pytest.raises(ValueError, match="timezone-aware"):
        assemble_product_frame(
            "SI",
            _si_volume_20260225(),
            _load("settlements_si_final_20260225.json"),
            preliminary=False,
            pulled_at=datetime(2026, 2, 25, 23, 59),  # naive
        )


def test_is_realtime_boundary():
    # Lag of exactly REALTIME_MAX_LAG_DAYS days is still real-time...
    on_time = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)  # 2026-02-25 + 4 days
    df, _ = assemble_product_frame(
        "SI",
        _si_volume_20260225(),
        _load("settlements_si_final_20260225.json"),
        preliminary=False,
        pulled_at=on_time,
    )
    assert df["is_realtime"].all()
    # ...one more day is retro capture, permanently second-class.
    late = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)
    df_late, meta_late = assemble_product_frame(
        "SI",
        _si_volume_20260225(),
        _load("settlements_si_final_20260225.json"),
        preliminary=False,
        pulled_at=late,
    )
    assert not df_late["is_realtime"].any()
    assert meta_late["is_realtime"] is False


def test_pull_before_trade_date_raises():
    with pytest.raises(ValueError, match="after pull time"):
        assemble_product_frame(
            "SI",
            _si_volume_20260225(),
            _load("settlements_si_final_20260225.json"),
            preliminary=False,
            pulled_at=datetime(2026, 2, 24, 12, 0, tzinfo=UTC),
        )


# --------------------------------------------------------------------------
# DB round trip (temp DuckDB via METALS_DB_PATH)
# --------------------------------------------------------------------------


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("METALS_DB_PATH", str(tmp_path / "t.duckdb"))
    from metals.data.migrations import runner

    runner.apply_migrations(verbose=False)
    return tmp_path / "t.duckdb"


def _count(sql: str, params: list | None = None) -> int:
    from metals.data.db import connection

    with connection() as conn:
        return conn.execute(sql, params or []).fetchone()[0]


def test_upsert_idempotent(temp_db):
    df = _si_frame()
    assert upsert_cme_daily(df) == len(df)
    assert upsert_cme_daily(df) == len(df)  # second run updates, never duplicates
    assert _count("SELECT COUNT(*) FROM cme_daily") == len(df)

    from metals.data.db import connection

    with connection() as conn:
        settle, volume, is_prelim = conn.execute(
            "SELECT settle, volume, is_preliminary FROM cme_daily "
            "WHERE product = 'SI' AND contract_month = '2026-03'"
        ).fetchone()
        agg_settle = conn.execute(
            "SELECT settle FROM cme_daily WHERE product = 'SI' AND contract_month = ?",
            [AGG_MONTH],
        ).fetchone()[0]
    assert settle == pytest.approx(90.988)
    assert volume == 67446
    assert is_prelim is False
    assert agg_settle is None


def test_upsert_keeps_preliminary_and_final_rows_distinct(temp_db):
    n_final = upsert_cme_daily(_si_frame(preliminary=False))
    n_prelim = upsert_cme_daily(_si_frame(preliminary=True))
    assert _count("SELECT COUNT(*) FROM cme_daily") == n_final + n_prelim
    assert _count("SELECT COUNT(*) FROM cme_daily WHERE is_preliminary") == n_prelim


def test_upsert_never_demotes_realtime_flag(temp_db):
    realtime = _si_frame()
    assert bool(realtime["is_realtime"].all())
    upsert_cme_daily(realtime)

    retro = realtime.copy()
    retro["is_realtime"] = False
    retro["volume"] = retro["volume"] + 1
    upsert_cme_daily(retro)

    assert _count("SELECT COUNT(*) FROM cme_daily WHERE NOT is_realtime") == 0
    # ...while the figures themselves did update.
    assert (
        _count("SELECT volume FROM cme_daily WHERE product='SI' AND contract_month='2026-03'")
        == 67447
    )


def test_upsert_realtime_row_keeps_first_capture_pulled_at(temp_db):
    """A retro re-pull must not restamp a real-time row's pulled_at.

    Regression: pulled_at = EXCLUDED.pulled_at combined with the sticky
    is_realtime flag produced rows like trade_date=2026-07-10,
    pulled_at=2026-09-01, is_realtime=true — incoherent provenance.
    """
    realtime = _si_frame()
    assert bool(realtime["is_realtime"].all())
    upsert_cme_daily(realtime)

    much_later = _si_frame(pulled_at=datetime(2026, 9, 1, 12, 0, tzinfo=UTC))
    assert not much_later["is_realtime"].any()
    much_later["volume"] = much_later["volume"] + 1
    upsert_cme_daily(much_later)

    from metals.data.db import connection

    with connection() as conn:
        pulled_at, is_realtime, volume = conn.execute(
            "SELECT pulled_at, is_realtime, volume FROM cme_daily "
            "WHERE product = 'SI' AND contract_month = '2026-03'"
        ).fetchone()
    assert pulled_at == datetime(2026, 2, 25, 23, 59)  # original capture time survives
    assert is_realtime is True
    assert volume == 67447  # ...while the figures themselves did update


def test_upsert_retro_row_may_refresh_pulled_at(temp_db):
    retro = _si_frame(pulled_at=datetime(2026, 4, 1, 12, 0, tzinfo=UTC))
    assert not retro["is_realtime"].any()
    upsert_cme_daily(retro)

    later = _si_frame(pulled_at=datetime(2026, 9, 1, 12, 0, tzinfo=UTC))
    upsert_cme_daily(later)

    from metals.data.db import connection

    with connection() as conn:
        pulled_at, is_realtime = conn.execute(
            "SELECT pulled_at, is_realtime FROM cme_daily "
            "WHERE product = 'SI' AND contract_month = '2026-03'"
        ).fetchone()
    assert pulled_at == datetime(2026, 9, 1, 12, 0)  # retro provenance may refresh
    assert is_realtime is False


def test_gap_check_flags_missing_nyse_days(temp_db):
    from metals.data.db import connection

    # Window 2026-06-28 .. 2026-07-07: NYSE days are Jun 29, 30, Jul 1, 2, 6, 7
    # (Jul 3 is the observed Independence Day holiday, Jul 4 being a Saturday).
    covered = [date(2026, 6, 29), date(2026, 6, 30), date(2026, 7, 2), date(2026, 7, 6),
               date(2026, 7, 7)]  # fmt: skip
    with connection() as conn:
        for day in covered:
            conn.execute(
                "INSERT INTO cme_daily VALUES (?, 'GC', ?, NULL, 1, 1, 0, FALSE, ?, ?, TRUE)",
                [day, AGG_MONTH, SOURCE_TAG, datetime(2026, 7, 8)],
            )

    report = gap_check(n_days=10, products=["GC", "SI"], as_of=date(2026, 7, 8))
    assert report["expected_days"] == [
        "2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07",
    ]  # fmt: skip
    assert report["missing"]["GC"] == ["2026-07-01"]
    assert report["missing"]["SI"] == report["expected_days"]
