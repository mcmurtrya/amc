"""Tests for the Johnson Matthey PGM price collector (no network).

Fixtures under ``tests/fixtures/jm_pgm/`` are real bytes captured from
matthey.com's price portlet on 2026-07-12 (trimmed to representative rows),
plus one hand-broken variant for the schema-drift failure mode.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from metals.data.jm_pgm import (
    REALTIME_MAX_LAG_DAYS,
    SOURCE_TAG,
    _extract_csv_url,
    parse_daily_csv,
    stale_run_report,
    upsert_pgm_prices,
)
from metals.data.migrations import runner

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "jm_pgm"


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "t.duckdb"
    monkeypatch.setenv("METALS_DB_PATH", str(db_file))
    runner.apply_migrations(verbose=False)
    return db_file


def _all_rows(db_file) -> list[tuple]:
    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        return conn.execute(
            """
            SELECT price_date, metal, quote, price_usd_oz, source, pulled_at, is_realtime
            FROM pgm_prices ORDER BY price_date, metal, quote
            """
        ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------- parsing


def test_parse_real_hk_open_fixture():
    """Real 2026 HK-opening excerpt: 10 quote dates x 5 metals, long format."""
    df = parse_daily_csv(_fixture_bytes("daily_hk_open.csv"), quote="hk_open")
    assert list(df.columns) == ["price_date", "metal", "quote", "price_usd_oz"]
    assert len(df) == 10 * 5
    assert (df["quote"] == "hk_open").all()
    assert set(df["metal"]) == {"platinum", "palladium", "rhodium", "iridium", "ruthenium"}
    # Spot-check values against the raw file.
    pt = df[(df["price_date"] == date(2026, 7, 10)) & (df["metal"] == "platinum")]
    assert pt["price_usd_oz"].item() == 1635.0
    rh = df[(df["price_date"] == date(2026, 6, 26)) & (df["metal"] == "rhodium")]
    assert rh["price_usd_oz"].item() == 7750.0
    # 2026-07-01 was a Hong Kong holiday: absent from this region's CSV.
    assert date(2026, 7, 1) not in set(df["price_date"])


def test_parse_dates_are_day_first():
    """JM writes DD/MM/YYYY: 01/07/1992 is 1 July, not 7 January."""
    df = parse_daily_csv(_fixture_bytes("daily_london_1992.csv"), quote="london")
    assert df["price_date"].min() == date(1992, 7, 1)
    assert df["price_date"].max() == date(1992, 7, 14)
    ir = df[(df["price_date"] == date(1992, 7, 1)) & (df["metal"] == "iridium")]
    assert ir["price_usd_oz"].item() == 200.0


def test_parse_rejects_header_drift():
    with pytest.raises(ValueError, match="header drift"):
        parse_daily_csv(_fixture_bytes("daily_bad_header.csv"), quote="london")


def test_parse_rejects_empty_and_headerless_bodies():
    with pytest.raises(ValueError, match="header drift"):
        parse_daily_csv(b"", quote="london")
    with pytest.raises(ValueError, match="no data rows"):
        parse_daily_csv(
            b"Daily PGM prices for London\nDate,Platinum,Palladium,Rhodium,Iridium,Ruthenium\n",
            quote="london",
        )


def test_parse_rejects_malformed_rows():
    header = b"Daily PGM prices for London\nDate,Platinum,Palladium,Rhodium,Iridium,Ruthenium\n"
    with pytest.raises(ValueError, match="not DD/MM/YYYY"):
        parse_daily_csv(header + b"2026-07-10,1,2,3,4,5\n", quote="london")
    with pytest.raises(ValueError, match="not numeric"):
        parse_daily_csv(header + b"10/07/2026,1635.0,n/a,3,4,5\n", quote="london")
    with pytest.raises(ValueError, match="no data rows"):
        # a body that is ONLY a short row parses to nothing and fails loudly
        parse_daily_csv(header + b"10/07/2026,1635.0,1268.0\n", quote="london")
    with pytest.raises(ValueError, match="repeats"):
        parse_daily_csv(header + b"10/07/2026,1,2,3,4,5\n10/07/2026,1,2,3,4,5\n", quote="london")


def _london_body(n_rows: int) -> list[str]:
    """n well-formed body rows on consecutive January/February 2026 days."""
    days = pd.date_range("2026-01-01", periods=n_rows, freq="D")
    return [f"{d.strftime('%d/%m/%Y')},1635.0,1268.0,4800.0,4100.0,525.0" for d in days]


def test_parse_skips_sporadic_short_row_loudly(capsys):
    """Regression for the 13/06/2014 vendor defect: JM drops an EMPTY field
    entirely, shifting values left, so a short row is ambiguous and must be
    skipped (loudly) rather than repaired or allowed to shift columns."""
    header = "Daily PGM prices for London\nDate,Platinum,Palladium,Rhodium,Iridium,Ruthenium\n"
    rows = _london_body(120)
    rows[60] = "02/03/2026,843.0,1100.0,600.0,70.0"  # 5 fields: one dropped mid-row
    df = parse_daily_csv((header + "\n".join(rows) + "\n").encode(), quote="london")
    assert df.attrs["n_malformed"] == 1
    assert date(2026, 3, 2) not in set(df["price_date"])  # skipped, not mis-assigned
    assert len(df) == 119 * 5
    out = capsys.readouterr().out
    assert "WARNING" in out and "malformed row" in out and "843.0" in out


def test_parse_raises_when_malformed_fraction_exceeds_tolerance():
    """Many short rows = schema drift, not a sporadic defect: fail loudly."""
    header = "Daily PGM prices for London\nDate,Platinum,Palladium,Rhodium,Iridium,Ruthenium\n"
    rows = _london_body(3)
    rows[1] = "02/01/2026,843.0,1100.0,600.0,70.0"
    with pytest.raises(ValueError, match="schema drift"):
        parse_daily_csv((header + "\n".join(rows) + "\n").encode(), quote="london")


def test_parse_rejects_unknown_quote_code():
    with pytest.raises(ValueError, match="Unknown quote"):
        parse_daily_csv(_fixture_bytes("daily_hk_open.csv"), quote="tokyo")


def test_parse_empty_price_cell_becomes_null():
    header = b"Daily PGM prices for London\nDate,Platinum,Palladium,Rhodium,Iridium,Ruthenium\n"
    df = parse_daily_csv(header + b"10/07/2026,1635.0,,8200.0,7650.0,1525.0\n", quote="london")
    pd_row = df[df["metal"] == "palladium"]
    assert pd_row["price_usd_oz"].isna().all()


# ------------------------------------------------- portlet JSON responses


def test_extract_csv_url_from_real_response():
    url = _extract_csv_url(_fixture_bytes("post_success.json").decode())
    assert url.startswith("https://matthey.com/documents/")
    assert ".csv" in url


def test_extract_csv_url_raises_on_error_status():
    with pytest.raises(RuntimeError, match="refused"):
        _extract_csv_url(_fixture_bytes("post_error.json").decode())


def test_extract_csv_url_raises_on_non_json():
    with pytest.raises(ValueError, match="non-JSON"):
        _extract_csv_url("<html>maintenance page</html>")


# ------------------------------------------------------------------ upsert


def _frame(rows: list[tuple[date, float]]) -> pd.DataFrame:
    """One-metal london frame from (price_date, price) pairs."""
    return pd.DataFrame(
        {
            "price_date": [d for d, _ in rows],
            "metal": ["platinum"] * len(rows),
            "quote": ["london"] * len(rows),
            "price_usd_oz": [p for _, p in rows],
        }
    )


def test_upsert_idempotent_and_utc_stamped(tmp_db):
    df = parse_daily_csv(_fixture_bytes("daily_hk_open.csv"), quote="hk_open")
    before = datetime.now(UTC).replace(tzinfo=None)
    assert upsert_pgm_prices(df) == 50
    assert upsert_pgm_prices(df) == 50  # rerun: same keys, no duplicates
    rows = _all_rows(tmp_db)
    assert len(rows) == 50
    after = datetime.now(UTC).replace(tzinfo=None)
    for price_date, _metal, quote, price, source, pulled_at, is_realtime in rows:
        assert source == SOURCE_TAG
        assert quote == "hk_open"
        assert price is None or price > 0
        assert isinstance(price_date, date)
        # pulled_at is stored naive UTC and stamped at write time.
        assert pulled_at.tzinfo is None
        assert before <= pulled_at <= after
        # The flag derives from capture lag, never from a run mode.
        assert is_realtime is ((pulled_at.date() - price_date).days <= REALTIME_MAX_LAG_DAYS)


def test_upsert_gap_fill_stamps_flags_per_row_from_lag(tmp_db):
    """A window spanning old + recent dates gets mixed flags: only rows pulled
    within REALTIME_MAX_LAG_DAYS of their quote date count as real-time."""
    pulled = datetime(2026, 7, 12, 9, 0, 0)
    df = _frame(
        [
            (date(2026, 5, 1), 1600.0),  # 72-day lag: retro gap-fill
            (date(2026, 6, 27), 1610.0),  # 15-day lag: just outside the window
            (date(2026, 6, 28), 1620.0),  # 14-day lag: boundary, still real-time
            (date(2026, 7, 10), 1635.0),  # 2-day lag: real-time
        ]
    )
    assert upsert_pgm_prices(df, pulled_at=pulled) == 4
    flags = {r[0]: r[6] for r in _all_rows(tmp_db)}
    assert flags == {
        date(2026, 5, 1): False,
        date(2026, 6, 27): False,
        date(2026, 6, 28): True,
        date(2026, 7, 10): True,
    }


def test_upsert_realtime_never_demotes(tmp_db):
    df = _frame([(date(2026, 7, 10), 1635.0)])
    # Retro capture first (months after the quote date): second-class.
    upsert_pgm_prices(df, pulled_at=datetime(2026, 11, 2, 9, 0, 0))
    assert {r[6] for r in _all_rows(tmp_db)} == {False}
    # A capture within the lag window promotes to real-time...
    upsert_pgm_prices(df, pulled_at=datetime(2026, 7, 12, 9, 0, 0))
    assert {r[6] for r in _all_rows(tmp_db)} == {True}
    # ...and a later retro re-pull cannot demote it back.
    upsert_pgm_prices(df, pulled_at=datetime(2027, 1, 4, 9, 0, 0))
    assert {r[6] for r in _all_rows(tmp_db)} == {True}
    assert len(_all_rows(tmp_db)) == 1


def test_upsert_realtime_row_keeps_first_capture_pulled_at(tmp_db):
    """A retro re-pull may revise the price, but a real-time row's pulled_at
    stays frozen at first live capture — realtime=True with a months-later
    pulled_at would be incoherent provenance."""
    first_pull = datetime(2026, 7, 12, 9, 0, 0)
    upsert_pgm_prices(_frame([(date(2026, 7, 10), 1635.0)]), pulled_at=first_pull)

    revised = _frame([(date(2026, 7, 10), 1640.0)])
    upsert_pgm_prices(revised, pulled_at=datetime(2026, 12, 1, 9, 0, 0))

    ((_, _, _, price, _, pulled_at, is_realtime),) = _all_rows(tmp_db)
    assert price == 1640.0  # values still follow the latest pull
    assert is_realtime is True  # never demoted
    assert pulled_at == first_pull  # provenance frozen at first live capture


def test_upsert_retro_row_pulled_at_follows_latest_pull(tmp_db):
    df = _frame([(date(2026, 5, 1), 1600.0)])
    upsert_pgm_prices(df, pulled_at=datetime(2026, 11, 2, 9, 0, 0))
    later = datetime(2026, 12, 1, 9, 0, 0)
    upsert_pgm_prices(df, pulled_at=later)
    ((_, _, _, _, _, pulled_at, is_realtime),) = _all_rows(tmp_db)
    assert is_realtime is False
    assert pulled_at == later


def test_upsert_rejects_duplicate_keys(tmp_db):
    df = parse_daily_csv(_fixture_bytes("daily_hk_open.csv"), quote="hk_open")
    doubled = pd.concat([df, df], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        upsert_pgm_prices(doubled)


def test_upsert_empty_writes_nothing(tmp_db):
    empty = pd.DataFrame(columns=["price_date", "metal", "quote", "price_usd_oz"])
    assert upsert_pgm_prices(empty) == 0
    assert _all_rows(tmp_db) == []


# ------------------------------------------------------------- stale runs


def test_stale_run_report_flags_plateaus():
    dates = [date(2026, 7, d) for d in range(1, 11)]
    frame = pd.DataFrame(
        {
            "price_date": dates * 2,
            "metal": ["ruthenium"] * 10 + ["platinum"] * 10,
            "quote": ["london"] * 20,
            # Ru: flat all ten days; Pt: changes daily.
            "price_usd_oz": [1525.0] * 10 + [1600.0 + i for i in range(10)],
        }
    )
    report = stale_run_report(frame)
    ru = report[report["metal"] == "ruthenium"].iloc[0]
    assert ru["longest_run"] == 10 and ru["tail_run"] == 10
    pt = report[report["metal"] == "platinum"].iloc[0]
    assert pt["longest_run"] == 1 and pt["tail_run"] == 1
    # Sorted stalest-first, so ruthenium leads.
    assert report.iloc[0]["metal"] == "ruthenium"


def test_stale_run_report_empty_input():
    empty = pd.DataFrame(columns=["price_date", "metal", "quote", "price_usd_oz"])
    assert stale_run_report(empty).empty
