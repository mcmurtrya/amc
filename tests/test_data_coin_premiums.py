"""Tests for the retail coin-premium panel collector (no network; fixture-driven).

Fixtures under tests/fixtures/coin_premiums/ are trimmed captures of the real
dealer pages (Internet Archive snapshots of the live URLs, JSON-LD blocks kept
verbatim) — see the HTML comments inside each fixture for provenance.
"""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml
from bs4 import BeautifulSoup

from metals.data.coin_premiums import (
    SOURCE_TAG,
    ParsedQuote,
    SchemaDriftError,
    _product_documents,
    collect,
    compute_premiums,
    load_basket,
    parse_product_page,
    save_raw_page,
    upsert_coin_premiums,
)

FIXTURES = Path(__file__).parent / "fixtures" / "coin_premiums"

APMEX_HINTS = {"ask_source": "json_ld", "ask_css": None, "bid_source": None}
JMB_HINTS = {"ask_source": "json_ld", "ask_css": None, "bid_source": "buyback_payload"}


def _fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Basket config
# ---------------------------------------------------------------------------


def test_load_basket_real_config_validates():
    """The committed configs/premium_basket.yaml must load and cover the basket."""
    cfg = load_basket()
    assert set(cfg["dealers"]) == {"apmex", "jmbullion"}
    products = {p["product_id"]: p for p in cfg["products"]}
    assert set(products) == {
        "gold_eagle_1oz_random",
        "silver_eagle_1oz_random",
        "gold_maple_1oz_random",
        "junk_silver_90pct_100fv",
        "silver_round_1oz_generic",
        "gold_bar_1oz_generic",
    }
    # Junk silver carries the explicit per-face-value melt basis.
    junk = products["junk_silver_90pct_100fv"]
    assert junk["fine_troy_oz"] == pytest.approx(71.5)
    assert junk["face_value_usd"] * junk["fine_troy_oz_per_face_dollar"] == pytest.approx(71.5)
    # Every product has both dealer URLs, https, on the expected hosts
    # (jmbullion via the AMP mirror — the www storefront blocks non-browser TLS).
    for product in cfg["products"]:
        assert set(product["urls"]) == {"apmex", "jmbullion"}
        assert product["urls"]["apmex"].startswith("https://www.apmex.com/product/")
        assert product["urls"]["jmbullion"].startswith("https://amp.jmbullion.com/")
    # Neither dealer is robots-disallowed as of the 2026-07-12 check.
    assert not cfg["dealers"]["apmex"]["disallowed"]
    assert not cfg["dealers"]["jmbullion"]["disallowed"]


def test_load_basket_rejects_inconsistent_face_value_basis(tmp_path):
    cfg = {
        "dealers": {"apmex": {}},
        "products": [
            {
                "product_id": "junk",
                "metal": "silver",
                "face_value_usd": 100,
                "fine_troy_oz_per_face_dollar": 0.715,
                "fine_troy_oz": 70.0,  # != 100 * 0.715
                "urls": {"apmex": "https://www.apmex.com/product/27/x"},
            }
        ],
    }
    path = tmp_path / "basket.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="fine_troy_oz"):
        load_basket(path)


def test_load_basket_rejects_unknown_dealer(tmp_path):
    cfg = {
        "dealers": {"apmex": {}},
        "products": [
            {
                "product_id": "age",
                "metal": "gold",
                "fine_troy_oz": 1.0,
                "urls": {"shadydealer": "https://example.com/age"},
            }
        ],
    }
    path = tmp_path / "basket.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="shadydealer"):
        load_basket(path)


# ---------------------------------------------------------------------------
# Page parsing (real fixture bytes)
# ---------------------------------------------------------------------------


def test_parse_apmex_fixture_json_ld_ask_no_bid():
    """APMEX: ask from the JSON-LD Product offer; no buyback published -> bid None."""
    quote = parse_product_page(
        _fixture_text("apmex_gold_eagle.html"),
        dealer="apmex",
        product_id="gold_eagle_1oz_random",
        parse_hints=APMEX_HINTS,
    )
    assert quote.ask_usd == pytest.approx(4893.64)
    assert quote.bid_usd is None


def test_parse_jmbullion_fixture_ask_and_buyback_bid():
    """JM Bullion: ask from JSON-LD; buyback bid from the Next.js flight payload."""
    quote = parse_product_page(
        _fixture_text("jmbullion_gold_eagle.html"),
        dealer="jmbullion",
        product_id="gold_eagle_1oz_random",
        parse_hints=JMB_HINTS,
    )
    assert quote.ask_usd == pytest.approx(5695.09)
    assert quote.bid_usd == pytest.approx(5110.98)
    # Sanity: dealer bid sits below dealer ask.
    assert quote.bid_usd < quote.ask_usd


def test_parse_jmbullion_amp_fixture_ask_only():
    """JM Bullion AMP mirror (the production path): JSON-LD ask; buyback is
    client-rendered on AMP, so with bid_source null the bid stays None."""
    quote = parse_product_page(
        _fixture_text("jmbullion_amp_gold_eagle.html"),
        dealer="jmbullion",
        product_id="gold_eagle_1oz_random",
        parse_hints={"ask_source": "json_ld", "ask_css": None, "bid_source": None},
    )
    assert quote.ask_usd == pytest.approx(4514.52)
    assert quote.bid_usd is None


def test_parse_apmex_json_ld_with_raw_control_characters():
    """Live APMEX Product JSON-LD can carry raw control chars inside strings
    (strict JSON parsers reject it) — regression for strict=False parsing."""
    quote = parse_product_page(
        _fixture_text("apmex_junk_silver.html"),
        dealer="apmex",
        product_id="junk_silver_90pct_100fv",
        parse_hints=APMEX_HINTS,
    )
    assert quote.ask_usd == pytest.approx(4434.49)
    assert quote.bid_usd is None


def test_parse_page_without_price_raises_named_drift_error():
    """A page that fetches but yields no price must raise, naming dealer/product."""
    with pytest.raises(SchemaDriftError, match=r"apmex/gold_eagle_1oz_random"):
        parse_product_page(
            _fixture_text("apmex_gold_eagle_no_price.html"),
            dealer="apmex",
            product_id="gold_eagle_1oz_random",
            parse_hints=APMEX_HINTS,
        )


def test_parse_missing_expected_buyback_raises():
    """If the config expects a buyback bid and the payload lost it, that is drift."""
    with pytest.raises(SchemaDriftError, match=r"jmbullion/gold_eagle_1oz_random"):
        parse_product_page(
            _fixture_text("jmbullion_gold_eagle_no_buyback.html"),
            dealer="jmbullion",
            product_id="gold_eagle_1oz_random",
            parse_hints=JMB_HINTS,
        )


def test_parse_unknown_bid_source_rejected():
    with pytest.raises(ValueError, match="bid_source"):
        parse_product_page(
            _fixture_text("apmex_gold_eagle.html"),
            dealer="apmex",
            product_id="gold_eagle_1oz_random",
            parse_hints={"bid_source": "telepathy"},
        )


# Synthetic two-Product page: the main Product followed by a related-item Product
# (as cross-sell carousels emit) in one top-level JSON-LD array. A LIFO traversal
# used to surface the *related* product's price; the parser must now refuse to
# pick between disagreeing Product prices.
def _two_product_page(main_price: str, related_price: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<title>1 oz American Gold Eagle Coin BU (Random Year) | APMEX</title>
<script type="application/ld+json">
[
  {{
    "@context": "http://schema.org/",
    "@type": "Product",
    "name": "1 oz American Gold Eagle Coin BU (Random Year)",
    "sku": "1",
    "offers": {{"@type": "Offer", "priceCurrency": "USD", "price": "{main_price}",
                "availability": "http://schema.org/InStock"}}
  }},
  {{
    "@context": "http://schema.org/",
    "@type": "Product",
    "name": "1 oz Gold Bar - Secondary Market",
    "sku": "22222",
    "offers": {{"@type": "Offer", "priceCurrency": "USD", "price": "{related_price}",
                "availability": "http://schema.org/InStock"}}
  }}
]
</script>
</head><body></body></html>"""


def test_parse_two_product_page_with_disagreeing_prices_raises():
    """Regression: a related-item Product's price must never silently win the row."""
    with pytest.raises(SchemaDriftError, match=r"apmex/gold_eagle_1oz_random.*disagree"):
        parse_product_page(
            _two_product_page("4893.64", "4787.25"),
            dealer="apmex",
            product_id="gold_eagle_1oz_random",
            parse_hints=APMEX_HINTS,
        )


def test_parse_two_product_page_with_agreeing_prices_still_parses():
    """Duplicated Product markup at the same price is not drift — only disagreement is."""
    quote = parse_product_page(
        _two_product_page("4893.64", "4893.64"),
        dealer="apmex",
        product_id="gold_eagle_1oz_random",
        parse_hints=APMEX_HINTS,
    )
    assert quote.ask_usd == pytest.approx(4893.64)
    assert quote.bid_usd is None


def test_product_documents_preserve_page_order():
    """Regression: LIFO traversal reversed top-level arrays and @graph blocks."""
    html = (
        '<script type="application/ld+json">'
        '[{"@type": "Product", "sku": "main"}, {"@type": "Product", "sku": "related"}]'
        "</script>"
        '<script type="application/ld+json">'
        '{"@graph": [{"@type": "Product", "sku": "g1"}, {"@type": "Product", "sku": "g2"}]}'
        "</script>"
    )
    docs = _product_documents(BeautifulSoup(html, "lxml"))
    assert [doc["sku"] for doc in docs] == ["main", "related", "g1", "g2"]


# ---------------------------------------------------------------------------
# Premium arithmetic
# ---------------------------------------------------------------------------


def test_compute_premiums_per_unit():
    ask_pct, bid_pct = compute_premiums(
        ask_usd=4893.64, bid_usd=4700.0, spot_usd_oz=4660.0, fine_troy_oz=1.0
    )
    assert ask_pct == pytest.approx((4893.64 / 4660.0 - 1) * 100)
    assert bid_pct == pytest.approx((4700.0 / 4660.0 - 1) * 100)


def test_compute_premiums_junk_silver_face_value_basis():
    """$100 face at 0.715 ozt/$1 -> melt = spot * 71.5, premium against the bag."""
    ask_pct, bid_pct = compute_premiums(
        ask_usd=2359.5, bid_usd=None, spot_usd_oz=30.0, fine_troy_oz=71.5
    )
    assert ask_pct == pytest.approx(10.0)  # melt 2145.0, ask 10% over
    assert bid_pct is None


def test_compute_premiums_rejects_bad_melt_basis():
    with pytest.raises(ValueError):
        compute_premiums(ask_usd=100.0, bid_usd=None, spot_usd_oz=0.0, fine_troy_oz=1.0)
    with pytest.raises(ValueError):
        compute_premiums(ask_usd=100.0, bid_usd=None, spot_usd_oz=30.0, fine_troy_oz=-1.0)


# ---------------------------------------------------------------------------
# collect() — offline, fixture bytes through the full pipeline
# ---------------------------------------------------------------------------


def _test_basket(tmp_path: Path) -> Path:
    cfg = {
        "spot_tickers": {"gold": "GC=F"},
        "dealers": {
            "apmex": {"disallowed": False, "parse": dict(APMEX_HINTS)},
            "jmbullion": {"disallowed": False, "parse": dict(JMB_HINTS)},
        },
        "products": [
            {
                "product_id": "gold_eagle_1oz_random",
                "metal": "gold",
                "fine_troy_oz": 1.0,
                "urls": {
                    "apmex": "https://www.apmex.com/product/1/age",
                    "jmbullion": "https://www.jmbullion.com/age/",
                },
            }
        ],
    }
    path = tmp_path / "basket.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _fixture_fetch(url: str) -> bytes:
    name = "apmex_gold_eagle.html" if "apmex" in url else "jmbullion_gold_eagle.html"
    return (FIXTURES / name).read_bytes()


def _collect_frame(tmp_path: Path, is_realtime: bool = True) -> pd.DataFrame:
    return collect(
        basket_path=_test_basket(tmp_path),
        raw_dir=tmp_path / "raw",
        delay_s=0.0,
        fetch=_fixture_fetch,
        spot={"gold": 4660.0},
        is_realtime=is_realtime,
    )


def test_collect_rows_spot_and_premiums(tmp_path):
    df = _collect_frame(tmp_path)
    assert len(df) == 2
    by_dealer = df.set_index("dealer")
    assert by_dealer.loc["apmex", "ask_usd"] == pytest.approx(4893.64)
    assert pd.isna(by_dealer.loc["apmex", "bid_usd"])  # no APMEX buyback published
    assert by_dealer.loc["jmbullion", "ask_usd"] == pytest.approx(5695.09)
    assert by_dealer.loc["jmbullion", "bid_usd"] == pytest.approx(5110.98)
    assert (df["spot_usd_oz"] == 4660.0).all()
    assert by_dealer.loc["apmex", "ask_premium_pct"] == pytest.approx((4893.64 / 4660.0 - 1) * 100)
    assert by_dealer.loc["jmbullion", "bid_premium_pct"] == pytest.approx(
        (5110.98 / 4660.0 - 1) * 100
    )
    assert (df["source"] == SOURCE_TAG).all()


def test_collect_pulled_at_is_naive_utc(tmp_path):
    before = datetime.now(UTC).replace(tzinfo=None)
    df = _collect_frame(tmp_path)
    after = datetime.now(UTC).replace(tzinfo=None)
    ts = df["pulled_at"].iloc[0]
    assert ts.tzinfo is None  # stored naive-UTC, like every other table
    assert before <= ts.to_pydatetime() <= after
    assert df["pulled_at"].nunique() == 1  # one panel snapshot per run


def test_collect_realtime_flag(tmp_path):
    assert _collect_frame(tmp_path, is_realtime=True)["is_realtime"].all()
    # Retro re-parses (e.g. Wayback material) must be flagged non-real-time.
    assert not _collect_frame(tmp_path, is_realtime=False)["is_realtime"].any()


def test_collect_archives_raw_pages_gzipped(tmp_path):
    df = _collect_frame(tmp_path)
    pulled_at = df["pulled_at"].iloc[0]
    day = pulled_at.strftime("%Y-%m-%d")
    hhmmss = pulled_at.strftime("%H%M%S")
    for dealer in ("apmex", "jmbullion"):
        path = tmp_path / "raw" / day / f"{dealer}_gold_eagle_1oz_random_{hhmmss}.html.gz"
        assert path.exists()
        with gzip.open(path, "rb") as f:
            assert f.read() == _fixture_fetch(dealer)


def test_save_raw_page_second_run_same_day_does_not_overwrite(tmp_path):
    """Two runs on the same UTC day must archive side by side, not clobber."""
    first = save_raw_page(b"run-1", "apmex", "age", datetime(2026, 7, 12, 9, 0, 0), tmp_path)
    second = save_raw_page(b"run-2", "apmex", "age", datetime(2026, 7, 12, 15, 30, 45), tmp_path)
    assert first != second
    assert first.parent == second.parent  # same YYYY-MM-DD day directory
    with gzip.open(first, "rb") as f:
        assert f.read() == b"run-1"
    with gzip.open(second, "rb") as f:
        assert f.read() == b"run-2"


# ---------------------------------------------------------------------------
# Upsert (temp DuckDB through the real migrations)
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("METALS_DB_PATH", str(tmp_path / "t.duckdb"))
    from metals.data.migrations.runner import apply_migrations

    apply_migrations(verbose=False)


def test_upsert_idempotent_and_null_bid(temp_db, tmp_path):
    from metals.data.db import connection

    df = _collect_frame(tmp_path)
    assert upsert_coin_premiums(df) == 2
    assert upsert_coin_premiums(df) == 2  # same keys — updates, no duplicates
    with connection(read_only=True) as conn:
        n = conn.execute("SELECT COUNT(*) FROM coin_premiums").fetchone()[0]
        assert n == 2
        apmex_bid = conn.execute(
            "SELECT bid_usd FROM coin_premiums WHERE dealer = 'apmex'"
        ).fetchone()[0]
        assert apmex_bid is None  # SQL NULL, not NaN
        jmb = conn.execute(
            """
            SELECT ask_usd, bid_usd, is_realtime, source
            FROM coin_premiums WHERE dealer = 'jmbullion'
            """
        ).fetchone()
        assert jmb[0] == pytest.approx(5695.09)
        assert jmb[1] == pytest.approx(5110.98)
        assert jmb[2] is True
        assert jmb[3] == SOURCE_TAG


def test_upsert_never_demotes_realtime_flag(temp_db, tmp_path):
    """is_realtime is first-capture honesty: OR semantics, never demoted (as cme_daily)."""
    from metals.data.db import connection

    live = _collect_frame(tmp_path, is_realtime=True)
    assert upsert_coin_premiums(live) == 2
    demoted = live.copy()
    demoted["is_realtime"] = False
    assert upsert_coin_premiums(demoted) == 2  # pulled_at in PK: same-instant re-insert
    retro = live.copy()
    retro["pulled_at"] = retro["pulled_at"] + pd.Timedelta(minutes=1)
    retro["is_realtime"] = False
    assert upsert_coin_premiums(retro) == 2
    promoted = retro.copy()
    promoted["is_realtime"] = True
    assert upsert_coin_premiums(promoted) == 2
    with connection(read_only=True) as conn:
        live_flags = conn.execute(
            "SELECT is_realtime FROM coin_premiums WHERE pulled_at = ?",
            [live["pulled_at"].iloc[0].to_pydatetime()],
        ).fetchall()
        retro_flags = conn.execute(
            "SELECT is_realtime FROM coin_premiums WHERE pulled_at = ?",
            [retro["pulled_at"].iloc[0].to_pydatetime()],
        ).fetchall()
    assert live_flags == [(True,), (True,)]  # re-run with False must not demote
    assert retro_flags == [(True,), (True,)]  # real-time recapture promotes retro rows


def test_upsert_empty_frame_writes_nothing(temp_db):
    assert upsert_coin_premiums(pd.DataFrame()) == 0


def test_parsed_quote_is_frozen():
    quote = ParsedQuote(ask_usd=1.0, bid_usd=None)
    with pytest.raises(AttributeError):
        quote.ask_usd = 2.0  # type: ignore[misc]
