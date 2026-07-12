"""Tests for GDELT URL -> readable-text preparation (pure, no network/DB)."""

from __future__ import annotations

from metals.data.text_prep import (
    clean_slug_text,
    prepare_texts,
    slug_from_url,
    truncate_words,
    url_to_text,
)


def test_basic_slug_with_trailing_numeric_segment():
    u = "https://www.ballinaadvocate.com.au/news/sweet-lineup-drops-for-big-pineapple-music-fest/3941230/"
    assert url_to_text(u) == "sweet lineup drops for big pineapple music fest"


def test_slug_after_numeric_id_section():
    u = "https://www.newjerseytelegraph.com/news/265569381/bolton-book-provides-new-insight-on-trump-thinking-on-venezuela"
    assert url_to_text(u) == "bolton book provides new insight on trump thinking on venezuela"


def test_drops_trailing_hyphen_id_and_html_ext():
    u = "https://www.mittelstandcafe.de/was-der-juli-in-sachen-goldpreis-bringen-k-nnte-1916622.html"
    out = url_to_text(u)
    assert "goldpreis" in out
    assert "1916622" not in out and ".html" not in out


def test_ece_extension_and_id_token_dropped():
    u = "https://www.thehindu.com/news/cities/puducherry/conditional-nod-for-industries/article31370978.ece"
    assert url_to_text(u) == "conditional nod for industries"


def test_camelcase_section_is_split():
    u = "https://af.reuters.com/article/currenciesNews/idAFL8N29J1Q0"
    out = url_to_text(u)
    assert "currencies" in out and "news" in out
    assert "idafl8n29j1q0" not in out  # the id token is dropped


def test_query_string_is_ignored():
    u = "http://www.dagenstv.com/se/chart/?cha=7&dat=2020-11-25&event=1405745277"
    assert url_to_text(u) == "chart"  # query params never enter the text


def test_pure_numeric_path_yields_empty():
    assert url_to_text("http://finance.eastmoney.com/a/202104081876721865.html") == ""
    assert url_to_text("http://life.eastmoney.com/a/202001191361935865.html") == ""


def test_section_only_path_degrades_to_section_word():
    # No real slug, only a section + numeric id -> the section word survives.
    # Acceptable: weak single-word signal, averaged away in daily aggregation.
    assert url_to_text("https://pantip.com/topic/39625122") == "topic"


def test_edge_cases_do_not_crash():
    assert slug_from_url("") == ""
    assert slug_from_url("not a url") == ""  # no netloc -> empty
    assert url_to_text(None) == ""  # type: ignore[arg-type]
    assert url_to_text("https://example.com") == ""  # domain only, no path
    assert clean_slug_text("") == ""


def test_truncate_words():
    assert truncate_words("a b c d e", max_words=3) == "a b c"
    assert truncate_words("", 10) == ""


def test_prepare_texts_batch():
    urls = [
        "https://x.com/news/gold-prices-surge-on-fed-pivot/55",
        "https://y.com/a/202001011234567.html",
    ]
    out = prepare_texts(urls)
    assert out[0] == "gold prices surge on fed pivot"
    assert out[1] == ""
