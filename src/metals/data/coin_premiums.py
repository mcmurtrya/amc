"""Retail coin-premium panel collector (Phase 7.1, collector 2).

Scrapes a fixed basket of benchmark bullion products (``configs/premium_basket.yaml``)
from APMEX and JM Bullion once daily: dealer ask (schema.org JSON-LD Product/Offer),
dealer buyback bid where published (JM Bullion's ``buybackPrice`` payload field),
spot at capture (yfinance GC=F / SI=F last close), and the implied premiums over
melt. Every fetched page is stored gzipped under
``data/raw/premium_panel/YYYY-MM-DD/<dealer>_<product_id>_<HHMMSS>.html.gz`` so
premiums can be re-parsed after site redesigns.

Rows are append-only with ``source``/``pulled_at`` provenance. Live pulls are
``is_realtime=True``; any retro-captured history (e.g. Wayback material) must be
written with ``is_realtime=False``, permanently. A page that fetches but yields no
ask price raises ``SchemaDriftError`` naming the dealer and product — never a
silently-empty row.

Run as:
    uv run python -m metals.data.coin_premiums
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

from metals.data.db import connection

SOURCE_TAG = "coin_premium_panel"
USER_AGENT = "AMCResearchCollector/0.1 (internal research)"
FETCH_DELAY_S = 2.0
REQUEST_TIMEOUT_S = 30

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASKET_PATH = _REPO_ROOT / "configs" / "premium_basket.yaml"
DEFAULT_RAW_DIR = _REPO_ROOT / "data" / "raw" / "premium_panel"

KNOWN_METALS = {"gold", "silver", "platinum", "palladium"}
SPOT_TICKERS = {"gold": "GC=F", "silver": "SI=F"}

# JM Bullion's buyback bid, e.g. "buybackPrice":5110.98 — the quote characters are
# backslash-escaped when the field sits inside the Next.js flight-payload string.
BUYBACK_RE = re.compile(r'buybackPrice\\?"\s*:\s*([0-9]+(?:\.[0-9]+)?)')

COLUMNS = [
    "pulled_at",
    "dealer",
    "product_id",
    "metal",
    "fine_troy_oz",
    "ask_usd",
    "bid_usd",
    "spot_usd_oz",
    "ask_premium_pct",
    "bid_premium_pct",
    "url",
    "source",
    "is_realtime",
]


class SchemaDriftError(RuntimeError):
    """A dealer page fetched fine but no longer parses — site layout drift."""


@dataclass(frozen=True)
class ParsedQuote:
    """Prices parsed from one dealer product page."""

    ask_usd: float
    bid_usd: float | None


def load_basket(path: str | Path = DEFAULT_BASKET_PATH) -> dict[str, Any]:
    """Load and validate the premium-basket config. Raises ValueError on problems."""
    with Path(path).open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "dealers" not in cfg or "products" not in cfg:
        raise ValueError(f"basket config {path} must define 'dealers' and 'products'")
    dealers = cfg["dealers"]
    products = cfg["products"]
    if not dealers or not products:
        raise ValueError(f"basket config {path} has empty 'dealers' or 'products'")
    seen_ids: set[str] = set()
    for product in products:
        pid = product.get("product_id")
        for key in ("product_id", "metal", "fine_troy_oz", "urls"):
            if key not in product:
                raise ValueError(f"basket product {pid!r} is missing required key {key!r}")
        if pid in seen_ids:
            raise ValueError(f"duplicate product_id {pid!r} in basket config")
        seen_ids.add(pid)
        if product["metal"] not in KNOWN_METALS:
            raise ValueError(f"product {pid!r} has unknown metal {product['metal']!r}")
        fine_oz = float(product["fine_troy_oz"])
        if fine_oz <= 0:
            raise ValueError(f"product {pid!r} has non-positive fine_troy_oz {fine_oz}")
        # Per-face-value melt basis (junk silver) must be encoded consistently.
        has_face = "face_value_usd" in product
        has_per_dollar = "fine_troy_oz_per_face_dollar" in product
        if has_face != has_per_dollar:
            raise ValueError(
                f"product {pid!r} must define face_value_usd and "
                "fine_troy_oz_per_face_dollar together"
            )
        if has_face:
            implied = float(product["face_value_usd"]) * float(
                product["fine_troy_oz_per_face_dollar"]
            )
            if abs(implied - fine_oz) > 1e-9:
                raise ValueError(
                    f"product {pid!r}: fine_troy_oz {fine_oz} != face_value_usd * "
                    f"fine_troy_oz_per_face_dollar = {implied}"
                )
        if not product["urls"]:
            raise ValueError(f"product {pid!r} has no dealer urls")
        for dealer in product["urls"]:
            if dealer not in dealers:
                raise ValueError(f"product {pid!r} references unknown dealer {dealer!r}")
    return cfg


def fetch_page(url: str) -> bytes:
    """Fetch one dealer page with the identified research User-Agent."""
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    if not resp.content:
        raise SchemaDriftError(f"empty response body from {url}")
    return resp.content


def _product_documents(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """All schema.org Product dicts across the page's JSON-LD blocks, in page order.

    Document order matters: a top-level array or ``@graph`` can carry the main
    Product alongside related-item Products, so traversal is FIFO (a LIFO stack
    would surface the *last* — typically related — Product first). Non-JSON
    blocks and non-Product documents (breadcrumbs etc.) are skipped here; the
    *absence of any parsable Product price* is escalated by
    ``parse_product_page``.
    """
    docs: list[dict[str, Any]] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        text = tag.get_text()
        if not text.strip():
            continue
        try:
            # strict=False: dealer JSON-LD can carry raw control characters
            # inside strings (observed on live APMEX product descriptions).
            payload = json.loads(text, strict=False)
        except json.JSONDecodeError:
            continue
        queue: deque[Any] = deque([payload])
        while queue:
            item = queue.popleft()
            if isinstance(item, list):
                queue.extend(item)
                continue
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                queue.extend(graph)
            types = item.get("@type")
            type_list = types if isinstance(types, list) else [types]
            if "Product" in type_list:
                docs.append(item)
    return docs


def _to_price(value: Any) -> float:
    """Coerce a JSON-LD / display price ('4,893.64', 4893.64) to a positive float."""
    if isinstance(value, str):
        value = value.replace(",", "").replace("$", "").strip()
    price = float(value)
    if price <= 0:
        raise SchemaDriftError(f"non-positive price {price!r} parsed")
    return price


def _product_offer_prices(soup: BeautifulSoup) -> list[float]:
    """One offer price per JSON-LD Product that carries one, in page order.

    Cross-sell carousels can emit related items as sibling Product documents;
    ``parse_product_page`` refuses to choose when their prices disagree.
    """
    prices: list[float] = []
    for product in _product_documents(soup):
        offers = product.get("offers")
        offer_list = offers if isinstance(offers, list) else [offers]
        for offer in offer_list:
            if not isinstance(offer, dict):
                continue
            price = offer.get("price") or offer.get("lowPrice")
            if price is not None:
                prices.append(_to_price(price))
                break
    return prices


def _ask_from_css(soup: BeautifulSoup, selector: str) -> float | None:
    """Fallback: first $-amount inside the configured CSS selector, if any."""
    node = soup.select_one(selector)
    if node is None:
        return None
    match = re.search(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)", node.get_text(" ", strip=True))
    if match is None:
        return None
    return _to_price(match.group(1))


def parse_product_page(
    html: str,
    dealer: str,
    product_id: str,
    parse_hints: dict[str, Any] | None = None,
) -> ParsedQuote:
    """Parse ask (and buyback bid where configured) from one dealer product page.

    Prefers the schema.org JSON-LD Product/Offer price; falls back to the
    dealer's documented ``ask_css`` selector. Raises ``SchemaDriftError`` naming
    the dealer/product when a page parses but yields no price, or when several
    Product documents carry *disagreeing* offer prices (a related-item Product
    must never silently poison the series) — the fail-loud alarm for site
    redesigns.
    """
    hints = parse_hints or {}
    soup = BeautifulSoup(html, "lxml")
    json_ld_prices = _product_offer_prices(soup)
    if len(set(json_ld_prices)) > 1:
        raise SchemaDriftError(
            f"{dealer}/{product_id}: {len(json_ld_prices)} JSON-LD Products carry "
            f"disagreeing offer prices {sorted(set(json_ld_prices))} — related-item "
            "markup? refusing to pick one"
        )
    ask = json_ld_prices[0] if json_ld_prices else None
    if ask is None and hints.get("ask_css"):
        ask = _ask_from_css(soup, hints["ask_css"])
    if ask is None:
        raise SchemaDriftError(
            f"{dealer}/{product_id}: page fetched but no ask price found "
            "(no JSON-LD Product offer price; CSS fallback missed) — site layout drift?"
        )

    bid: float | None = None
    bid_source = hints.get("bid_source")
    if bid_source == "buyback_payload":
        pattern = re.compile(hints["bid_regex"]) if hints.get("bid_regex") else BUYBACK_RE
        match = pattern.search(html)
        if match is None:
            raise SchemaDriftError(
                f"{dealer}/{product_id}: buyback bid expected (bid_source="
                "'buyback_payload') but no buybackPrice found — site layout drift?"
            )
        bid = _to_price(match.group(1))
    elif bid_source is not None:
        raise ValueError(f"unknown bid_source {bid_source!r} for dealer {dealer!r}")
    return ParsedQuote(ask_usd=ask, bid_usd=bid)


def compute_premiums(
    ask_usd: float,
    bid_usd: float | None,
    spot_usd_oz: float,
    fine_troy_oz: float,
) -> tuple[float, float | None]:
    """Premiums over melt in percent: (price / (spot * fine_oz) - 1) * 100."""
    if spot_usd_oz <= 0:
        raise ValueError(f"non-positive spot {spot_usd_oz}")
    if fine_troy_oz <= 0:
        raise ValueError(f"non-positive fine_troy_oz {fine_troy_oz}")
    melt = spot_usd_oz * fine_troy_oz
    ask_premium = (ask_usd / melt - 1.0) * 100.0
    bid_premium = (bid_usd / melt - 1.0) * 100.0 if bid_usd is not None else None
    return ask_premium, bid_premium


def fetch_spot(tickers: dict[str, str] | None = None) -> dict[str, float]:
    """Last available close per metal from yfinance (lazy import, as in prices.py)."""
    import yfinance as yf

    tickers = tickers or dict(SPOT_TICKERS)
    out: dict[str, float] = {}
    for metal, ticker in tickers.items():
        raw = yf.download(
            tickers=[ticker],
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no rows for spot ticker {ticker!r}")
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw[ticker]["Close"].dropna()
        else:
            closes = raw["Close"].dropna()
        if closes.empty:
            raise RuntimeError(f"yfinance returned no non-null closes for {ticker!r}")
        out[metal] = float(closes.iloc[-1])
    return out


def save_raw_page(
    content: bytes,
    dealer: str,
    product_id: str,
    pulled_at: datetime,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> Path:
    """Gzip the fetched page to data/raw/premium_panel/YYYY-MM-DD/ for re-parsing.

    The filename carries the pull time (HHMMSS) so a second run the same UTC
    day archives alongside the first instead of silently overwriting it.
    """
    day_dir = Path(raw_dir) / pulled_at.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{dealer}_{product_id}_{pulled_at.strftime('%H%M%S')}.html.gz"
    with gzip.open(path, "wb") as f:
        f.write(content)
    return path


def collect(
    basket_path: str | Path = DEFAULT_BASKET_PATH,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    delay_s: float = FETCH_DELAY_S,
    fetch: Callable[[str], bytes] | None = None,
    spot: dict[str, float] | None = None,
    is_realtime: bool = True,
) -> pd.DataFrame:
    """Fetch, archive, and parse the whole basket into a coin_premiums frame.

    ``fetch`` and ``spot`` are injectable for offline tests / raw re-parses;
    live runs use ``fetch_page`` and ``fetch_spot``. ``is_realtime`` must stay
    True for live pulls and be False for any retro-captured material — that
    flag is permanent in the table. Any parse failure raises; catch-and-continue
    is forbidden inside collectors (the runner isolates failures).
    """
    cfg = load_basket(basket_path)
    fetch_fn = fetch if fetch is not None else fetch_page
    spot_map = spot if spot is not None else fetch_spot(cfg.get("spot_tickers"))
    pulled_at = datetime.now(UTC).replace(tzinfo=None)  # naive UTC, like the rest of the store

    rows: list[dict[str, Any]] = []
    first_fetch = True
    for product in cfg["products"]:
        pid = product["product_id"]
        metal = product["metal"]
        if metal not in spot_map:
            raise RuntimeError(f"no spot price for metal {metal!r} (product {pid!r})")
        fine_oz = float(product["fine_troy_oz"])
        for dealer in sorted(product["urls"]):
            dealer_cfg = cfg["dealers"][dealer]
            if dealer_cfg.get("disallowed"):
                print(f"SKIPPING {dealer}/{pid}: robots/terms disallow (see basket config)")
                continue
            if not first_fetch and delay_s > 0:
                time.sleep(delay_s)
            first_fetch = False
            url = product["urls"][dealer]
            content = fetch_fn(url)
            save_raw_page(content, dealer, pid, pulled_at, raw_dir)
            quote = parse_product_page(
                content.decode("utf-8", errors="replace"),
                dealer=dealer,
                product_id=pid,
                parse_hints=dealer_cfg.get("parse"),
            )
            ask_premium, bid_premium = compute_premiums(
                quote.ask_usd, quote.bid_usd, spot_map[metal], fine_oz
            )
            rows.append(
                {
                    "pulled_at": pd.Timestamp(pulled_at),
                    "dealer": dealer,
                    "product_id": pid,
                    "metal": metal,
                    "fine_troy_oz": fine_oz,
                    "ask_usd": quote.ask_usd,
                    "bid_usd": quote.bid_usd,
                    "spot_usd_oz": spot_map[metal],
                    "ask_premium_pct": ask_premium,
                    "bid_premium_pct": bid_premium,
                    "url": url,
                    "source": SOURCE_TAG,
                    "is_realtime": is_realtime,
                }
            )
    if not rows:
        raise RuntimeError("premium panel collected zero rows — every dealer skipped?")
    return pd.DataFrame(rows, columns=COLUMNS)


def upsert_coin_premiums(df: pd.DataFrame) -> int:
    """Idempotent upsert into the coin_premiums table. Returns rows written.

    ``pulled_at`` is in the primary key, so conflicts are same-instant
    re-inserts; figures update in place but ``is_realtime`` can never be
    demoted (the never-demote OR pattern, as in ``cme_daily``).
    """
    if df.empty:
        return 0
    insert_df = df.copy()
    for col in ("ask_usd", "bid_usd", "spot_usd_oz", "ask_premium_pct", "bid_premium_pct"):
        # Nullable Float64 so missing bids land as SQL NULL, not NaN doubles.
        insert_df[col] = insert_df[col].astype("Float64")
    with connection() as conn:
        conn.register("incoming_coin_premiums", insert_df)
        conn.execute(
            """
            INSERT INTO coin_premiums
                (pulled_at, dealer, product_id, metal, fine_troy_oz, ask_usd, bid_usd,
                 spot_usd_oz, ask_premium_pct, bid_premium_pct, url, source, is_realtime)
            SELECT pulled_at, dealer, product_id, metal, fine_troy_oz, ask_usd, bid_usd,
                   spot_usd_oz, ask_premium_pct, bid_premium_pct, url, source, is_realtime
            FROM incoming_coin_premiums
            ON CONFLICT (pulled_at, dealer, product_id) DO UPDATE SET
                metal           = EXCLUDED.metal,
                fine_troy_oz    = EXCLUDED.fine_troy_oz,
                ask_usd         = EXCLUDED.ask_usd,
                bid_usd         = EXCLUDED.bid_usd,
                spot_usd_oz     = EXCLUDED.spot_usd_oz,
                ask_premium_pct = EXCLUDED.ask_premium_pct,
                bid_premium_pct = EXCLUDED.bid_premium_pct,
                url             = EXCLUDED.url,
                source          = EXCLUDED.source,
                is_realtime     = coin_premiums.is_realtime OR EXCLUDED.is_realtime
            """
        )
        conn.unregister("incoming_coin_premiums")
    return len(insert_df)


def refresh(
    basket_path: str | Path = DEFAULT_BASKET_PATH,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    delay_s: float = FETCH_DELAY_S,
) -> dict:
    """Collect the live panel and upsert. Return a summary dict."""
    df = collect(basket_path=basket_path, raw_dir=raw_dir, delay_s=delay_s)
    n = upsert_coin_premiums(df)
    return {
        "rows_written": n,
        "pulled_at": df["pulled_at"].iloc[0].isoformat(),
        "dealers": sorted(df["dealer"].unique()),
        "products": int(df["product_id"].nunique()),
        "spot_usd_oz": {
            metal: float(df.loc[df["metal"] == metal, "spot_usd_oz"].iloc[0])
            for metal in sorted(df["metal"].unique())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the retail coin-premium panel.")
    parser.add_argument("--basket", default=str(DEFAULT_BASKET_PATH), help="Basket YAML path.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Raw page archive dir.")
    parser.add_argument(
        "--delay", type=float, default=FETCH_DELAY_S, help="Seconds between page fetches (>= 2)."
    )
    args = parser.parse_args()
    summary = refresh(basket_path=args.basket, raw_dir=args.raw_dir, delay_s=args.delay)
    print(f"Rows written:  {summary['rows_written']}")
    print(f"Pulled at:     {summary['pulled_at']} UTC")
    print(f"Dealers:       {', '.join(summary['dealers'])}")
    print(f"Products:      {summary['products']}")
    for metal, px in summary["spot_usd_oz"].items():
        print(f"Spot {metal:9s} ${px:,.2f}/ozt")


if __name__ == "__main__":
    main()
