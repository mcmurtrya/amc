"""Inventory-VaR spread-floor engine (first increment).

Computes, per metal per day, the maximum defensible buy price for scrap/coin
inventory held over AMC's days-to-weeks float:

    max_buy = exit_floor - cushion - carry
    cushion = k * tail_vol_daily * sqrt(float_days) * spot

This is the market-derived FIRST INCREMENT of
``results/amc_spread_floor_engine_spec.md`` (§11): every term degrades to a
documented fallback, recorded per row in ``flags``, so the sheet ships on
already-owned data and sharpens *in place* as rhodium history (real tails),
Databento (real carry), Greysheet (real exit levels) and AMC's ledger (real
float and the book VaR) land — no rewrite. Volatility is the classical
downside estimator Phase 6 blessed over ML for exactly this; no regime/sentiment
feature enters.

Leakage: the floor for a given day is a *contemporaneous* decision object (given
today's spot and trailing volatility, how wide to quote a lot bought now), not a
forward forecast — so there is no future-dated target. The only guard needed is
strictly-trailing windows (``min_periods = window``) and a chronological index;
both are enforced below.

Run as:
    uv run python -m metals.models.spread_floor
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from metals.data.db import connection
from metals.eval.harness import register_run
from metals.features.inventory import ASSUMED_FLOAT_DAYS, expected_float_days
from metals.features.leakage import assert_chronological
from metals.features.loaders import load_macro, load_prices

TRADING_DAYS = 252
CAL_DAYS_PER_YEAR = 365.0
TRADING_TO_CAL = 7.0 / 5.0

# Metal -> canonical price ticker (Yahoo continuous front future).
METAL_TICKERS: dict[str, str] = {
    "gold": "GC=F",
    "silver": "SI=F",
    "platinum": "PL=F",
    "palladium": "PA=F",
}
GVZ_SERIES = "GVZCLS"  # Cboe gold implied-vol index (annualized %), gold only
RF_SERIES = "DGS3MO"  # 3-month T-bill, the financing-cost proxy (annualized %)

DEFAULT_K = 1.645  # normal-approximation 5% one-sided quantile (tail=normal_approx)
DEFAULT_VOL_WINDOW = 60  # trailing trading days for the downside-vol estimate

# Placeholder wholesale-exit haircuts (fraction below spot), by metal. These are
# the exit=fixed_haircut FALLBACK, replaced later by the trailing Greysheet bid
# (coin) and AMC's own refining-payable curve (scrap).
DEFAULT_HAIRCUT: dict[str, float] = {
    "gold": 0.02,
    "silver": 0.05,
    "platinum": 0.05,
    "palladium": 0.08,
}

SOURCE_TAG = "spread_floor_v1"

_OUTPUT_COLUMNS = [
    "date_utc",
    "metal",
    "spot_usd_oz",
    "tail_vol_daily",
    "float_days",
    "k",
    "cushion_usd_oz",
    "carry_usd_oz",
    "exit_floor_usd_oz",
    "max_buy_usd_oz",
    "max_buy_frac",
    "flags",
]


def one_day_log_returns(close_wide: pd.DataFrame) -> pd.DataFrame:
    """One-day log returns per ticker. Non-positive prices mask to NaN (no -inf)."""
    log_p = np.log(close_wide.where(close_wide > 0))
    return log_p - log_p.shift(1)


def downside_vol(returns_1d: pd.DataFrame, window: int = DEFAULT_VOL_WINDOW) -> pd.DataFrame:
    """Rolling lower-tail (downside) deviation of one-day returns, in daily units.

    Defined as ``sqrt(mean over window of min(r, 0)^2)`` — a second lower partial
    moment about zero, using the full window in the denominator (so it is stable
    and never divides by a tiny negative-day count). The floor protects against a
    *fall*, so only downside moves feed the cushion. Warmup rows (< ``window``)
    are NaN via ``min_periods = window``.
    """
    neg_sq = returns_1d.clip(upper=0.0) ** 2
    return np.sqrt(neg_sq.rolling(window=window, min_periods=window).mean())


def implied_daily_vol(macro_wide: pd.DataFrame) -> pd.Series | None:
    """Gold implied vol from GVZ as a *daily*-return standard deviation, or None.

    GVZ is an annualized volatility in percent (like the VIX), so divide by 100
    to a fraction and by ``sqrt(252)`` to daily units. Returns None when GVZ has
    not been ingested, in which case the caller uses realized downside vol.
    """
    if GVZ_SERIES not in macro_wide.columns:
        return None
    return (macro_wide[GVZ_SERIES] / 100.0) / np.sqrt(TRADING_DAYS)


def risk_free_annual(macro_wide: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Annualized risk-free rate (fraction) aligned to ``index``, forward-filled.

    Uses the 3-month T-bill (``DGS3MO``, in percent). Absent → a flat 0.0, which
    zeroes the (fallback) carry term rather than erroring.
    """
    if RF_SERIES not in macro_wide.columns:
        return pd.Series(0.0, index=index)
    rf = (macro_wide[RF_SERIES] / 100.0).reindex(index).ffill()
    return rf.fillna(0.0)


def compute_spread_floor(
    close_wide: pd.DataFrame,
    macro_wide: pd.DataFrame,
    *,
    k: float = DEFAULT_K,
    vol_window: int = DEFAULT_VOL_WINDOW,
    haircut: dict[str, float] | None = None,
    float_map: dict[str, tuple[float, str]] | None = None,
) -> pd.DataFrame:
    """Compute the daily spread floor per metal. Returns a long DataFrame.

    Pure function of the input frames (no I/O), so it is unit-testable on toy
    data. ``float_map`` is ``{metal: (float_days, flag)}``; when omitted every
    metal uses the assumed float. Warmup rows without a volatility estimate are
    dropped.
    """
    haircut = {**DEFAULT_HAIRCUT, **(haircut or {})}
    if float_map is None:
        float_map = {m: (ASSUMED_FLOAT_DAYS, "float=assumed") for m in METAL_TICKERS}

    returns = one_day_log_returns(close_wide)
    realized_daily = downside_vol(returns, window=vol_window)
    gvz_daily = implied_daily_vol(macro_wide)
    rf_annual = risk_free_annual(macro_wide, close_wide.index)

    frames: list[pd.DataFrame] = []
    for metal, ticker in METAL_TICKERS.items():
        if ticker not in close_wide.columns:
            continue
        spot = close_wide[ticker]
        realized_v = realized_daily[ticker]

        # Tail vol: implied (gold, where GVZ present) else realized downside.
        if metal == "gold" and gvz_daily is not None:
            implied_v = gvz_daily.reindex(spot.index)
            missing = implied_v.isna()
            tail_v = implied_v.where(~missing, realized_v)
            vol_flag = pd.Series("vol=implied", index=spot.index).where(
                ~missing, "vol=realized_downside"
            )
        else:
            tail_v = realized_v
            vol_flag = pd.Series("vol=realized_downside", index=spot.index)

        float_days, float_flag = float_map[metal]
        cal_days = float_days * TRADING_TO_CAL

        cushion = k * tail_v * np.sqrt(float_days) * spot
        carry = spot * rf_annual * (cal_days / CAL_DAYS_PER_YEAR)
        exit_floor = spot * (1.0 - haircut[metal])
        max_buy = exit_floor - cushion - carry

        flags = vol_flag + f"|tail=normal_approx|{float_flag}|carry=rf_only|exit=fixed_haircut"

        frame = pd.DataFrame(
            {
                "date_utc": spot.index,
                "metal": metal,
                "spot_usd_oz": spot.to_numpy(),
                "tail_vol_daily": tail_v.to_numpy(),
                "float_days": float(float_days),
                "k": float(k),
                "cushion_usd_oz": cushion.to_numpy(),
                "carry_usd_oz": carry.to_numpy(),
                "exit_floor_usd_oz": exit_floor.to_numpy(),
                "max_buy_usd_oz": max_buy.to_numpy(),
                "max_buy_frac": (max_buy / spot).to_numpy(),
                "flags": flags.to_numpy(),
            }
        )
        frame = frame.dropna(subset=["tail_vol_daily", "cushion_usd_oz", "max_buy_usd_oz"])
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def write_spread_floor(df: pd.DataFrame, source: str = SOURCE_TAG) -> int:
    """Idempotent upsert of a computed floor frame into ``spread_floor_daily``.

    Keyed ``(date_utc, metal, source)`` so re-running the same engine version
    overwrites in place and a later version writes alongside. Returns rows written.
    """
    if df.empty:
        return 0
    out = df.copy()
    out["date_utc"] = pd.to_datetime(out["date_utc"]).dt.date
    out["source"] = source
    out["computed_at"] = datetime.now(UTC).replace(tzinfo=None)
    cols = [*_OUTPUT_COLUMNS, "source", "computed_at"]
    out = out[cols]
    col_list = ", ".join(cols)
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in cols if c not in ("date_utc", "metal", "source")
    )
    with connection() as conn:
        conn.register("incoming_floor", out)
        conn.execute(
            f"""
            INSERT INTO spread_floor_daily ({col_list})
            SELECT {col_list} FROM incoming_floor
            ON CONFLICT (date_utc, metal, source) DO UPDATE SET {updates}
            """
        )
        conn.unregister("incoming_floor")
    return int(len(out))


def run(
    k: float = DEFAULT_K,
    vol_window: int = DEFAULT_VOL_WINDOW,
    source: str = SOURCE_TAG,
    notes: str | None = None,
) -> dict:
    """End-to-end: load owned data, compute the floor, persist, register the run."""
    close = load_prices(tickers=list(METAL_TICKERS.values()), column="close")
    if close.empty:
        raise RuntimeError(
            "No metals prices in DuckDB. Run `uv run python -m metals.data.prices` first."
        )
    assert_chronological(close)
    macro = load_macro(series_ids=[GVZ_SERIES, RF_SERIES])
    float_map = expected_float_days()

    df = compute_spread_floor(close, macro, k=k, vol_window=vol_window, float_map=float_map)
    n = write_spread_floor(df, source=source)

    latest = (
        df.sort_values("date_utc").groupby("metal").tail(1).set_index("metal")
        if not df.empty
        else pd.DataFrame()
    )
    date_range = (
        [
            str(pd.to_datetime(df["date_utc"]).min().date()),
            str(pd.to_datetime(df["date_utc"]).max().date()),
        ]
        if not df.empty
        else []
    )
    run_id = register_run(
        name=f"{source}_{datetime.now(UTC):%Y%m%d_%H%M}",
        model_type="spread_floor",
        target_type="max_buy_floor",
        config={
            "source": source,
            "k": k,
            "vol_window": vol_window,
            "float_days": {m: v for m, (v, _) in float_map.items()},
            "float_flags": {m: f for m, (_, f) in float_map.items()},
            "haircut": DEFAULT_HAIRCUT,
            "gvz_ingested": GVZ_SERIES in macro.columns,
            "rf_ingested": RF_SERIES in macro.columns,
            "n_rows": int(n),
            "metals": list(latest.index) if not latest.empty else [],
            "date_range": date_range,
        },
        notes=notes,
    )
    return {"run_id": run_id, "rows_written": n, "latest": latest, "date_range": date_range}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory-VaR spread-floor engine (increment 1).")
    parser.add_argument(
        "--k", type=float, default=DEFAULT_K, help="Tail multiplier (cushion conservatism)."
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=DEFAULT_VOL_WINDOW,
        help="Trailing downside-vol window (trading days).",
    )
    parser.add_argument("--source", default=SOURCE_TAG, help="Engine version tag (PK component).")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    summary = run(k=args.k, vol_window=args.vol_window, source=args.source, notes=args.notes)
    print(f"Rows written:  {summary['rows_written']}")
    print(f"Date range:    {summary['date_range']}")
    print(f"Run id:        {summary['run_id']}")
    latest = summary["latest"]
    if not latest.empty:
        print("\nLatest floor (per fine troy oz):")
        show = latest[
            [
                "spot_usd_oz",
                "tail_vol_daily",
                "float_days",
                "cushion_usd_oz",
                "max_buy_usd_oz",
                "max_buy_frac",
                "flags",
            ]
        ].copy()
        show["discount_pct"] = (1.0 - show["max_buy_frac"]) * 100.0
        with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
            print(show.to_string())


if __name__ == "__main__":
    main()
