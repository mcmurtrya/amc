"""Tests for the inventory-VaR spread-floor engine (first increment)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metals.features.inventory import ASSUMED_FLOAT_DAYS
from metals.models.spread_floor import (
    DEFAULT_HAIRCUT,
    METAL_TICKERS,
    compute_spread_floor,
    downside_vol,
    implied_daily_vol,
    one_day_log_returns,
)


def _toy_prices(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = list(METAL_TICKERS.values())
    rets = rng.normal(loc=0.0, scale=0.012, size=(n, len(cols)))
    levels = 1000 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(levels, index=idx, columns=cols)


def test_downside_vol_ignores_upside():
    # A series that only ever rises has zero downside deviation.
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    rising = pd.DataFrame({"GC=F": np.linspace(100, 200, 50)}, index=idx)
    r = one_day_log_returns(rising)
    dv = downside_vol(r, window=20)
    assert (dv["GC=F"].dropna() == 0).all()


def test_downside_vol_warmup_is_nan():
    p = _toy_prices(n=100)
    r = one_day_log_returns(p)
    dv = downside_vol(r, window=60)
    # returns lose 1 row to the shift; downside vol needs a full 60-row window.
    assert dv["GC=F"].iloc[:60].isna().all()
    assert dv["GC=F"].iloc[61:].notna().any()


def test_compute_columns_and_shape():
    p = _toy_prices()
    df = compute_spread_floor(p, pd.DataFrame(index=p.index), vol_window=60)
    assert not df.empty
    assert set(df["metal"].unique()) == set(METAL_TICKERS)
    for col in ("spot_usd_oz", "tail_vol_daily", "cushion_usd_oz", "max_buy_usd_oz", "flags"):
        assert col in df.columns


def test_max_buy_identity():
    p = _toy_prices()
    df = compute_spread_floor(p, pd.DataFrame(index=p.index), vol_window=60)
    recomputed = df["exit_floor_usd_oz"] - df["cushion_usd_oz"] - df["carry_usd_oz"]
    assert np.allclose(df["max_buy_usd_oz"], recomputed)


def test_floor_is_below_spot_and_cushion_nonneg():
    p = _toy_prices()
    df = compute_spread_floor(p, pd.DataFrame(index=p.index), vol_window=60)
    assert (df["max_buy_usd_oz"] < df["spot_usd_oz"]).all()
    assert (df["cushion_usd_oz"] >= 0).all()
    assert (df["max_buy_frac"] < 1.0).all()


def test_higher_k_lowers_floor():
    p = _toy_prices()
    lo = compute_spread_floor(p, pd.DataFrame(index=p.index), k=1.0, vol_window=60)
    hi = compute_spread_floor(p, pd.DataFrame(index=p.index), k=3.0, vol_window=60)
    lo_g = lo[lo["metal"] == "gold"].set_index("date_utc")["max_buy_usd_oz"]
    hi_g = hi[hi["metal"] == "gold"].set_index("date_utc")["max_buy_usd_oz"]
    assert (hi_g < lo_g).all()


def test_default_flags_are_all_fallbacks():
    p = _toy_prices()
    df = compute_spread_floor(p, pd.DataFrame(index=p.index), vol_window=60)
    flags = df["flags"].iloc[0]
    assert "vol=realized_downside" in flags
    assert "tail=normal_approx" in flags
    assert "float=assumed" in flags
    assert "carry=rf_only" in flags
    assert "exit=fixed_haircut" in flags
    # No GVZ supplied, so no row should claim implied vol.
    assert not df["flags"].str.contains("vol=implied").any()


def test_exit_floor_uses_haircut():
    p = _toy_prices()
    df = compute_spread_floor(p, pd.DataFrame(index=p.index), vol_window=60)
    g = df[df["metal"] == "gold"]
    assert np.allclose(g["exit_floor_usd_oz"], g["spot_usd_oz"] * (1 - DEFAULT_HAIRCUT["gold"]))


def test_gvz_path_flags_implied_for_gold_only():
    p = _toy_prices()
    # GVZ ~ 16% annualized, constant, on the price calendar.
    macro = pd.DataFrame({"GVZCLS": np.full(len(p), 16.0)}, index=p.index)
    df = compute_spread_floor(p, macro, vol_window=60)
    gold = df[df["metal"] == "gold"]
    silver = df[df["metal"] == "silver"]
    assert (gold["flags"].str.contains("vol=implied")).all()
    assert (silver["flags"].str.contains("vol=realized_downside")).all()
    # 16% annualized -> daily fraction.
    expected_daily = (16.0 / 100.0) / np.sqrt(252)
    assert np.allclose(gold["tail_vol_daily"], expected_daily)


def test_implied_daily_vol_none_without_gvz():
    assert implied_daily_vol(pd.DataFrame(index=pd.date_range("2020-01-01", periods=3))) is None


def test_carry_zero_without_rf():
    p = _toy_prices()
    df = compute_spread_floor(p, pd.DataFrame(index=p.index), vol_window=60)
    assert (df["carry_usd_oz"] == 0).all()


def test_carry_positive_with_rf():
    p = _toy_prices()
    macro = pd.DataFrame({"DGS3MO": np.full(len(p), 5.0)}, index=p.index)
    df = compute_spread_floor(p, macro, vol_window=60)
    assert (df["carry_usd_oz"] > 0).all()


def test_assumed_float_used_by_default():
    p = _toy_prices()
    df = compute_spread_floor(p, pd.DataFrame(index=p.index), vol_window=60)
    assert (df["float_days"] == ASSUMED_FLOAT_DAYS).all()


def test_missing_ticker_is_skipped():
    p = _toy_prices().drop(columns=["PA=F"])
    df = compute_spread_floor(p, pd.DataFrame(index=p.index), vol_window=60)
    assert "palladium" not in set(df["metal"].unique())
    assert "gold" in set(df["metal"].unique())
