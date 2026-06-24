"""Tests for the Phase 5 scenario/treatment builder (pure functions)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metals.features.scenarios import (
    ScenarioSpec,
    align_to_trading_days,
    build_confounders,
    build_treatment,
    confounder_exclusions,
    load_scenario_config,
    load_scenarios,
)


def _spec(**kw) -> ScenarioSpec:
    base = dict(
        id="s",
        name="S",
        definition_type="event",
        economic_family="x",
        available=True,
        source_table="macro",
        source_field="DTWEXBGS",
        transform="level",
        periods=1,
        rule="tercile_high",
        pct=None,
        k=None,
    )
    base.update(kw)
    return ScenarioSpec(**base)


# --- align_to_trading_days -------------------------------------------------


def test_align_rolls_forward_to_next_trading_day():
    trading = pd.bdate_range("2020-01-01", "2020-01-31")
    # 2020-01-04 is a Saturday -> next business day is Monday 2020-01-06
    out = align_to_trading_days([pd.Timestamp("2020-01-04")], trading)
    assert list(out) == [pd.Timestamp("2020-01-06")]


def test_align_keeps_exact_trading_day():
    trading = pd.bdate_range("2020-01-01", "2020-01-31")
    out = align_to_trading_days([pd.Timestamp("2020-01-06")], trading)
    assert list(out) == [pd.Timestamp("2020-01-06")]


def test_align_drops_events_after_last_trading_day():
    trading = pd.bdate_range("2020-01-01", "2020-01-31")
    out = align_to_trading_days([pd.Timestamp("2020-02-15")], trading)
    assert len(out) == 0


def test_align_requires_sorted_index():
    trading = pd.DatetimeIndex(["2020-01-03", "2020-01-02"])
    with pytest.raises(ValueError, match="sorted ascending"):
        align_to_trading_days([pd.Timestamp("2020-01-02")], trading)


# --- build_treatment: daily macro driver -----------------------------------


def test_build_treatment_macro_sigma_high_flags_the_spike():
    idx = pd.bdate_range("2009-06-01", periods=400)  # spans pre/post 2010
    level = pd.Series(100.0, index=idx)
    step = idx[300]
    level.loc[step:] = 105.0  # +5% step -> 5-day pct_change spikes for 5 days
    macro = pd.DataFrame({"DTWEXBGS": level})
    spec = _spec(
        id="dxy_up",
        source_field="DTWEXBGS",
        transform="pct_change",
        periods=5,
        rule="sigma_high",
        k=2.0,
    )

    t = build_treatment(spec, idx, macro=macro, window_start="2010-01-01")
    assert t.name == "dxy_up"
    assert set(np.unique(t.to_numpy())) <= {0, 1}
    assert t.loc[step] == 1
    assert t.loc[idx[290]] == 0
    assert 1 <= int(t.sum()) <= 6


def test_build_treatment_macro_requires_field():
    idx = pd.bdate_range("2010-01-01", periods=50)
    macro = pd.DataFrame({"DTWEXBGS": np.arange(50.0)}, index=idx)
    spec = _spec(source_field="MISSING", rule="sigma_high", k=2.0)
    with pytest.raises(ValueError, match="not in"):
        build_treatment(spec, idx, macro=macro)


# --- build_treatment: sparse FOMC event driver -----------------------------


def test_build_treatment_fomc_tercile_high_selects_and_aligns():
    fomc_dates = pd.to_datetime(
        [
            "2010-03-16",
            "2010-04-28",
            "2010-06-23",
            "2010-08-10",
            "2010-09-21",
            "2010-11-03",
            "2010-12-14",
            "2011-01-26",
            "2011-03-15",
        ]
    )
    mps = pd.Series(
        [0.5, -0.4, 0.6, -0.1, 0.05, -0.5, 0.55, -0.3, 0.45],
        index=fomc_dates,
        name="mps_orth",
    )
    fomc = mps.to_frame()
    trading = pd.bdate_range("2010-01-01", "2011-06-01")
    spec = _spec(
        id="hawkish",
        source_table="fomc_surprises",
        source_field="mps_orth",
        transform="level",
        rule="tercile_high",
    )

    t = build_treatment(spec, trading, fomc=fomc, window_start="2010-01-01")
    # Upper-tercile of mps_orth picks {0.6, 0.55, 0.5}.
    assert int(t.sum()) == 3
    assert t.loc[pd.Timestamp("2010-06-23")] == 1  # 0.60, a weekday -> itself
    assert t.loc[pd.Timestamp("2010-12-14")] == 1  # 0.55
    assert t.loc[pd.Timestamp("2010-04-28")] == 0  # -0.40 (dovish)


def test_build_treatment_threshold_is_in_window_only():
    # A huge pre-window value must not set the threshold for in-window dates.
    idx = pd.bdate_range("2009-01-01", periods=500)
    s = pd.Series(0.0, index=idx)
    s.iloc[5] = 100.0  # pre-2010 outlier
    s.iloc[400] = 1.0  # modest in-window value
    macro = pd.DataFrame({"DTWEXBGS": s})
    spec = _spec(id="x", source_field="DTWEXBGS", transform="level", rule="pct_high", pct=0.95)
    t = build_treatment(spec, idx, macro=macro, window_start="2010-01-01")
    # The in-window 95th pct is ~0, so the modest in-window 1.0 is flagged,
    # and the pre-window outlier (excluded from the index window) is never 1.
    assert t.loc[idx[400]] == 1
    assert t.loc[idx[5]] == 0


# --- confounders -----------------------------------------------------------


def test_build_confounders_columns_lagging_and_exclude():
    idx = pd.bdate_range("2020-01-01", periods=120)
    prices = pd.DataFrame({"GC=F": np.linspace(1800, 1900, 120)}, index=idx)
    macro = pd.DataFrame(
        {
            "DTWEXBGS": np.linspace(95, 100, 120),
            "VIXCLS": np.linspace(15, 20, 120),
            "DGS10": np.linspace(2.0, 3.0, 120),
            "T10YIE": np.linspace(1.5, 2.0, 120),
        },
        index=idx,
    )

    c = build_confounders("GC=F", idx, prices=prices, macro=macro)
    assert list(c.columns) == [
        "ret_5d_lag",
        "rvol_20d_lag",
        "dxy_5d_chg",
        "vix",
        "real_yield",
    ]
    assert pd.isna(c["ret_5d_lag"].iloc[0])  # lagged -> NaN at the start
    assert c["real_yield"].iloc[-1] == pytest.approx(3.0 - 2.0)

    c2 = build_confounders("GC=F", idx, prices=prices, macro=macro, exclude=("dxy_5d_chg",))
    assert "dxy_5d_chg" not in c2.columns


def test_build_confounders_unknown_ticker_raises():
    idx = pd.bdate_range("2020-01-01", periods=30)
    prices = pd.DataFrame({"GC=F": np.ones(30)}, index=idx)
    macro = pd.DataFrame({"DGS10": np.ones(30)}, index=idx)
    with pytest.raises(ValueError, match="not in prices"):
        build_confounders("ZZ=F", idx, prices=prices, macro=macro)


def test_confounder_exclusions_only_for_dxy():
    assert confounder_exclusions(_spec(source_field="DTWEXBGS")) == ("dxy_5d_chg",)
    assert (
        confounder_exclusions(_spec(source_table="fomc_surprises", source_field="mps_orth")) == ()
    )


# --- config parsing --------------------------------------------------------


def test_load_scenario_config_from_raw():
    raw = {
        "modelling": {"window_start": "2011-01-01", "horizons": [1, 5]},
        "scenarios": [
            {
                "id": "a",
                "name": "A",
                "definition_type": "event",
                "economic_family": "usd",
                "available": True,
                "source": "macro:DTWEXBGS",
                "transform": "pct_change",
                "periods": 5,
                "rule": "sigma_high",
                "k": 2.0,
            },
            {"id": "b", "available": False, "source": "events:CPI", "rule": "tercile_high"},
        ],
    }
    cfg = load_scenario_config(raw)
    assert cfg.window_start == "2011-01-01"
    assert cfg.horizons == (1, 5)
    a = cfg.scenarios[0]
    assert (a.source_table, a.source_field, a.k) == ("macro", "DTWEXBGS", 2.0)
    assert cfg.scenarios[1].available is False


def test_load_scenario_config_bad_source_raises():
    with pytest.raises(ValueError, match="<table>:<field>"):
        load_scenario_config(
            {
                "scenarios": [
                    {"id": "x", "source": "nocolon", "rule": "tercile_high"},
                ]
            }
        )


def test_load_scenario_config_rejects_unknown_rule():
    with pytest.raises(ValueError, match="unknown rule"):
        load_scenario_config(
            {
                "scenarios": [
                    {"id": "x", "source": "macro:DTWEXBGS", "rule": "bogus"},
                ]
            }
        )


def test_real_yaml_registry_has_expected_scenarios():
    available = {s.id for s in load_scenarios(available_only=True)}
    assert {
        "hawkish_fomc",
        "dovish_fomc",
        "gpr_spike",
        "dxy_up_shock",
        "dxy_down_shock",
    } <= available
    all_specs = load_scenarios()
    all_ids = {s.id for s in all_specs}
    assert {"cpi_upside", "nfp_upside"} <= all_ids
    assert all(not s.available for s in all_specs if s.id.startswith(("cpi", "nfp")))
