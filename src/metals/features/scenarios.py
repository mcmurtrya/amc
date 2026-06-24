"""Scenario / treatment construction for Phase 5 (causal ML).

A *scenario* is a binary treatment indicator ``T_t`` defined by thresholding a
driver series. This module lifts the treatment-construction logic that used to
live inline in the Phase 2 notebooks (03 hawkish/dovish FOMC, 04 GPR/DXY) into
reusable, tested functions driven by ``configs/scenarios.yaml`` — so the Phase 5
DoubleML estimates and the Phase 2 local projections operate on *identical*
treatment dates, which is what makes the triangulation honest.

Two driver kinds, two code paths:

* **Event drivers** (e.g. FOMC ``mps_orth``) are sparse — one observation per
  announcement. We threshold the surprise distribution, then roll each treated
  announcement FORWARD to the next trading day ``>=`` it. The cumulative outcome
  in the LP/DoubleML regression then sums ``r_{t+1..t+h}``, excluding the
  announcement-day move.
* **Daily macro drivers** (e.g. GPR daily index, broad USD index) are already
  on the trading calendar; after ``reindex(trading_idx).ffill()`` and the
  configured transform we threshold them directly.

Leakage contract (enforced by construction, mirroring ``lp.py``): the treatment
is measured *at* time ``t``; confounders are either lagged own-asset features or
contemporaneous macro state; the outcome is strictly forward
(``lp.cumulative_log_returns``). Thresholds are fit *in-window*
(``rows >= window_start``), the same modest in-sample convention as Phase 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from metals.data import config

# Lowercase COT metal name -> price-table ticker (per spreads.DEFAULT_SPREADS).
METAL_TO_TICKER: dict[str, str] = {
    "gold": "GC=F",
    "silver": "SI=F",
    "platinum": "PL=F",
    "palladium": "PA=F",
}
TICKER_TO_METAL: dict[str, str] = {v: k for k, v in METAL_TO_TICKER.items()}

_VALID_RULES = frozenset({"tercile_high", "tercile_low", "pct_high", "sigma_high", "sigma_low"})
_VALID_TRANSFORMS = frozenset({"level", "diff", "pct_change"})


@dataclass(frozen=True)
class ScenarioSpec:
    """A single scenario definition parsed from ``configs/scenarios.yaml``.

    Fields
    ------
    id, name
        Stable identifier and human-readable label.
    definition_type
        ``"event"`` (from a release/index) or ``"cluster"`` (from Phase 3/4).
    economic_family
        Coarse grouping (monetary / usd / geopolitical / ...) used later for
        the cross-metal consistency check.
    available
        Whether the treatment can actually be built from ingested data. CPI/NFP
        scenarios are listed but ``available=False`` (no consensus ingestion).
    source_table, source_field
        Parsed from the ``"<table>:<field>"`` source string.
    transform, periods
        Driver transform applied before thresholding (``level``/``diff``/
        ``pct_change``); ``periods`` is the lag for diff/pct_change.
    rule, pct, k
        Thresholding rule and its parameters (``pct`` for ``pct_high``; ``k``
        for the ``sigma_*`` rules).
    """

    id: str
    name: str
    definition_type: str
    economic_family: str
    available: bool
    source_table: str
    source_field: str
    transform: str
    periods: int
    rule: str
    pct: float | None
    k: float | None


@dataclass(frozen=True)
class ScenarioConfig:
    """Parsed ``configs/scenarios.yaml``: modelling knobs + scenario list."""

    window_start: str
    horizons: tuple[int, ...]
    scenarios: list[ScenarioSpec]


def _parse_spec(row: dict[str, Any]) -> ScenarioSpec:
    source = str(row["source"])
    if ":" not in source:
        raise ValueError(
            f"scenario {row.get('id')!r}: source must be '<table>:<field>', got {source!r}"
        )
    table, field = source.split(":", 1)
    transform = row.get("transform", "level")
    if transform not in _VALID_TRANSFORMS:
        raise ValueError(
            f"scenario {row.get('id')!r}: unknown transform {transform!r}; "
            f"choose from {sorted(_VALID_TRANSFORMS)}"
        )
    rule = row["rule"]
    if rule not in _VALID_RULES:
        raise ValueError(
            f"scenario {row.get('id')!r}: unknown rule {rule!r}; choose from {sorted(_VALID_RULES)}"
        )
    return ScenarioSpec(
        id=str(row["id"]),
        name=str(row.get("name", row["id"])),
        definition_type=str(row.get("definition_type", "event")),
        economic_family=str(row.get("economic_family", "unknown")),
        available=bool(row.get("available", True)),
        source_table=table,
        source_field=field,
        transform=transform,
        periods=int(row.get("periods", 1)),
        rule=rule,
        pct=None if row.get("pct") is None else float(row["pct"]),
        k=None if row.get("k") is None else float(row["k"]),
    )


def load_scenario_config(raw: dict[str, Any] | None = None) -> ScenarioConfig:
    """Load and validate ``configs/scenarios.yaml`` into a ``ScenarioConfig``.

    Pass ``raw`` (an already-parsed mapping) to bypass disk — used in tests.
    """
    cfg = raw if raw is not None else config.scenarios()
    modelling = cfg.get("modelling", {})
    horizons = tuple(int(h) for h in modelling.get("horizons", (1, 5, 20)))
    specs = [_parse_spec(row) for row in cfg.get("scenarios", [])]
    return ScenarioConfig(
        window_start=str(modelling.get("window_start", "2010-01-01")),
        horizons=horizons,
        scenarios=specs,
    )


def load_scenarios(available_only: bool = False) -> list[ScenarioSpec]:
    """Return the scenario list (optionally only the buildable ones)."""
    specs = load_scenario_config().scenarios
    return [s for s in specs if s.available] if available_only else specs


def _safe(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a column if present, else an all-NaN series aligned to the index."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index, name=col)


def align_to_trading_days(
    event_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    trading_idx: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Map each event timestamp to the first trading day ``>=`` it (roll forward).

    Events falling after the last trading day are dropped. The mapping never
    rolls backward, so an announcement is always attributed to the session in
    which it could first be traded on. Returns the unique, sorted trading days.
    """
    ti = pd.DatetimeIndex(trading_idx)
    if not ti.is_monotonic_increasing:
        raise ValueError("trading_idx must be sorted ascending.")
    ev = pd.DatetimeIndex(pd.to_datetime(list(event_dates)))
    if len(ev) == 0:
        return pd.DatetimeIndex([])
    pos = ti.searchsorted(ev.sort_values(), side="left")
    keep = pos < len(ti)
    return pd.DatetimeIndex(ti[pos[keep]]).unique()


def _apply_transform(s: pd.Series, transform: str, periods: int) -> pd.Series:
    if transform == "level":
        return s
    if transform == "diff":
        return s.diff(periods)
    if transform == "pct_change":
        return s.pct_change(periods)
    raise ValueError(f"unknown transform {transform!r}")


def _threshold_mask(
    driver: pd.Series,
    rule: str,
    in_window: np.ndarray,
    pct: float | None,
    k: float | None,
) -> pd.Series:
    """Boolean mask of `driver` crossing the rule's in-window threshold.

    Thresholds are fit on ``driver[in_window]`` only; the mask itself is
    evaluated over the whole index. Comparisons against NaN driver values are
    ``False`` by construction.
    """
    win_vals = driver[in_window].dropna()
    if win_vals.empty:
        raise ValueError("no in-window observations to fit the threshold.")
    if rule == "tercile_high":
        return driver > win_vals.quantile(2.0 / 3.0)
    if rule == "tercile_low":
        return driver < win_vals.quantile(1.0 / 3.0)
    if rule == "pct_high":
        if pct is None:
            raise ValueError("rule 'pct_high' requires a `pct` parameter.")
        return driver > win_vals.quantile(pct)
    if rule == "sigma_high":
        if k is None:
            raise ValueError("rule 'sigma_high' requires a `k` parameter.")
        return driver > k * float(win_vals.std())
    if rule == "sigma_low":
        if k is None:
            raise ValueError("rule 'sigma_low' requires a `k` parameter.")
        return driver < -k * float(win_vals.std())
    raise ValueError(f"unknown rule {rule!r}")


def build_treatment(
    spec: ScenarioSpec,
    trading_idx: pd.DatetimeIndex,
    *,
    fomc: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    window_start: str | pd.Timestamp = "2010-01-01",
) -> pd.Series:
    """Build a binary (int8) treatment series for ``spec`` on ``trading_idx``.

    Parameters
    ----------
    spec
        The scenario definition.
    trading_idx
        The trading-day index to define the treatment on (e.g. the price index).
    fomc
        FOMC-surprises frame indexed by announcement date (from
        ``loaders.load_fomc_surprises``). Required for ``fomc_surprises``
        sources.
    macro
        Wide macro frame indexed by ``timestamp_utc`` (from
        ``loaders.load_macro``). Required for ``macro`` sources.
    window_start
        Left edge of the in-window threshold-fitting period.

    Returns
    -------
    pd.Series
        int8 0/1 series named ``spec.id``, indexed by ``trading_idx``.
    """
    ti = pd.DatetimeIndex(trading_idx)
    if not ti.is_monotonic_increasing:
        raise ValueError("trading_idx must be sorted ascending.")
    win_start = pd.Timestamp(window_start)

    if spec.source_table == "fomc_surprises":
        if fomc is None or fomc.empty:
            raise ValueError(f"scenario {spec.id!r} needs FOMC surprises but none provided.")
        if spec.source_field not in fomc.columns:
            raise ValueError(
                f"scenario {spec.id!r}: field {spec.source_field!r} not in "
                f"fomc columns {list(fomc.columns)}"
            )
        driver = _apply_transform(fomc[spec.source_field].dropna(), spec.transform, spec.periods)
        in_window = np.asarray(driver.index >= win_start)
        mask = _threshold_mask(driver, spec.rule, in_window, spec.pct, spec.k)
        # Only in-window announcements are treated (pre-window dates have no
        # defined regime — the threshold was not fit on them).
        treated_dates = driver.index[mask.fillna(False).to_numpy() & in_window]
        aligned = align_to_trading_days(treated_dates, ti)
        out = pd.Series(0, index=ti, dtype="int8", name=spec.id)
        out.loc[ti.intersection(aligned)] = np.int8(1)
        return out

    if spec.source_table == "macro":
        if macro is None or macro.empty:
            raise ValueError(f"scenario {spec.id!r} needs macro data but none provided.")
        if spec.source_field not in macro.columns:
            raise ValueError(
                f"scenario {spec.id!r}: field {spec.source_field!r} not in macro columns."
            )
        series = macro[spec.source_field].reindex(ti).ffill()
        driver = _apply_transform(series, spec.transform, spec.periods)
        in_window = np.asarray(driver.index >= win_start)
        mask = _threshold_mask(driver, spec.rule, in_window, spec.pct, spec.k)
        # Treatment is only active in-window (the threshold is undefined outside).
        flagged = mask.fillna(False).to_numpy() & in_window
        return pd.Series(flagged.astype("int8"), index=ti, name=spec.id)

    raise ValueError(
        f"scenario {spec.id!r}: cannot build treatment for source table "
        f"{spec.source_table!r} (only fomc_surprises and macro are wired)."
    )


def confounder_exclusions(spec: ScenarioSpec) -> tuple[str, ...]:
    """Confounder columns to drop because they coincide with the driver.

    A scenario must not control for its own treatment variable (notebook 04's
    ``controls_for(ticker, exclude=...)`` pattern). Only the USD shocks overlap
    the default control set (``dxy_5d_chg``); FOMC and GPR drivers are not in it.
    """
    if spec.source_table == "macro" and spec.source_field == "DTWEXBGS":
        return ("dxy_5d_chg",)
    return ()


def build_confounders(
    ticker: str,
    trading_idx: pd.DatetimeIndex,
    *,
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """Phase 2 control set ``X_t`` on ``trading_idx`` for ``ticker``.

    Columns (Phase 2 step 2.7): ``ret_5d_lag``, ``rvol_20d_lag``,
    ``dxy_5d_chg``, ``vix``, ``real_yield``. Own-asset features are lagged one
    day so they use only past information; macro controls are contemporaneous
    (available same-day). Columns named in ``exclude`` are dropped — pass
    ``confounder_exclusions(spec)`` to remove the scenario's own driver.

    ``prices`` is a wide adj-close frame (ticker columns); ``macro`` is a wide
    FRED frame (series_id columns).
    """
    from metals.features.returns import ANNUALIZATION

    if ticker not in prices.columns:
        raise ValueError(f"ticker {ticker!r} not in prices columns.")
    ti = pd.DatetimeIndex(trading_idx)
    px = prices[ticker].where(prices[ticker] > 0)
    own_ret = np.log(px).diff().reindex(ti)
    macro_a = macro.reindex(ti).ffill()

    out = pd.DataFrame(index=ti)
    out["ret_5d_lag"] = own_ret.rolling(5, min_periods=5).sum().shift(1)
    out["rvol_20d_lag"] = (
        own_ret.rolling(20, min_periods=20).std() * float(np.sqrt(ANNUALIZATION))
    ).shift(1)
    out["dxy_5d_chg"] = _safe(macro_a, "DTWEXBGS").pct_change(5)
    out["vix"] = _safe(macro_a, "VIXCLS")
    out["real_yield"] = _safe(macro_a, "DGS10") - _safe(macro_a, "T10YIE")

    drop = [c for c in exclude if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    return out
