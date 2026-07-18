"""Ledger-derived inventory features for the spread-floor engine.

Reads AMC's own transaction ledger — the local-only ``amc_scrap_lots`` /
``amc_coin_trades`` tables (Collector 1, migration ``008``) — to estimate the
holding-period ("float") distribution that scales the spread-floor cushion.

Until the ledger is populated these functions fall back to a documented
**assumed** float, flagged so no downstream number is mistaken for one
calibrated to AMC's book. The first increment uses a simple realized mean; the
survival (Kaplan–Meier) estimator the full spec calls for — which accounts for
still-open lots (right censoring) — is a later increment.
"""

from __future__ import annotations

from metals.data.db import connection

METALS: tuple[str, ...] = ("gold", "silver", "platinum", "palladium")

# Trading days (~2 calendar weeks). The pre-ledger fallback holding period.
ASSUMED_FLOAT_DAYS: float = 10.0
# Below this many closed lots, a per-metal realized float is too thin to trust.
MIN_CLOSED_LOTS: int = 20
# Ledger durations are calendar days; the cushion scales daily (trading-day) vol.
CAL_TO_TRADING: float = 5.0 / 7.0


def realized_float_days(min_lots: int = MIN_CLOSED_LOTS) -> dict[str, float]:
    """Mean realized holding period (trading days) per metal from CLOSED scrap lots.

    A lot is closed when ``disposed_utc`` is set. Only metals with at least
    ``min_lots`` closed lots are returned; the rest are absent (the caller falls
    back to the assumed float). Open (undisposed) lots are censored and excluded
    here — a known understatement the survival estimator will later correct.

    Returns an empty dict when the ledger table is absent or empty, so this is
    safe to call before AMC's first export lands.
    """
    sql = """
        SELECT metal,
               count(*)                                          AS n,
               avg(date_diff('day', purchased_utc, disposed_utc)) AS cal_days
        FROM amc_scrap_lots
        WHERE disposed_utc IS NOT NULL
          AND disposed_utc >= purchased_utc
        GROUP BY metal
    """
    try:
        with connection(read_only=True) as conn:
            df = conn.execute(sql).fetchdf()
    except Exception:
        # Table missing (migration not applied) or DB unavailable: fall back.
        return {}
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        n = int(r["n"])
        cal_days = r["cal_days"]
        if n >= min_lots and cal_days is not None and float(cal_days) > 0:
            out[str(r["metal"])] = float(cal_days) * CAL_TO_TRADING
    return out


def expected_float_days(min_lots: int = MIN_CLOSED_LOTS) -> dict[str, tuple[float, str]]:
    """Per-metal ``(float_days, flag)``.

    ``flag`` is ``"float=ledger_mean"`` where the ledger has enough closed lots,
    else ``"float=assumed"``. Every metal in :data:`METALS` is present in the
    result so the caller never has to special-case a missing key.
    """
    realized = realized_float_days(min_lots=min_lots)
    out: dict[str, tuple[float, str]] = {}
    for metal in METALS:
        if metal in realized:
            out[metal] = (realized[metal], "float=ledger_mean")
        else:
            out[metal] = (ASSUMED_FLOAT_DAYS, "float=assumed")
    return out
