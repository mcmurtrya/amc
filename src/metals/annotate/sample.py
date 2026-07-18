"""Stratified ~80-day sample for the Stage-0 pilot.

Three strata, chosen so the five checks have something to bite on cheaply:

- ``event``  — FOMC days (external ground truth from ``fomc_surprises``) for the
  known-event-recall check.
- ``pgm``    — days inside known PGM-stress windows, to measure whether the
  per-metal PGM channel is non-empty at all (it is expected to be sparse).
- ``random`` — uniform trading days, to measure baseline coverage.

Deterministic: seeded ``random.Random`` so the sample (and any pre-registration
built on it) is reproducible.

Coverage-aware: sample days are drawn only from dates that actually have titled
headlines, so a corpus ingestion gap (e.g. GDELT dropped all of 2024-01 but
2024-01-15) can't be drawn and silently yield an empty annotation input.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from metals.data.db import connection

TITLE_ERA_START = "2019-09-22"  # real page_title coverage begins here
TITLE_ERA_END = "2026-06-19"  # corpus max

# Known PGM-stress windows (approximate) — the channel AMC most needs and where a
# supply/demand signal is most likely to surface if it surfaces anywhere.
PGM_WINDOWS: list[tuple[str, str]] = [
    ("2019-12-01", "2020-02-28"),  # palladium squeeze
    ("2020-03-01", "2020-04-15"),  # COVID market dislocation
    ("2022-09-01", "2023-03-31"),  # South Africa power crisis (platinum)
    ("2023-04-01", "2023-11-30"),  # palladium price collapse
]

DEFAULT_N_EVENT = 20
DEFAULT_N_PGM = 15
DEFAULT_N_RANDOM = 45


@dataclass(frozen=True)
class Stratum:
    n_event: int = DEFAULT_N_EVENT
    n_pgm: int = DEFAULT_N_PGM
    n_random: int = DEFAULT_N_RANDOM


# Exclusive upper bound (day after the corpus max) so a half-open range includes
# the corpus-max day's intraday rows — ``timestamp_utc BETWEEN start AND end``
# would cast the varchar end to midnight and drop them.
_ERA_HI_EXCL = (pd.Timestamp(TITLE_ERA_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _distinct_dates(table: str, ticker: str | None = None) -> list[str]:
    """Distinct calendar dates present in ``table`` within the title era.

    Half-open ``>= start AND < end+1day`` on the raw timestamp: a fast range scan
    that (unlike ``BETWEEN``) is inclusive of the corpus-max day. ``table``/
    ``ticker`` are internal literals, never user input.
    """
    where = "timestamp_utc >= ? AND timestamp_utc < ?"
    params: list[str] = [TITLE_ERA_START, _ERA_HI_EXCL]
    if ticker is not None:
        where = "ticker = ? AND " + where
        params = [ticker, *params]
    sql = f"SELECT DISTINCT CAST(timestamp_utc AS DATE) AS d FROM {table} WHERE {where} ORDER BY d"
    with connection() as conn:
        df = conn.execute(sql, params).fetchdf()
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in df["d"].tolist()]


def _trading_days() -> list[str]:
    return _distinct_dates("prices", ticker="GC=F")


def _fomc_days() -> list[str]:
    return _distinct_dates("fomc_surprises")


def _covered_days() -> set[str]:
    """Title-era dates with >= 1 headline row (excludes corpus gaps).

    Omits the ``page_title IS NOT NULL`` filter deliberately: within the title era
    ~99.5% of rows are titled, so "has any row" yields the identical covered-day
    set as "has titled rows" (verified) at ~10x the speed — a gap day has 0 rows
    either way, and the rare all-NULL-title day is caught by ``load_day_titles``.
    """
    return set(_distinct_dates("headlines"))


def _in_windows(day: str, windows: list[tuple[str, str]]) -> bool:
    ts = pd.Timestamp(day)
    return any(pd.Timestamp(lo) <= ts <= pd.Timestamp(hi) for lo, hi in windows)


def _roll_forward(day: str, trading: list[str]) -> str | None:
    """Map an arbitrary date to the first trading day >= it (None if past the era)."""
    for td in trading:
        if td >= day:
            return td
    return None


def _assemble_sample(
    trading: list[str],
    fomc: list[str],
    covered: set[str],
    seed: int,
    cfg: Stratum,
) -> pd.DataFrame:
    """Pure sampler: build the stratified frame from prepared inputs.

    ``covered`` is the set of dates with titled headlines; every stratum draws
    only from it, so corpus-gap days are never sampled. A FOMC that maps onto a
    gap day is dropped (not rolled to an unrelated covered day). The ``pgm`` and
    ``random`` strata exclude ALL FOMC-mapped days (not just the 20 selected), so
    they stay event-free baselines; ``event`` wins ties over ``pgm`` over
    ``random`` so each day carries exactly one label.
    """
    rng = random.Random(seed)
    trading_set = set(trading)
    trading_cov = [d for d in trading if d in covered]

    # Event days: FOMC dates mapped onto trading days, kept only if covered.
    event_pool: list[str] = []
    for fd in fomc:
        td = fd if fd in trading_set else _roll_forward(fd, trading)
        if td is not None and td in covered and td not in event_pool:
            event_pool.append(td)
    event_all = set(event_pool)  # every FOMC day, selected or not — kept out of pgm/random
    rng.shuffle(event_pool)
    event_days = sorted(event_pool[: cfg.n_event])

    pgm_pool = [d for d in trading_cov if _in_windows(d, PGM_WINDOWS) and d not in event_all]
    rng.shuffle(pgm_pool)
    pgm_days = sorted(pgm_pool[: cfg.n_pgm])
    pgm_set = set(pgm_days)

    random_pool = [d for d in trading_cov if d not in event_all and d not in pgm_set]
    rng.shuffle(random_pool)
    random_days = sorted(random_pool[: cfg.n_random])

    rows = (
        [{"date": d, "stratum": "event"} for d in event_days]
        + [{"date": d, "stratum": "pgm"} for d in pgm_days]
        + [{"date": d, "stratum": "random"} for d in random_days]
    )
    return (
        pd.DataFrame(rows, columns=["date", "stratum"]).sort_values("date").reset_index(drop=True)
    )


def draw_sample(
    seed: int = 42,
    strata: Stratum | None = None,
    require_coverage: bool = True,
) -> pd.DataFrame:
    """Draw the stratified day sample. Returns columns ``date``, ``stratum``.

    ``require_coverage`` (default) restricts every stratum to dates that have
    titled headlines, so a corpus ingestion gap can't be drawn; set False to
    sample from all trading days regardless of headline coverage.
    """
    cfg = strata or Stratum()
    trading = _trading_days()
    fomc = _fomc_days()
    covered = _covered_days() if require_coverage else set(trading)
    return _assemble_sample(trading, fomc, covered, seed, cfg)
