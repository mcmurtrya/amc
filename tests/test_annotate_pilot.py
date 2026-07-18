"""Offline tests for the Stage-0 annotator pilot (no DB / API needed).

The live paths (DuckDB queries, Batch API) are exercised separately; these cover
the pure logic: prompt/schema fingerprinting, date-blinding, request shaping, the
cost-estimate math, and the report-card checks on synthetic results.
"""

from __future__ import annotations

import json

import pandas as pd

from metals.annotate import checks, pilot, schema
from metals.annotate.sample import PGM_WINDOWS, Stratum, _in_windows, _roll_forward
from metals.annotate.titles import (
    _STOP_RE,
    METAL_TITLE_RE,
    DayTitles,
    _normalize,
    load_day_titles,
)


def test_prompt_hash_stable_and_model_sensitive():
    assert schema.prompt_hash("claude-opus-4-8") == schema.prompt_hash("claude-opus-4-8")
    assert schema.prompt_hash("claude-opus-4-8") != schema.prompt_hash("claude-sonnet-5")


def test_build_user_message_is_date_blind_by_default():
    titles = ["Gold rises ahead of Fed", "Palladium supply squeeze"]
    blind = schema.build_user_message(titles)
    assert "1. Gold rises ahead of Fed" in blind
    assert "2. Palladium supply squeeze" in blind
    assert "2024-01-15" not in blind
    dated = schema.build_user_message(titles, show_date=True, date="2024-01-15")
    assert "Date: 2024-01-15" in dated


def test_schema_is_strict():
    assert schema.ANNOTATION_SCHEMA["additionalProperties"] is False
    per_title = schema.ANNOTATION_SCHEMA["properties"]["titles"]["items"]
    assert per_title["additionalProperties"] is False
    assert "relevant" in per_title["required"]
    assert "none" in schema.EVENT_TYPES


def test_build_params_shape():
    params = pilot.build_params(["a title"], model="claude-opus-4-8")
    assert params["model"] == "claude-opus-4-8"
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["max_tokens"] == pilot.MAX_OUTPUT_TOKENS


def test_estimate_run_offline_math():
    sample = pd.DataFrame({"date": ["2020-01-02", "2020-01-03"], "stratum": ["random", "random"]})
    day_titles = {
        "2020-01-02": DayTitles("2020-01-02", ["h1"], ["Gold up on Fed"], 1, 0),
        "2020-01-03": DayTitles(
            "2020-01-03", ["h2", "h3"], ["Silver dips", "Platinum strike"], 2, 0
        ),
    }
    est = pilot.estimate_run(sample, n_variants=2, day_titles=day_titles, use_api_count=False)
    assert est.n_days == 2
    assert est.n_variants == 2
    assert {r.model for r in est.sample_rows} == set(pilot.PRICING)
    opus = next(r for r in est.sample_rows if r.model == "claude-opus-4-8")
    # 3 titles total x 2 variants = 6 per-title records + 2 days x 2 variants overhead.
    assert (
        opus.output_tokens
        == (3 * pilot.PER_TITLE_OUTPUT_TOKENS + 2 * pilot.DAY_OVERHEAD_OUTPUT_TOKENS) * 2
    )
    assert opus.batch_usd == round(opus.standard_usd * 0.5, 2)
    # Full-run extrapolation scales up from the 2-day sample.
    full = next(r for r in est.full_run_rows if r.model == "claude-opus-4-8")
    assert full.output_tokens > opus.output_tokens


def test_normalize_collapses_whitespace_and_case():
    assert _normalize("  Gold   RISES  ") == "gold rises"


def test_normalize_hardened_dedup_key():
    # Outlet suffix, punctuation, digit-grouping, and curly quotes all fold so
    # syndication copies of one story collapse to a single de-dup key.
    a = _normalize("Gold hits $2,000! - Reuters")
    b = _normalize("Gold hits $2000 — Kitco")
    assert a == b == "gold hits 2000"
    assert _normalize("Gold’s rally") == _normalize("Gold's rally")
    # Non-English titles are not erased to empty (unicode-safe).
    assert _normalize("黄金上涨") != ""


def test_stop_phrase_regex():
    for junk in [
        "Team wins gold medal",
        "Silver Alert issued",
        "Gold Coast tourism",
        "Platinum jubilee celebrations",
        "silver lining for markets",
    ]:
        assert _STOP_RE.search(junk), junk
    for real in ["Gold hits record high", "Palladium supply squeeze", "Silver rallies"]:
        assert not _STOP_RE.search(real), real


def test_metal_regex_covers_amc_terms():
    for hit in [
        "Iridium prices hit record",
        "Krugerrand premiums surge",
        "US Mint suspends Silver Eagle sales",
        "PGM basket weakens",
        "COMEX gold falls",
        "Ruthenium demand",
    ]:
        assert METAL_TITLE_RE.search(hit), hit
    # Precision guards: bare "coin"/"sovereign" must NOT match.
    assert not METAL_TITLE_RE.search("Bitcoin coin rally")
    assert not METAL_TITLE_RE.search("Sovereign debt crisis")


def test_pre_title_era_flag():
    dt = load_day_titles("2017-05-01")  # before 2019-09-22, no DB query needed
    assert dt.pre_title_era is True
    assert dt.titles == []


def test_time_stratified_cap_reserves_us_session():
    from metals.annotate.titles import _select_capped

    # 50 early-UTC (hours 0-6) then 50 US-session (hours 13-22) titles, time-sorted.
    early = pd.date_range("2024-02-15 00:00", periods=50, freq="7min", tz=None)
    uslate = pd.date_range("2024-02-15 13:00", periods=50, freq="10min", tz=None)
    df = pd.DataFrame({"timestamp_utc": list(early) + list(uslate), "page_title": range(100)})
    kept = _select_capped(df, 20)
    assert len(kept) == 20  # budget filled exactly
    us_kept = ((kept["timestamp_utc"].dt.hour >= 13) & (kept["timestamp_utc"].dt.hour <= 22)).sum()
    # earliest-20 would keep 0 US-session rows; the reserve guarantees ~half.
    assert us_kept == 10


def test_assemble_sample_excludes_corpus_gaps():
    from metals.annotate.sample import Stratum, _assemble_sample

    trading = sorted([f"2024-02-{d:02d}" for d in range(1, 21)] + ["2024-01-31"])
    covered = set(trading) - {"2024-01-31"}  # 2024-01-31 is a corpus gap
    fomc = ["2024-01-31"]  # a FOMC lands on the gap day
    df = _assemble_sample(
        trading, fomc, covered, seed=1, cfg=Stratum(n_event=2, n_pgm=0, n_random=5)
    )
    assert "2024-01-31" not in set(df["date"])  # gap day never drawn
    assert (df["stratum"] == "event").sum() == 0  # gap FOMC dropped, not rolled
    assert set(df["date"]).issubset(covered)


def test_assemble_sample_pgm_random_are_event_free():
    from metals.annotate.sample import Stratum, _assemble_sample

    trading = [f"2024-02-{d:02d}" for d in range(1, 26)]  # 25 covered trading days
    fomc = ["2024-02-05", "2024-02-06", "2024-02-07"]  # 3 FOMC days; only 1 selected
    df = _assemble_sample(
        trading, fomc, set(trading), seed=3, cfg=Stratum(n_event=1, n_pgm=0, n_random=10)
    )
    sampled = dict(zip(df["date"], df["stratum"], strict=False))
    for fd in fomc:  # a FOMC day is 'event' or unsampled — never random/pgm
        assert sampled.get(fd) in (None, "event")
    assert (df["stratum"] == "random").sum() == 10


def test_sample_window_helpers():
    assert _in_windows("2020-03-15", PGM_WINDOWS)
    assert not _in_windows("2021-07-01", PGM_WINDOWS)
    trading = ["2020-01-02", "2020-01-06", "2020-01-07"]
    assert _roll_forward("2020-01-03", trading) == "2020-01-06"
    assert _roll_forward("2030-01-01", trading) is None
    assert Stratum().n_event + Stratum().n_pgm + Stratum().n_random == 80


def _synthetic_results() -> pd.DataFrame:
    def rec(rel, metal, narrative):
        return json.dumps(
            {
                "gold_narrative_regime": narrative,
                "monetary_stance_day": "none",
                "titles": [
                    {
                        "id": 1,
                        "relevant": rel,
                        "metal_reads": [{"metal": metal, "direction": 1}] if rel else [],
                        "event_type": "none",
                        "event_entity": "",
                        "supply_demand_side": "none",
                        "framing": "neither",
                        "monetary_stance": "none",
                    }
                ],
            }
        )

    rows = []
    for date, stratum, metal in [
        ("2020-01-06", "random", "gold"),
        ("2020-02-20", "pgm", "palladium"),
    ]:
        for variant in ("blind", "dated"):
            rows.append(
                {
                    "date": date,
                    "variant": variant,
                    "stratum": stratum,
                    "model": "claude-opus-4-8",
                    "ok": True,
                    "gold_narrative_regime": "safe_haven",
                    "monetary_stance_day": "none",
                    "raw_json": rec(True, metal, "safe_haven"),
                }
            )
    return pd.DataFrame(rows)


def test_coverage_and_drift_and_card():
    df = _synthetic_results()
    cov = {c.name: c for c in checks.coverage(df)}
    assert cov["any_metal_coverage"].value == 1.0
    assert cov["pgm_stress_coverage"].value == 1.0  # the pgm day has a palladium read
    drift = checks.date_blind_drift(df)
    assert drift.value == 0.0  # identical blind/dated labels
    card = checks.report_card(df)
    assert "GATE:" in card
    assert "any_metal_coverage" in card
