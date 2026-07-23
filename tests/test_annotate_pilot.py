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
    # Batch is half price. Assert the invariant with a cent of slack rather than
    # `round(standard * 0.5, 2)`: both fields are rounded from the SAME unrounded
    # cost (pilot.py), so re-rounding an already-rounded standard double-rounds
    # and disagrees whenever the true cost sits near a half-cent.
    assert abs(opus.batch_usd - opus.standard_usd / 2) <= 0.01
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


# --- schema v3.0 ------------------------------------------------------------


def _event_results(**title_overrides) -> pd.DataFrame:
    """Blind results frame carrying one event-bearing title."""
    title = {
        "id": 1,
        "relevant": True,
        "metal_reads": [{"metal": "palladium", "direction": -2}],
        "event_type": "pgm_supply_disruption",
        "event_entity": "Nornickel",
        "supply_demand_side": "supply",
        "framing": "reaction",
        "monetary_stance": "none",
    }
    title.update(title_overrides)
    raw = json.dumps(
        {
            "gold_narrative_regime": "safe_haven",
            "monetary_stance_day": "none",
            # A recap title with no event: the v3 keys must be absent here.
            "titles": [
                title,
                {
                    "id": 2,
                    "relevant": True,
                    "metal_reads": [{"metal": "gold", "direction": 0}],
                    "event_type": "none",
                    "event_entity": "",
                    "supply_demand_side": "none",
                    "framing": "neither",
                    "monetary_stance": "none",
                },
            ],
        }
    )
    return pd.DataFrame(
        [
            {
                "date": "2020-02-20",
                "variant": "blind",
                "stratum": "pgm",
                "model": "claude-opus-4-8",
                "ok": True,
                "gold_narrative_regime": "safe_haven",
                "monetary_stance_day": "none",
                "raw_json": raw,
            }
        ]
    )


def test_v3_fields_are_optional_not_required():
    """The conditional fields must be omittable, or the token saving is lost."""
    item = schema.ANNOTATION_SCHEMA["properties"]["titles"]["items"]
    optional = set(item["properties"]) - set(item["required"])
    assert optional == {"novelty", "event_time_ref", "physical_tightness", "region"}


def test_v3_vocabularies_are_wired_into_the_schema():
    props = schema.ANNOTATION_SCHEMA["properties"]["titles"]["items"]["properties"]
    assert props["novelty"]["enum"] == schema.NOVELTY
    assert props["event_time_ref"]["enum"] == schema.EVENT_TIME_REFS
    assert props["physical_tightness"]["enum"] == schema.PHYSICAL_TIGHTNESS
    assert props["region"]["enum"] == schema.REGION_TAGS


def test_scrap_recycling_is_an_event_type_and_is_documented():
    assert "scrap_recycling_flow" in schema.EVENT_TYPES
    assert "scrap_recycling_flow" in schema.SYSTEM_PROMPT


def test_prompt_documents_the_conditional_fields_and_says_to_omit_them():
    for field in ("novelty", "event_time_ref", "physical_tightness", "region"):
        assert f"`{field}`" in schema.SYSTEM_PROMPT
    assert "OMIT THE KEYS ENTIRELY" in schema.SYSTEM_PROMPT


def test_task_version_bumped_so_stale_caches_are_invalidated():
    assert schema.TASK_VERSION == "v3.3"


def test_v3_usage_counts_only_event_titles():
    """The recap title must not dilute the denominator."""
    df = _event_results(novelty="first_report", event_time_ref="today", region="russia_cis")
    res = {c.name: c for c in checks.v3_field_usage(df)}
    assert res["novelty_fill"].value == 1.0  # 1/1 event titles, not 1/2 titles
    assert res["novelty_fill"].passed is True
    assert res["event_time_ref_fill"].value == 1.0
    assert res["region_informative"].value == 1.0
    assert res["region_informative"].passed is None  # report-only, no gate


def test_v3_usage_flags_an_ignored_instruction():
    """An event title missing the required-by-prompt fields must FAIL."""
    df = _event_results()
    res = {c.name: c for c in checks.v3_field_usage(df)}
    assert res["novelty_fill"].value == 0.0
    assert res["novelty_fill"].passed is False


def test_v3_usage_treats_none_as_uninformative():
    df = _event_results(physical_tightness="none", region="none")
    res = {c.name: c for c in checks.v3_field_usage(df)}
    assert res["physical_tightness_informative"].value == 0.0
    assert res["region_informative"].value == 0.0


def test_v3_usage_reports_scrap_channel_firing():
    df = _event_results(event_type="scrap_recycling_flow")
    res = {c.name: c for c in checks.v3_field_usage(df)}
    assert res["scrap_recycling_fires"].value == 1.0


def test_v3_usage_pending_when_no_events():
    df = _synthetic_results()  # every title is event_type "none"
    res = checks.v3_field_usage(df)
    assert len(res) == 1 and res[0].passed is None


def test_report_card_includes_v3_checks():
    card = checks.report_card(_event_results(novelty="first_report", event_time_ref="today"))
    assert "novelty_fill" in card
    assert "scrap_recycling_fires" in card


# --- review fixes (2026-07-21): currency gate, per-title A/B drift, emission --


def test_prompt_carries_named_date_and_region_precedence_rules():
    assert "you cannot compute the distance without today's date" in schema.SYSTEM_PROMPT
    assert "not the actor imposing it" in schema.SYSTEM_PROMPT
    # The one-week boundary must be exclusive, not overlapping. (Match on
    # single-line fragments — the prompt hard-wraps at ~80 columns.)
    assert "including one week out" in schema.SYSTEM_PROMPT
    assert "more than one week out" in schema.SYSTEM_PROMPT


def _stamped(df: pd.DataFrame, task_version=None, model="claude-opus-4-8") -> pd.DataFrame:
    tv = task_version or schema.TASK_VERSION
    df = df.copy()
    df["task_version"] = tv
    df["prompt_hash"] = schema.prompt_hash(model)
    df["model"] = model
    return df


def test_results_currency_passes_on_current_stamps():
    res = checks.results_currency(_stamped(_event_results()))
    assert res.passed is True and res.value == 1.0


def test_results_currency_fails_on_stale_task_version():
    res = checks.results_currency(_stamped(_event_results(), task_version="v2.0"))
    assert res.passed is False
    assert "v2.0" in res.detail


def test_results_currency_fails_on_prompt_hash_drift():
    df = _stamped(_event_results())
    df["prompt_hash"] = "0123456789abcdef"  # same version, edited prompt
    res = checks.results_currency(df)
    assert res.passed is False


def test_results_currency_pending_without_provenance_columns():
    res = checks.results_currency(_event_results())
    assert res.passed is None


def _ab_results(blind_novelty: str, dated_novelty: str) -> pd.DataFrame:
    def rec(novelty):
        return json.dumps(
            {
                "gold_narrative_regime": "safe_haven",
                "monetary_stance_day": "none",
                "titles": [
                    {
                        "id": 1,
                        "relevant": True,
                        "metal_reads": [{"metal": "palladium", "direction": -2}],
                        "event_type": "pgm_supply_disruption",
                        "event_entity": "Nornickel",
                        "supply_demand_side": "supply",
                        "framing": "reaction",
                        "monetary_stance": "none",
                        "novelty": novelty,
                        "event_time_ref": "today",
                    }
                ],
            }
        )

    rows = []
    for variant, novelty in (("blind", blind_novelty), ("dated", dated_novelty)):
        rows.append(
            {
                "date": "2020-02-20",
                "variant": variant,
                "stratum": "pgm",
                "model": "claude-opus-4-8",
                "ok": True,
                "gold_narrative_regime": "safe_haven",
                "monetary_stance_day": "none",
                "raw_json": rec(novelty),
            }
        )
    return pd.DataFrame(rows)


def test_v3_ab_drift_detects_novelty_disagreement():
    res = {c.name: c for c in checks.v3_date_blind_drift(_ab_results("first_report", "followup"))}
    assert res["novelty_ab_drift"].value == 1.0
    assert res["event_time_ref_ab_drift"].value == 0.0  # "today" in both


def test_v3_ab_drift_pending_without_both_variants():
    res = checks.v3_date_blind_drift(_event_results())  # blind-only fixture
    assert len(res) == 1 and res[0].passed is None


def test_v3_spurious_emission_counts_non_event_titles_only():
    # _event_results has 1 event title (omitted keys don't matter) and 1
    # non-event title with NO v3 keys -> 0/1 spurious.
    res = checks.v3_spurious_emission(_event_results())
    assert res.value == 0.0
    # Now a frame whose non-event title wrongly carries a v3 key.
    raw = json.loads(_event_results().iloc[0]["raw_json"])
    raw["titles"][1]["novelty"] = "unclear"  # spurious on the event_type=none title
    df = _event_results()
    df.loc[0, "raw_json"] = json.dumps(raw)
    res = checks.v3_spurious_emission(df)
    assert res.value == 1.0
    assert res.passed is None  # report-only


def test_report_card_includes_currency_and_emission_checks():
    card = checks.report_card(_stamped(_event_results(novelty="first_report")))
    assert "results_current" in card
    assert "v3_spurious_emission" in card


# --- schema v3.2: the language bridge ---------------------------------------


def _day_frame(rows):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp(f"2024-02-15 {h:02d}:00"),
                "headline_id": f"h{i}",
                "page_title": title,
                "themes": "[]",
                "src_lang": lang,
            }
            for i, (h, title, lang) in enumerate(rows)
        ]
    )


def test_bridge_langs_is_the_measured_nine():
    from metals.annotate.multilang import BRIDGE_LANGS

    assert BRIDGE_LANGS == {"zho", "vie", "ara", "tur", "tha", "kor", "ind", "ron", "jpn"}


def test_admit_bridge_and_base_paths():
    from metals.annotate.titles import _admit

    df = _day_frame(
        [
            (1, "黄金价格创新高", "zho"),  # bridge: zho terms
            (2, "Liderii AUR țin cu Trump", "ron"),  # party — must NOT admit
            (3, "Prețul la aur crește", "ron"),  # metal — bridge admits
            (4, "El precio del oro sube", "spa"),  # spa dropped from bridge — no admit
            (5, "Gold hits record high", "eng"),  # base gate
            (6, "Goldpreis steigt weiter", "deu"),  # base gate cross-language (\bgold... no —
            # "Goldpreis" is one word; \bgold\b does NOT match inside it. Base won't admit;
            # deu is not in the bridge either. Verifies the drop is real.
            (7, "Team wins gold medal again", "eng"),  # stop-phrase veto
        ]
    )
    keep, via_bridge = _admit(df)
    assert bool(keep.iloc[0]) and bool(via_bridge.iloc[0])
    assert not bool(keep.iloc[1])
    assert bool(keep.iloc[2]) and bool(via_bridge.iloc[2])
    assert not bool(keep.iloc[3])  # dropped language stays dropped
    assert bool(keep.iloc[4]) and not bool(via_bridge.iloc[4])
    assert not bool(keep.iloc[5])
    assert not bool(keep.iloc[6])


def test_allocate_proportional_with_ceiling_and_remainders():
    from metals.annotate.titles import _allocate

    # zho dominates but the ceiling holds; remainder goes to the next largest.
    alloc = _allocate({"zho": 100, "vie": 30, "kor": 2}, budget=50, ceiling=20)
    assert alloc["zho"] == 20  # capped
    assert alloc["kor"] == 2  # never more than its count
    assert sum(alloc.values()) <= 50
    # Deterministic:
    assert alloc == _allocate({"zho": 100, "vie": 30, "kor": 2}, budget=50, ceiling=20)


def test_allocate_leaves_slack_when_everything_caps():
    from metals.annotate.titles import _allocate

    alloc = _allocate({"a": 3, "b": 2}, budget=50, ceiling=50)
    assert alloc == {"a": 3, "b": 2}  # slack remains for the caller


def test_language_stratified_cap_reserves_base_floor():

    from metals.annotate.titles import _select_language_stratified

    rows = [(9 + (i % 12), f"Gold headline {i}", "eng") for i in range(200)]
    rows += [(i % 24, f"黄金标题 {i}", "zho") for i in range(300)]
    df = _day_frame(rows)
    df["_via_bridge"] = df["src_lang"] == "zho"
    kept = _select_language_stratified(df, 100)
    assert len(kept) == 100
    n_base = int((~kept["_via_bridge"]).sum())
    assert n_base >= 50  # base floor honored
    n_zho = int((kept["src_lang"] == "zho").sum())
    assert n_zho <= 50  # bridge share bounded
    assert list(kept["timestamp_utc"]) == sorted(kept["timestamp_utc"])  # chronological


def test_language_stratified_slack_flows_to_bridge():
    from metals.annotate.titles import _select_language_stratified

    # Base has only 10 titles; bridge should absorb the leftover budget.
    rows = [(13, f"Gold headline {i}", "eng") for i in range(10)]
    rows += [(i % 24, f"黄金标题 {i}", "zho") for i in range(120)]
    rows += [(i % 24, f"giá vàng {i}", "vie") for i in range(120)]
    df = _day_frame(rows)
    df["_via_bridge"] = df["src_lang"] != "eng"
    kept = _select_language_stratified(df, 100)
    assert len(kept) == 100
    assert int((~kept["_via_bridge"]).sum()) == 10  # all of base, no more exists


def test_prompt_declares_any_language_rule():
    assert "Titles may be in ANY language" in schema.SYSTEM_PROMPT
    assert "original script" in schema.SYSTEM_PROMPT


def test_offtopic_by_lang_pending_without_run_pilot_frame():
    res = checks.offtopic_by_lang(_event_results())  # fixture lacks n_titles
    assert len(res) == 1 and res[0].passed is None


def test_report_card_includes_offtopic_split_row():
    card = checks.report_card(_stamped(_event_results()))
    assert "offtopic_by_lang" in card


# --- v3.3 pre-spend review fixes ---------------------------------------------


def test_mask_dates_full_dates_years_and_price_protection():
    from metals.annotate.titles import _mask_dates

    assert _mask_dates("Gold Rate - March 30, 2020") == "Gold Rate - [DATE]"
    assert _mask_dates("Giá vàng hôm nay 28/10/2019") == "Giá vàng hôm nay [DATE]"
    assert _mask_dates("Fed on 2023-06-14") == "Fed on [DATE]"
    assert _mask_dates("2024年10月28日黄金") == "[DATE]黄金"
    assert _mask_dates("Gold outlook 2024") == "Gold outlook [YEAR]"
    # Prices survive: comma-grouped and currency-prefixed forms are protected.
    assert _mask_dates("Gold falls to 1,950 an ounce") == "Gold falls to 1,950 an ounce"
    assert _mask_dates("Gold at $1950 resistance") == "Gold at $1950 resistance"


def test_mask_applies_to_both_variants_identically():
    # Masking happens in titles.py BEFORE build_params, so blind and dated
    # variants receive identical (masked) text — only the Date: header differs.
    masked = "Gold Rate - [DATE]"
    blind = schema.build_user_message([masked])
    dated = schema.build_user_message([masked], show_date=True, date="2020-03-30")
    assert masked in blind and masked in dated
    assert blind == dated.replace("Date: 2020-03-30\n", "")


def test_allocate_zero_total_guard():
    from metals.annotate.titles import _allocate

    assert _allocate({"a": 0, "b": 0}, budget=5, ceiling=3) == {"a": 0, "b": 0}


def test_report_card_not_green_when_gated_checks_pending():
    """An all-error batch must yield INCOMPLETE, never GREEN."""
    df = _stamped(_event_results())
    df["ok"] = False  # every request errored
    card = checks.report_card(df)
    assert "GREEN" not in card
    assert "INCOMPLETE" in card


def test_report_card_red_on_any_failure_still_wins():
    df = _stamped(_event_results(), task_version="v0.0")  # results_current FAIL
    card = checks.report_card(df)
    assert "RED" in card


def test_audit_accuracy_agreement_math(tmp_path):
    import pandas as pd

    df = _stamped(_event_results())  # blind row: id1 relevant=True, id2 relevant=True
    gold = pd.DataFrame(
        [
            {"date": "2020-02-20", "id": 1, "relevant": 1},  # agree
            {"date": "2020-02-20", "id": 2, "relevant": 0},  # disagree
            {"date": "2099-01-01", "id": 1, "relevant": 1},  # unjoinable
        ]
    )
    csv = tmp_path / "gold.csv"
    gold.to_csv(csv, index=False)
    res = checks.audit_accuracy(df, csv)
    assert res.value == 0.5
    assert res.passed is False
    assert "1 audit rows unjoinable" in res.detail


def test_repro_agreement_math_and_sha_exclusion():
    import pandas as pd

    def frame(narrative, relevant2, sha):
        raw = json.dumps(
            {
                "gold_narrative_regime": narrative,
                "monetary_stance_day": "none",
                "titles": [
                    {"id": 1, "relevant": True},
                    {"id": 2, "relevant": relevant2},
                ],
            }
        )
        return pd.DataFrame(
            [
                {
                    "date": "2020-02-20",
                    "variant": "blind",
                    "ok": True,
                    "gold_narrative_regime": narrative,
                    "monetary_stance_day": "none",
                    "title_sha256": sha,
                    "raw_json": raw,
                }
            ]
        )

    same = checks.repro_agreement(frame("safe_haven", True, "s1"), frame("safe_haven", True, "s1"))
    assert "= 1.00  PASS" in same
    diverged = checks.repro_agreement(
        frame("safe_haven", True, "s1"), frame("cb_demand", False, "s1")
    )
    assert "FAIL" in diverged
    excluded = checks.repro_agreement(
        frame("safe_haven", True, "s1"), frame("safe_haven", True, "s2")
    )
    assert "EXCLUDED" in excluded  # different title lists never compared


def test_run_pilot_refuses_overwrite(tmp_path):
    import pandas as pd

    out = tmp_path / "results.parquet"
    out.write_bytes(b"existing")
    sample = pd.DataFrame({"date": ["2020-01-02"], "stratum": ["random"]})
    import pytest as _pytest

    with _pytest.raises(FileExistsError):
        pilot.run_pilot(sample, out_path=out)


def test_date_in_title_share_diagnostic():
    df = _stamped(_event_results())
    df["n_titles"] = 2
    df["n_date_masked"] = 1
    res = checks.date_in_title_share(df)
    assert res.value == 0.5 and res.passed is None
    legacy = checks.date_in_title_share(_stamped(_event_results()))
    assert legacy.passed is None and "predates" in legacy.detail
