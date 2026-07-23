"""Offline tests for the multilingual precision mini-batch (no DB / API needed)."""

from __future__ import annotations

import json
import re

import pandas as pd

from metals.annotate import precision
from metals.annotate.multilang import LANG_TERMS


def test_lang_terms_compile_and_cover_expected_languages():
    for pat in LANG_TERMS.values():
        re.compile(pat)
    assert {"zho", "spa", "vie", "rus", "ara", "deu"} <= set(LANG_TERMS)


def test_judge_schema_is_strict_and_enum_wired():
    item = precision.JUDGE_SCHEMA["properties"]["titles"]["items"]
    assert precision.JUDGE_SCHEMA["additionalProperties"] is False
    assert item["additionalProperties"] is False
    assert item["properties"]["fp_reason"]["enum"] == precision.FP_REASONS
    assert "not_applicable" in precision.FP_REASONS


def test_judge_prompt_declares_language_neutrality_and_taxonomy_rules():
    p = precision.JUDGE_SYSTEM_PROMPT
    assert "in its own language" in p
    assert "not_applicable" in p
    # The known ambiguity traps must be named for the judge.
    assert "vàng" in p and "plata" in p and "złoty" in p


def test_judge_prompt_hash_stable_and_model_sensitive():
    a = precision.judge_prompt_hash("claude-opus-4-8")
    assert a == precision.judge_prompt_hash("claude-opus-4-8")
    assert a != precision.judge_prompt_hash("claude-sonnet-5")


def _sample(n_per_lang: dict[str, int]) -> pd.DataFrame:
    rows = []
    for lang, n in n_per_lang.items():
        for i in range(n):
            rows.append({"lang": lang, "date": "2024-01-02", "title": f"{lang} title {i}"})
    return pd.DataFrame(rows)


def test_chunking_math_and_custom_id_roundtrip():
    sample = _sample({"zho": 100, "eng": 100, "vie": 30})
    chunks = precision._chunks(sample)
    # 100 -> 2 chunks, 100 -> 2, 30 -> 1
    assert len(chunks) == 5
    assert all(len(grp) <= precision.CHUNK_TITLES for _, _, grp in chunks)
    ids = [f"{lang}__{idx}" for lang, idx, _ in chunks]
    assert len(ids) == len(set(ids))
    for cid in ids:
        lang, idx = cid.split("__", 1)
        assert lang in {"zho", "eng", "vie"} and int(idx) >= 0


def test_build_judge_params_shape():
    params = precision.build_judge_params(["Gold up", "黄金上涨"], model="claude-opus-4-8")
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "1. Gold up" in params["messages"][0]["content"]
    assert "2. 黄金上涨" in params["messages"][0]["content"]


def test_estimate_is_offline_and_mentions_batch_discount():
    out = precision.estimate(_sample({"zho": 100, "eng": 50}))
    assert "batch = 50%" in out
    assert "150 titles" in out


def _results(rows) -> pd.DataFrame:
    base = {
        "model": "m",
        "judge_version": precision.JUDGE_VERSION,
        "judge_prompt_hash": "x",
        "batch_id": "b",
        "pulled_at": "t",
        "date": "2024-01-02",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_report_precision_and_verdicts():
    rows = []
    # eng anchor: 9/10 relevant
    for i in range(10):
        rows.append(
            {
                "lang": "eng",
                "title": f"e{i}",
                "ok": True,
                "relevant": i != 0,
                "fp_reason": "not_applicable" if i != 0 else "place_name",
            }
        )
    # zho: 8/10 -> KEEP
    for i in range(10):
        rows.append(
            {
                "lang": "zho",
                "title": f"z{i}",
                "ok": True,
                "relevant": i >= 2,
                "fp_reason": "not_applicable" if i >= 2 else "person_or_brand",
            }
        )
    # vie: 3/10 -> STOPLIST-OR-DROP
    for i in range(10):
        rows.append(
            {
                "lang": "vie",
                "title": f"v{i}",
                "ok": True,
                "relevant": i >= 7,
                "fp_reason": "not_applicable" if i >= 7 else "color_or_adjective",
            }
        )
    out = precision.report(_results(rows))
    assert "ANCHOR" in out
    assert "KEEP" in out
    assert "STOPLIST-OR-DROP" in out
    assert "color_or_adjective" in out  # FP taxonomy surfaces
    assert "eng anchor precision 0.90" in out


def test_report_flags_unjudged_titles():
    rows = [
        {"lang": "zho", "title": "a", "ok": True, "relevant": True, "fp_reason": "not_applicable"},
        {"lang": "zho", "title": "b", "ok": False, "relevant": None, "fp_reason": "__error__:x"},
    ]
    out = precision.report(_results(rows))
    assert "WARNING: 1 titles unjudged" in out


def test_report_empty_frame_degrades():
    df = _results([{"lang": "zho", "title": "a", "ok": False, "relevant": None, "fp_reason": None}])
    assert "nothing to report" in precision.report(df)


def test_min_precision_bar_matches_documented_decision():
    assert precision.MIN_LANG_PRECISION == 0.60


def test_lang_gate_count_script_uses_shared_terms():
    src = open("scripts/lang_gate_count.py").read()
    assert "from metals.annotate.multilang import LANG_TERMS" in src
    assert '"zho":' not in src  # inline dict removed — single source of truth


def test_judge_schema_serializes():
    json.dumps(precision.JUDGE_SCHEMA)


# --- terms v2 (2026-07-23): measured-FP stop-lists + case fixes -------------


def test_ron_case_distinction_kills_the_party():
    """`aur` the metal matches; `AUR` the political party must not."""
    pat = re.compile(LANG_TERMS["ron"])
    assert pat.search("Prețul la aur atinge un nou record")
    assert pat.search("Aurul se scumpește")
    assert pat.search("aurului")  # genitive, case-folded stem group
    assert not pat.search("Liderii AUR țin cu Donald Trump")
    assert not pat.search("George Simion (AUR) a declarat")


def test_jpn_gold_katakana_removed():
    """ゴールド was 70/100 fashion/brand FPs — dropped from the vocabulary."""
    assert "ゴールド" not in LANG_TERMS["jpn"]
    pat = re.compile(LANG_TERMS["jpn"])
    assert pat.search("金価格が上昇")  # the precise compounds survive
    assert not pat.search("ゴールドマン傘下マーカス")  # Goldman Sachs transliterated


def test_stop_terms_compile_and_only_cover_known_languages():
    from metals.annotate.multilang import LANG_STOP_TERMS

    for pat in LANG_STOP_TERMS.values():
        re.compile(pat)
    assert set(LANG_STOP_TERMS) <= set(LANG_TERMS)
    # The six languages that passed outright are deliberately untouched.
    assert not {"zho", "vie", "ara", "tur", "tha", "kor"} & set(LANG_STOP_TERMS)


def test_measured_fp_patterns_are_vetoed():
    from metals.annotate.multilang import LANG_STOP_TERMS

    cases = {
        "spa": ["El arzobispo de La Plata", "Mar del Plata", "medalla de oro"],
        "ita": ["Medaglia d'argento di Benemerenza", "Pomodorino d'Oro 2019 premia"],
        "rus": ["золотая медаль", "Золотой глобус"],
        "fra": ["l'or noir du Sahara", "médaille d'or olympique"],
        "ben": ["সোনালী ব্যাংক"],
        "pol": ["5 mln złotych"],
        "ell": ["Χρυσή Αυγή", "χρυσό μετάλλιο"],
        "por": ["Ouro Preto recebe turistas"],
    }
    for lang, titles in cases.items():
        stop = re.compile(LANG_STOP_TERMS[lang])
        for title in titles:
            assert stop.search(title), f"{lang} stop misses: {title}"


def test_stops_do_not_kill_relevant_titles():
    from metals.annotate.multilang import LANG_STOP_TERMS

    keep = {
        "spa": "El precio del oro alcanza un máximo histórico",
        "ita": "Quotazione dell'oro in rialzo, lingotti richiesti",
        "rus": "Цены на золото выросли на бирже",
        "fra": "Le cours de l'or bat un record",
        "por": "Preço do ouro sobe com procura por refúgio",
    }
    for lang, title in keep.items():
        term = re.compile(LANG_TERMS[lang])
        stop = re.compile(LANG_STOP_TERMS[lang])
        assert term.search(title), f"{lang} term misses relevant: {title}"
        assert not stop.search(title), f"{lang} stop kills relevant: {title}"


def test_multi_admit_case_param_count_and_no_flag_contract():
    from metals.annotate.multilang import LANG_STOP_TERMS, multi_admit_case

    arms, params = multi_admit_case()
    assert len(params) == len(LANG_TERMS) + len(LANG_STOP_TERMS)
    # The contract: callers must not pass a flags argument (ron's case fix).
    assert ", 'i')" not in arms


def test_terms_version_bumped_and_stamped():
    from metals.annotate.multilang import TERMS_VERSION

    assert TERMS_VERSION == "v2"


def test_gate_count_script_uses_shared_admit_helper():
    src = open("scripts/lang_gate_count.py").read()
    assert "multi_admit_case()" in src
    assert "regexp_matches(page_title, ?, 'i')\" for lang in LANG_TERMS" not in src
