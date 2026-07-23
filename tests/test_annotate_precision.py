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
