"""Stage-0 checks + pass/fail report card (plans/phase_8_ssl_probing.md §8.1).

Consumes the ``run_pilot`` results frame and computes the automatable checks:
coverage, known-event recall, and the date-blind A/B drift (the parametric-
leakage control). Human-audit accuracy and multi-seed reproducibility are
surfaced when their inputs are present, else reported as pending.

Thresholds below are the pre-registered gate — record them in journal.md before
looking at any label (§4.4). A red gate stops the program cheaply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from metals.annotate import schema as sch
from metals.data.db import connection

# Pre-registered Stage-0 gate.
MIN_ANY_METAL_COVERAGE = 0.40
MIN_PGM_STRESS_COVERAGE = 0.20
MAX_DATE_BLIND_DRIFT = 0.10
MIN_FOMC_RECALL = 0.50  # fraction of FOMC days where a monetary stance fires
# schema v3.0: the prompt requires `novelty` and `event_time_ref` on EVERY
# event-bearing title, so a low fill rate means the instruction was ignored — a
# schema problem, not a sparse-world problem. `physical_tightness` and `region`
# are genuinely sparse and are reported without a gate.
MIN_V3_FILL = 0.80

_PGM = {"platinum", "palladium"}

# The pre-registered AUTO-gated checks (journal 2026-07-23): the card may print
# GREEN only when EVERY one of these is computed AND passed. A gated check that
# is merely PENDING (e.g. an all-error batch leaves coverage uncomputable) must
# yield INCOMPLETE, never GREEN — report-only diagnostics stay excluded.
# human_audit_accuracy is gated but manual, so it does not block the interim
# GREEN (whose message explicitly says "proceed to human audit").
GATED_AUTO = (
    "results_current",
    "any_metal_coverage",
    "pgm_stress_coverage",
    "fomc_recall",
    "date_blind_drift",
    "novelty_fill",
    "event_time_ref_fill",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    value: float | None
    threshold: str
    passed: bool | None  # None == pending / not computable
    detail: str


def _titles(raw_json: object) -> list[dict]:
    if not isinstance(raw_json, str) or raw_json.startswith("__error__"):
        return []
    try:
        parsed = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return []
    titles = parsed.get("titles", [])
    return [t for t in titles if isinstance(t, dict)]


def _blind(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["variant"] == "blind") & df["ok"]].reset_index(drop=True)


def _day_has_metal(row: pd.Series, metals: set[str] | None = None) -> bool:
    for t in _titles(row["raw_json"]):
        if not t.get("relevant"):
            continue
        for mr in t.get("metal_reads", []):
            m = mr.get("metal") if isinstance(mr, dict) else None
            if metals is None:
                if m:
                    return True
            elif m in metals:
                return True
    return False


def coverage(df: pd.DataFrame) -> list[CheckResult]:
    blind = _blind(df)
    if blind.empty:
        return [CheckResult("coverage", None, "-", None, "no successful blind rows")]
    any_metal = blind.apply(lambda r: _day_has_metal(r), axis=1)
    pgm_days = blind[blind["stratum"] == "pgm"]
    pgm_cov = (
        pgm_days.apply(lambda r: _day_has_metal(r, _PGM), axis=1).mean()
        if not pgm_days.empty
        else None
    )
    results = [
        CheckResult(
            "any_metal_coverage",
            float(any_metal.mean()),
            f">= {MIN_ANY_METAL_COVERAGE:.0%}",
            bool(any_metal.mean() >= MIN_ANY_METAL_COVERAGE),
            f"{int(any_metal.sum())}/{len(blind)} days carry a per-metal read",
        )
    ]
    results.append(
        CheckResult(
            "pgm_stress_coverage",
            None if pgm_cov is None else float(pgm_cov),
            f">= {MIN_PGM_STRESS_COVERAGE:.0%}",
            None if pgm_cov is None else bool(pgm_cov >= MIN_PGM_STRESS_COVERAGE),
            "share of PGM-stress days with a platinum/palladium read"
            + ("" if pgm_cov is not None else " (no pgm days in sample)"),
        )
    )
    return results


def _fomc_sign_map(dates: list[str], window_days: int = 4) -> dict[str, int]:
    """Sign of the nearest FOMC surprise per date (+1 hawkish, -1 dovish, 0 flat).

    Sample dates are roll-forwarded onto trading days (sample.py), so a
    holiday-announcement FOMC date won't match its surprise row exactly; match on
    the nearest surprise within ``window_days`` instead of exact equality.
    """
    if not dates:
        return {}
    lo = (pd.Timestamp(min(dates)) - pd.Timedelta(days=window_days + 3)).date()
    hi = (pd.Timestamp(max(dates)) + pd.Timedelta(days=window_days + 3)).date()
    with connection() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(fomc_surprises)").fetchall()}
        surprise_col = next((c for c in ("mps_orth", "mps", "surprise") if c in cols), None)
        if surprise_col is None:
            return {}
        df = conn.execute(
            f"SELECT CAST(timestamp_utc AS DATE) AS d, {surprise_col} AS s "
            f"FROM fomc_surprises WHERE CAST(timestamp_utc AS DATE) BETWEEN ? AND ?",
            [str(lo), str(hi)],
        ).fetchdf()
    events = [
        (pd.Timestamp(r.d), 0 if r.s == 0 else (1 if r.s > 0 else -1))
        for r in df.itertuples(index=False)
        if pd.notna(r.s)
    ]
    out: dict[str, int] = {}
    for d in dates:
        td = pd.Timestamp(d)
        near = sorted(
            ((abs((td - ed).days), sg) for ed, sg in events if abs((td - ed).days) <= window_days)
        )
        if near:
            out[d] = near[0][1]
    return out


def known_event_recall(df: pd.DataFrame) -> CheckResult:
    blind = _blind(df)
    event = blind[blind["stratum"] == "event"]
    if event.empty:
        return CheckResult("fomc_recall", None, "-", None, "no event-stratum rows")
    fires = event["monetary_stance_day"].isin(["hawkish", "dovish", "mixed"])
    recall = float(fires.mean())
    # Directional agreement where both the annotator and the surprise are signed.
    signs = _fomc_sign_map(event["date"].tolist())
    stance_sign = {"hawkish": 1, "dovish": -1}
    agree = tot = 0
    for _, r in event.iterrows():
        a = stance_sign.get(r["monetary_stance_day"])
        b = signs.get(r["date"])
        if a is not None and b not in (None, 0):
            tot += 1
            agree += int(a == b)
    agree_txt = f"; sign agreement {agree}/{tot} vs fomc_surprises" if tot else ""
    return CheckResult(
        "fomc_recall",
        recall,
        f">= {MIN_FOMC_RECALL:.0%}",
        bool(recall >= MIN_FOMC_RECALL),
        f"{int(fires.sum())}/{len(event)} FOMC days fire a monetary stance{agree_txt} "
        "(recall confounded by parametric leakage — see date-blind drift)",
    )


def date_blind_drift(df: pd.DataFrame) -> CheckResult:
    ok = df[df["ok"]]
    piv = ok.pivot_table(
        index="date",
        columns="variant",
        values=["gold_narrative_regime", "monetary_stance_day"],
        aggfunc="first",
    )
    if ("gold_narrative_regime", "blind") not in piv.columns or (
        "gold_narrative_regime",
        "dated",
    ) not in piv.columns:
        return CheckResult(
            "date_blind_drift", None, "-", None, "date-blind A/B not run (need both variants)"
        )
    disagree = tot = 0
    for field in ("gold_narrative_regime", "monetary_stance_day"):
        both = piv[[(field, "blind"), (field, "dated")]].dropna()
        tot += len(both)
        disagree += int((both[(field, "blind")] != both[(field, "dated")]).sum())
    drift = float(disagree / tot) if tot else None
    return CheckResult(
        "date_blind_drift",
        drift,
        f"<= {MAX_DATE_BLIND_DRIFT:.0%}",
        None if drift is None else bool(drift <= MAX_DATE_BLIND_DRIFT),
        "share of day-labels that change when the date is revealed "
        f"({disagree}/{tot}); high drift = parametric (hindsight) leakage",
    )


def _event_titles(df: pd.DataFrame) -> list[dict]:
    """Every relevant, event-bearing title across the blind rows."""
    out: list[dict] = []
    for _, row in _blind(df).iterrows():
        for t in _titles(row["raw_json"]):
            if t.get("relevant") and t.get("event_type") not in (None, "none"):
                out.append(t)
    return out


def v3_field_usage(df: pd.DataFrame) -> list[CheckResult]:
    """Did the annotator actually populate the schema-v3.0 conditional fields?

    These fields are omitted by design on non-event titles, so the denominator is
    event-bearing titles only. A field that never fires cost output tokens and
    bought nothing — that is a schema finding worth having before the full run.
    """
    events = _event_titles(df)
    if not events:
        return [CheckResult("v3_field_usage", None, "-", None, "no event-bearing titles in sample")]
    n = len(events)
    results: list[CheckResult] = []
    for field in ("novelty", "event_time_ref"):
        # Keep the integer count: reconstructing it as int(fill * n) truncates
        # under float error (1/49*49 == 0.999...), printing a wrong numerator.
        filled = sum(1 for t in events if t.get(field))
        fill = filled / n
        results.append(
            CheckResult(
                f"{field}_fill",
                float(fill),
                f">= {MIN_V3_FILL:.0%}",
                bool(fill >= MIN_V3_FILL),
                f"{filled}/{n} event titles carry `{field}`",
            )
        )
    for field in ("physical_tightness", "region"):
        informative = sum(1 for t in events if t.get(field) not in (None, "none"))
        results.append(
            CheckResult(
                f"{field}_informative",
                float(informative / n),
                "report-only",
                None,
                f"{informative}/{n} event titles carry a non-'none' `{field}` "
                "(sparse by design — no gate)",
            )
        )
    scrap = sum(1 for t in events if t.get("event_type") == "scrap_recycling_flow")
    results.append(
        CheckResult(
            "scrap_recycling_fires",
            float(scrap / n),
            "report-only",
            None,
            f"{scrap}/{n} event titles typed `scrap_recycling_flow` "
            "(new in v3.0 — zero means the channel is absent from the corpus)",
        )
    )
    return results


def results_currency(df: pd.DataFrame) -> CheckResult:
    """Are these results from the CURRENT instrument? (Gated.)

    ``run_pilot`` stamps every row with ``task_version``/``prompt_hash``, but
    until this check nothing ever read them back: the cache path is not keyed by
    the hash, so a schema bump could silently pass stale parquet through the
    card. A version/hash mismatch is a RED gate — re-run before trusting
    anything below it.
    """
    if "task_version" not in df.columns or "prompt_hash" not in df.columns:
        return CheckResult(
            "results_current",
            None,
            "-",
            None,
            "frame predates provenance stamping — cannot confirm instrument version",
        )
    expected_hash = {m: sch.prompt_hash(m) for m in set(df.get("model", []))}
    current = df.apply(
        lambda r: (
            r["task_version"] == sch.TASK_VERSION
            and r["prompt_hash"] == expected_hash.get(r.get("model"))
        ),
        axis=1,
    )
    if not len(df):
        return CheckResult("results_current", None, "-", None, "empty results frame")
    share = float(current.mean())
    stale = sorted(set(df.loc[~current, "task_version"].dropna()) - {sch.TASK_VERSION})
    detail = (
        f"all rows from the current instrument ({sch.TASK_VERSION})"
        if share == 1.0
        else (
            f"{int((~current).sum())}/{len(df)} rows from a stale instrument "
            f"(versions {stale or ['prompt-hash drift']}; current {sch.TASK_VERSION}) "
            "— re-run the batch before trusting this card"
        )
    )
    return CheckResult("results_current", share, "== 100%", bool(share == 1.0), detail)


def v3_date_blind_drift(df: pd.DataFrame) -> list[CheckResult]:
    """Per-title A/B drift on the v3.0 dating fields (report-only).

    The day-level :func:`date_blind_drift` gate never reads a per-title field,
    yet ``novelty`` is the field the schema flags as most likely to invite
    parametric recall. Both variants annotate the same numbered list, so titles
    join across variants by ``(date, id)``; drift is measured on titles that are
    relevant and event-bearing in BOTH variants.
    """
    ok = df[df["ok"]] if "ok" in df.columns else df.iloc[0:0]
    by: dict[tuple, dict[str, dict]] = {}
    for _, row in ok.iterrows():
        for t in _titles(row["raw_json"]):
            by.setdefault((row["date"], t.get("id")), {})[row["variant"]] = t
    pairs = [
        v
        for v in by.values()
        if "blind" in v
        and "dated" in v
        and all(x.get("relevant") and x.get("event_type") not in (None, "none") for x in v.values())
    ]
    if not pairs:
        return [
            CheckResult(
                "v3_ab_drift",
                None,
                "-",
                None,
                "date-blind A/B not run, or no title is event-bearing in both variants",
            )
        ]
    out: list[CheckResult] = []
    for field in ("novelty", "event_time_ref"):
        both = [
            (p["blind"].get(field), p["dated"].get(field))
            for p in pairs
            if p["blind"].get(field) is not None and p["dated"].get(field) is not None
        ]
        if not both:
            out.append(
                CheckResult(
                    f"{field}_ab_drift",
                    None,
                    "report-only",
                    None,
                    f"no joined event title carries `{field}` in both variants",
                )
            )
            continue
        dis = sum(1 for a, b in both if a != b)
        out.append(
            CheckResult(
                f"{field}_ab_drift",
                float(dis / len(both)),
                "report-only",
                None,
                f"{dis}/{len(both)} joined event titles change `{field}` when the date "
                "is revealed (high drift = the field leans on calendar knowledge)",
            )
        )
    return out


def v3_spurious_emission(df: pd.DataFrame) -> CheckResult:
    """Share of NON-event titles emitting any conditional v3 key (report-only).

    The prompt stakes the token budget on "OMIT THE KEYS ENTIRELY", and the fill
    gates cannot see over-emission because their denominator is event-bearing
    titles. This is the number that says whether the 60-tokens/title cost model
    (``pilot.PER_TITLE_OUTPUT_TOKENS``) will hold on the full run.
    """
    keys = ("novelty", "event_time_ref", "physical_tightness", "region")
    non_event = emitting = 0
    for _, row in _blind(df).iterrows():
        for t in _titles(row["raw_json"]):
            if t.get("relevant") and t.get("event_type") not in (None, "none"):
                continue
            non_event += 1
            emitting += int(any(k in t for k in keys))
    if not non_event:
        return CheckResult(
            "v3_spurious_emission", None, "report-only", None, "no non-event titles in sample"
        )
    return CheckResult(
        "v3_spurious_emission",
        float(emitting / non_event),
        "report-only",
        None,
        f"{emitting}/{non_event} non-event titles emit a conditional v3 key despite the "
        "OMIT rule (over-emission inflates output cost above the 60-tokens/title model)",
    )


def offtopic_by_lang(df: pd.DataFrame, max_langs: int = 12) -> list[CheckResult]:
    """Report-only (v3.2): judged-irrelevant share per src_lang on blind rows.

    Joins per-title records to languages by position. v3.3: the language list
    comes from the row's PERSISTED ``langs_json`` (written at submit time), so
    the check needs no database and cannot drift from what was actually
    annotated. Legacy frames without the column fall back to reloading
    ``load_day_titles(date)``, guarded by the stored ``n_titles`` count.
    """
    if "n_titles" not in df.columns:
        return [
            CheckResult(
                "offtopic_by_lang", None, "-", None, "frame lacks n_titles — not a run_pilot frame"
            )
        ]

    def _row_langs(row: pd.Series) -> list[str] | None:
        raw = row.get("langs_json")
        if isinstance(raw, str) and raw:
            try:
                langs = json.loads(raw)
            except json.JSONDecodeError:
                return None
            return langs if len(langs) == int(row["n_titles"]) else None
        # Legacy frame (pre-v3.3): re-derive, guarded against drift.
        from metals.annotate.titles import load_day_titles

        try:
            dt = load_day_titles(row["date"])
        except Exception:
            return None
        if not dt.langs or len(dt.titles) != int(row["n_titles"]):
            return None
        return dt.langs

    tallies: dict[str, list[int]] = {}  # lang -> [irrelevant, total]
    skipped = 0
    for _, row in _blind(df).iterrows():
        langs = _row_langs(row)
        if langs is None:
            skipped += 1
            continue
        for t in _titles(row["raw_json"]):
            i = t.get("id")
            if not isinstance(i, int) or not 1 <= i <= len(langs):
                continue
            bucket = tallies.setdefault(langs[i - 1], [0, 0])
            bucket[1] += 1
            bucket[0] += int(not t.get("relevant"))
    if not tallies:
        return [
            CheckResult(
                "offtopic_by_lang", None, "-", None, f"no joinable days ({skipped} skipped)"
            )
        ]
    results: list[CheckResult] = []
    ranked = sorted(tallies.items(), key=lambda kv: -kv[1][1])[:max_langs]
    for lang, (irr, tot) in ranked:
        results.append(
            CheckResult(
                f"offtopic[{lang}]",
                float(irr / tot),
                "report-only",
                None,
                f"{irr}/{tot} titles judged irrelevant"
                + (f" ({skipped} days skipped)" if skipped else ""),
            )
        )
    return results


def date_in_title_share(df: pd.DataFrame) -> CheckResult:
    """Report-only (v3.3): share of annotated titles whose text carried a masked
    date/year. Measures how much blindness the masking actually bought."""
    if "n_date_masked" not in df.columns or "n_titles" not in df.columns:
        return CheckResult(
            "date_in_title_share", None, "-", None, "frame predates v3.3 mask provenance"
        )
    blind = _blind(df)
    tot = int(blind["n_titles"].sum())
    if not tot:
        return CheckResult("date_in_title_share", None, "-", None, "no blind titles")
    masked = int(blind["n_date_masked"].sum())
    return CheckResult(
        "date_in_title_share",
        float(masked / tot),
        "report-only",
        None,
        f"{masked}/{tot} annotated titles had a date/year masked from their text",
    )


def audit_accuracy(df: pd.DataFrame, audit_csv: str | Path) -> CheckResult:
    """The human-audit gate: per-title `relevant` agreement vs a hand-labelled CSV.

    CSV columns: ``date, id, relevant`` (0/1 or true/false), labelled from an
    ``audit-template`` export — the auditor sees titles only, never model
    outputs. Joins to BLIND rows by (date, id). Gate: >= 0.80 agreement.
    """
    gold = pd.read_csv(audit_csv)
    need = {"date", "id", "relevant"}
    if not need <= set(gold.columns):
        return CheckResult(
            "human_audit_accuracy", None, ">= 80%", None, f"audit csv missing columns {need}"
        )
    model: dict[tuple[str, int], bool] = {}
    for _, row in _blind(df).iterrows():
        for t in _titles(row["raw_json"]):
            if isinstance(t.get("id"), int):
                model[(str(row["date"]), int(t["id"]))] = bool(t.get("relevant"))
    agree = tot = 0
    for _, g in gold.iterrows():
        key = (str(g["date"]), int(g["id"]))
        if key not in model:
            continue
        tot += 1
        agree += int(bool(model[key]) == bool(g["relevant"]))
    if tot == 0:
        return CheckResult(
            "human_audit_accuracy", None, ">= 80%", None, "no joinable audited titles"
        )
    acc = agree / tot
    return CheckResult(
        "human_audit_accuracy",
        float(acc),
        ">= 80%",
        bool(acc >= 0.80),
        f"{agree}/{tot} audited titles agree on `relevant` "
        f"({len(gold) - tot} audit rows unjoinable)",
    )


def repro_agreement(df_a: pd.DataFrame, df_b: pd.DataFrame) -> str:
    """The pre-registered reproducibility addendum, computed.

    Day labels (gold_narrative_regime + monetary_stance_day) joined on date over
    blind rows: agreement >= 0.80. Per-title `relevant` joined on (date, id):
    agreement >= 0.90. Titles must be identical (title_sha256) or the pair is
    excluded and reported — agreement across different inputs is meaningless.
    """
    a, b = _blind(df_a), _blind(df_b)
    on_date_a = a.set_index("date")
    on_date_b = b.set_index("date")
    shared = sorted(set(on_date_a.index) & set(on_date_b.index))
    mismatched = [
        d
        for d in shared
        if "title_sha256" in a.columns
        and "title_sha256" in b.columns
        and on_date_a.loc[d, "title_sha256"] != on_date_b.loc[d, "title_sha256"]
    ]
    usable = [d for d in shared if d not in mismatched]
    day_tot = day_agree = 0
    for d in usable:
        for col in ("gold_narrative_regime", "monetary_stance_day"):
            day_tot += 1
            day_agree += int(on_date_a.loc[d, col] == on_date_b.loc[d, col])
    rel_a: dict[tuple[str, int], bool] = {}
    for _, row in a[a["date"].isin(usable)].iterrows():
        for t in _titles(row["raw_json"]):
            if isinstance(t.get("id"), int):
                rel_a[(str(row["date"]), int(t["id"]))] = bool(t.get("relevant"))
    rel_tot = rel_agree = 0
    for _, row in b[b["date"].isin(usable)].iterrows():
        for t in _titles(row["raw_json"]):
            key = (str(row["date"]), t.get("id"))
            if isinstance(t.get("id"), int) and key in rel_a:
                rel_tot += 1
                rel_agree += int(rel_a[key] == bool(t.get("relevant")))
    day_rate = day_agree / day_tot if day_tot else float("nan")
    rel_rate = rel_agree / rel_tot if rel_tot else float("nan")
    lines = [
        f"Reproducibility over {len(usable)} shared days"
        + (
            f" ({len(mismatched)} EXCLUDED: title lists differ — {mismatched})"
            if mismatched
            else ""
        ),
        f"  day-label agreement:        {day_agree}/{day_tot} = {day_rate:.2f}  "
        + ("PASS (>= 0.80)" if day_rate >= 0.80 else "FAIL (< 0.80)"),
        f"  per-title relevant agree:   {rel_agree}/{rel_tot} = {rel_rate:.2f}  "
        + ("PASS (>= 0.90)" if rel_rate >= 0.90 else "FAIL (< 0.90)"),
    ]
    return "\n".join(lines)


def report_card(df: pd.DataFrame) -> str:
    """Assemble the pre-registered pass/fail card."""
    results = [
        results_currency(df),
        *coverage(df),
        known_event_recall(df),
        date_blind_drift(df),
        *v3_field_usage(df),
        *v3_date_blind_drift(df),
        v3_spurious_emission(df),
        date_in_title_share(df),
        *offtopic_by_lang(df),
    ]
    results.append(
        CheckResult(
            "human_audit_accuracy",
            None,
            ">= 80%",
            None,
            "manual gold set — run `check --audit <csv>` after hand-labelling ~30 days",
        )
    )
    n_ok = int(df["ok"].sum()) if "ok" in df else 0
    lines = [
        "Stage-0 pilot report card",
        f"  rows: {len(df)}  successful: {n_ok}  models: {sorted(set(df.get('model', [])))}",
        "",
        f"  {'check':<32}{'value':>10}  {'gate':<12} verdict",
    ]
    for r in results:
        v = "n/a" if r.value is None else f"{r.value:.3f}"
        verdict = "PENDING" if r.passed is None else ("PASS" if r.passed else "FAIL")
        lines.append(f"  {r.name:<32}{v:>10}  {r.threshold:<12} {verdict}")
        lines.append(f"      {r.detail}")
    by_name = {r.name: r for r in results}
    gated = [by_name.get(n) for n in GATED_AUTO]
    if any(r is not None and r.passed is False for r in results):
        gate = "RED (a computed check failed — stop or scope down; see details)"
    elif all(r is not None and r.passed is True for r in gated):
        gate = "GREEN (all auto-gated checks pass — proceed to human audit, then Stage 1)"
    else:
        pending = [
            n for n, r in zip(GATED_AUTO, gated, strict=True) if r is None or r.passed is None
        ]
        gate = f"INCOMPLETE (gated checks not yet computable: {', '.join(pending)})"
    lines += ["", f"  GATE: {gate}"]
    return "\n".join(lines)
