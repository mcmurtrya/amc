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

import pandas as pd

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
        fill = sum(1 for t in events if t.get(field)) / n
        results.append(
            CheckResult(
                f"{field}_fill",
                float(fill),
                f">= {MIN_V3_FILL:.0%}",
                bool(fill >= MIN_V3_FILL),
                f"{int(fill * n)}/{n} event titles carry `{field}`",
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


def report_card(df: pd.DataFrame) -> str:
    """Assemble the pre-registered pass/fail card."""
    results = [
        *coverage(df),
        known_event_recall(df),
        date_blind_drift(df),
        *v3_field_usage(df),
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
    computed = [r for r in results if r.passed is not None]
    if computed and all(r.passed for r in computed):
        gate = "GREEN (computed checks pass — proceed to human audit, then Stage 1)"
    elif any(r.passed is False for r in computed):
        gate = "RED (a computed check failed — stop or scope down; see details)"
    else:
        gate = "INCOMPLETE (run the batch and/or the date-blind A/B first)"
    lines += ["", f"  GATE: {gate}"]
    return "\n".join(lines)
