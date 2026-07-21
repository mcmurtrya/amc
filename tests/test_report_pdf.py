"""Tests for the client-report presentation layer and live-facts getters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from metals.report import facts
from metals.report.pdf import CALLOUT_COLORS, Report, stamp


def _minimal(tmp_path: Path, name: str = "t.pdf") -> Path:
    rep = Report("Title", subtitle="Sub", author="A", footer="F")
    rep.title_page(summary="Summary paragraph.", meta=[("Prepared for", "Owner")])
    rep.h1("Section")
    rep.para("Body <b>bold</b> text.")
    return rep.build(tmp_path / name)


def test_build_writes_a_pdf(tmp_path: Path) -> None:
    out = _minimal(tmp_path)
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")


def test_build_creates_missing_parent_dirs(tmp_path: Path) -> None:
    rep = Report("T")
    rep.para("x")
    out = rep.build(tmp_path / "deep" / "nested" / "o.pdf")
    assert out.exists()


def test_all_flowable_kinds_render(tmp_path: Path) -> None:
    """Every builder method must survive a real reportlab build."""
    rep = Report("Kitchen sink", subtitle="all blocks", footer="f")
    rep.title_page(summary="s")
    rep.h1("H1")
    rep.h2("H2")
    rep.para("para")
    rep.lead("lead")
    rep.small("small")
    rep.bullets(["one", "two"])
    rep.table(
        header=["A", "B"],
        rows=[["1", "2"], ["3", "4"], ["5", "6"]],
        align_right=[1],
        notes="a note",
    )
    rep.definition_list([("Term", "meaning")])
    for kind in CALLOUT_COLORS:
        rep.callout("label", "body", kind=kind)  # type: ignore[arg-type]
    rep.spacer()
    rep.page_break()
    rep.para("after break")
    out = rep.build(tmp_path / "sink.pdf")
    assert out.read_bytes().startswith(b"%PDF-")
    # A multi-block document must not collapse to a single page.
    assert out.read_bytes().count(b"/Type /Page") > 1


def test_story_is_not_consumed_by_build(tmp_path: Path) -> None:
    """reportlab mutates the list it is handed; building twice must still work."""
    rep = Report("T")
    rep.h1("S")
    rep.para("p")
    first = rep.build(tmp_path / "a.pdf")
    second = rep.build(tmp_path / "b.pdf")
    assert first.exists() and second.exists()
    assert second.stat().st_size > 0


def test_callout_rejects_unknown_kind(tmp_path: Path) -> None:
    rep = Report("T")
    with pytest.raises(KeyError):
        rep.callout("l", "b", kind="nonsense")  # type: ignore[arg-type]


def test_stamp_includes_commit_when_given() -> None:
    when = datetime(2026, 7, 20)
    assert stamp("abc1234", now=when) == "Generated 20 July 2026 · build abc1234"
    assert stamp("", now=when) == "Generated 20 July 2026"


def test_ledger_status_populated_flag() -> None:
    assert not facts.LedgerStatus(0, 0, 0).populated
    assert facts.LedgerStatus(0, 0, 1).populated


def test_fact_getters_return_usable_types() -> None:
    """Getters must degrade rather than raise when a table is absent."""
    assert isinstance(facts.ledger_status().scrap_lots, int)
    assert isinstance(facts.headline_count(), int)
    assert isinstance(facts.quarantined_rows(), int)
    first, last, n = facts.price_coverage()
    assert isinstance(first, str) and isinstance(last, str) and isinstance(n, int)


def test_scalar_helper_falls_back_on_bad_sql() -> None:
    """A missing table yields the default, not an exception."""
    assert facts._scalar("SELECT count(*) FROM table_that_does_not_exist", -1) == -1
    assert facts._frame("SELECT * FROM table_that_does_not_exist").empty


# -- owner report -----------------------------------------------------------


def test_owner_report_builds(tmp_path: Path) -> None:
    from metals.report import owner_report

    out = owner_report.build(tmp_path / "owner.pdf")
    assert out.read_bytes().startswith(b"%PDF-")
    assert out.stat().st_size > 10_000


def test_owner_report_text_is_plain_language(tmp_path: Path) -> None:
    """Quant jargon must not reach the owner outside the glossary."""
    import subprocess

    from metals.report import owner_report

    out = owner_report.build(tmp_path / "o.pdf")
    text = subprocess.run(
        ["pdftotext", str(out), "-"], capture_output=True, text=True, check=True
    ).stdout.lower()
    for jargon in ("rmse", "p-value", "local projection", "doubleml", "lightgbm"):
        assert jargon not in text, f"jargon leaked into owner report: {jargon}"


def test_owner_report_states_ledger_is_missing_when_empty(tmp_path: Path) -> None:
    """The gate must be visible; an empty ledger cannot render as silence."""
    import subprocess

    from metals.report import facts, owner_report

    if facts.ledger_status().populated:
        pytest.skip("ledger is populated in this database")
    out = owner_report.build(tmp_path / "o.pdf")
    text = subprocess.run(
        ["pdftotext", str(out), "-"], capture_output=True, text=True, check=True
    ).stdout
    assert "bottleneck" in text.lower()
    assert "zero rows" in text.lower()


def test_source_list_deduplicates_individual_files() -> None:
    from metals.report.owner_report import _source_list

    listed = _source_list()
    parts = [p.strip() for p in listed.split(",")]
    assert len(parts) == len(set(parts)), "duplicate source files"
    assert all(p.endswith(".md") for p in parts)


def test_flag_explanations_surface_unknown_flags() -> None:
    """An unexplained flag must appear verbatim, never be silently dropped."""
    import pandas as pd

    from metals.report.owner_report import _flag_explanations

    df = pd.DataFrame({"flags": ["float=assumed|brand_new_flag"]})
    out = _flag_explanations(df)
    assert "brand_new_flag" in out
    assert any("ASSUMED" in o for o in out)


def test_every_finding_carries_a_caveat() -> None:
    """Editorial rule 1: no finding ships without its caveat."""
    from metals.report.owner_report import FINDINGS, NULLS

    for f in FINDINGS + NULLS:
        assert f.caveat.strip(), f"{f.headline} has no caveat"
        assert f.source.strip(), f"{f.headline} has no source"
