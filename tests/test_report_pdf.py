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


# --- review fixes (2026-07-21) -----------------------------------------------


def test_scalar_and_frame_propagate_binder_errors():
    """Schema drift (a renamed column) must fail loudly, never read as empty.

    A silently-defaulted fact becomes a false statement in a client-facing
    PDF ("we hold zero rows of your transaction data"), so only a MISSING
    TABLE may degrade to the default.
    """
    import duckdb

    with pytest.raises(duckdb.BinderException):
        facts._scalar("SELECT no_such_column FROM prices", -1)
    with pytest.raises(duckdb.BinderException):
        facts._frame("SELECT no_such_column FROM prices")


def test_live_db_getters_return_nondefault_values():
    """On a box with the research DB, every getter must bind and return real data.

    This is the regression net for the all-getters-failed report: with the old
    blanket `except Exception` a locked/missing DB produced zeros and every
    type-only assertion still passed.
    """
    try:
        n = facts._scalar("SELECT count(*) FROM prices", None)
    except Exception:
        pytest.skip("live DB unavailable")
    if not n:
        pytest.skip("prices table empty on this box")
    first, last, count = facts.price_coverage()
    assert count == n and first != "n/a" and last != "n/a"
    assert facts.headline_count() >= 0  # binds (raises on drift), value may vary
    floors = facts.latest_spread_floors()
    if not floors.empty:
        assert {"metal", "date_utc", "max_buy_frac", "flags"} <= set(floors.columns)


def test_default_out_is_repo_anchored_not_cwd():
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "make_owner_report", repo / "scripts" / "make_owner_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEFAULT_OUT.is_absolute()
    assert mod.DEFAULT_OUT == repo / "results" / "amc_owner_briefing.pdf"
