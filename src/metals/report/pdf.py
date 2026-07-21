"""Presentation primitives for client-facing PDF reports.

A thin wrapper over reportlab's platypus flowables, tuned for one job: a
document a non-technical business reader will actually finish. Long measure,
generous leading, few rules, and a small fixed palette.

Nothing in this module knows anything about metals research. Content lives in
the report modules; this is layout only, so a second report reuses it as-is.

Two conventions carry the project's honesty discipline into the page itself:

* :meth:`Report.callout` renders a labelled box. The ``kind="caution"`` variant
  is the house style for "this number rests on an assumption" — findings that
  need a caveat carry it visually adjacent, not in a footnote.
* :meth:`Report.table` accepts ``notes`` so a figure and its provenance cannot
  be separated by an edit.

Usage::

    rep = Report("Title", subtitle="...", author="...")
    rep.title_page(summary="one paragraph")
    rep.h1("Section")
    rep.para("Body text with <b>bold</b> allowed.")
    rep.build(Path("out.pdf"))
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# A small palette. Ink is near-black rather than black (softer on paper);
# accent is used only for headings and rules, never for body text.
INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5f6368")
ACCENT = colors.HexColor("#1f4e5f")
RULE = colors.HexColor("#d4d4d4")
BAND = colors.HexColor("#f2f4f5")

CALLOUT_COLORS: dict[str, tuple[colors.Color, colors.Color]] = {
    # kind -> (border/label colour, fill)
    "note": (colors.HexColor("#1f4e5f"), colors.HexColor("#eef4f6")),
    "caution": (colors.HexColor("#8a6100"), colors.HexColor("#fdf6e6")),
    "good": (colors.HexColor("#2e6b3e"), colors.HexColor("#eef6f0")),
    "blocked": (colors.HexColor("#8a2f2f"), colors.HexColor("#fbefef")),
}

CalloutKind = Literal["note", "caution", "good", "blocked"]

BODY_SIZE = 10.5
LEADING = 15.5


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=BODY_SIZE,
        leading=LEADING,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=9,
    )
    return {
        "body": base,
        "title": ParagraphStyle(
            "title",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=30,
            textColor=ACCENT,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base,
            fontSize=13,
            leading=18,
            textColor=MUTED,
            spaceAfter=22,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=21,
            textColor=ACCENT,
            spaceBefore=20,
            spaceAfter=8,
        ),
        # keepWithNext stops a heading stranding itself at the foot of a page.
        "h2": ParagraphStyle(
            "h2",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=16,
            textColor=INK,
            spaceBefore=13,
            spaceAfter=5,
            keepWithNext=1,
        ),
        "lead": ParagraphStyle(
            "lead",
            parent=base,
            fontSize=12,
            leading=18,
            spaceAfter=12,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base,
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
        ),
        "cell": ParagraphStyle("cell", parent=base, fontSize=9, leading=12.5, spaceAfter=0),
        "cellhead": ParagraphStyle(
            "cellhead",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12.5,
            spaceAfter=0,
            textColor=colors.white,
        ),
        "calloutlabel": ParagraphStyle(
            "calloutlabel",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            spaceAfter=3,
        ),
        "calloutbody": ParagraphStyle(
            "calloutbody",
            parent=base,
            fontSize=9.5,
            leading=14,
            spaceAfter=0,
        ),
    }


class HRule(Flowable):
    """A hairline rule the width of the frame."""

    def __init__(self, thickness: float = 0.6, color: colors.Color = RULE, space: float = 6):
        super().__init__()
        self.thickness = thickness
        self.color = color
        self.space = space
        self.width = 0.0
        self.height = space

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:  # noqa: N803
        self.width = availWidth
        return availWidth, self.space

    def draw(self) -> None:
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space / 2, self.width, self.space / 2)


@dataclass
class Report:
    """Accumulates flowables, then renders to a PDF.

    ``footer`` appears on every page after the first alongside the page number.
    """

    title: str
    subtitle: str = ""
    author: str = ""
    footer: str = ""
    pagesize: tuple[float, float] = LETTER
    story: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.s = _styles()

    # -- text ---------------------------------------------------------------

    def h1(self, text: str) -> None:
        self.story.append(Paragraph(text, self.s["h1"]))
        self.story.append(HRule(space=8))

    def h2(self, text: str) -> None:
        self.story.append(Paragraph(text, self.s["h2"]))

    def para(self, text: str, style: str = "body") -> None:
        self.story.append(Paragraph(text, self.s[style]))

    def lead(self, text: str) -> None:
        self.para(text, style="lead")

    def small(self, text: str) -> None:
        self.para(text, style="small")

    def bullets(self, items: list[str], bullet: str = "•") -> None:
        self.story.append(
            ListFlowable(
                [ListItem(Paragraph(i, self.s["body"]), leftIndent=16) for i in items],
                bulletType="bullet",
                start=bullet,
                bulletFontSize=BODY_SIZE,
                leftIndent=14,
            )
        )
        self.story.append(Spacer(1, 5))

    def spacer(self, height: float = 10) -> None:
        self.story.append(Spacer(1, height))

    @contextmanager
    def keep_together(self) -> Iterator[None]:
        """Group everything emitted inside the block onto one page.

        ``keepWithNext`` on a heading style does not survive a following
        :class:`KeepTogether` (a table), which is exactly the case that strands a
        heading at the foot of a page. This is the explicit remedy::

            with rep.keep_together():
                rep.h2("Heading")
                rep.table(...)

        A block taller than one page falls back to normal splitting rather than
        overflowing.
        """
        outer = self.story
        self.story = []
        try:
            yield
        finally:
            inner, self.story = self.story, outer
            self.story.append(KeepTogether(inner))

    def page_break(self) -> None:
        self.story.append(PageBreak())

    # -- blocks -------------------------------------------------------------

    def callout(self, label: str, body: str, kind: CalloutKind = "note") -> None:
        """A labelled box. Use ``caution`` for assumption-dependent numbers."""
        edge, fill = CALLOUT_COLORS[kind]
        label_style = ParagraphStyle("cl", parent=self.s["calloutlabel"], textColor=edge)
        inner = [
            Paragraph(label.upper(), label_style),
            Paragraph(body, self.s["calloutbody"]),
        ]
        t = Table([[inner]], colWidths=["100%"])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), fill),
                    ("LINEBEFORE", (0, 0), (0, -1), 2.5, edge),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        self.story.append(KeepTogether(t))
        self.story.append(Spacer(1, 10))

    def table(
        self,
        header: list[str],
        rows: list[list[str]],
        col_widths: list[float] | None = None,
        align_right: list[int] | None = None,
        notes: str = "",
    ) -> None:
        """A banded table. ``align_right`` lists 0-based numeric column indexes.

        ``notes`` renders directly beneath, so provenance travels with the data.
        """
        align_right = align_right or []
        data = [[Paragraph(h, self.s["cellhead"]) for h in header]]
        for r in rows:
            data.append([Paragraph(str(c), self.s["cell"]) for c in r])

        t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
        ]
        for i in range(1, len(data)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), BAND))
        for c in align_right:
            style.append(("ALIGN", (c, 1), (c, -1), "RIGHT"))
        t.setStyle(TableStyle(style))

        block: list[Any] = [t]
        if notes:
            block.append(Spacer(1, 4))
            block.append(Paragraph(notes, self.s["small"]))
        self.story.append(KeepTogether(block))
        self.story.append(Spacer(1, 12))

    def definition_list(self, pairs: list[tuple[str, str]]) -> None:
        """Term/definition pairs — the glossary shape."""
        rows = [[f"<b>{term}</b>", body] for term, body in pairs]
        self.table(
            header=["Term", "What it means"],
            rows=rows,
            col_widths=[1.55 * inch, 4.75 * inch],
        )

    def title_page(self, summary: str = "", meta: list[tuple[str, str]] | None = None) -> None:
        self.story.append(Spacer(1, 0.9 * inch))
        self.story.append(Paragraph(self.title, self.s["title"]))
        if self.subtitle:
            self.story.append(Paragraph(self.subtitle, self.s["subtitle"]))
        self.story.append(HRule(thickness=1.4, color=ACCENT, space=16))
        if summary:
            self.story.append(Paragraph(summary, self.s["lead"]))
        if meta:
            self.story.append(Spacer(1, 12))
            for k, v in meta:
                self.story.append(Paragraph(f"<b>{k}:</b> {v}", self.s["small"]))
        self.page_break()

    # -- render -------------------------------------------------------------

    def _decorate(self, canvas: Any, doc: Any) -> None:
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(MUTED)
            canvas.drawString(0.9 * inch, 0.6 * inch, self.footer)
            canvas.drawRightString(self.pagesize[0] - 0.9 * inch, 0.6 * inch, f"Page {doc.page}")
            canvas.setStrokeColor(RULE)
            canvas.setLineWidth(0.4)
            canvas.line(0.9 * inch, 0.78 * inch, self.pagesize[0] - 0.9 * inch, 0.78 * inch)
        canvas.restoreState()

    def build(self, path: Path) -> Path:
        """Render to ``path`` and return it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = BaseDocTemplate(
            str(path),
            pagesize=self.pagesize,
            leftMargin=0.9 * inch,
            rightMargin=0.9 * inch,
            topMargin=0.9 * inch,
            bottomMargin=0.95 * inch,
            title=self.title,
            author=self.author,
            subject=self.subtitle,
        )
        frame = Frame(
            doc.leftMargin,
            doc.bottomMargin,
            doc.width,
            doc.height,
            id="main",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=self._decorate)])
        doc.build(list(self.story))
        return path


def stamp(commit: str = "", now: datetime | None = None) -> str:
    """Footer/provenance string: generation time and, if known, the git commit."""
    when = (now or datetime.now()).strftime("%d %B %Y")
    return f"Generated {when}" + (f" · build {commit}" if commit else "")
