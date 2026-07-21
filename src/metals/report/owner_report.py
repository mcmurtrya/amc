"""The AMC owner's briefing: research findings in business language.

This module holds *wording*, not analysis. Every empirical number is quoted from
a committed write-up in ``results/`` (cited inline in ``FINDINGS`` below) or read
live from DuckDB via :mod:`metals.report.facts`. Nothing is recomputed here, so
this file can never disagree with the research by accident — only by a stale
citation, which the source constants make easy to audit.

Editorial rules, which exist because the reader is a business owner acting on
this and cannot audit it himself:

1. **No finding appears without its caveat.** The caveat renders in the same
   visual block, not a footnote.
2. **Negative results get equal billing.** Four of this project's most valuable
   results are things that did not work; presenting only the positive finding
   would misrepresent the state of knowledge.
3. **Uncalibrated numbers are labelled uncalibrated,** in the same table cell
   where they appear. The spread-floor figures are placeholders and the document
   says so wherever they are shown.
4. **Plain words.** No "IC", "RMSE", "local projection", or "p-value" outside
   the glossary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from metals.report import facts
from metals.report.pdf import Report, stamp

TITLE = "What the Research Says About Your Metal Risk"
SUBTITLE = "A plain-language briefing for AMC Company"

# Human-readable expansions of the spread-floor engine's honesty flags. Each
# says what the term currently rests on, in the owner's terms.
FLAG_MEANINGS: dict[str, str] = {
    "vol=implied": "<b>Gold</b> uses a forward-looking price-swing estimate taken "
    "from the gold options market.",
    "vol=realized_downside": "<b>Silver, platinum and palladium</b> use a backward-"
    "looking estimate taken from each metal's own recent declines — no equivalent "
    "options market exists for them.",
    "tail=normal_approx": "Bad-case size uses a textbook bell-curve rule, which "
    "understates genuine panics.",
    "float=assumed": "Holding period is ASSUMED at 10 trading days. Not measured "
    "from your books, because we do not have them yet.",
    "carry=rf_only": "Financing cost counts the Treasury rate only — not your "
    "insurance, labour, or shop overhead.",
    "exit=fixed_haircut": "Exit price is a placeholder haircut, not your actual "
    "refiner payable or dealer bid.",
}


@dataclass(frozen=True)
class Finding:
    """One result, with the caveat that must always travel with it."""

    headline: str
    plain: str
    numbers: str
    confidence: str
    caveat: str
    source: str


FINDINGS: tuple[Finding, ...] = (
    Finding(
        headline="A hawkish Fed surprise pushes metal prices down for about a week",
        plain=(
            "When the Federal Reserve surprises the market by sounding tougher on "
            "interest rates than expected, gold, silver and platinum reliably fall "
            "over the following week. This is the one result we would put real "
            "weight behind."
        ),
        numbers=(
            "Five trading days after a hawkish surprise: gold −1.5%, silver −3.0%, "
            "platinum −1.7%, palladium −1.6% (palladium not statistically reliable). "
            "Over twenty days the moves roughly double: gold −1.8%, silver −3.7%, "
            "platinum −3.0%."
        ),
        confidence=(
            "Highest. Three independent methods agree, including one that never "
            "looks at Fed meeting dates at all. Holds in all three time periods we "
            "split the data into. Shuffling the event dates at random destroys the "
            "effect, as it should."
        ),
        caveat=(
            "The effect has weakened by roughly 2.3× since the money-printing era — "
            "a surprise today moves prices less than the same surprise in 2010. "
            "Based on 35 events. And critically: we could NOT re-test this on recent "
            "data, because the dataset measuring Fed surprises stops in December "
            "2023. The entire 2024–2026 rate-cutting cycle is untested."
        ),
        source="results/phase5_triangulation.md, results/phase6_findings.md",
    ),
    Finding(
        headline="Metals punish bad Fed news far harder than they reward good Fed news",
        plain=(
            "A tough Fed surprise hurts. A friendly Fed surprise barely helps. Metal "
            "behaves like insurance: it reacts sharply when the reason to hold it is "
            "threatened, and only mildly when that reason is reaffirmed."
        ),
        numbers=(
            "Hawkish surprise: gold −1.4% over five days. Dovish (friendly) surprise: "
            "gold +0.6%, and not statistically distinguishable from zero. No dovish "
            "result for any metal reached significance."
        ),
        confidence="High. Both main methods agree on the asymmetry.",
        caveat=(
            "Our sample period (2010–2023) was tilted toward tightening cycles, so "
            "some of the muted upside may reflect that easy money was already priced "
            "in rather than a permanent behavioural rule."
        ),
        source="results/phase2_scenarios.md, results/phase6_findings.md",
    ),
    Finding(
        headline="Silver is your most volatile exposure; gold is your steadiest",
        plain=(
            "When a Fed shock hits, silver swings hardest, platinum next, gold and "
            "palladium least. This ranking held up across every method and every time "
            "period we tested — it is one of the most stable patterns in the work."
        ),
        numbers=(
            "Silver moves roughly twice as far as gold on the same shock "
            "(−3.0% vs −1.5% at five days)."
        ),
        confidence="High. Ordering preserved across both estimators and all three eras.",
        caveat=(
            "Palladium is the unreliable member of the group: its response to Fed news "
            "collapses to near zero after 2015, almost certainly because the 2018–2022 "
            "industrial supply squeeze overwhelmed anything the Fed was doing."
        ),
        source="results/phase6_findings.md",
    ),
)

NULLS: tuple[Finding, ...] = (
    Finding(
        headline="Free news-mood tracking does not help predict prices — it hurts",
        plain=(
            "We collected 139.9 million news articles and built market-mood measures "
            "from them. They did not improve volatility forecasts. On fresh data they "
            "made forecasts significantly WORSE. We set the pass mark before running "
            "the test, and the test failed it."
        ),
        numbers=(
            "The pre-set bar was a 1.0% accuracy improvement and wins in 60% of test "
            "periods. Actual: 0.37% WORSE, winning 4 of 11. Two separate approaches were "
            "then re-tested on the untouched hold-out data, and both degraded accuracy "
            "by a statistically significant margin: the news-mood measure, and a "
            "market-regime measure that sorts each day into a “type of market” using "
            "price behaviour, economic conditions and news together."
        ),
        confidence=(
            "High, and unusually clean: the standard was fixed in advance, then "
            "independently reconfirmed on data never used in development."
        ),
        caveat=(
            "Be precise about what was tested. The mood score is a free, general-purpose "
            "one that counts positive and negative words against a standard dictionary. "
            "It is not finance-specific, it reads whole articles rather than headlines, "
            "and it produces one market-wide reading per day rather than a separate one "
            "per metal. So the finding is that THIS measure does not forecast — not that "
            "no news measure could. Nor is any of it worthless: the market-regime "
            "grouping fails at forecasting, yet is genuinely useful for explaining why a "
            "past move was unusually large. Prediction and explanation are different jobs."
        ),
        source=(
            "results/phase3_writeup.md, results/phase6_validation.md, "
            "results/amc_paid_data_review.md"
        ),
    ),
    Finding(
        headline="The machine-learning model did not beat old-fashioned statistics",
        plain=(
            "On data the models had never seen, two decades-old statistical methods "
            "produced lower forecast errors than the machine-learning model. We are "
            "reporting the simpler tools as the better ones."
        ),
        numbers=(
            "Forecast error on the hold-out: classical methods 0.127 and 0.131, "
            "machine learning 0.140. A naive guess was worst at 0.186 — so the "
            "modelling is doing something, just not better than the simple version."
        ),
        confidence=(
            "Moderate. The test window contains only about two or three genuinely "
            "independent observations, which is too few to settle the question."
        ),
        caveat=(
            "No model was tuned, by design, to keep the comparison fair. A tuned "
            "machine-learning model might do better — we have not tested that."
        ),
        source="results/phase6_validation.md",
    ),
    Finding(
        headline="“A weak dollar is good for metals” did not survive testing",
        plain=(
            "This common rule of thumb came out backwards in our data, and then "
            "behaved even more strangely on fresh data. We recommend not trading on "
            "it in this simple form."
        ),
        numbers=(
            "Sharp dollar declines were followed by metal prices FALLING, not rising "
            "(platinum −1.6%, palladium −2.2%). On fresh data the direction repeated "
            "but the size was 5 to 90 times too large — a warning sign, not a "
            "confirmation."
        ),
        confidence=(
            "We are confident the simple version is broken, and we understand why: "
            "big dollar moves cluster inside panics, so the measure picks up the "
            "panic rather than the currency."
        ),
        caveat=(
            "A properly isolated dollar effect DOES exist and points the textbook way. "
            "The problem is the crude trigger, not the underlying economics."
        ),
        source="results/phase2_scenarios.md, results/phase6_findings.md",
    ),
    Finding(
        headline="Geopolitical risk headlines are not a usable gold signal",
        plain=(
            "The belief that spiking geopolitical tension reliably lifts gold did not "
            "hold. The index we used cannot tell a genuine crisis from a merely noisy "
            "news week."
        ),
        numbers=(
            "Across 207 spike events, gold's five-day move was +0.05% — indistinguishable "
            "from nothing. On fresh data the signal was a coin flip: two of four metals "
            "moved the predicted way."
        ),
        confidence="High that the signal as constructed is unusable.",
        caveat=(
            "Flight-to-safety is real — we can see it when we identify crises by market "
            "behaviour instead of headline counts. The measuring instrument was at fault."
        ),
        source="results/phase2_scenarios.md, results/phase6_validation.md",
    ),
)

GLOSSARY: tuple[tuple[str, str], ...] = (
    (
        "Hawkish surprise",
        "The Fed signals tighter money — higher rates, or rates "
        "staying high longer — than the market expected that morning.",
    ),
    ("Dovish surprise", "The opposite: easier money than expected."),
    (
        "Float",
        "How long you own metal between buying it and disposing of it. Risk "
        "grows with the square root of this period, so doubling it raises required "
        "cushion by about 41%, not 100%.",
    ),
    (
        "Hold-out test",
        "Data locked away and never looked at while building the "
        "models, then used once at the end as a closed-book exam.",
    ),
    (
        "Pre-registered test",
        "Writing down the pass mark before running the test, so "
        "the result cannot be rationalised afterwards.",
    ),
    ("Spot", "The live market price per troy ounce of pure metal."),
    ("Payable", "The share of metal value your refiner actually pays you."),
)


def _source_list() -> str:
    """Distinct write-ups behind the findings, one entry each.

    ``Finding.source`` may name several files, so split before deduplicating —
    otherwise combined strings and their components both survive as "distinct".
    """
    files: set[str] = set()
    for f in FINDINGS + NULLS:
        files.update(part.strip() for part in f.source.split(",") if part.strip())
    return ", ".join(sorted(files))


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _floor_rows(df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                str(r["metal"]).capitalize(),
                f"${r['spot_usd_oz']:,.2f}",
                f"${r['max_buy_usd_oz']:,.2f}",
                _pct(float(r["max_buy_frac"])),
                _pct(1 - float(r["max_buy_frac"])),
            ]
        )
    return rows


def _flag_explanations(df: pd.DataFrame) -> list[str]:
    """Distinct honesty flags across the live rows, in plain English.

    Ordered by :data:`FLAG_MEANINGS` rather than by row, so related items (the
    two volatility sources) sit together regardless of which metal sorts first.
    Unrecognised flags are surfaced verbatim rather than dropped — an unexplained
    flag in the document is a prompt to add wording, whereas a silently missing
    one would hide a caveat.
    """
    seen: set[str] = set()
    for raw in df["flags"]:
        seen.update(str(raw).split("|"))
    known = [FLAG_MEANINGS[f] for f in FLAG_MEANINGS if f in seen]
    unknown = sorted(f for f in seen if f not in FLAG_MEANINGS)
    return known + unknown


def build(out_path: Path, now: datetime | None = None) -> Path:
    """Assemble the owner briefing and write it to ``out_path``."""
    commit = facts.git_commit()
    ledger = facts.ledger_status()
    floors = facts.latest_spread_floors()
    first, last, n_price = facts.price_coverage()
    generated = stamp(commit, now=now)

    rep = Report(
        title=TITLE,
        subtitle=SUBTITLE,
        author="Metals research programme",
        footer=f"AMC Company — confidential · {generated}",
    )

    # -- cover -------------------------------------------------------------
    rep.title_page(
        summary=(
            "This briefing explains what a multi-year study of precious-metals price "
            "behaviour found, what it failed to find, and what both mean for how AMC "
            "buys and carries metal. It is written to be read start to finish in about "
            "fifteen minutes, with no statistical background assumed."
        ),
        meta=[
            ("Prepared for", "The owner, AMC Company"),
            ("Scope", "Gold, silver, platinum, palladium — scrap intake and coin trading"),
            ("Price data behind it", f"{n_price:,} daily records, {first} to {last}"),
            ("News data behind it", f"{facts.headline_count():,} articles"),
            ("Model runs logged", f"{facts.model_run_count():,}"),
            ("Status", generated),
        ],
    )

    # -- the short version -------------------------------------------------
    rep.h1("If you read nothing else")
    rep.bullets(
        [
            "<b>One reliable warning signal exists.</b> When the Fed surprises the "
            "market by sounding tougher on rates, metal prices fall over the next week "
            "or so — silver about twice as hard as gold. This is the single result that "
            "survived every test we threw at it.",
            "<b>Most popular market wisdom did not survive.</b> “Weak dollar lifts "
            "metals” and “geopolitical tension lifts gold” both failed. "
            "Knowing which rules of thumb are false is worth as much as the one that is "
            "true, because it stops you acting on them.",
            "<b>Free news-mood tracking does not predict prices.</b> We tested this "
            "properly, with the pass mark set in advance, on 139.9 million articles. It "
            "failed, and on fresh data it actively made forecasts worse. We also "
            "recommend against buying a paid news-sentiment subscription — but on "
            "separate grounds: those products cost five figures a year, and their "
            "licences bar commercial use by a business like yours.",
            "<b>Simple beats sophisticated.</b> On data the models had never seen, "
            "decades-old statistical methods forecast better than the machine learning. "
            "We are recommending the simpler tool.",
            "<b>The biggest gap is your own books.</b> Everything above describes the "
            "market. Converting it into dollars of risk on YOUR inventory requires your "
            "transaction ledger, which we do not yet have. That is the single highest-value "
            "thing you can provide.",
        ]
    )

    rep.callout(
        "The honest headline",
        "This programme produced one durable trading-relevant fact, several useful "
        "refutations, and a clear-eyed account of what it could not establish. If a "
        "research report ever tells you everything worked, be suspicious of it.",
        kind="note",
    )

    rep.page_break()

    # -- what holds up ------------------------------------------------------
    rep.h1("What holds up")
    rep.para(
        "Each finding below survived multiple independent methods. The caveat printed "
        "with each one is not boilerplate — it is the specific condition under which "
        "the finding stops being true."
    )
    for f in FINDINGS:
        rep.h2(f.headline)
        rep.para(f.plain)
        rep.para(f"<b>The numbers.</b> {f.numbers}")
        rep.para(f"<b>How confident.</b> {f.confidence}")
        rep.callout("Caveat that travels with this", f.caveat, kind="caution")

    rep.page_break()

    # -- what failed --------------------------------------------------------
    rep.h1("What failed, and why that is worth money to you")
    rep.para(
        "A study that only reports what worked is selling you something. These are the "
        "ideas we tested and rejected. Each one represents money or attention you no "
        "longer need to spend."
    )
    for f in NULLS:
        rep.h2(f.headline)
        rep.para(f.plain)
        rep.para(f"<b>The numbers.</b> {f.numbers}")
        rep.para(f"<b>How confident.</b> {f.confidence}")
        rep.callout("Important nuance", f.caveat, kind="note")

    rep.page_break()

    # -- the tool -----------------------------------------------------------
    rep.h1("The buy-spread tool, and why you cannot use it yet")
    rep.para(
        "The practical product of this work is a calculator that answers your daily "
        "question: <i>how far below spot must I buy so that a bad week does not turn "
        "this lot into a loss?</i> It works from the exit price you can realise, minus "
        "a cushion for how far the metal could fall while you hold it, minus your cost "
        "of carrying it."
    )

    if floors.empty:
        rep.callout(
            "Not yet computed",
            "The spread-floor engine has not been run against this database, so no "
            "figures can be shown. Re-run it and regenerate this briefing.",
            kind="blocked",
        )
    else:
        as_of = pd.Timestamp(floors.iloc[0]["date_utc"]).strftime("%d %B %Y")
        rep.callout(
            "Do not quote these numbers to a customer",
            "The figures below are placeholders, not prices. Every one of them rests "
            "on an assumed two-week holding period and a placeholder exit price, "
            "because your ledger and refiner settlements are not yet loaded. They are "
            "shown so you can see the machinery working and judge whether the shape is "
            "right — they are deliberately too conservative, and using them as quotes "
            "would cost you deals.",
            kind="blocked",
        )
        rep.table(
            header=[
                "Metal",
                "Spot price",
                "Maximum buy",
                "% of spot",
                "Discount",
            ],
            rows=_floor_rows(floors),
            align_right=[1, 2, 3, 4],
            notes=(
                f"Computed {as_of}, the last date with complete price data. "
                "Placeholder values — see the caution above."
            ),
        )
        rep.h2("What each number currently rests on")
        rep.bullets(_flag_explanations(floors))
        rep.para(
            "Read that list as a to-do rather than a disclaimer. Every item on it "
            "tightens once real data replaces the placeholder — and a tighter floor "
            "means you can bid more competitively without taking on more risk."
        )

    with rep.keep_together():
        rep.h2("What would make these numbers real")
        rep.table(
            header=["What we need", "What it fixes", "Where it comes from"],
            rows=[
                [
                    "Your scrap ledger — purchase and disposal dates, weights, "
                    "fineness, price paid, proceeds",
                    "Replaces the assumed two-week holding period with your actual "
                    "one. This is the single largest source of excess conservatism.",
                    "Your bookkeeping system. Template and importer are already built and tested.",
                ],
                [
                    "Refiner settlement statements",
                    "Replaces the placeholder exit price with your true payable.",
                    "Your refiner invoices.",
                ],
                [
                    "Your coin buy and sell records",
                    "Measures the premium you actually realise, rather than a "
                    "published benchmark that may not reflect your market.",
                    "Your bookkeeping system.",
                ],
            ],
            col_widths=[2.0 * 72, 2.4 * 72, 1.9 * 72],
        )

    rep.page_break()

    # -- what we need -------------------------------------------------------
    rep.h1("What we need from you")
    if ledger.populated:
        rep.callout(
            "Ledger received",
            f"Your books are loaded: {ledger.scrap_lots:,} scrap lots, "
            f"{ledger.coin_trades:,} coin trades, {ledger.till_days:,} daily counts. "
            "The placeholder holding period can now be replaced with your measured one.",
            kind="good",
        )
    else:
        rep.callout(
            "Nothing received yet — this is the bottleneck",
            "We hold zero rows of your transaction data. The import tool, the file "
            "format, and the validation checks are all built and tested; they are "
            "waiting on a first export from your bookkeeping. Until then every "
            "dollar-denominated figure in this briefing rests on an assumption rather "
            "than on your business.",
            kind="blocked",
        )
    rep.para(
        "Three practical points about that data. <b>First, it never leaves the "
        "machine.</b> Your books are held on one local computer, are never uploaded to "
        "any cloud service, and are excluded from any copy that goes off-site. "
        "<b>Second, it does not need to be clean.</b> The importer checks every row and "
        "reports every problem at once, and corrected files can simply be re-imported. "
        "<b>Third, a partial history is still useful</b> — even a few hundred closed "
        "lots would replace the assumed holding period with a measured one."
    )

    rep.h2("What you would get back, quickly")
    rep.bullets(
        [
            "How long each metal actually sits in your inventory, which is the number "
            "that sets how wide your buying spread must be.",
            "Your realised margin per lot, by metal and by lot size.",
            "The premium you genuinely capture on coins, by product — measured, not benchmarked.",
            "A dollar figure for how much a bad week costs your current book, which is "
            "the number that tells you whether hedging is worth its cost.",
        ]
    )

    # -- limits -------------------------------------------------------------
    rep.h1("What this work cannot tell you")
    rep.bullets(
        [
            "<b>It does not predict prices.</b> No part of this study forecasts the "
            "direction of gold. That is not a shortcoming of the effort; it is what the "
            "evidence supports.",
            "<b>The main finding is untested on the last two years.</b> The dataset "
            "measuring Fed surprises ends in December 2023, so the whole 2024–2026 "
            "rate-cutting cycle is unexamined. This is the most valuable gap to close.",
            "<b>Palladium is the weakest case throughout.</b> Its industrial supply "
            "story overwhelms the financial patterns that fit the other three metals. "
            "Treat palladium conclusions with extra caution.",
            "<b>The hold-out test was small.</b> About two or three genuinely "
            "independent observations. It was enough to rule things out, not enough to "
            "confirm fine distinctions.",
            "<b>Some data collection is paused on legal grounds.</b> Several automated "
            "price and premium feeds were found to prohibit commercial use, so they "
            f"were stopped and {facts.quarantined_rows():,} already-collected rows were "
            "quarantined out of all analysis pending a licence. Nothing in this briefing "
            "relies on them.",
        ]
    )

    rep.h1("Terms used in this briefing")
    rep.definition_list(list(GLOSSARY))

    rep.spacer(6)
    rep.small(
        "Every figure in this document is either quoted from a dated internal write-up "
        "or read directly from the research database at the moment of generation. "
        f"{generated}. Findings sourced from: " + _source_list() + "."
    )

    return rep.build(out_path)
