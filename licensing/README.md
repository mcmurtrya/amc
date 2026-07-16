# Data-source licence requests

Drafted 2026-07-16 after the Terms-of-Use audit of the Phase 7.1 collectors
(journal.md, same date) found all four live sources bar AMC's use as built. Three
of the four have a realistic consent or licensing path; this folder holds the
drafts to send.

These are **drafts for a human to send** from an AMC address — not automated, and
not to be sent by any tooling. Fill the bracketed placeholders (sender name, AMC
contact details, account/subscriber IDs where relevant) before sending. Sender
identity matters: each ask rests on AMC being a real, identifiable commercial
dealer making a narrow internal-research request.

| # | Source | Collector | Ask | Cost if granted |
|---|--------|-----------|-----|-----------------|
| 1 | Greysheet / CDN Publishing | replaces `coin_premiums` | Subscribe; confirm the API licence permits storage + modelling | $299/yr (already budgeted) |
| 2 | Fair Economy, Inc. (ForexFactory) | `consensus` | Written consent to retain a 6-field slice, internal only | $0 |
| 3 | Johnson Matthey Plc | `jm_pgm` | Confirm non-benchmark internal analytical use is permitted | $0 |

Not included: APMEX and JM Bullion (the `coin_premiums` dealers). Their asks are
for *dealer-specific posted asks*, which Greysheet does not replace; both ToUs
contemplate written consent, so those requests are worth making too, but they are
lower priority than confirming the Greysheet licence — draft them if the panel's
dealer-specific construct turns out to matter more than CDN's retail benchmark.

Also not here: **Google Trends** (`trends` collector) — the fourth barred source
needs no licence. Google grants use of the data; only the scraper's transport was
non-compliant, so it was rewritten as an importer of Google's sanctioned manual
CSV export (done 2026-07-16, `src/metals/data/trends.py`). No email required.

**Before sending #1:** read the Greysheet **API Terms of Use and License
Agreement** for the same three properties that sank the scrape — commercial use,
storage/caching into a local database, and model training. A paid API is not
automatically clear on any of them (that was the CME lesson: a licence can permit
access while barring the archive and the model). Email #1 makes those three
questions explicit so the answer is in writing before money changes hands.
