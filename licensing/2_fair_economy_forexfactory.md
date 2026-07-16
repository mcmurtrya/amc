# Fair Economy, Inc. (ForexFactory) — consent to retain a calendar-consensus slice

**To:** Fair Economy, Inc. — via forexfactory.com/contact (the site's contact form)
**From:** [your name], AMC Company — [email]
**Subject:** Written-consent request: internal research use of a small consensus slice from the weekly calendar feed

---

Hello,

I do quantitative research for AMC Company, a precious-metals dealer in the US. We
study how scheduled US data releases move metals prices, to inform our own hedging
and inventory timing.

For that work I've been using the published **weekly calendar export**
(`nfs.faireconomy.media/ff_calendar_thisweek.json`, the JSON linked from the
calendar's own Weekly Export panel). We access it through that published export
only — with an honestly identified client, well within your stated download
limits — and we do not touch any internal endpoint.

I'm writing because I understand from your notices page that copying or retaining
FEED content requires prior written consent, and I'd like to ask for it rather
than assume it. Our use is deliberately narrow:

- We keep **only** a small slice of each week's US releases — for a handful of
  events (e.g. CPI, the Employment Situation), we record the **release date/time,
  the event name, and the forecast (consensus) and previous values**. Nothing
  else from FEED (no descriptions, specs, links, historic tables, impact ratings,
  news, forum content).
- It is stored in our **internal** database and used solely for our own analysis.
  We do **not** republish, redistribute, resell, or display it to anyone outside
  AMC.
- Frequency: one pull per week (we can align to whatever cadence you prefer).

Would Fair Economy be willing to grant written consent for that limited internal
use? If it helps, I'm happy to work within specific conditions — a capped field
set, an agreed cadence, attribution, or a short data-use letter on your terms.

If this isn't something you permit, I understand, and I'll stop retaining the data
and find another source for the consensus figures.

Thank you for considering it,
[your name]
AMC Company
[email]

---

*Internal notes (delete before sending):*
- *Verdict was BARRED on one clause only: FEED's "copying ... in part or in whole
  ... is explicitly prohibited," where FEED is defined to include event names,
  release datetimes, and assembled data. There is NO non-commercial limit and NO
  ML clause here — "Business Use" is expressly contemplated — so the only thing in
  the way is consent to retain, which is exactly what this asks for. Good odds.*
- *FEI is a ~13-person Tampa company; a narrow, polite, specific ask to a small
  publisher is the right register. Governing law is Florida.*
- *`macro_consensus` keys on `consensus_source`, so if consent is granted, nothing
  in the schema changes — just un-skip at weekly cadence and record the grant in
  results/amc_paid_data_review.md.*
- *Parallel fallbacks that don't need this consent, worth starting regardless: the
  Cleveland Fed inflation nowcast (vintaged, daily) covers the CPI half cleanly;
  Trading Economics has a free trial to TEST whether its point-in-time consensus
  timestamps genuinely precede releases (the FXMacroData test). NFP consensus is
  the irreducible gap if both this and TE fall through.*
