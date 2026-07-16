# Greysheet / CDN Publishing — subscription + API licence clarification

**To:** CDN Publishing sales / support (via greysheet.com/contact, or the sales
contact on greysheet.com/publications/api-pricing)
**From:** [your name], AMC Company — [email], [phone]
**Subject:** Coin Dealer Digital subscription + CDN Public API terms question (commercial dealer)

---

Hello,

I run pricing and inventory research for AMC Company, a precious-metals dealer
(scrap gold/silver/platinum/palladium plus gold coin and specie). I'd like to
subscribe to Coin Dealer Digital and use the **CDN Public API V2** to pull CPG
retail values and Greysheet wholesale bid/ask for a fixed basket of bullion and
generic products, once daily, for our own internal pricing and analysis.

Before subscribing I want to be sure our intended use is within the API licence,
because it goes a little beyond simple display. Specifically, we would:

1. **Store** each daily pull in our own internal database, building a historical
   time series (kept for internal reference; **not** redistributed, resold, or
   shown to anyone outside AMC).
2. **Use** that series as an input to internal statistical/quantitative models
   that inform our buy/sell spreads and inventory decisions.

Could you confirm whether the CDN Public API Terms of Use and License Agreement
permit those two uses — internal storage of a historical series, and use of the
data in internal models — under the Coin Dealer Digital tier? If they require a
higher tier (I've seen the Advanced API referenced) or a separate data-licensing
arrangement, I'd be glad to be pointed to the right one and its pricing.

A few specifics that may help you route this:

- Basket is roughly [6–12] products; one pull per day; no redistribution.
- We do not need real-time or intraday data — end-of-day values are fine.
- If it's easier to answer against the actual agreement, a copy of the API Terms
  of Use / License Agreement would be welcome and I'll confirm our use against it.

Thanks very much — happy to set up the subscription as soon as the licence point
is clear.

Best regards,
[your name]
AMC Company
[email] · [phone]

---

*Internal notes (delete before sending):*
- *This is the licensed replacement for the barred `coin_premiums` scraper. Caveat
  to hold in mind: CPG retail is CDN's published benchmark, NOT APMEX's/JM
  Bullion's posted asks — so this changes the panel's construct from
  dealer-specific spreads to a retail benchmark. Fine for spread-floor grounding,
  but note it in the plan rather than silently substituting.*
- *The two questions above (storage, modelling) are the exact two the CME episode
  taught us to get in writing before paying. Do not subscribe until they're
  answered — a paid API can still bar caching or training.*
- *Cost: $299/yr Coin Dealer Digital, already budgeted in
  results/amc_paid_data_review.md. Confirm whether Basic tier includes API access
  or whether Advanced (Dealer+/Pro) is required — pricing unconfirmed.*
