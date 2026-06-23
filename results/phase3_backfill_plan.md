# Phase 3 — GDELT backfill plan

The Phase 3 news corpus (`headlines` table) has a 28-month hole. Before any
clustering can be built (steps 3.7–3.15), the corpus must be backfilled to
continuous coverage. This is the plan; `scripts/backfill_gdelt.py` is the tool.

## Current coverage (the problem)

`headlines` has only **21 months** of data, all in two disconnected pieces:

| Range | Status |
|---|---|
| 2020-01 .. 2021-08 | present (~14.1M rows, continuous) |
| 2021-09 .. 2023-12 | **empty (28 months)** |
| 2024-01            | 32k-row fragment (looks like the end-to-end test pull) |
| 2024-02 .. present | **empty** |
| pre-2020-01        | **empty** |

The continuous window is ~20 months, entirely COVID/post-COVID. Clustering on
that would only rediscover "COVID vs recovery"; the plan's regime sanity-check
(2022 inflation, 2023 banking stress, the 2018–22 palladium squeeze) is mostly
in the empty stretch. The bulk backfill clearly stalled — likely an interrupted
run or a BigQuery quota stop.

`scripts/backfill_gdelt.py` (no credentials needed for this part) reports the
gaps for a target range:

```
$ uv run python scripts/backfill_gdelt.py --start 2015-02-18 --end <today>
Missing months: 116, in 3 gap range(s):
  - 2015-02-01 .. 2019-12-31
  - 2021-09-01 .. 2023-12-31
  - 2024-02-01 .. <today>
```

## Date-range decision

**Pull 2015-02-18 → present.** Rationale:

- GDELT **GKG 2.0** (the `gdelt-bq.gdeltv2.gkg_partitioned` table the fetcher
  uses, with `V2Themes` / `V2Tone`) begins **2015-02-18**. That is the earliest
  date compatible with the existing schema.
- 2015–present covers five of the plan's six regime checks: 2020 COVID, 2022
  inflation, 2023 banking stress, and — crucially — the **2018–22 palladium
  supply squeeze** in full. The two it cannot reach (2011 gold peak, 2013 taper
  tantrum) predate GKG 2.0 and would require GKG 1.0, which has a different schema
  and is out of scope.
- If 2015–2019 is deemed not worth the spend, the **minimum viable** backfill is
  the two gaps that bracket the existing block: **2021-09 → present** (and keep
  the existing 2020–2021 block). That still yields a continuous 2020→present
  window. The 2015–2019 range is the optional extension that buys the palladium
  squeeze and more macro-regime variety.

## Cost

BigQuery on-demand pricing is **$6.25 / TB scanned** (US), with the **first 1 TB
each month free**. The fetcher prunes by `_PARTITIONTIME` and scans only five columns
(`DATE`, `SourceCommonName`, `DocumentIdentifier`, `V2Themes`, `V2Tone`), so a
date-bounded pull scans only that range's partitions.

Rough order of magnitude: the five scanned columns run on the order of
~0.1–0.3 TB per year. So:

| Backfill | Approx. scan | Approx. cost (after 1 TB/mo free) |
|---|---|---|
| 2021-09 → present (~58 mo) | ~0.5–1.5 TB | ~$0–6 |
| + 2015–2019 (full history) | ~1–3 TB total | ~$0–13 |

These are estimates. **The authoritative number comes from a free dry run** —
`--estimate` sums BigQuery's `total_bytes_processed` per chunk without running or
billing anything:

```
$ uv run python scripts/backfill_gdelt.py --estimate          # whole gap set
$ uv run python scripts/backfill_gdelt.py --start 2021-09-01 --estimate   # just the post-2021 gap
```

Spreading the pull across calendar months stretches the 1 TB/month free tier and
can bring the real cost to ~$0.

## Cost guards

- **Dry-run first.** Always `--estimate` before `--execute`.
- **Per-chunk cap.** `--execute` sets `maximum_bytes_billed` per chunk
  (`--max-gb`, default 100 GB). A pathological full-table scan (TBs) aborts
  instead of billing. Legitimate monthly chunks scan only tens of GB.
- **Idempotent.** Months already present are skipped; the upsert is
  `ON CONFLICT`, so re-running is safe and never double-counts.
- **Chunked download.** `--chunk-days` (default 30) keeps each BigQuery result
  set small enough to download reliably via the Storage API.

## Runbook

```bash
# 1. See the gaps (no credentials needed)
uv run python scripts/backfill_gdelt.py --start 2015-02-18

# 2. Estimate the cost to fill them (free dry run; needs GCP creds)
uv run python scripts/backfill_gdelt.py --start 2015-02-18 --estimate

# 3. Fill them, capped (needs GCP creds). Start with the post-2021 gap if you
#    want to keep the first run small:
uv run python scripts/backfill_gdelt.py --start 2021-09-01 --execute --max-gb 100

# 4. Then the optional pre-2020 history:
uv run python scripts/backfill_gdelt.py --start 2015-02-18 --end 2019-12-31 --execute

# 5. Re-check coverage
uv run python scripts/backfill_gdelt.py --start 2015-02-18
```

Record the dry-run bytes and actual cost in `journal.md` (plan step 3.3).

## What this unblocks, and what it does not

Filling the corpus is necessary but not sufficient for good clusters. Three
quality issues (detailed in the Phase 3 review) remain and should be handled
*before* the embedding/clustering steps:

- **GKG stores URLs, not headline text.** Embedding raw URLs is weak signal.
  Build `text_prep.py` (step 3.5) to extract the human-readable URL slug first;
  this recovers most of the lost semantics cheaply.
- **Weak per-metal specificity.** Only `ECON_GOLDPRICE` (~4.4k rows) is
  metal-specific; the corpus is dominated by generic mining + macro themes. Per-
  (date, metal) news vectors will be nearly identical across metals — manage
  expectations for cross-metal differentiation.
- **Coverage density grows over time.** Normalize count / theme-prevalence
  features by daily total article count (plan pitfall 3.x).

## Open decisions

1. Full history (2015→present) vs minimum viable (2021-09→present)? Decide after
   seeing the `--estimate` numbers.
2. Add the Kitco RSS supplement (step 3.4) for real headline text + metals-
   specific commentary, or proceed with GKG URL-slugs only?
