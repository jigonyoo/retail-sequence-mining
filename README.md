# Retail Sequence & Churn-Association Mining

A data-engineering code sample: given a retail transaction log, recover
(1) frequent **sequential purchase patterns**, (2) signup-**cohort
retention curves**, and (3) **churn associations** — which prior
sequences or RFM buckets co-occur with customer churn, with counts and
confidence intervals attached.

**Positioning: this is a data engineering sample, not a causal-inference
or ML product.** Every association reported here is correlational. See
[Limitations](#limitations) below before reading too much into any
single number.

## What it does

1. **`mining/sequences.py`** — builds one ordered event sequence per
   customer from the transaction log, then mines frequent ordered
   subsequences (length 1–3) using an explicit, small-alphabet,
   Apriori-style prefix-extension approach. Support = number of
   distinct customers whose history contains the pattern in order (not
   necessarily contiguous).
2. **`mining/cohorts.py`** — groups customers into signup-month
   cohorts and computes month-by-month retention, correctly excluding
   cohort-months that haven't happened yet (instead of miscounting them
   as churned).
3. **`mining/churn.py`** — labels each customer churned/active using a
   fixed inactivity-window rule, then measures the association between
   churn and (a) frequent sequence patterns and (b) frequency/monetary
   RFM buckets. Every association reports support, churn rate with vs.
   without the trait (each with a 95% Wilson confidence interval), and
   lift.
4. **`mining/stats.py`** — guard rails: a minimum-support threshold so
   thin-evidence groups are omitted rather than reported, and Wilson
   score intervals so small groups' rates come with visibly wide
   uncertainty bands instead of a bare, over-confident percentage.
5. **`mining/report.py`** — renders the above into Markdown + CSV.
6. **`data/generate_transactions.py`** — a deterministic synthetic
   transaction-log generator with a planted signal (customers who
   purchase category A then B are more likely, though not certain, to
   churn afterward) plus noise, so the pipeline has something real to
   recover.

## Quickstart

```bash
python3 run_demo.py
```

Writes `sample_output/mining_report.md`, `sequences.csv`,
`churn_associations.csv`, and `run_summary.txt`. Re-running is always
safe: the script never deletes `sample_output/`, it only creates it
(`os.makedirs(exist_ok=True)`) and overwrites individual files.

Run it twice and diff — output is byte-identical:

```bash
python3 run_demo.py
md5sum sample_output/*
python3 run_demo.py
md5sum sample_output/*   # same hashes
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

34 tests covering: sequence extraction and tie-breaking, subsequence
containment, support counting, min-support filtering, cohort
construction and retention math (including the "not yet observed"
edge case), churn boundary conditions (exact-window vs. one-day-short),
RFM bucket boundaries, Wilson interval behavior (including that it
widens for small n), association guard-rail behavior, recovery of the
planted A→B→churn signal from the synthetic generator, and two flavors
of determinism (generator output and full pipeline output, both
md5-compared across repeated runs).

## Docker

```bash
docker compose up --build
```

Runs the test suite then the demo inside a container with
`network_mode: none` — there is nothing in this project that needs
network access, so the compose file enforces that at the container
level as well as in code.

## Project layout

```
mining/               core library (stdlib only)
  sequences.py         sequential pattern mining
  cohorts.py            cohort retention
  churn.py               churn labeling + association analysis
  stats.py                 Wilson intervals, min-support guard
  report.py                  Markdown/CSV rendering
  run.py                       pipeline orchestration
  util.py                        shared I/O + date helpers
data/
  generate_transactions.py   deterministic synthetic data generator
  synthetic/                    (generated at runtime, gitignored)
tests/
  test_mining.py              15+ unittest tests
sample_output/                 output of `python3 run_demo.py`
run_demo.py
SCHEMA.md
requirements.txt (stdlib only, no third-party deps)
Dockerfile / docker-compose.yml (network_mode: none)
LICENSE (MIT)
```

## Limitations

- **Synthetic data.** The transaction log is generated, not observed.
  The planted A→B→churn signal and its magnitude were chosen by the
  generator's author, not discovered in a real business — this
  demonstrates that the *pipeline* correctly recovers a known signal
  from noisy data, not that the signal itself is realistic.
- **Correlational only, not causal.** Every number in
  `churn_associations.csv` and the report's churn-association table
  describes co-occurrence within this historical log. None of it
  establishes that a purchase sequence or an RFM bucket *causes*
  churn — a shared underlying cause (e.g. a product-quality issue, a
  pricing change, a competitor promotion) could produce the same
  pattern without any of the measured traits being the driver.
- **RFM buckets can be partially mechanical.** A churned customer, by
  construction, stops transacting earlier — so "low frequency" and
  "low monetary value" are partly *definitionally* entangled with
  churn, not purely behavioral predictors of it. Recency-based buckets
  are excluded entirely from the association analysis for this reason
  (recency is literally part of the churn definition); frequency and
  monetary buckets are kept but should be read with this caveat in
  mind — they're the weakest evidence of "prediction" in this report,
  even though their lift numbers look the largest.
- **Small groups are guarded, not hidden well.** Groups below the
  minimum support threshold are dropped from the association table
  entirely. Groups that pass the threshold but are still small will
  show a wide Wilson interval — treat a wide interval as "not enough
  data to act on," not as a green light because the point estimate
  looks dramatic.
- **Sequential pattern mining is intentionally simple.** This is a
  small-alphabet (5 symbols), explicit, Apriori/prefix-extension
  implementation capped at length 3, built for clarity over
  performance. It has no timing-window constraints (e.g. "B within 14
  days of A"), no gap constraints, and would not scale as-is to a
  large product catalog or very long customer histories — a
  production system would want PrefixSpan/SPADE or a windowed variant.
- **Churn is one fixed definition.** "Churned" here means "no
  transaction for N days as of the observation cutoff," a common but
  simplistic operationalization. Real churn programs typically use
  business-specific definitions (e.g. subscription lapses, contract
  non-renewal) that this generic inactivity rule does not capture.
- **No causal design, no experiment.** Nothing here randomizes
  treatment or holds anything constant. Before acting on any
  association surfaced by this pipeline (e.g. targeting A→B customers
  with a retention campaign), run a controlled experiment (e.g. an
  A/B test withholding the intervention from a comparable holdout) to
  confirm the pattern actually predicts — let alone causes — the
  outcome you care about.

## About this sample

This is sample #2 in a small "retail data mining" portfolio set. Sample
#1 covers association-rule mining and RFM segmentation as a snapshot;
this one adds the *time/sequence* dimension (order matters) and a
*churn* outcome tied to specific prior behavior.
