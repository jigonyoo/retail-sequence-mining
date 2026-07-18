"""Render pipeline results to Markdown (report) and CSV (data tables).

All rendering here is pure string formatting from already-computed,
already-sorted data structures -- no fresh randomness, no timestamps,
so two runs on the same input produce byte-identical files.
"""
from __future__ import annotations

from typing import List

from .util import write_csv


def _fmt_pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%"


def _fmt_rate4(x) -> str:
    return f"{x:.4f}"


def _fmt_lift(x) -> str:
    if x is None:
        return "n/a (baseline group has zero churn)"
    return f"{x:.2f}x"


def write_sequences_csv(path: str, frequent_patterns: List[tuple]) -> None:
    """frequent_patterns: list of (pattern_tuple, support_count), already
    sorted by the caller (descending support, then pattern).
    """
    rows = []
    for pattern, support in frequent_patterns:
        rows.append([len(pattern), " -> ".join(pattern), support])
    write_csv(path, ["length", "pattern", "support_customers"], rows)


def write_churn_associations_csv(path: str, associations: List) -> None:
    rows = []
    for a in associations:
        rows.append(
            [
                a.label,
                a.support,
                a.n_total,
                a.n_with,
                a.churn_with,
                _fmt_rate4(a.rate_with),
                _fmt_rate4(a.wilson_with.lower),
                _fmt_rate4(a.wilson_with.upper),
                a.n_without,
                a.churn_without,
                _fmt_rate4(a.rate_without),
                _fmt_rate4(a.wilson_without.lower),
                _fmt_rate4(a.wilson_without.upper),
                ("n/a" if a.lift is None else f"{a.lift:.4f}"),
                a.note,
            ]
        )
    write_csv(
        path,
        [
            "group_label",
            "support_count",
            "n_total_customers",
            "n_with_trait",
            "churned_with_trait",
            "churn_rate_with",
            "churn_rate_with_wilson_lower",
            "churn_rate_with_wilson_upper",
            "n_without_trait",
            "churned_without_trait",
            "churn_rate_without",
            "churn_rate_without_wilson_lower",
            "churn_rate_without_wilson_upper",
            "lift",
            "note",
        ],
        rows,
    )


def render_markdown(result) -> str:
    """result: mining.run.PipelineResult"""
    p = result.params
    lines = []
    lines.append("# Retail Sequence & Churn-Association Mining Report")
    lines.append("")
    lines.append(
        "This report is a data-engineering sample. All patterns and "
        "associations below are **correlational**: they describe what "
        "co-occurred in the historical log, not what caused it. Small "
        "groups are flagged with wide confidence intervals rather than "
        "hidden, and groups below the minimum support threshold are "
        "omitted entirely rather than reported on thin evidence."
    )
    lines.append("")

    lines.append("## Dataset summary")
    lines.append("")
    lines.append(f"- Customers: {result.n_customers}")
    lines.append(f"- Transactions: {result.n_transactions}")
    lines.append(f"- Observation cutoff (day-offset): {result.cutoff_day}")
    lines.append(f"- Churn inactivity window: {p.inactivity_window_days} days")
    lines.append(
        f"- Minimum support: {p.min_support_fraction * 100:.1f}% of customers "
        f"({result.min_support_count} customers)"
    )
    lines.append(f"- Churned customers: {result.n_churned} / {result.n_customers} "
                 f"({_fmt_pct(result.n_churned / result.n_customers)})")
    lines.append("")

    lines.append("## Frequent sequential patterns")
    lines.append("")
    lines.append(
        "Ordered subsequences of purchase events shared by at least "
        f"{result.min_support_count} customers (min support). Support = "
        "number of distinct customers whose purchase history contains "
        "the pattern in order (not necessarily adjacent)."
    )
    lines.append("")
    lines.append("| Length | Pattern | Support (customers) | Support (%) |")
    lines.append("|---|---|---|---|")
    for pattern, support in result.frequent_patterns[:25]:
        pct = support / result.n_customers if result.n_customers else 0
        lines.append(f"| {len(pattern)} | {' -> '.join(pattern)} | {support} | {_fmt_pct(pct)} |")
    lines.append("")

    lines.append("## Cohort retention")
    lines.append("")
    lines.append(
        "Percentage of each signup-month cohort with at least one "
        "transaction in month *m* since signup. Cohorts that have not "
        "yet reached month *m* by the observation cutoff are excluded "
        "from that month's figure (shown as n/a) rather than counted "
        "as churned."
    )
    lines.append("")
    for cohort_key in sorted(result.retention.keys()):
        rows = result.retention[cohort_key]
        cohort_size = rows[0]["cohort_size"] if rows else 0
        lines.append(f"### Cohort {cohort_key} (n={cohort_size} at signup)")
        lines.append("")
        lines.append("| Month | Cohort size (observed) | Retained | Retention rate |")
        lines.append("|---|---|---|---|")
        for row in rows:
            rate = _fmt_pct(row["retention_rate"])
            lines.append(f"| {row['month']} | {row['cohort_size']} | {row['retained_count']} | {rate} |")
        lines.append("")

    lines.append("## Churn associations")
    lines.append("")
    lines.append(
        "For each behavioral trait (a frequent sequence pattern or an "
        "RFM bucket), churn rate among customers *with* the trait vs. "
        "*without* it, each with a 95% Wilson confidence interval, plus "
        "lift = rate(with) / rate(without). Recency-based buckets are "
        "excluded here because recency is part of the churn definition "
        "itself and would make the association circular."
    )
    lines.append("")
    lines.append(
        "| Trait | Support | Churn rate w/ (95% CI) | Churn rate w/o (95% CI) | Lift | n |"
    )
    lines.append("|---|---|---|---|---|---|")
    for a in result.associations:
        with_ci = f"{_fmt_pct(a.rate_with)} [{_fmt_pct(a.wilson_with.lower)}, {_fmt_pct(a.wilson_with.upper)}]"
        without_ci = (
            f"{_fmt_pct(a.rate_without)} [{_fmt_pct(a.wilson_without.lower)}, "
            f"{_fmt_pct(a.wilson_without.upper)}]"
        )
        lines.append(
            f"| {a.label} | {a.support} | {with_ci} | {without_ci} | "
            f"{_fmt_lift(a.lift)} | {a.n_with}/{a.n_without} |"
        )
    lines.append("")
    lines.append(
        "**Every row above is correlational, not causal.** Before "
        "acting on any association (e.g. targeting customers who show "
        "pattern A -> B with a retention offer), confirm it with a "
        "controlled experiment (e.g. an A/B test of an intervention "
        "on a held-out sample) rather than treating the historical "
        "association as proof the sequence *causes* churn."
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Synthetic data: signal magnitude and noise level were chosen "
        "by the generator, not observed in a real business."
    )
    lines.append(
        "- Sequential-pattern mining here is a small-alphabet, explicit "
        "prefix-extension approach, not a general-purpose miner (e.g. "
        "no PrefixSpan/GSP gap constraints, no timing-window constraints "
        "between events)."
    )
    lines.append(
        "- Churn is a fixed inactivity-window definition, not a "
        "validated business definition of 'lost customer'."
    )
    lines.append(
        "- All associations are correlational; they do not establish "
        "that a sequence or bucket causes churn, only that it "
        "co-occurred with churn more or less often in this log."
    )
    lines.append(
        "- Wilson intervals bound sampling uncertainty within this "
        "dataset; they do not correct for confounders (e.g. a pattern "
        "and churn might share a common cause not modeled here)."
    )
    lines.append("")

    return "\n".join(lines)


def write_run_summary(path: str, result, generated_by: str) -> None:
    lines = []
    lines.append(f"generated_by: {generated_by}")
    lines.append(f"customers: {result.n_customers}")
    lines.append(f"transactions: {result.n_transactions}")
    lines.append(f"cutoff_day: {result.cutoff_day}")
    lines.append(f"inactivity_window_days: {result.params.inactivity_window_days}")
    lines.append(f"min_support_count: {result.min_support_count}")
    lines.append(f"churned_customers: {result.n_churned}")
    lines.append(f"frequent_patterns_found: {len(result.frequent_patterns)}")
    lines.append(f"churn_associations_reported: {len(result.associations)}")
    lines.append("note: all associations are correlational, not causal.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
