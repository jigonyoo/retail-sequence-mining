"""Pipeline orchestration: wire sequences + cohorts + churn + report
together into one deterministic run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import cohorts as cohorts_mod
from . import sequences as sequences_mod
from .churn import (
    association_for_group,
    compute_churn_labels,
    frequency_bucket,
    monetary_bucket,
    observation_cutoff_day,
)
from .stats import min_support_count_from_fraction
from .util import Customer, Transaction, group_by_customer


@dataclass(frozen=True)
class Params:
    min_support_fraction: float = 0.05
    max_pattern_length: int = 3
    inactivity_window_days: int = 90
    max_retention_months: int = 6
    # minimum length-2+ pattern support used as churn-association traits
    min_association_pattern_length: int = 2


@dataclass
class PipelineResult:
    params: Params
    n_customers: int
    n_transactions: int
    cutoff_day: int
    min_support_count: int
    n_churned: int
    frequent_patterns: List[Tuple[Tuple[str, ...], int]]
    retention: Dict[str, List[Dict]]
    associations: List
    churn_labels: Dict[str, bool]
    sequences: Dict[str, List[str]]


def percentile(sorted_values: List[int], pct: float) -> int:
    """Nearest-rank percentile over an already-sorted list of ints.
    Deterministic, no interpolation ambiguity.
    """
    if not sorted_values:
        return 0
    n = len(sorted_values)
    rank = max(1, min(n, math.ceil(pct * n)))
    return sorted_values[rank - 1]


def run_pipeline(
    customers: Dict[str, Customer],
    txns: List[Transaction],
    params: Optional[Params] = None,
) -> PipelineResult:
    if params is None:
        params = Params()

    txns_by_customer = group_by_customer(txns)
    n_customers = len(customers)

    # --- sequences ---------------------------------------------------
    sequences = sequences_mod.build_sequences(txns)
    # customers with zero transactions still get an empty sequence entry
    for cid in customers:
        sequences.setdefault(cid, [])

    min_support_count = min_support_count_from_fraction(n_customers, params.min_support_fraction)

    frequent_patterns = sequences_mod.frequent_sequential_patterns(
        sequences, min_support_count, max_length=params.max_pattern_length
    )

    # --- cohorts / retention ------------------------------------------
    cutoff_day = observation_cutoff_day(txns)
    retention = cohorts_mod.all_cohort_retention(
        customers, txns_by_customer, cutoff_day, params.max_retention_months
    )

    # --- churn ----------------------------------------------------------
    churn_labels = compute_churn_labels(
        customers, txns_by_customer, cutoff_day, params.inactivity_window_days
    )
    n_churned = sum(1 for v in churn_labels.values() if v)

    all_customer_ids = set(customers.keys())

    associations = []

    # Sequence-pattern traits (length >= min_association_pattern_length):
    # "customers whose history contains this ordered pattern".
    pattern_traits = [
        (pattern, support)
        for pattern, support in frequent_patterns
        if len(pattern) >= params.min_association_pattern_length
    ]
    for pattern, _support in pattern_traits:
        members = {
            cid for cid, seq in sequences.items() if sequences_mod.sequence_contains(seq, pattern)
        }
        label = "sequence " + " -> ".join(pattern)
        result = association_for_group(
            label, members, all_customer_ids, churn_labels, min_support_count
        )
        if result is not None:
            associations.append(result)

    # RFM (frequency, monetary) bucket traits. Recency is intentionally
    # excluded -- see churn.py module docstring.
    txn_counts = {cid: len(txns_by_customer.get(cid, [])) for cid in customers}
    monetary_totals = {
        cid: sum(t.amount_cents for t in txns_by_customer.get(cid, [])) for cid in customers
    }
    sorted_counts = sorted(txn_counts.values())
    sorted_monetary = sorted(monetary_totals.values())
    freq_low_max = percentile(sorted_counts, 1 / 3)
    freq_medium_max = percentile(sorted_counts, 2 / 3)
    mon_low_max = percentile(sorted_monetary, 1 / 3)
    mon_medium_max = percentile(sorted_monetary, 2 / 3)

    for bucket_name in ("low", "medium", "high"):
        members = {
            cid
            for cid in customers
            if frequency_bucket(txn_counts[cid], freq_low_max, freq_medium_max) == bucket_name
        }
        label = f"frequency bucket = {bucket_name}"
        result = association_for_group(
            label, members, all_customer_ids, churn_labels, min_support_count
        )
        if result is not None:
            associations.append(result)

    for bucket_name in ("low", "medium", "high"):
        members = {
            cid
            for cid in customers
            if monetary_bucket(monetary_totals[cid], mon_low_max, mon_medium_max) == bucket_name
        }
        label = f"monetary bucket = {bucket_name}"
        result = association_for_group(
            label, members, all_customer_ids, churn_labels, min_support_count
        )
        if result is not None:
            associations.append(result)

    # Deterministic ordering: descending lift (None sorts last), then label.
    associations.sort(
        key=lambda a: (a.lift is None, -(a.lift or 0.0), a.label)
    )

    return PipelineResult(
        params=params,
        n_customers=n_customers,
        n_transactions=len(txns),
        cutoff_day=cutoff_day,
        min_support_count=min_support_count,
        n_churned=n_churned,
        frequent_patterns=frequent_patterns,
        retention=retention,
        associations=associations,
        churn_labels=churn_labels,
        sequences=sequences,
    )
