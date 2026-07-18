"""Churn labeling and churn-association analysis.

Definitions
-----------
* Observation cutoff = the last day-offset present anywhere in the
  transaction log (derived from the data, never from wall-clock time).
* A customer's "last activity day" = the day-offset of their most
  recent transaction (or their signup day if -- unusually -- they have
  no transactions at all).
* A customer is CHURNED if
      cutoff_day - last_activity_day >= inactivity_window_days
  i.e. the boundary is inclusive: a customer whose gap since last
  activity exactly equals the window is counted as churned; a customer
  one day short of the window is not.

Association analysis
---------------------
For a candidate group (customers who exhibit some sequence pattern, or
who fall in an RFM bucket), we report:
  - support: how many customers are in the group
  - churn rate with the trait vs. churn rate without it, each with a
    Wilson confidence interval
  - lift = rate_with / rate_without

We deliberately do NOT build an association predictor out of recency
buckets: recency is part of the churn definition itself, so a
recency-based "association" would be circular. Frequency, monetary
value, and sequence-pattern membership are used instead -- these are
independent of how churn is defined.

Every result produced here is explicitly correlational: it describes
co-occurrence in historical data, not a causal driver of churn. See
README.md Limitations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from .stats import WilsonInterval, meets_min_support, wilson_interval
from .util import Customer, Transaction

CORRELATIONAL_NOTE = (
    "Correlational only: co-occurrence in historical data, not a proven "
    "cause of churn. Confirm with a controlled experiment before acting."
)


def observation_cutoff_day(txns: Iterable[Transaction]) -> int:
    days = [t.day_offset for t in txns]
    if not days:
        raise ValueError("cannot compute observation cutoff from an empty transaction log")
    return max(days)


def last_activity_day(customer: Customer, txns_for_customer: List[Transaction]) -> int:
    if not txns_for_customer:
        return customer.signup_day
    return max(t.day_offset for t in txns_for_customer)


def compute_churn_labels(
    customers: Dict[str, Customer],
    txns_by_customer: Dict[str, List[Transaction]],
    cutoff_day: int,
    inactivity_window_days: int,
) -> Dict[str, bool]:
    labels: Dict[str, bool] = {}
    for cid, cust in customers.items():
        last_day = last_activity_day(cust, txns_by_customer.get(cid, []))
        gap = cutoff_day - last_day
        labels[cid] = gap >= inactivity_window_days
    return labels


def frequency_bucket(txn_count: int, low_max: int, medium_max: int) -> str:
    """Bucket a customer's total transaction count into low/medium/high.
    Thresholds are inclusive upper bounds for low/medium; anything above
    medium_max is high.
    """
    if txn_count <= low_max:
        return "low"
    if txn_count <= medium_max:
        return "medium"
    return "high"


def monetary_bucket(total_cents: int, low_max: int, medium_max: int) -> str:
    if total_cents <= low_max:
        return "low"
    if total_cents <= medium_max:
        return "medium"
    return "high"


@dataclass(frozen=True)
class AssociationResult:
    label: str
    support: int
    n_total: int
    n_with: int
    churn_with: int
    rate_with: float
    wilson_with: WilsonInterval
    n_without: int
    churn_without: int
    rate_without: float
    wilson_without: WilsonInterval
    lift: Optional[float]
    note: str = CORRELATIONAL_NOTE


def association_for_group(
    label: str,
    group_members: Set[str],
    all_customer_ids: Set[str],
    churn_labels: Dict[str, bool],
    min_support_count: int,
) -> Optional[AssociationResult]:
    """Compute the churn association for one binary group membership
    (e.g. "has pattern A->B" / "does not"). Returns None if the group
    (or its complement) is below the minimum support guard rail.
    """
    n_total = len(all_customer_ids)
    with_ids = group_members & all_customer_ids
    without_ids = all_customer_ids - with_ids

    n_with = len(with_ids)
    n_without = len(without_ids)

    if not meets_min_support(n_with, min_support_count):
        return None
    if not meets_min_support(n_without, min_support_count):
        return None

    churn_with = sum(1 for cid in with_ids if churn_labels.get(cid, False))
    churn_without = sum(1 for cid in without_ids if churn_labels.get(cid, False))

    wilson_with = wilson_interval(churn_with, n_with)
    wilson_without = wilson_interval(churn_without, n_without)

    rate_with = wilson_with.point_estimate
    rate_without = wilson_without.point_estimate

    lift = (rate_with / rate_without) if rate_without > 0 else None

    return AssociationResult(
        label=label,
        support=n_with,
        n_total=n_total,
        n_with=n_with,
        churn_with=churn_with,
        rate_with=rate_with,
        wilson_with=wilson_with,
        n_without=n_without,
        churn_without=churn_without,
        rate_without=rate_without,
        wilson_without=wilson_without,
        lift=lift,
    )
