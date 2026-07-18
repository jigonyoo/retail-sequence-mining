"""Signup-month cohorts and retention curves.

A customer is "active" in month m (m = 0, 1, 2, ...) relative to their
own signup date if they made at least one transaction whose day-offset
falls in calendar-month (signup_month + m).

Retention for month m is only computed over customers whose month m has
actually elapsed by the observation cutoff (the last day present in the
dataset). Customers who signed up too recently to have reached month m
yet are excluded from that month's denominator -- including them as
"not retained" would understate retention purely due to recency, not
behavior.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from .util import Customer, Transaction, day_to_month_key, months_between


def build_cohorts(customers: Dict[str, Customer]) -> Dict[str, List[str]]:
    """{cohort_month ('YYYY-MM'): [customer_id, ...] sorted} """
    cohorts: Dict[str, List[str]] = {}
    for cid, cust in customers.items():
        key = day_to_month_key(cust.signup_day)
        cohorts.setdefault(key, []).append(cid)
    for key in cohorts:
        cohorts[key].sort()
    return cohorts


def customer_active_months(signup_day: int, txn_days: List[int]) -> Set[int]:
    """Set of month-indices (0 = signup month) in which the customer had
    at least one transaction.
    """
    return {months_between(signup_day, d) for d in txn_days}


def observable_month_count(signup_day: int, cutoff_day: int) -> int:
    """Highest month-index that has fully elapsed for this customer by
    cutoff_day (inclusive). E.g. 0 means only the signup month itself is
    observable so far.
    """
    if cutoff_day < signup_day:
        return -1
    return months_between(signup_day, cutoff_day)


def retention_curve(
    customers: Dict[str, Customer],
    txns_by_customer: Dict[str, List[Transaction]],
    cohort_customer_ids: List[str],
    cutoff_day: int,
    max_months: int,
) -> List[Dict[str, Optional[float]]]:
    """Retention curve for one cohort.

    Returns a list (one row per month 0..max_months) of dicts:
    {month, cohort_size, retained_count, retention_rate}
    retention_rate is None when no customer in the cohort has reached
    that month yet (avoids a fake 0% data point).
    """
    active_months_by_cust: Dict[str, Set[int]] = {}
    for cid in cohort_customer_ids:
        days = [t.day_offset for t in txns_by_customer.get(cid, [])]
        active_months_by_cust[cid] = customer_active_months(customers[cid].signup_day, days)

    rows: List[Dict[str, Optional[float]]] = []
    for m in range(0, max_months + 1):
        eligible = [
            cid
            for cid in cohort_customer_ids
            if observable_month_count(customers[cid].signup_day, cutoff_day) >= m
        ]
        retained = [cid for cid in eligible if m in active_months_by_cust[cid]]
        rate = (len(retained) / len(eligible)) if eligible else None
        rows.append(
            {
                "month": m,
                "cohort_size": len(eligible),
                "retained_count": len(retained),
                "retention_rate": rate,
            }
        )
    return rows


def all_cohort_retention(
    customers: Dict[str, Customer],
    txns_by_customer: Dict[str, List[Transaction]],
    cutoff_day: int,
    max_months: int,
) -> Dict[str, List[Dict[str, Optional[float]]]]:
    """Retention curves for every cohort, keyed by cohort month,
    iterated/returned in sorted cohort-month order for determinism.
    """
    cohorts = build_cohorts(customers)
    result: Dict[str, List[Dict[str, Optional[float]]]] = {}
    for key in sorted(cohorts.keys()):
        result[key] = retention_curve(
            customers, txns_by_customer, cohorts[key], cutoff_day, max_months
        )
    return result
