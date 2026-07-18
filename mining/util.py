"""Shared helpers: deterministic CSV I/O and small data structures.

No wall-clock calls anywhere in this module. Dates are always computed
from a fixed EPOCH plus an integer day-offset that is stored in the
input data itself, never from ``datetime.now()``/``date.today()``.
"""
from __future__ import annotations

import csv
import datetime
from dataclasses import dataclass, field
from typing import Dict, List

# Fixed reference epoch used to turn integer day-offsets in the synthetic
# data into calendar dates. This is a constant, not a wall-clock read, so
# output stays deterministic across runs.
EPOCH = datetime.date(2024, 1, 1)


def day_to_date(day_offset: int) -> datetime.date:
    """Map an integer day-offset (>=0) to a calendar date from EPOCH."""
    return EPOCH + datetime.timedelta(days=int(day_offset))


def day_to_month_key(day_offset: int) -> str:
    """Map a day-offset to a 'YYYY-MM' cohort key."""
    d = day_to_date(day_offset)
    return "%04d-%02d" % (d.year, d.month)


def months_between(day_a: int, day_b: int) -> int:
    """Whole calendar months between day_a and day_b (day_b - day_a), >=0
    assumed day_b >= day_a. Used for cohort "months since signup" bucketing.
    """
    da = day_to_date(day_a)
    db = day_to_date(day_b)
    return (db.year - da.year) * 12 + (db.month - da.month)


@dataclass(frozen=True)
class Transaction:
    customer_id: str
    order_index: int
    day_offset: int
    event: str
    amount_cents: int


@dataclass(frozen=True)
class Customer:
    customer_id: str
    signup_day: int


def read_customers(path: str) -> Dict[str, Customer]:
    customers: Dict[str, Customer] = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row["customer_id"]
            customers[cid] = Customer(
                customer_id=cid,
                signup_day=int(row["signup_day"]),
            )
    return customers


def read_transactions(path: str) -> List[Transaction]:
    txns: List[Transaction] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            txns.append(
                Transaction(
                    customer_id=row["customer_id"],
                    order_index=int(row["order_index"]),
                    day_offset=int(row["day_offset"]),
                    event=row["event"],
                    amount_cents=int(row["amount_cents"]),
                )
            )
    return txns


def write_csv(path: str, header: List[str], rows: List[List]) -> None:
    """Write rows deterministically. Rows must already be sorted by the
    caller -- this function does not re-sort, so output ordering is
    fully controlled by pipeline logic (needed for byte-identical reruns).
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def customer_sequence(
    txns_by_customer: Dict[str, List[Transaction]], customer_id: str
) -> List[str]:
    """Ordered list of event codes for one customer, sorted by
    (day_offset, order_index) for a fully deterministic tie-break.
    """
    rows = txns_by_customer.get(customer_id, [])
    rows_sorted = sorted(rows, key=lambda t: (t.day_offset, t.order_index))
    return [t.event for t in rows_sorted]


def group_by_customer(
    txns: List[Transaction],
) -> Dict[str, List[Transaction]]:
    out: Dict[str, List[Transaction]] = {}
    for t in txns:
        out.setdefault(t.customer_id, []).append(t)
    return out
