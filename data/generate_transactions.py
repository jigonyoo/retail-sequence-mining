#!/usr/bin/env python3
"""Deterministic synthetic retail transaction log generator.

Pure standard library (random.Random with a fixed integer seed only --
no time.time(), no os.urandom). Running this script twice produces
byte-identical CSV output.

Planted signal
--------------
A configurable fraction of customers ("signal customers") are made to
purchase category A followed later by category B early in their
history. Signal customers are given an elevated (but not 100%)
probability of becoming inactive early ("churning" under
mining/churn.py's inactivity-window definition). Non-signal customers
churn at a lower baseline probability, and because event categories are
otherwise drawn uniformly at random, a small fraction of non-signal
customers will exhibit A -> B purely by chance too. This keeps the
recovered association correlational and noisy rather than a
deterministic 1-to-1 rule, which is the honest way to demonstrate a
sequence-mining + churn-association pipeline.

Output: two CSVs (customers.csv, transactions.csv) written to the
--out-dir directory, sorted deterministically.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from typing import List, Tuple

SEED = 9001
N_CUSTOMERS = 600
ALPHABET = ["A", "B", "C", "D", "E"]

SIGNUP_DAY_MIN = 0
SIGNUP_DAY_MAX = 150
SIM_END_DAY = 400

SIGNAL_FRACTION = 0.35
SIGNAL_CHURN_PROB = 0.75
BASELINE_CHURN_PROB = 0.15

CHURNER_LAST_DAY_OFFSET_RANGE = (10, 60)   # added to signup_day
ACTIVE_LAST_DAY_RANGE = (340, 400)         # absolute day, independent of signup

CHURNER_EVENT_COUNT_RANGE = (3, 8)
ACTIVE_EVENT_COUNT_RANGE = (6, 14)

AMOUNT_CENTS_RANGE = (500, 15000)
SIGNAL_AB_GAP_RANGE = (1, 5)


def _spaced_days(rng: random.Random, start: int, end: int, count: int) -> List[int]:
    """count day-offsets in [start, end], first=start, last=end,
    middle ones roughly evenly spaced with small jitter, always sorted.
    """
    if count <= 1:
        return [start]
    if count == 2:
        return [start, end]
    days = [start]
    span = end - start
    for k in range(1, count - 1):
        base = start + round(span * k / (count - 1))
        jitter = rng.randint(-2, 2)
        day = min(max(base + jitter, start), end)
        days.append(day)
    days.append(end)
    days.sort()
    return days


def _gen_customer_txns(
    rng: random.Random, customer_id: str, signup_day: int, is_signal: bool
) -> Tuple[List[dict], bool]:
    """Returns (list of txn dicts without order_index, intended_churn flag)."""
    if is_signal:
        will_churn = rng.random() < SIGNAL_CHURN_PROB
    else:
        will_churn = rng.random() < BASELINE_CHURN_PROB

    if will_churn:
        lo, hi = CHURNER_LAST_DAY_OFFSET_RANGE
        last_day = min(signup_day + rng.randint(lo, hi), SIM_END_DAY)
        count = rng.randint(*CHURNER_EVENT_COUNT_RANGE)
    else:
        lo, hi = ACTIVE_LAST_DAY_RANGE
        last_day = max(rng.randint(lo, hi), signup_day + 30)
        last_day = min(last_day, SIM_END_DAY)
        count = rng.randint(*ACTIVE_EVENT_COUNT_RANGE)

    days = _spaced_days(rng, signup_day, last_day, count)

    events = [rng.choice(ALPHABET) for _ in days]

    if is_signal:
        # Force the planted A -> B pattern in the first two slots.
        events[0] = "A"
        if len(events) >= 2:
            events[1] = "B"
            gap = rng.randint(*SIGNAL_AB_GAP_RANGE)
            forced_b_day = min(days[0] + gap, days[-1])
            # keep monotonic non-decreasing order after the forced edit
            days[1] = max(forced_b_day, days[0])
            if len(days) > 2:
                days[1] = min(days[1], days[2])
            days.sort()

    rows = []
    for day, event in zip(days, events):
        amount = rng.randint(*AMOUNT_CENTS_RANGE)
        rows.append({"customer_id": customer_id, "day_offset": day, "event": event, "amount_cents": amount})

    return rows, will_churn


def generate(seed: int = SEED, n_customers: int = N_CUSTOMERS):
    rng = random.Random(seed)

    customers = []
    all_txn_rows = []

    for i in range(n_customers):
        cid = f"C{i:04d}"
        signup_day = rng.randint(SIGNUP_DAY_MIN, SIGNUP_DAY_MAX)
        is_signal = rng.random() < SIGNAL_FRACTION

        rows, _intended_churn = _gen_customer_txns(rng, cid, signup_day, is_signal)

        customers.append({"customer_id": cid, "signup_day": signup_day})
        all_txn_rows.extend(rows)

    # Assign order_index deterministically per customer: sort each
    # customer's rows by day, tie-break by original generation order
    # (stable sort preserves that).
    by_customer = {}
    for row in all_txn_rows:
        by_customer.setdefault(row["customer_id"], []).append(row)

    final_txns = []
    for cid in sorted(by_customer.keys()):
        rows = sorted(by_customer[cid], key=lambda r: r["day_offset"])
        for idx, row in enumerate(rows):
            final_txns.append(
                {
                    "customer_id": cid,
                    "order_index": idx,
                    "day_offset": row["day_offset"],
                    "event": row["event"],
                    "amount_cents": row["amount_cents"],
                }
            )

    customers.sort(key=lambda c: c["customer_id"])

    return customers, final_txns


def write_outputs(out_dir: str, customers, txns) -> None:
    os.makedirs(out_dir, exist_ok=True)

    customers_path = os.path.join(out_dir, "customers.csv")
    with open(customers_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["customer_id", "signup_day"])
        for c in customers:
            writer.writerow([c["customer_id"], c["signup_day"]])

    txns_path = os.path.join(out_dir, "transactions.csv")
    with open(txns_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["customer_id", "order_index", "day_offset", "event", "amount_cents"])
        for t in txns:
            writer.writerow(
                [t["customer_id"], t["order_index"], t["day_offset"], t["event"], t["amount_cents"]]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "synthetic"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-customers", type=int, default=N_CUSTOMERS)
    args = parser.parse_args()

    customers, txns = generate(seed=args.seed, n_customers=args.n_customers)
    write_outputs(args.out_dir, customers, txns)
    print(f"Wrote {len(customers)} customers and {len(txns)} transactions to {args.out_dir}")


if __name__ == "__main__":
    main()
