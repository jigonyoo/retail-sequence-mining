#!/usr/bin/env python3
"""Build the synthetic retail transaction log, run the mining pipeline,
and write sample_output/. Deterministic: running this twice produces
byte-identical files in sample_output/.

IMPORTANT: this script never deletes the output directory. It creates
it with os.makedirs(exist_ok=True) and overwrites individual files with
open(path, "w"), so re-running is always safe.
"""
from __future__ import annotations

import os

from data.generate_transactions import generate, write_outputs
from mining import report as report_mod
from mining.run import Params, run_pipeline
from mining.util import Customer, Transaction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic")
OUTPUT_DIR = os.path.join(BASE_DIR, "sample_output")


def _to_customers(raw_customers) -> dict:
    return {
        r["customer_id"]: Customer(customer_id=r["customer_id"], signup_day=r["signup_day"])
        for r in raw_customers
    }


def _to_transactions(raw_txns) -> list:
    return [
        Transaction(
            customer_id=r["customer_id"],
            order_index=r["order_index"],
            day_offset=r["day_offset"],
            event=r["event"],
            amount_cents=r["amount_cents"],
        )
        for r in raw_txns
    ]


def main() -> None:
    raw_customers, raw_txns = generate()

    # Write the synthetic source CSVs too, for transparency / reuse.
    # write_outputs() uses makedirs(exist_ok=True) + open(...,"w") only.
    write_outputs(SYNTHETIC_DIR, raw_customers, raw_txns)

    customers = _to_customers(raw_customers)
    txns = _to_transactions(raw_txns)

    result = run_pipeline(customers, txns, Params())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    md_path = os.path.join(OUTPUT_DIR, "mining_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(report_mod.render_markdown(result))

    report_mod.write_sequences_csv(
        os.path.join(OUTPUT_DIR, "sequences.csv"), result.frequent_patterns
    )
    report_mod.write_churn_associations_csv(
        os.path.join(OUTPUT_DIR, "churn_associations.csv"), result.associations
    )
    report_mod.write_run_summary(
        os.path.join(OUTPUT_DIR, "run_summary.txt"), result, generated_by="run_demo.py"
    )

    print(f"Customers: {result.n_customers}")
    print(f"Transactions: {result.n_transactions}")
    print(f"Churned: {result.n_churned} ({result.n_churned / result.n_customers:.1%})")
    print(f"Frequent patterns: {len(result.frequent_patterns)}")
    print(f"Churn associations reported: {len(result.associations)}")
    print(f"Wrote output to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
