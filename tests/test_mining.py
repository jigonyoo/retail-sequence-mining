"""Unit + integration tests for the retail sequence/churn mining sample.

Run with:
    python3 -m unittest discover -s tests -v
(from the project root, so `mining` and `data` are importable).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest

from data.generate_transactions import generate, write_outputs
from mining import cohorts as cohorts_mod
from mining import sequences as sequences_mod
from mining.churn import (
    association_for_group,
    compute_churn_labels,
    frequency_bucket,
    monetary_bucket,
    observation_cutoff_day,
)
from mining.run import Params, percentile, run_pipeline
from mining.stats import (
    meets_min_support,
    min_support_count_from_fraction,
    wilson_interval,
)
from mining.util import Customer, Transaction, day_to_month_key, months_between


def T(cid, order_index, day_offset, event, amount_cents=1000):
    return Transaction(
        customer_id=cid, order_index=order_index, day_offset=day_offset, event=event, amount_cents=amount_cents
    )


class SequenceExtractionTests(unittest.TestCase):
    def test_build_sequences_orders_by_day_offset(self):
        txns = [
            T("C1", 2, 40, "C"),
            T("C1", 0, 10, "A"),
            T("C1", 1, 20, "B"),
        ]
        seqs = sequences_mod.build_sequences(txns)
        self.assertEqual(seqs["C1"], ["A", "B", "C"])

    def test_build_sequences_tie_break_uses_order_index(self):
        # Two transactions on the same day: order_index must break the tie.
        txns = [
            T("C1", 1, 10, "B"),
            T("C1", 0, 10, "A"),
        ]
        seqs = sequences_mod.build_sequences(txns)
        self.assertEqual(seqs["C1"], ["A", "B"])

    def test_sequence_contains_non_contiguous(self):
        seq = ["A", "C", "B", "D"]
        self.assertTrue(sequences_mod.sequence_contains(seq, ("A", "B")))
        self.assertTrue(sequences_mod.sequence_contains(seq, ("A", "B", "D")))
        self.assertFalse(sequences_mod.sequence_contains(seq, ("B", "A")))
        self.assertFalse(sequences_mod.sequence_contains(seq, ("A", "E")))

    def test_sequence_contains_empty_pattern_is_true(self):
        self.assertTrue(sequences_mod.sequence_contains(["A", "B"], tuple()))

    def test_sequence_contains_repeated_symbol_requires_two_occurrences(self):
        self.assertTrue(sequences_mod.sequence_contains(["A", "B", "A"], ("A", "A")))
        self.assertFalse(sequences_mod.sequence_contains(["A", "B"], ("A", "A")))


class FrequentPatternTests(unittest.TestCase):
    def setUp(self):
        # 5 customers: 3 contain A->B, 2 do not.
        self.sequences = {
            "C1": ["A", "B", "C"],
            "C2": ["A", "X", "B"],
            "C3": ["A", "B"],
            "C4": ["B", "A"],
            "C5": ["C", "C"],
        }

    def test_support_count_counts_distinct_customers_not_occurrences(self):
        # C1 contains "A" once; make sure repeats within one customer
        # don't inflate support beyond 1 per customer.
        seqs = {"C1": ["A", "A", "A"], "C2": ["B"]}
        self.assertEqual(sequences_mod.support_count(seqs, ("A",)), 1)

    def test_support_count_two_length_pattern(self):
        self.assertEqual(sequences_mod.support_count(self.sequences, ("A", "B")), 3)

    def test_min_support_filtering_excludes_rare_patterns(self):
        patterns = sequences_mod.frequent_sequential_patterns(
            self.sequences, min_support_count=3, max_length=2
        )
        pattern_tuples = [p for p, _s in patterns]
        self.assertIn(("A", "B"), pattern_tuples)
        # "B","A" has support 1 (only C4) -> must be excluded at min_support=3
        self.assertNotIn(("B", "A"), pattern_tuples)

    def test_frequent_patterns_sorted_by_descending_support(self):
        patterns = sequences_mod.frequent_sequential_patterns(
            self.sequences, min_support_count=1, max_length=1
        )
        supports = [s for _p, s in patterns]
        self.assertEqual(supports, sorted(supports, reverse=True))


class CohortRetentionTests(unittest.TestCase):
    def setUp(self):
        # EPOCH = 2024-01-01. Day 0 -> Jan, day 31 -> Feb, day 62 -> Mar (approx).
        self.customers = {
            "A": Customer("A", signup_day=0),
            "B": Customer("B", signup_day=5),
            "C": Customer("C", signup_day=100),  # much later cohort
        }

    def test_build_cohorts_groups_by_signup_month(self):
        cohorts = cohorts_mod.build_cohorts(self.customers)
        self.assertIn("2024-01", cohorts)
        self.assertEqual(sorted(cohorts["2024-01"]), ["A", "B"])

    def test_retention_curve_month_zero_is_full_cohort(self):
        txns_by_customer = {
            "A": [T("A", 0, 0, "X")],
            "B": [T("B", 0, 5, "X")],
        }
        rows = cohorts_mod.retention_curve(
            self.customers, txns_by_customer, ["A", "B"], cutoff_day=10, max_months=1
        )
        self.assertEqual(rows[0]["cohort_size"], 2)
        self.assertEqual(rows[0]["retained_count"], 2)
        self.assertEqual(rows[0]["retention_rate"], 1.0)

    def test_retention_curve_excludes_not_yet_observed_months(self):
        # cutoff_day is only 10 days after signup -> month 1+ hasn't
        # happened yet for these customers, so it must be excluded
        # (cohort_size 0, rate None), not counted as churned/0%.
        txns_by_customer = {"A": [T("A", 0, 0, "X")], "B": [T("B", 0, 5, "X")]}
        rows = cohorts_mod.retention_curve(
            self.customers, txns_by_customer, ["A", "B"], cutoff_day=10, max_months=3
        )
        month_3_row = rows[3]
        self.assertEqual(month_3_row["cohort_size"], 0)
        self.assertIsNone(month_3_row["retention_rate"])

    def test_retention_curve_customer_inactive_in_month_not_retained(self):
        # Customer A only transacts at signup, never again -> month 1
        # retention should show them as not retained (once observable).
        txns_by_customer = {"A": [T("A", 0, 0, "X")]}
        rows = cohorts_mod.retention_curve(
            self.customers, txns_by_customer, ["A"], cutoff_day=100, max_months=2
        )
        self.assertEqual(rows[1]["retained_count"], 0)
        self.assertEqual(rows[1]["cohort_size"], 1)
        self.assertEqual(rows[1]["retention_rate"], 0.0)


class ChurnDefinitionTests(unittest.TestCase):
    def test_churn_boundary_inclusive_at_exact_window(self):
        customers = {"A": Customer("A", signup_day=0)}
        txns_by_customer = {"A": [T("A", 0, 0, "X")]}
        cutoff = 90  # exactly the window
        labels = compute_churn_labels(customers, txns_by_customer, cutoff, inactivity_window_days=90)
        self.assertTrue(labels["A"])

    def test_churn_boundary_not_churned_one_day_short(self):
        customers = {"A": Customer("A", signup_day=0)}
        txns_by_customer = {"A": [T("A", 0, 1, "X")]}  # last activity day 1
        cutoff = 90  # gap = 89 < 90
        labels = compute_churn_labels(customers, txns_by_customer, cutoff, inactivity_window_days=90)
        self.assertFalse(labels["A"])

    def test_observation_cutoff_is_max_day_offset(self):
        txns = [T("A", 0, 5, "X"), T("B", 0, 42, "Y"), T("A", 1, 10, "Z")]
        self.assertEqual(observation_cutoff_day(txns), 42)

    def test_frequency_bucket_boundaries(self):
        self.assertEqual(frequency_bucket(2, low_max=2, medium_max=5), "low")
        self.assertEqual(frequency_bucket(3, low_max=2, medium_max=5), "medium")
        self.assertEqual(frequency_bucket(5, low_max=2, medium_max=5), "medium")
        self.assertEqual(frequency_bucket(6, low_max=2, medium_max=5), "high")

    def test_monetary_bucket_boundaries(self):
        self.assertEqual(monetary_bucket(1000, low_max=1000, medium_max=5000), "low")
        self.assertEqual(monetary_bucket(5001, low_max=1000, medium_max=5000), "high")


class StatsGuardRailTests(unittest.TestCase):
    def test_wilson_interval_widens_for_small_groups(self):
        # Same observed rate (50%), very different n -> small-n interval
        # must be strictly wider.
        small = wilson_interval(successes=5, n=10)
        large = wilson_interval(successes=500, n=1000)
        self.assertGreater(small.width, large.width)

    def test_wilson_interval_bounds_are_sane(self):
        w = wilson_interval(successes=3, n=20)
        self.assertGreaterEqual(w.lower, 0.0)
        self.assertLessEqual(w.upper, 1.0)
        self.assertLessEqual(w.lower, w.point_estimate)
        self.assertGreaterEqual(w.upper, w.point_estimate)

    def test_wilson_interval_zero_n_is_maximally_uncertain(self):
        w = wilson_interval(successes=0, n=0)
        self.assertEqual(w.lower, 0.0)
        self.assertEqual(w.upper, 1.0)

    def test_min_support_count_from_fraction_rounds_up(self):
        # 2% of 101 = 2.02 -> must round up to 3, never down to 2.
        self.assertEqual(min_support_count_from_fraction(101, 0.02), 3)

    def test_meets_min_support(self):
        self.assertTrue(meets_min_support(10, 10))
        self.assertFalse(meets_min_support(9, 10))


class AssociationTests(unittest.TestCase):
    def test_association_returns_none_below_min_support(self):
        churn_labels = {f"C{i}": (i % 2 == 0) for i in range(10)}
        all_ids = set(churn_labels.keys())
        group = {"C0", "C2"}  # only 2 members, below threshold
        result = association_for_group("tiny group", group, all_ids, churn_labels, min_support_count=5)
        self.assertIsNone(result)

    def test_association_counts_and_lift_correct(self):
        # 4 customers with trait, all churned; 4 without, none churned.
        churn_labels = {"W1": True, "W2": True, "W3": True, "W4": True, "N1": False, "N2": False, "N3": False, "N4": False}
        all_ids = set(churn_labels.keys())
        group = {"W1", "W2", "W3", "W4"}
        result = association_for_group("trait X", group, all_ids, churn_labels, min_support_count=2)
        self.assertIsNotNone(result)
        self.assertEqual(result.support, 4)
        self.assertEqual(result.churn_with, 4)
        self.assertEqual(result.churn_without, 0)
        self.assertAlmostEqual(result.rate_with, 1.0)
        self.assertAlmostEqual(result.rate_without, 0.0)
        # lift undefined (division by zero rate) -> None
        self.assertIsNone(result.lift)

    def test_association_note_is_labeled_correlational(self):
        churn_labels = {f"C{i}": (i < 5) for i in range(10)}
        all_ids = set(churn_labels.keys())
        group = {f"C{i}" for i in range(5)}
        result = association_for_group("half", group, all_ids, churn_labels, min_support_count=2)
        self.assertIn("Correlational", result.note)
        self.assertIn("not", result.note.lower())


class UtilTests(unittest.TestCase):
    def test_day_to_month_key(self):
        self.assertEqual(day_to_month_key(0), "2024-01")

    def test_months_between(self):
        # day 0 = 2024-01-01, day 31 = 2024-02-01 -> 1 month apart
        self.assertEqual(months_between(0, 31), 1)
        self.assertEqual(months_between(0, 0), 0)

    def test_percentile_nearest_rank(self):
        values = sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(percentile(values, 0.5), 5)
        self.assertEqual(percentile(values, 1.0), 10)
        self.assertEqual(percentile([], 0.5), 0)


class GeneratorAndPipelineIntegrationTests(unittest.TestCase):
    def test_generator_is_deterministic(self):
        c1, t1 = generate()
        c2, t2 = generate()
        self.assertEqual(c1, c2)
        self.assertEqual(t1, t2)

    def test_full_run_recovers_planted_ab_churn_signal(self):
        raw_customers, raw_txns = generate()
        customers = {
            r["customer_id"]: Customer(r["customer_id"], r["signup_day"]) for r in raw_customers
        }
        txns = [
            Transaction(r["customer_id"], r["order_index"], r["day_offset"], r["event"], r["amount_cents"])
            for r in raw_txns
        ]
        result = run_pipeline(customers, txns, Params())

        ab_assoc = next((a for a in result.associations if a.label == "sequence A -> B"), None)
        self.assertIsNotNone(ab_assoc, "expected 'sequence A -> B' to be a reported association")
        self.assertGreater(ab_assoc.support, result.min_support_count)
        self.assertIsNotNone(ab_assoc.lift)
        self.assertGreater(
            ab_assoc.lift, 1.3, "planted A->B->churn signal should show meaningfully elevated lift"
        )
        # The confidence intervals should not overlap the null (lift=1)
        # story completely -- with-trait lower bound should exceed
        # without-trait's point estimate isn't guaranteed, but the with
        # rate should clearly exceed the without rate.
        self.assertGreater(ab_assoc.rate_with, ab_assoc.rate_without)

    def test_pipeline_determinism_md5_identical_across_runs(self):
        raw_customers, raw_txns = generate()
        customers = {
            r["customer_id"]: Customer(r["customer_id"], r["signup_day"]) for r in raw_customers
        }
        txns = [
            Transaction(r["customer_id"], r["order_index"], r["day_offset"], r["event"], r["amount_cents"])
            for r in raw_txns
        ]

        from mining import report as report_mod

        def render():
            result = run_pipeline(customers, txns, Params())
            md = report_mod.render_markdown(result)
            return md, result

        md1, result1 = render()
        md2, result2 = render()
        self.assertEqual(hashlib.md5(md1.encode("utf-8")).hexdigest(), hashlib.md5(md2.encode("utf-8")).hexdigest())
        self.assertEqual(md1, md2)

    def test_generate_output_files_are_byte_identical_across_writes(self):
        tmp1 = tempfile.mkdtemp()
        tmp2 = tempfile.mkdtemp()
        try:
            c1, t1 = generate()
            c2, t2 = generate()
            write_outputs(tmp1, c1, t1)
            write_outputs(tmp2, c2, t2)
            for fname in ("customers.csv", "transactions.csv"):
                with open(os.path.join(tmp1, fname), "rb") as fh:
                    md5_1 = hashlib.md5(fh.read()).hexdigest()
                with open(os.path.join(tmp2, fname), "rb") as fh:
                    md5_2 = hashlib.md5(fh.read()).hexdigest()
                self.assertEqual(md5_1, md5_2, f"{fname} differs across runs")
        finally:
            shutil.rmtree(tmp1, ignore_errors=True)
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_all_reported_associations_carry_counts_and_correlational_label(self):
        raw_customers, raw_txns = generate()
        customers = {
            r["customer_id"]: Customer(r["customer_id"], r["signup_day"]) for r in raw_customers
        }
        txns = [
            Transaction(r["customer_id"], r["order_index"], r["day_offset"], r["event"], r["amount_cents"])
            for r in raw_txns
        ]
        result = run_pipeline(customers, txns, Params())
        self.assertGreater(len(result.associations), 0)
        for a in result.associations:
            self.assertGreater(a.n_with, 0)
            self.assertGreaterEqual(a.n_without, 0)
            self.assertIsInstance(a.churn_with, int)
            self.assertIsInstance(a.churn_without, int)
            self.assertIn("Correlational", a.note)


if __name__ == "__main__":
    unittest.main()
