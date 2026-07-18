"""Per-customer purchase sequences and frequent sequential patterns.

Method (deliberately simple and explicit -- this is a small-alphabet
retail catalogue, not a general sequence-mining library):

1. Build one ordered event list per customer from the transaction log.
2. Count 1-length patterns: how many *distinct customers* ever bought
   event E (support = customer count, not event count).
3. Build 2-length ordered patterns (a, b) meaning "a occurs somewhere
   before b in the customer's history" (not necessarily adjacent).
   Candidates are restricted to events that are individually frequent
   (Apriori-style downward-closure pruning): a frequent pair implies
   both its elements are frequent on their own.
4. Extend frequent 2-patterns to 3-patterns the same way (prefix
   extension), and so on up to ``max_length``.

Support is always "number of distinct customers whose sequence contains
this ordered subsequence", counted at most once per customer.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .util import Transaction, customer_sequence, group_by_customer

Pattern = Tuple[str, ...]


def build_sequences(txns: List[Transaction]) -> Dict[str, List[str]]:
    """Return {customer_id: [event, event, ...]} ordered by
    (day_offset, order_index). Customers with zero transactions are not
    included (nothing to build a sequence from).
    """
    by_cust = group_by_customer(txns)
    return {cid: customer_sequence(by_cust, cid) for cid in by_cust}


def alphabet_of(sequences: Dict[str, List[str]]) -> List[str]:
    """Sorted distinct event codes seen across all sequences."""
    seen = set()
    for seq in sequences.values():
        seen.update(seq)
    return sorted(seen)


def sequence_contains(seq: Sequence[str], pattern: Pattern) -> bool:
    """True if `pattern` occurs as an ordered (not necessarily
    contiguous) subsequence of `seq`. Greedy left-to-right matching is
    correct here because we only need existence, not the matching
    positions.
    """
    if not pattern:
        return True
    idx = 0
    n = len(seq)
    for p in pattern:
        while idx < n and seq[idx] != p:
            idx += 1
        if idx >= n:
            return False
        idx += 1  # consume this element, next pattern symbol must come after it
    return True


def support_count(sequences: Dict[str, List[str]], pattern: Pattern) -> int:
    """Number of distinct customers whose sequence contains `pattern`."""
    return sum(1 for seq in sequences.values() if sequence_contains(seq, pattern))


def _frequent_length_1(
    sequences: Dict[str, List[str]], alphabet: List[str], min_support_count: int
) -> List[Tuple[Pattern, int]]:
    out = []
    for e in alphabet:
        sc = support_count(sequences, (e,))
        if sc >= min_support_count:
            out.append(((e,), sc))
    return out


def _extend(
    sequences: Dict[str, List[str]],
    prefixes: List[Tuple[Pattern, int]],
    alphabet: List[str],
    min_support_count: int,
) -> List[Tuple[Pattern, int]]:
    """Extend each frequent prefix pattern by one more (frequent)
    alphabet symbol, keeping only extensions that themselves meet
    min_support_count (Apriori downward-closure pruning).
    """
    out = []
    for prefix, _prefix_support in prefixes:
        for e in alphabet:
            candidate = prefix + (e,)
            sc = support_count(sequences, candidate)
            if sc >= min_support_count:
                out.append((candidate, sc))
    return out


def frequent_sequential_patterns(
    sequences: Dict[str, List[str]],
    min_support_count: int,
    max_length: int = 3,
) -> List[Tuple[Pattern, int]]:
    """All frequent ordered patterns of length 1..max_length.

    Returns a flat list of (pattern_tuple, support_count) sorted by
    descending support then ascending pattern (lexicographic) so output
    order is fully deterministic.
    """
    if max_length < 1:
        return []

    alphabet = alphabet_of(sequences)
    all_patterns: List[Tuple[Pattern, int]] = []

    level = _frequent_length_1(sequences, alphabet, min_support_count)
    all_patterns.extend(level)

    length = 1
    while length < max_length and level:
        level = _extend(sequences, level, alphabet, min_support_count)
        all_patterns.extend(level)
        length += 1

    all_patterns.sort(key=lambda item: (-item[1], item[0]))
    return all_patterns


def customer_has_pattern(sequences: Dict[str, List[str]], customer_id: str, pattern: Pattern) -> bool:
    seq = sequences.get(customer_id, [])
    return sequence_contains(seq, pattern)
