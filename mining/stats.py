"""Statistical guard rails: minimum support and Wilson score intervals.

Small groups produce noisy rates. Rather than reporting a bare
percentage for e.g. "3 out of 4 customers churned", every churn rate in
this project is reported alongside a Wilson score confidence interval,
which widens automatically as the sample size shrinks. Callers should
treat a wide interval as "not enough data to act on" rather than
suppressing the row outright -- suppression is left to min_support.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_Z = 1.959963985  # ~95% two-sided normal quantile


@dataclass(frozen=True)
class WilsonInterval:
    point_estimate: float
    lower: float
    upper: float
    n: int
    successes: int

    @property
    def width(self) -> float:
        return self.upper - self.lower


def wilson_interval(successes: int, n: int, z: float = DEFAULT_Z) -> WilsonInterval:
    """Wilson score interval for a binomial proportion.

    Preferred over the naive normal-approximation interval because it
    stays inside [0, 1] and behaves sensibly for small n or proportions
    near 0 or 1 -- both common in small churn cohorts.
    """
    if n <= 0:
        return WilsonInterval(point_estimate=0.0, lower=0.0, upper=1.0, n=0, successes=0)
    if successes < 0 or successes > n:
        raise ValueError("successes must be within [0, n]")

    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z2 / (4 * n * n)))

    lower = (center - margin) / denom
    upper = (center + margin) / denom
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    return WilsonInterval(point_estimate=p_hat, lower=lower, upper=upper, n=n, successes=successes)


def meets_min_support(count: int, min_support_count: int) -> bool:
    """Guard rail: a pattern/group below the minimum absolute support
    count should not be reported as a finding, regardless of how
    dramatic its rate looks -- a rate computed from 2 customers is not
    a pattern, it's an anecdote.
    """
    return count >= min_support_count


def min_support_count_from_fraction(total_n: int, min_support_fraction: float) -> int:
    """Convert a fractional minimum support (e.g. 0.02 = 2% of
    customers) into an absolute integer count, rounded up so the
    threshold is never weaker than requested.
    """
    return max(1, math.ceil(total_n * min_support_fraction))
