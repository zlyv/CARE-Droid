#!/usr/bin/env python3
"""Artifact-independent checks for future routing helpers."""

from __future__ import annotations

import numpy as np

from future_router_helpers import (
    exact_top_fraction,
    groupwise_top_fraction,
    routed_probability,
    source_quantile_threshold,
)


def main() -> None:
    score = np.arange(20, dtype=np.float64)
    route = exact_top_fraction(score, 0.25)
    assert int(np.sum(route)) == 5
    assert np.flatnonzero(route).tolist() == [15, 16, 17, 18, 19]

    group = np.asarray([2018] * 8 + [2019] * 12)
    annual = groupwise_top_fraction(score, group, 0.25)
    assert int(np.sum(annual[group == 2018])) == 2
    assert int(np.sum(annual[group == 2019])) == 3

    threshold = source_quantile_threshold(score, 0.25)
    assert threshold == 15.0

    base = np.asarray([0.1, 0.2, 0.3])
    expert = np.asarray([0.8, 0.9, 1.0])
    merged = routed_probability(base, expert, np.asarray([False, True, False]))
    assert np.allclose(merged, [0.1, 0.9, 0.3])
    print("future router self-test passed")


if __name__ == "__main__":
    main()

