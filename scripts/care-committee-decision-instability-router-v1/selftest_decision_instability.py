#!/usr/bin/env python3
"""Fast formula and budget checks for decision-instability routing."""

from __future__ import annotations

import numpy as np

from decision_instability_core import (
    decision_instability,
    exact_top_fraction,
    groupwise_top_fraction,
)


def one(base: float, values: list[float]) -> float:
    detail = decision_instability(
        np.asarray([base]), np.asarray([values], dtype=np.float64)
    )
    return float(detail["uncertainty"][0])


def main() -> None:
    assert abs(one(0.52, [0.48, 0.53, 0.50, 0.55]) - 0.485) < 1e-12
    assert abs(one(0.93, [0.95, 0.91, 0.12, 0.18]) - 0.46) < 1e-12
    assert abs(one(0.51, [0.91, 0.88, 0.94, 0.90]) - 0.0925) < 1e-12
    # Benign decision: opposition is the mean malware probability.
    assert abs(one(0.20, [0.10, 0.20, 0.30, 0.40]) - 0.25) < 1e-12
    # Malware decision: opposition is one minus the mean malware probability.
    assert abs(one(0.80, [0.60, 0.70, 0.80, 0.90]) - 0.25) < 1e-12

    for n, expected in ((799, 200), (793, 198), (399, 100), (397, 99), (400, 100)):
        assert int(np.sum(exact_top_fraction(np.arange(n), 0.25))) == expected
    groups = np.repeat(np.asarray([2018, 2019, 2020, 2021]), [793, 399, 397, 400])
    route = groupwise_top_fraction(np.arange(len(groups)), groups, 0.25)
    assert [int(np.sum(route[groups == year])) for year in range(2018, 2022)] == [
        198, 100, 99, 100
    ]
    print("decision-instability self-test passed", flush=True)


if __name__ == "__main__":
    main()
