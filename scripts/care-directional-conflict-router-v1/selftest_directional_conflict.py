#!/usr/bin/env python3
"""Artifact-independent checks for the directional conflict router."""

import numpy as np

from directional_conflict_core import (
    boundary_uncertainty,
    build_route_scores,
    estimate_class_conditional_reliability,
    exact_top_fraction,
)


def main() -> None:
    probability = np.asarray([
        [0.9, 0.8, 0.2, 0.1],
        [0.8, 0.7, 0.3, 0.2],
        [0.7, 0.4, 0.6, 0.3],
        [0.2, 0.3, 0.8, 0.9],
    ])
    labels = np.asarray([1, 1, 0, 0])
    positive, negative, audit = estimate_class_conditional_reliability(
        probability, labels
    )
    assert len(audit) == 4
    assert np.all((positive > 0.0) & (positive < 1.0))
    assert np.all((negative > 0.0) & (negative < 1.0))

    base = np.asarray([0.51, 0.10])
    auxiliary = np.asarray([
        [0.51, 0.48, 0.55, 0.46],
        [0.08, 0.15, 0.91, 0.86],
    ])
    scores, conflict = build_route_scores(
        base, auxiliary, np.ones(4), np.ones(4)
    )
    assert np.isclose(boundary_uncertainty(base)[0], 0.98)
    assert np.isclose(scores["ConflictMax"][1], 0.82)
    assert scores["SoftOR"][1] > scores["ConflictMax"][1]
    assert scores["ConflictMax"][0] <= 0.08
    assert conflict.shape == (2, 4)

    route = exact_top_fraction(np.linspace(0.0, 1.0, 20), 0.25)
    assert int(np.sum(route)) == 5
    print("directional conflict router self-test passed")


if __name__ == "__main__":
    main()
