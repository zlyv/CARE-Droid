#!/usr/bin/env python3
"""Fast protocol and formula checks for committee uncertainty router v1."""

from __future__ import annotations

import inspect

import numpy as np

from committee_uncertainty_core import (
    FutureLabelGate,
    assert_disjoint,
    binary_entropy,
    committee_scores,
    exact_top_fraction,
    fit_calibrated_committee,
    groupwise_top_fraction,
)


def expect_failure(function, *args) -> None:
    try:
        function(*args)
    except (RuntimeError, ValueError):
        return
    raise AssertionError("expected guard to reject unsafe input")


def main() -> None:
    assert abs(float(binary_entropy(np.asarray([0.5]))[0]) - 1.0) < 1e-10
    edge = binary_entropy(np.asarray([1e-12, 1.0 - 1e-12]))
    assert np.max(edge) < 1e-8

    ambiguous = committee_scores(np.asarray([[0.49, 0.51, 0.48, 0.52]]))
    conflict = committee_scores(np.asarray([[0.05, 0.10, 0.90, 0.95]]))
    assert float(conflict["mutual_information"][0]) > 0.5
    assert float(conflict["mutual_information"][0]) > (
        100.0 * float(ambiguous["mutual_information"][0])
    )
    identical = committee_scores(np.asarray([[0.2, 0.2, 0.2, 0.2]]))
    assert abs(float(identical["mutual_information"][0])) < 1e-12

    for n, expected in ((799, 200), (793, 198), (399, 100), (397, 99), (400, 100)):
        route = exact_top_fraction(np.arange(n, dtype=float), 0.25)
        assert int(np.sum(route)) == expected
    groups = np.repeat(np.asarray([2018, 2019, 2020, 2021]), [793, 399, 397, 400])
    group_route = groupwise_top_fraction(
        np.arange(len(groups), dtype=float), groups, 0.25
    )
    assert [int(np.sum(group_route[groups == year])) for year in range(2018, 2022)] == [
        198, 100, 99, 100
    ]

    assert_disjoint(np.asarray([0, 1]), np.asarray([2, 3]))
    expect_failure(assert_disjoint, np.asarray([0, 1]), np.asarray([1, 2]))

    # The only labels accepted by the fit API are y_train.  There is no target-
    # label parameter, so an outer validation/future label cannot be passed into
    # member fitting or Platt calibration by this public path.
    parameters = set(inspect.signature(fit_calibrated_committee).parameters)
    assert "y_train" in parameters
    assert not any(
        name in parameters for name in ("y_target", "target_labels", "y_validation")
    )

    gate = FutureLabelGate([{"label": 0}, {"label": 1}])
    expect_failure(gate.labels, lambda row: int(row["label"]))
    gate.freeze_predictions_and_routes()
    assert np.array_equal(
        gate.labels(lambda row: int(row["label"])), np.asarray([0, 1])
    )
    print("committee uncertainty self-test passed", flush=True)


if __name__ == "__main__":
    main()
