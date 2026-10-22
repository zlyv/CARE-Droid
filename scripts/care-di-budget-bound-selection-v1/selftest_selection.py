#!/usr/bin/env python3
"""Small deterministic tests for routing and bounded-selection helpers."""

from __future__ import annotations

import numpy as np

from selection_core import cap_tag, decision_instability, exact_top_fraction, groupwise_top_fraction
from collect_source_grid import candidate_metrics, nested_crossfit_selection, select_row


def main() -> None:
    base = np.asarray([0.52, 0.93, 0.51, 0.20])
    members = np.asarray([
        [0.48, 0.53, 0.50, 0.55],
        [0.95, 0.91, 0.12, 0.18],
        [0.91, 0.88, 0.94, 0.90],
        [0.10, 0.20, 0.30, 0.40],
    ])
    expected = np.asarray([0.485, 0.46, 0.0925, 0.25])
    assert np.allclose(decision_instability(base, members), expected)
    assert int(np.sum(exact_top_fraction(np.arange(799), 0.25))) == 200
    groups = np.repeat([1, 2], [7, 9])
    route = groupwise_top_fraction(np.arange(16), groups, 0.25)
    assert int(np.sum(route[groups == 1])) == 2
    assert int(np.sum(route[groups == 2])) == 2
    assert cap_tag(1.5) == "max_delta_1p5"
    assert cap_tag(0.25) == "max_delta_0p25"

    # End-to-end in-memory grid and leave-one-fold-out selection smoke test.
    rng = np.random.default_rng(7)
    n = 100
    labels = np.tile(np.asarray([0, 1]), n // 2)
    probability = np.clip(0.2 + 0.6 * labels + rng.normal(0, 0.22, n), 0.01, 0.99)
    folds = np.tile(np.arange(1, 6), n // 5)
    source = {
        "labels": labels,
        "base": probability,
        "folds": folds,
        "scores": {
            "A25": 1.0 - 2.0 * np.abs(probability - 0.5),
            "DI": rng.random(n),
        },
    }
    experts = {
        0.5: np.clip(probability + 0.08 * (2 * labels - 1), 0.01, 0.99),
        1.0: np.clip(probability + 0.12 * (2 * labels - 1), 0.01, 0.99),
    }
    grid, per_fold = candidate_metrics(source, experts, [0.10, 0.25, 0.40])
    assert len(grid) == 2 * 2 * 3
    assert len(per_fold) == len(grid) * 5
    chosen = select_row(grid, 0.25)
    assert chosen["router"] == "DI" and float(chosen["budget"]) <= 0.25
    choices, metrics, final, routed = nested_crossfit_selection(
        source, experts, [0.10, 0.25, 0.40], 0.25
    )
    assert len(choices) == 5 and final.shape == routed.shape == (n,)
    assert metrics["selection"] == "leave-one-outer-fold-out"
    print("DI budget/bound selection self-test passed", flush=True)


if __name__ == "__main__":
    main()
