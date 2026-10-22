#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import run_asymmetric_a25_router_source_v1 as module


def main() -> None:
    probability = np.asarray([0.01, 0.20, 0.45, 0.49, 0.51, 0.56, 0.80, 0.99])
    predicted = (probability >= 0.5).astype(np.int64)
    score = module.ambiguity(probability)
    route = module.route_by_class_quota(score, predicted, total=4, k0=3)
    assert int(np.sum(route)) == 4
    assert int(np.sum(route & (predicted == 0))) == 3
    assert int(np.sum(route & (predicted == 1))) == 1
    # Within each predicted side, the samples nearest 0.5 must be selected.
    assert np.array_equal(np.flatnonzero(route), np.asarray([1, 2, 3, 4]))

    labels = np.asarray([0, 0, 1, 1, 0, 1, 1, 1])
    k0, total, stats = module.select_training_quota(labels, probability, 0.5)
    selected = module.route_by_class_quota(score, predicted, total, k0)
    assert int(np.sum(selected)) == total == 4
    assert stats["captured_error_n"] >= 0
    print("ASYMMETRIC_A25_SELFTEST_OK")


if __name__ == "__main__":
    main()
