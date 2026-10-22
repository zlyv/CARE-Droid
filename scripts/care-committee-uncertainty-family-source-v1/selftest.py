#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from uncertainty_family import exact_fold_top, scores


def main() -> None:
    base = np.array([0.9, 0.52, 0.9, 0.1])
    member = np.array([
        [0.9, 0.9, 0.9, 0.9],
        [0.48, 0.52, 0.49, 0.51],
        [0.95, 0.91, 0.12, 0.18],
        [0.05, 0.09, 0.88, 0.82],
    ])
    got = scores(base, member)
    assert got["MI_JSD25"][1] < got["MI_JSD25"][2]
    assert got["PredictiveEntropy25"][1] > 0.99
    assert got["MeanOpposition_DI25"][2] > got["MeanOpposition_DI25"][0]
    route = exact_fold_top(np.arange(8), np.array([1]*4+[2]*4),
                           np.array([f"x{i}" for i in range(8)]), .25)
    assert route.sum() == 2 and route[:4].sum() == route[4:].sum() == 1
    print("COMMITTEE_UNCERTAINTY_FAMILY_SELFTEST_PASS")
    print("candidate_scores=" + ",".join(got))
    print("future_rows_loaded=0")
    print("stage2_training_invoked=false")


if __name__ == "__main__":
    main()

