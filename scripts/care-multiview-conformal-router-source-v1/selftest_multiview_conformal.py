#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_multiview_conformal_router_source_v1 as route


def main() -> None:
    rng = np.random.RandomState(7)
    labels = np.asarray([0] * 30 + [1] * 30)
    probability = np.clip(rng.beta(2, 2, size=(60, 5)), 1e-4, 1 - 1e-4)
    weight = np.asarray([0.30, 0.20, 0.15, 0.20, 0.15])
    nonconformity = route.nonconformity(probability, weight, 1e-8)
    pvalue = route.conformal_pvalues(nonconformity, labels, nonconformity[:7])
    assert nonconformity.shape == (60, 2)
    assert pvalue.shape == (7, 2)
    assert np.all((pvalue > 0.0) & (pvalue <= 1.0))
    predicted = (probability[:7, 0] >= 0.5).astype(int)
    score = route.route_score(pvalue, predicted)
    assert np.all((score >= 0.0) & (score <= 2.0))
    selected = route.top_fraction(score, 0.25)
    assert int(np.sum(selected)) == 2
    print("SELFTEST_OK")


if __name__ == "__main__":
    main()
