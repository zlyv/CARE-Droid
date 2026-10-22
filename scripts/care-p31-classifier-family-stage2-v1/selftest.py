#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import ARMS, arm_probability, comparison_metrics, grouped_top25, mean_opposition


def main():
    member = np.asarray([[.1, .2, .3, .4], [.8, .7, .6, .9], [.4, .51, .45, .49], [.6, .4, .7, .3]])
    for arm in ARMS:
        probability = arm_probability(member, arm)
        assert probability.shape == (4,) and np.isfinite(probability).all()
        assert mean_opposition(member, probability).shape == (4,)
    score = np.arange(20, dtype=float); sha = np.asarray([f"{index:02d}" for index in range(20)])
    route = grouped_top25(score, sha, np.repeat([1, 2], 10))
    assert int(route.sum()) == 4 and int(np.sum(route[:10])) == 2 and int(np.sum(route[10:])) == 2
    y = np.asarray([0, 0, 1, 1]); base = np.asarray([.1, .9, .1, .9]); final = np.asarray([.1, .1, .9, .1])
    value = comparison_metrics(y, base, final, np.ones(4, bool))
    assert (value["fixed"], value["broken"], value["net"]) == (2, 1, 1)
    print("P31_CLASSIFIER_FAMILY_STAGE2_SELFTEST_PASS arms=LR,RF,XGB,SVM,MEAN4")


if __name__ == "__main__":
    main()
