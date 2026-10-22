#!/usr/bin/env python3
import numpy as np
from common import metrics, route_at_budget


def main():
    score = np.arange(20, dtype=float); sha = np.asarray([f"{x:064x}" for x in range(20)]); fold = np.repeat([1, 2], 10)
    r10 = route_at_budget(score, sha, fold, .10); r50 = route_at_budget(score, sha, fold, .50)
    assert r10.sum() == 2 and r50.sum() == 10 and np.all(~r10 | r50)
    y = np.asarray([0, 1, 0, 1]); base = np.asarray([.1, .4, .8, .9]); expert = np.asarray([.9, .8, .1, .2]); route = np.asarray([0, 1, 1, 0], bool)
    result = metrics(y, base, expert, route)
    assert result["fixed"] == 2 and result["broken"] == 0 and result["net"] == 2
    print("SVM_DIRECT_C43_BUDGET_FRONTIER_SELFTEST_PASS", flush=True)


if __name__ == "__main__": main()
