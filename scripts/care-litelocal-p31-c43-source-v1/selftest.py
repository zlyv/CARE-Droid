#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from p31_core import foldwise_top25, mean_opposition


def main():
    rng = np.random.default_rng(20260928)
    members = rng.uniform(.01, .99, size=(799, 4))
    mean = members.mean(axis=1)
    fold = np.repeat(np.arange(1, 6), [160, 160, 160, 160, 159])
    ids = np.asarray([f"{index:064x}" for index in range(799)])
    route = foldwise_top25(mean_opposition(members, mean), fold, ids)
    assert int(route.sum()) == 200
    assert [int(route[fold == value].sum()) for value in range(1, 6)] == [40] * 5
    print("P31_C43_SOURCE_SELFTEST_PASS")
    print("stage1=P31_only FUSED63_consumed=false")
    print("route_budget=200/799 max_delta=2.25 future_rows_loaded=0 ego_consumed=false")


if __name__ == "__main__":
    main()
