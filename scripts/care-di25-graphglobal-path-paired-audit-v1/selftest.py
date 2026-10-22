#!/usr/bin/env python3
"""Contract tests for paired statistics."""

import numpy as np

from audit_core import exact_mcnemar, metrics, stratified_bootstrap_indices


def main():
    assert exact_mcnemar(0, 0) == 1.0
    assert abs(exact_mcnemar(0, 5) - 0.0625) < 1e-12
    assert exact_mcnemar(2, 2) == 1.0
    y = np.asarray([0, 0, 1, 1])
    p0 = np.asarray([0.1, 0.2, 0.8, 0.9])
    p1 = np.asarray([0.1, 0.7, 0.8, 0.9])
    m = metrics(y, p1, p0)
    assert m["fixed"] == 0 and m["broken"] == 1 and m["net"] == -1
    draws = list(stratified_bootstrap_indices(y, 10, 17))
    assert len(draws) == 10
    assert all(np.sum(y[index] == 0) == 2 and np.sum(y[index] == 1) == 2 for index in draws)
    print("GRAPHGLOBAL_PATH_PAIRED_AUDIT_SELFTEST_PASS")


if __name__ == "__main__":
    main()

