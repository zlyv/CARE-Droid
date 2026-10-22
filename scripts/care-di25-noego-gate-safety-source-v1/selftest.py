#!/usr/bin/env python3
import numpy as np

from gate_safety_core import apply_rule, choose_threshold


def main():
    y = np.asarray([1, 1, 0, 0, 1, 0])
    p = np.asarray([.2, .3, .2, .8, .8, .2])
    q = np.asarray([.95, .90, .20, .55, .80, .70])
    route = np.ones(6, dtype=bool)
    best, rows = choose_threshold(y, p, q, route)
    assert len(rows) == 51
    assert .5 <= best["threshold"] <= 1.0
    final, active, limited = apply_rule(p, q, route, best["threshold"])
    assert final.shape == p.shape and active.shape == p.shape and not limited.any()
    print("GATE_SAFETY_SELFTEST_PASS")


if __name__ == "__main__": main()

