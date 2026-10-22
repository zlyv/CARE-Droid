#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from authority_core import CROSSING_MARGIN, MAX_DELTA, decoupled_correction, logit


def main():
    p = np.asarray([0.40, 0.60, 0.01, 0.90, 0.40])
    q = np.asarray([0.90, 0.90, 0.90, 0.49, 0.90])
    route = np.asarray([1, 1, 1, 1, 0], dtype=bool)
    out = decoupled_correction(p, q, route)
    assert out["probability"][0] > 0.5
    assert out["probability"][1] < 0.5
    assert out["bound_limited"][2]
    assert np.isclose(out["probability"][3], p[3], atol=1e-12)
    assert np.isclose(out["probability"][4], p[4], atol=1e-12)
    assert np.max(np.abs(out["delta"])) <= MAX_DELTA + 1e-12
    assert abs(abs(logit([out["probability"][0]])[0]) - CROSSING_MARGIN) < 1e-10
    print("DECOUPLED_AUTHORITY_SELFTEST_PASS")


if __name__ == "__main__":
    main()
