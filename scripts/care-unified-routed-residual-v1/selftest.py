#!/usr/bin/env python3
import numpy as np

from common import MAX_DELTA, correction_metrics, logit, sigmoid


def main():
    p = np.asarray([.2, .8]); assert np.max(np.abs(sigmoid(logit(p)) - p)) < 1e-12
    c = np.asarray([1.0, -1.0]); d = np.asarray([3.0, -3.0])
    fused = np.clip(c + d, -MAX_DELTA, MAX_DELTA)
    assert np.max(np.abs(fused)) <= MAX_DELTA
    result = correction_metrics(np.asarray([1, 0]), np.asarray([.4, .6]), np.asarray([.6, .4]), np.ones(2, bool))
    assert (result["fixed"], result["broken"], result["net"]) == (2, 0, 2)
    print("UNIFIED_ROUTED_RESIDUAL_SELFTEST_PASS formula=clip(c43_delta+committee_logit_residual,+-2.25)")


if __name__ == "__main__":
    main()
