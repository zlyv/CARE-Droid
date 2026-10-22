#!/usr/bin/env python3
import numpy as np
from run_audit import auc, oracle_threshold, threshold_net


def main():
    y = np.asarray([0, 0, 1, 1]); score = np.asarray([.1, .2, .8, .9])
    assert auc(y, score) == 1.0
    assert threshold_net(y, score, .5, 0) == (2, 0, 2)
    result = oracle_threshold(y, score, 0); assert result["oracle_net"] == 2
    print("DIRECT_C43_TEMPORAL_DIRECTION_SELFTEST_PASS")


if __name__ == "__main__": main()
