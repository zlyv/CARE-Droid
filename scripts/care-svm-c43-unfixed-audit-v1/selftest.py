#!/usr/bin/env python3
import numpy as np

from common import binary_auc, logit, metrics, sigmoid
from run_audit import correct_category, error_category


def main():
    p = np.asarray([.01, .2, .5, .8, .99])
    assert np.allclose(sigmoid(logit(p)), p)
    assert abs(binary_auc([0, 0, 1, 1], [.1, .2, .8, .9]) - 1.0) < 1e-12
    result = metrics([0, 1, 0, 1], [.1, .4, .8, .9], [.1, .6, .4, .9], [0, 1, 1, 0])
    assert (result["fixed"], result["broken"], result["net"]) == (2, 0, 2)
    assert error_category(True, 1.0, 1.0) == "FIXED"
    assert error_category(False, 1.0, 3.0) == "STAGE1_MARGIN_EXCEEDS_MAX_DELTA"
    assert error_category(False, .01, 1.0) == "NEAR_ZERO"
    assert error_category(False, -.2, 1.0) == "WRONG_DIRECTION"
    assert error_category(False, .2, 1.0) == "CORRECT_DIRECTION_TOO_WEAK"
    assert correct_category(True, 1.0) == "BROKEN"
    print("SVM_C43_UNFIXED_SELFTEST_PASS", flush=True)


if __name__ == "__main__": main()
