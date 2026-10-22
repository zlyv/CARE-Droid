#!/usr/bin/env python3
import numpy as np
from future_common import metrics


def main():
    y = np.asarray([0, 0, 1, 1]); base = np.asarray([.1, .8, .2, .9]); final = np.asarray([.1, .2, .8, .9])
    result = metrics(y, base, final, np.ones(4, bool))
    assert result["fixed"] == 2 and result["broken"] == 0 and result["net"] == 2
    assert result["BA"] == 1.0
    print("SVM_DIRECT_C43_FUTURE_SELFTEST_PASS")


if __name__ == "__main__":
    main()
