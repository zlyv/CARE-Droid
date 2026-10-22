#!/usr/bin/env python3
import numpy as np

from transforms import ARMS, transform


def main():
    x = np.zeros((3, 31), float)
    x[1, 0] = 2; x[1, 1] = 1
    x[2, :] = np.arange(1, 32)
    widths = {}
    for arm in ARMS:
        value = transform(x, arm)
        assert value.shape[0] == 3 and np.isfinite(value).all()
        widths[arm] = value.shape[1]
    assert widths["B5_PAIRWISE_HELLINGER"] == 527
    assert np.all(transform(x, "B1_HELLINGER_PRESENCE")[0] == 0)
    print("P31_LR_SOURCE_SAFE_FUSION_SELFTEST_PASS", widths)


if __name__ == "__main__": main()

