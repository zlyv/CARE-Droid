#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import binary_auc, bootstrap_auc_difference


def main() -> None:
    y = np.asarray([0, 0, 1, 1])
    perfect = np.asarray([.1, .2, .8, .9]); reversed_score = 1 - perfect
    assert binary_auc(y, perfect) == 1.0 and binary_auc(y, reversed_score) == 0.0
    result = bootstrap_auc_difference(y, perfect, reversed_score, 7, repeats=100)
    assert result["difference"] == 1.0 and result["ci95"][0] == 1.0
    print("CMW_POSTGATE_SELFTEST_PASS", flush=True)


if __name__ == "__main__":
    main()
