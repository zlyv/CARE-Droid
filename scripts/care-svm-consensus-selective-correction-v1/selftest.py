#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import binary_auc, metrics


def main():
    y = np.asarray([0, 0, 1, 1]); base = np.asarray([.1, .7, .3, .9]); candidate = np.asarray([.1, .2, .8, .9])
    result = metrics(y, base, candidate, np.ones(4, bool))
    assert result["fixed"] == 2 and result["broken"] == 0 and result["net"] == 2
    assert binary_auc(np.asarray([0, 1]), np.asarray([0.1, 0.9])) == 1.0
    members = np.asarray([[.1, .2, .3], [.8, .7, .2]])
    label = np.asarray([0, 1])
    support = np.mean((members >= .5) == label[:, None], axis=1)
    assert np.allclose(support, [1.0, 2 / 3])
    print("SVM_CONSENSUS_SELECTIVE_CORRECTION_SELFTEST_PASS")


if __name__ == "__main__":
    main()
