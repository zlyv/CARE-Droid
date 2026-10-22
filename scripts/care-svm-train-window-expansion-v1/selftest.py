#!/usr/bin/env python3
import numpy as np
from common import grouped_top25, comparison

route = grouped_top25(np.arange(8, dtype=float), np.asarray(list("abcdefgh")), np.zeros(8, int))
assert int(route.sum()) == 2
result = comparison(np.asarray([0, 0, 1, 1]), np.asarray([.2, .8, .2, .8]),
    np.asarray([.2, .2, .8, .2]), np.ones(4, bool))
assert result["fixed"] == 2 and result["broken"] == 1 and result["net"] == 1
assert result["fixed_FP"] == 1 and result["fixed_FN"] == 1 and result["broken_TP"] == 1
print("SVM_TRAIN_WINDOW_SELFTEST_PASS")
