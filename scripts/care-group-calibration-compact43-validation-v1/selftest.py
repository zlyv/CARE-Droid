#!/usr/bin/env python3
import numpy as np
from common import auc

assert abs(auc([0, 1], [0.1, 0.9]) - 1.0) < 1e-12
assert abs(auc([0, 1], [0.9, 0.1])) < 1e-12
print("CARE_GROUP_COMPACT_VALIDATION_SELFTEST_PASS")

