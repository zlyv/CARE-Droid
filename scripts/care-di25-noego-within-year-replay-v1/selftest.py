#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# Minimal arithmetic self-test for DI25 routing semantics.
y = np.array([0,0,1,1])
p1 = np.array([0.1,0.7,0.4,0.9])
pe = np.array([0.2,0.2,0.8,0.8])
route = np.array([0,1,1,0])
pf = np.where(route == 1, pe, p1)
assert np.array_equal((pf >= 0.5).astype(int), np.array([0,0,1,1]))
assert balanced_accuracy_score(y, pf >= 0.5) == 1.0
print('NOEGO_WITHIN_YEAR_SELFTEST_PASS')
