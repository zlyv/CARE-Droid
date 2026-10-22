#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import grouped_top25, mean_opposition, metric, route_metric

y = np.asarray([0, 0, 1, 1]); p = np.asarray([.1, .7, .4, .9]); members = np.column_stack([p, p, p, p])
value = metric(y, p)
assert value["TN"] == 1 and value["FP"] == 1 and value["FN"] == 1 and value["TP"] == 1
score = mean_opposition(members, p); route = grouped_top25(score, np.asarray(["a", "b", "c", "d"]), np.zeros(4, int))
assert int(route.sum()) == 1
routed = route_metric(y, p, members, np.asarray(["a", "b", "c", "d"]), np.zeros(4, int))
assert routed["routed"] == 1
print("P31_CLASSIFIER_FAMILY_SELFTEST_PASS members=LR,RF,XGB,SVM control=MEAN4", flush=True)

