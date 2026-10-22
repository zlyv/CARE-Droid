#!/usr/bin/env python3
import tempfile
from pathlib import Path

import numpy as np

from common import metrics
from stage1_core import mean_opposition, top25

y = np.asarray([0, 0, 1, 1])
base = np.asarray([.1, .8, .2, .9])
candidate = np.asarray([.1, .2, .8, .9])
m = metrics(y, base, candidate, np.ones(4, bool))
assert m["fixed"] == 2 and m["broken"] == 0 and m["net"] == 2
members = np.asarray([[.1, .2, .3, .4], [.6, .7, .8, .9]])
score = mean_opposition(members, np.asarray([.2, .8]))
assert np.allclose(score, [.25, .25])
route = top25(np.arange(8), np.asarray([str(x) for x in range(8)]))
assert route.sum() == 2
print("CARE_DROID_FORMAL_DIAGNOSTICS_SELFTEST_PASS")
print("stage1=P31_RF router=RF_anchored_MeanOpposition_DI25 stage2=A0_ROLE_C43_RQ max_delta=2.25")
