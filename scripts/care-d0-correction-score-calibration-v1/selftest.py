#!/usr/bin/env python3
import numpy as np
from common import logit
b=np.asarray([.2,.8]);c=np.asarray([.4,.6]);d=logit(c)-logit(b)
assert np.allclose(1/(1+np.exp(-(logit(b)+d))),c)
opp=np.where(b>=.5,-d,d)
assert np.all(opp>0)
assert np.isfinite(logit([.2,.8])).all()
print("D0_SCORE_CALIBRATION_SELFTEST_PASS")

