#!/usr/bin/env python3
import numpy as np
import torch

from common import binary_auc, correction_metrics
from models import fit_model, predict_model, standardize

rng = np.random.default_rng(7)
bags = [rng.normal(size=(n, 43)).astype(np.float32) for n in (1, 3, 7, 12, 2, 5)]
y = np.asarray([0, 0, 1, 1, 0, 1])
train, test, mean, scale = standardize(bags[:4], bags[4:])
assert mean.shape == (43,) and scale.shape == (43,)
for arm in ("M1_ABMIL", "M2_SIGNED_MIL", "M3_SIGNED_CLAM"):
    model = fit_model(arm, train, y[:4], 19, torch.device("cpu"), epochs=2, batch_size=2)
    output = predict_model(model, test, torch.device("cpu"))
    assert output["logit"].shape == (2,) and np.isfinite(output["logit"]).all()
assert np.isnan(binary_auc([0, 0], [.1, .2]))
metric = correction_metrics([0, 1], [.2, .8], [.7, .3], [1, 1])
assert metric["broken"] == 2 and metric["fixed"] == 0
print("SIGNED_HOST_MIL_SELFTEST_PASS")

