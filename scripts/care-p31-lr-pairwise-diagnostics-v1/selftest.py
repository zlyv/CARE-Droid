#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1] / "care-p31-lr-source-safe-fusion-v1"
sys.path.insert(0, str(root))
from transforms import transform

x = np.zeros((2,31)); x[1,:3] = [1,2,3]
value = transform(x, "B5_PAIRWISE_HELLINGER")
assert value.shape == (2,527) and np.isfinite(value).all()
print("P31_LR_PAIRWISE_DIAGNOSTICS_SELFTEST_PASS")

