#!/usr/bin/env python3
from __future__ import annotations
import numpy as np

ARMS=("H_HOST_ONLY","HR_HOST_ROLE","HC_HOST_CONTEXT","HRC_FULL")

def interaction(v):
    """2x2 R x C interaction with Host held present."""
    return float(v["HRC_FULL"]-v["HR_HOST_ROLE"]-v["HC_HOST_CONTEXT"]+v["H_HOST_ONLY"])

def clip_logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    return np.log(p/(1-p))
