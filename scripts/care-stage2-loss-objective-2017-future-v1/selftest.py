#!/usr/bin/env python3
"""Unit-test T0/T1/T2 objective formulas without reading formal APK/model assets."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import run_experiment as r
class V:pass
v=V();v.torch=torch
logits=torch.tensor([-2.,0.,2.]);labels=torch.tensor([0.,1.,1.])
t0,_=r.objective(v,logits,labels,'T0_W0_BCE');expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
assert torch.allclose(t0,expected)
t1,_=r.objective(v,logits,labels,'T1_FOCAL_GAMMA2');assert torch.isfinite(t1) and t1>0
t2,single=r.objective(v,logits,labels,'T2_BCE_RANK01');assert torch.isfinite(t2) and t2>t0 and not single
_,single=r.objective(v,logits[:1],labels[:1],'T2_BCE_RANK01');assert single
print('LOSS_OBJECTIVE_SELFTEST_PASS')
