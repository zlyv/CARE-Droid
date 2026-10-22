#!/usr/bin/env python3
"""Unit tests for grouping, metrics, bins, and protocol invariants of the audit."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location("audit",Path(__file__).with_name("audit.py")); a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)

assert a.outcome_group(1,.1,.9,True)=="G1_FIXED"
assert a.outcome_group(0,.9,.8,True)=="G3_BOTH_WRONG"
assert a.outcome_group(1,.9,.1,True)=="G2_BROKEN"
assert a.outcome_group(0,.1,.2,True)=="G4_BOTH_CORRECT"
assert a.outcome_group(0,.9,.2,False)=="G5_UNROUTED"
assert a.transition(1,.1,.9)=="FN_TO_TP"
assert a.transition(0,.9,.1)=="FP_TO_TN"
assert a.transition(1,.9,.1)=="TP_TO_FN"
assert a.transition(0,.1,.9)=="TN_TO_FP"
m=a.metric([0,0,1,1],[.1,.9,.1,.9]); assert abs(m["BA"]-.5)<1e-12 and (m["TN"],m["FP"],m["FN"],m["TP"])==(1,1,1,1)
assert a.risk_bin(.5)=="[0.5,0.6)" and a.risk_bin(.999)=="[0.9,1.0]"
assert a.jvalue(np.bool_(True)) is True
assert len(a.MANIFEST_SEEDS)==3 and len(a.MODEL_SEEDS)==3 and len(a.SPLITS)==5
print("UNIT_TEST_PASS",flush=True)
