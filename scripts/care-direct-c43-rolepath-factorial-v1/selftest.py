#!/usr/bin/env python3
from ablation_common import ARMS
from rolepath_models import ARMS as SWITCHES

assert tuple(SWITCHES)==ARMS
assert SWITCHES["R11_ROLE_QUERY_READOUT"]==(1.0,1.0)
assert SWITCHES["R10_ROLE_QUERY_ONLY"]==(1.0,0.0)
assert SWITCHES["R01_ROLE_READOUT_ONLY"]==(0.0,1.0)
assert SWITCHES["R00_ROLE_NEITHER"]==(0.0,0.0)
assert len(set(SWITCHES.values()))==4
print("DIRECT_C43_ROLEPATH_FACTORIAL_SELFTEST_PASS arms=4")
