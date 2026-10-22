#!/usr/bin/env python3
from core import failure_type
assert failure_type(1,.2,.1)[0]=="WRONG_DIRECTION"
assert failure_type(1,.2,.205)[0]=="NEAR_ZERO"
assert failure_type(1,.2,.4)[0]=="RIGHT_DIRECTION_TOO_WEAK"
assert failure_type(1,.05,.2)[0]=="BOUND_LIMIT"
assert failure_type(1,.2,.6)[0]=="FIXED"
print("STAGE2_FAILURE_MECHANISM_SELFTEST_PASS")
print("training_invoked=false")
print("inference_invoked=false")

