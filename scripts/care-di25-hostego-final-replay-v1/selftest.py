#!/usr/bin/env python3
import math
from run_final_replay import sign
assert sign(1)==1 and sign(-1)==-1 and sign(1e-13)==0
assert abs(2.25*math.tanh(.4))<2.25
print("FINAL_REPLAY_SELFTEST_PASS")
