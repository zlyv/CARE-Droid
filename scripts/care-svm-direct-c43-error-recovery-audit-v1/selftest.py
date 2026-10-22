#!/usr/bin/env python3
from audit import reason


def main():
    assert reason(1, .2, .7, True) == "FIXED"
    assert reason(0, .8, .3, False) == "UNROUTED_EXPERT_COULD_FIX"
    assert reason(1, .2, .4, True) == "ROUTED_CORRECT_DIRECTION_TOO_WEAK"
    assert reason(0, .8, .9, True) == "ROUTED_WRONG_DIRECTION"
    assert reason(1, .2, .205, True) == "ROUTED_NEAR_ZERO_MOVEMENT"
    print("DIRECT_C43_ERROR_RECOVERY_SELFTEST_PASS")


if __name__ == "__main__": main()
