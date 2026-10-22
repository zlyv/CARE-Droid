#!/usr/bin/env python3
from __future__ import annotations

from audit_core import bounded_failure


def main():
    fixed = bounded_failure(1, .20, .60)
    wrong = bounded_failure(1, .20, .10)
    weak = bounded_failure(1, .20, .30)
    unreachable = bounded_failure(1, .01, .08)
    assert fixed["primary_failure_type"] == "FIXED_BOUNDED"
    assert wrong["primary_failure_type"] == "WRONG_OR_ZERO_DIRECTION"
    assert weak["primary_failure_type"] == "RIGHT_DIRECTION_TOO_WEAK"
    assert unreachable["primary_failure_type"] == "BOUND_UNREACHABLE"
    print("P31_C43_FAILURE_AUDIT_SELFTEST_PASS")
    print("training_invoked=false inference_invoked=false route_modified=false")


if __name__ == "__main__":
    main()
