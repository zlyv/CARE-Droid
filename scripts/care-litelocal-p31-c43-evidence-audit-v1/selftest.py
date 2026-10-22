#!/usr/bin/env python3
from __future__ import annotations

def percentile(value, control):
    control = [float(item) for item in control]
    below = sum(item < value for item in control)
    equal = sum(item == value for item in control)
    return float((below + .5 * equal) / len(control))


def main():
    assert percentile(2.0, [1.0, 2.0, 3.0]) == .5
    assert percentile(0.0, [1.0, 2.0]) == 0.0
    assert percentile(3.0, [1.0, 2.0]) == 1.0
    print("P31_C43_EVIDENCE_AUDIT_SELFTEST_PASS")
    print("training_invoked=false inference_invoked=false route_modified=false")


if __name__ == "__main__":
    main()
