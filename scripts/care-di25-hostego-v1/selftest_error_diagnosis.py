#!/usr/bin/env python3
from error_diagnosis_core import MAX_DELTA, cohort, residual_primary_failure


def main():
    assert cohort(True, False, True) == "A"
    assert cohort(False, False, False) == "B"
    assert cohort(False, True, True) == "C"
    assert cohort(False, True, False) == "D"
    assert cohort(True, True, True) == "E"
    assert cohort(True, True, False) == "F"
    assert residual_primary_failure(1, 1.0, MAX_DELTA + 0.1).startswith("D3")
    assert residual_primary_failure(1, -0.2, 1.0).startswith("D1")
    assert residual_primary_failure(0, -0.01, 1.0).startswith("D4")
    assert residual_primary_failure(0, -0.5, 1.0).startswith("D2")
    print("ERROR_DIAGNOSIS_SELFTEST_PASS")


if __name__ == "__main__":
    main()
