#!/usr/bin/env python3
from __future__ import annotations

from common import exact_mcnemar_p, norm


def main() -> None:
    assert norm(" A B ") == "a b"
    assert exact_mcnemar_p(0, 10) < .01
    assert exact_mcnemar_p(5, 5) == 1.0
    assert exact_mcnemar_p(0, 0) == 1.0
    print("HIGH_CONFIDENCE_OBSERVABILITY_SELFTEST_PASS")


if __name__ == "__main__": main()
