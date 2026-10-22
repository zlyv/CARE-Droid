#!/usr/bin/env python3
"""Fast, data-free checks for final-validation helpers."""
from __future__ import annotations

import numpy as np

import audit_final_protocol as audit
import run_seed_robustness as seed
import summarize_temporal_diagnostic as temporal


def main() -> None:
    assert audit.exact_p(5, 2) == 0.453125
    assert seed.exact_p(5, 2) == 0.453125
    assert temporal.exact_p(11, 1) == 0.00634765625
    y = np.asarray([0, 0, 1, 1])
    assert audit.metric(y, np.asarray([.1, .2, .8, .9])) == (1.0, 1.0, 1.0)
    assert temporal.DISCLAIMER.startswith("These 2018–2021 sets were previously inspected")
    print("FINAL_VALIDATION_SELFTEST_PASS")


if __name__ == "__main__":
    main()
