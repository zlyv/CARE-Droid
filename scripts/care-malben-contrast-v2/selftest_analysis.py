#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from analyze_source_v2 import evaluate, exact_mcnemar_p


def main() -> None:
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    base = np.asarray([0.1, 0.9, 0.1, 0.9])
    expert = np.asarray([0.2, 0.2, 0.8, 0.8])
    route = np.asarray([0, 1, 1, 0], dtype=bool)
    row = evaluate(y, base, expert, route)
    assert row["fixed"] == 2 and row["broken"] == 0 and row["net"] == 2
    assert row["repairable_error_n"] == 2 and row["RER_at_25"] == 1.0
    assert exact_mcnemar_p(0, 2) == 0.5
    assert exact_mcnemar_p(0, 0) == 1.0
    print("source analysis self-test passed")


if __name__ == "__main__":
    main()

