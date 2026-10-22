#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from evidence import fit_positive_lift, transform_certificate


def main():
    # family 0 positively accompanies role 0; family 1 has insufficient support.
    x = np.asarray([[1,0],[1,0],[1,0],[0,1],[0,0],[0,0]], float)
    role = np.asarray([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]], float)
    model = fit_positive_lift(x, role, np.ones(len(x), bool), min_support=3)
    weight = np.asarray([[1,0],[1,0],[0,1],[0,1]], float)
    values = transform_certificate(model, np.asarray([[1,0],[0,0],[1,0],[0,1]], float), weight)
    assert values["certificate"][0] > 0
    assert values["certificate"][1] == 0  # no observed context is never benign evidence
    assert values["certificate"][2] == 0  # context does not explain the observed Role
    assert np.all(values["certificate"] >= 0)
    assert np.all(values["certificate"] <= 1 + 1e-12)
    print("SVM_POSITIVE_BENIGN_SELFTEST_PASS absence_is_zero=true low_anomaly_unused=true")


if __name__ == "__main__":
    main()

