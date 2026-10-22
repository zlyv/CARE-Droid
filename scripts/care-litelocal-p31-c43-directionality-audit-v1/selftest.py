#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from direction_core import aggregate_host_values, knn_rms


def main():
    query = np.asarray([[0., 0.], [2., 2.]])
    reference = np.asarray([[0., 0.], [1., 1.], [3., 3.]])
    distance = knn_rms(query, reference, k=1)
    if not np.allclose(distance, [0., 1.]):
        raise RuntimeError("kNN RMS selftest failed: " + repr(distance))
    summary = aggregate_host_values(np.asarray([-1., 2., 3., 4.]))
    if summary["top3_mean"] != 3.0 or summary["fraction_positive"] != .75:
        raise RuntimeError("aggregation selftest failed")
    print("P31_C43_DIRECTIONALITY_AUDIT_SELFTEST_PASS")
    print("representations=CORE43,RELATION31,JOINT74")
    print("training_invoked=false inference_invoked=false route_modified=false")


if __name__ == "__main__":
    main()
