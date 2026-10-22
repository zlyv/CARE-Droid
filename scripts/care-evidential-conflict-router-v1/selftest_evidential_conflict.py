#!/usr/bin/env python3
"""Artifact-independent formula checks for evidential conflict fusion."""

import numpy as np

from evidential_conflict_core import conflict_aware_fusion, exact_top_fraction


def main() -> None:
    # Strong agreement should be more certain than low evidence.
    agreement = np.asarray([[[12.0, 0.0]] * 4])
    ignorance = np.asarray([[[0.0, 0.0]] * 4])
    # Two strongly benign and two strongly malicious views create conflict.
    conflict = np.asarray([[
        [12.0, 0.0],
        [12.0, 0.0],
        [0.0, 12.0],
        [0.0, 12.0],
    ]])
    agree_result = conflict_aware_fusion(agreement)
    ignorant_result = conflict_aware_fusion(ignorance)
    conflict_result = conflict_aware_fusion(conflict)
    assert agree_result["uncertainty"][0] < ignorant_result["uncertainty"][0]
    assert conflict_result["conflict"][0] > agree_result["conflict"][0]
    assert conflict_result["uncertainty"][0] > agree_result["uncertainty"][0]
    assert np.isclose(ignorant_result["uncertainty"][0], 1.0)
    assert np.allclose(
        conflict_result["uncertainty"],
        2.0 / (np.sum(conflict_result["fused_evidence"], axis=1) + 2.0),
    )
    assert int(np.sum(exact_top_fraction(np.arange(20), 0.25))) == 5
    print("evidential conflict router self-test passed")


if __name__ == "__main__":
    main()
