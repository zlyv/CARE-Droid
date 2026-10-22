#!/usr/bin/env python3
"""Executable Role-Class-K1, independent-fallback and shared-scale contract."""

import numpy as np
from shared_role_prototypes import evidence_columns, fit_role_prototypes


def row(role, shift):
    value = np.zeros(74, dtype=np.float32)
    value[role] = 1.0
    value[31:] = shift
    return value[None, :]


def main():
    raw = []
    labels = []
    # role 0 has support on both sides; role 1 has benign support below 20
    # but malware support at 20. This exercises independent fallback.
    for label in (0, 1):
        for index in range(20):
            raw.append(row(0, float(label) + index * 0.01)); labels.append(label)
        count = 19 if label == 0 else 20
        for index in range(count):
            raw.append(row(1, float(label) + index * 0.01)); labels.append(label)
    scaled = [value.copy() for value in raw]
    state = fit_role_prototypes(scaled, raw, np.asarray(labels), 16, 20, 17)
    assert state.use_role[0, 0] and state.use_role[1, 0]
    assert not state.use_role[0, 1] and state.use_role[1, 1]
    assert state.role_tau[0] > 0
    assert state.role_tau[1] == state.global_tau
    columns = evidence_columns(scaled[0], raw[0], state)
    assert columns.shape == (1, 9) and np.isfinite(columns).all()
    assert int(columns[0, 7]) == 0 and int(columns[0, 8]) == 0
    assert all(row["tau_role"] > 0 for row in state.quality_rows)
    print("SHARED_ROLE_PROTOTYPE_CONTRACT_PASS")


if __name__ == "__main__":
    main()
