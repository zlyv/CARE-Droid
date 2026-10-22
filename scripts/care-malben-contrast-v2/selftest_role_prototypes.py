#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from role_conditioned_prototypes import evidence_columns, fit_role_prototypes


def make_bag(role: int, center: float, rng: np.random.RandomState) -> np.ndarray:
    bag = rng.normal(center, 0.1, size=(8, 74)).astype(np.float32)
    bag[:, :31] = 0.0
    bag[:, role] = 1.0
    return bag


def main() -> None:
    rng = np.random.RandomState(5)
    raw = []
    labels = []
    for label, center in ((0, -2.0), (1, 2.0)):
        for _ in range(5):
            raw.append(make_bag(3, center, rng))
            labels.append(label)
    scaled = [np.array(value, copy=True) for value in raw]
    state = fit_role_prototypes(scaled, raw, np.asarray(labels), 8, 20, 13)
    assert state.use_role[0, 3] and state.use_role[1, 3]
    assert not state.use_role[0, 4] and not state.use_role[1, 4]
    benign = evidence_columns(scaled[0], raw[0], state)
    malware = evidence_columns(scaled[-1], raw[-1], state)
    assert benign.shape == malware.shape == (8, 9)
    assert np.mean(benign[:, 0]) > np.mean(benign[:, 1])
    assert np.mean(malware[:, 1]) > np.mean(malware[:, 0])
    assert np.allclose(benign[:, 2], benign[:, 1] - benign[:, 0])
    assert np.all((benign[:, 3] >= 0.0) & (benign[:, 3] <= 1.0))
    print("role-conditioned prototype self-test passed")


if __name__ == "__main__":
    main()
