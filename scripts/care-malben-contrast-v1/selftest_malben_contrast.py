#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from contrast_prototype_core import contrast_columns, fit_contrast_state


def main() -> None:
    rng = np.random.RandomState(7)
    bags = []
    labels = []
    for label, center in ((0, -2.0), (1, 2.0)):
        for _ in range(8):
            bags.append(rng.normal(center, 0.25, size=(10, 6)).astype(np.float32))
            labels.append(label)
    state = fit_contrast_state(bags, np.asarray(labels), 8, 3, 3, 11)
    benign = contrast_columns(np.full((2, 6), -2.0, dtype=np.float32), state)
    malware = contrast_columns(np.full((2, 6), 2.0, dtype=np.float32), state)
    assert benign.shape == malware.shape == (2, 4)
    assert np.isfinite(benign).all() and np.isfinite(malware).all()
    assert np.all(benign[:, 0] > benign[:, 1])
    assert np.all(malware[:, 1] > malware[:, 0])
    assert np.allclose(benign[:, 2], benign[:, 1] - benign[:, 0])
    assert np.all((benign[:, 3] >= 0.0) & (benign[:, 3] <= 1.0))
    print("malicious/benign contrast self-test passed")


if __name__ == "__main__":
    main()

