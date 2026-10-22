#!/usr/bin/env python3
"""Formula check for routed absolute Stage-II classification."""

from __future__ import annotations

import numpy as np


def main() -> None:
    base = np.asarray([0.9, 0.1, 0.8, 0.2])
    direct = np.asarray([0.2, 0.8, 0.7, 0.3])
    route = np.asarray([True, True, False, False])
    final = np.where(route, direct, base)
    assert np.allclose(final, [0.2, 0.8, 0.8, 0.2])
    # Direct output is absolute; it is not added to Stage I as a residual.
    assert not np.allclose(final[:2], base[:2] + direct[:2])
    print("DI/Direct-L2 comparison self-test passed", flush=True)


if __name__ == "__main__":
    main()
