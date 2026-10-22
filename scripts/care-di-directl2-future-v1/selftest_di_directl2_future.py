#!/usr/bin/env python3
"""Small formula tests for DI25 Direct-L2 future diagnostic."""

from __future__ import annotations

import numpy as np


def main() -> None:
    base = np.asarray([0.9, 0.1, 0.52, 0.48])
    bounded = np.asarray([0.8, 0.2, 0.4, 0.6])
    direct = np.asarray([0.3, 0.7, 0.9, 0.1])
    route = np.asarray([True, True, False, False])
    bounded_final = np.where(route, bounded, base)
    direct_final = np.where(route, direct, base)
    assert np.allclose(bounded_final, [0.8, 0.2, 0.52, 0.48])
    assert np.allclose(direct_final, [0.3, 0.7, 0.52, 0.48])
    # Direct-L2 replaces the routed probability; it is not a residual update.
    assert not np.allclose(direct_final[:2], base[:2] + direct[:2])
    print("DI25 Direct-L2 future self-test passed", flush=True)


if __name__ == "__main__":
    main()
