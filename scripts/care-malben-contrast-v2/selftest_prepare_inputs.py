#!/usr/bin/env python3
"""Deterministic checks for frozen-input recovery helpers."""

from __future__ import annotations

import numpy as np

from prepare_frozen_inputs import derive_router, valid_router


def main() -> None:
    rows = []
    for index in range(799):
        fold = index % 5 + 1
        base = 0.8 if index % 2 else 0.2
        members = (0.1, 0.2, 0.8, 0.9) if index % 3 == 0 else (base,) * 4
        rows.append({
            "sha256": f"{index:064x}",
            "label": str(index % 2),
            "fold": str(fold),
            "p_stage1": str(base),
            "p_lr": str(members[0]),
            "p_rf": str(members[1]),
            "p_xgb": str(members[2]),
            "p_svm": str(members[3]),
            "route_A25": str(int(index < 200)),
        })
    output = derive_router(rows)
    assert valid_router(output)
    route = np.asarray([int(row["route_DI"]) for row in output])
    assert int(np.sum(route)) == 200
    assert all(0.0 <= float(row["U_DI"]) <= 1.0 for row in output)
    print("frozen-input preparation self-test passed")


if __name__ == "__main__":
    main()
