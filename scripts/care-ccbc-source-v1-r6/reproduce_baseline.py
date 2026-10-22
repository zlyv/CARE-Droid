#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import OUTPUT, dump_json, load_formal_rows, metrics


EXPECTED_STAGE1 = {
    "BA": 0.9273809523809524,
    "AUC": 0.9763784461152882,
    "F1": 0.9256410256410257,
    "errors": 58,
}
EXPECTED_FORMAL = {
    "BA": 0.9323840852130325,
    "AUC": 0.9756109022556392,
    "F1": 0.9305912596401028,
    "errors": 54,
    "fixed": 11,
    "broken": 7,
    "net": 4,
}


def close(observed, expected, tolerance=5e-6):
    return abs(float(observed) - float(expected)) <= tolerance


def main() -> None:
    rows = load_formal_rows()
    y = np.asarray([row["label"] for row in rows], int)
    fold = np.asarray([row["fold"] for row in rows], int)
    base = np.asarray([row["stage1_prob"] for row in rows], float)
    route = np.asarray([row["routed"] for row in rows], bool)
    final = np.asarray([row["formal_final_prob"] for row in rows], float)
    stage1 = metrics(y, base, base, np.zeros(len(rows), bool))
    formal = metrics(y, base, final, route)
    errors = (base >= .5) != y
    captured = int(np.sum(errors & route))
    route_n = int(route.sum())
    reproduction_errors = []
    if len(rows) != 799:
        reproduction_errors.append("source_rows")
    if [int(np.sum(fold == value)) for value in range(1, 6)] != [160, 160, 160, 160, 159]:
        reproduction_errors.append("fold_sizes")
    if [int(np.sum(route & (fold == value))) for value in range(1, 6)] != [40] * 5:
        reproduction_errors.append("route_per_fold")
    for key, expected in EXPECTED_STAGE1.items():
        if not close(stage1[key], expected):
            reproduction_errors.append(f"stage1_{key}")
    for key, expected in EXPECTED_FORMAL.items():
        if not close(formal[key], expected):
            reproduction_errors.append(f"formal_{key}")
    if captured != 45:
        reproduction_errors.append("captured_errors")
    result = {
        "status": "PASS" if not reproduction_errors else "FAIL",
        "source_rows": len(rows),
        "fold_sizes": [int(np.sum(fold == value)) for value in range(1, 6)],
        "stage1": stage1,
        "router": {
            "routed": route_n, "captured_errors": captured,
            "stage1_errors": int(errors.sum()),
            "error_recall_at_25": captured / max(int(errors.sum()), 1),
            "precision_at_25": captured / max(route_n, 1),
            "routed_stage1_correct": int(np.sum(route & ~errors)),
        },
        "formal": formal,
        "expected_stage1": EXPECTED_STAGE1,
        "expected_formal": EXPECTED_FORMAL,
        "mismatches": reproduction_errors,
        "future_rows_loaded": 0,
    }
    dump_json(OUTPUT / "baseline_reproduction.json", result)
    if reproduction_errors:
        print("CCBC_BASELINE_REPRODUCTION_FAIL", reproduction_errors)
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal baseline mismatch")
    print("CCBC_BASELINE_REPRODUCTION_PASS")
    print(f"stage1_BA={stage1['BA']:.12f} errors={stage1['errors']}")
    print(f"routed={route_n} captured={captured}/{int(errors.sum())}")
    print(f"formal_BA={formal['BA']:.12f} fixed={formal['fixed']} broken={formal['broken']} net={formal['net']}")


if __name__ == "__main__":
    main()

