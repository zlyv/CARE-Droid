#!/usr/bin/env python3
from __future__ import annotations

import json
import numpy as np

from common import BUDGETS, DIRECT_METHOD, OUTPUT, discover, ensure_boundary, metrics, norm, read_csv, route_at_budget


def main():
    ensure_boundary(); assets = discover(); rows = read_csv(assets["predictions"])
    if len(rows) != 799 or len({norm(row["sha256"]) for row in rows}) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: direct predictions are not the 799 source APKs")
    architecture = json.loads(assets["architecture"].read_text(encoding="utf-8"))
    method = DIRECT_METHOD
    required = ["sha256", "label", "fold", "Global24__prob", "A25__score", "A25__routed", method + "__expert_prob", method + "__prob"]
    missing = [key for key in required if key not in rows[0]]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing prediction columns " + repr(missing))
    values = np.asarray([[float(row[key]) for key in ("Global24__prob", "A25__score", method + "__expert_prob")] for row in rows])
    if not np.isfinite(values).all():
        raise RuntimeError("BLOCKED_BY_CONTRACT: nonfinite full-cohort expert prediction")
    y = np.asarray([int(float(row["label"])) for row in rows]); fold = np.asarray([int(float(row["fold"])) for row in rows])
    sha = np.asarray([norm(row["sha256"]) for row in rows]); base = values[:, 0]; score = values[:, 1]; expert = values[:, 2]
    route25 = route_at_budget(score, sha, fold, .25)
    stored = np.asarray([int(float(row["A25__routed"])) for row in rows], bool)
    if not np.array_equal(route25, stored):
        raise RuntimeError("BLOCKED_BY_CONTRACT: reconstructed 25% route differs from frozen route")
    reference = metrics(y, base, expert, route25)
    if (reference["fixed"], reference["broken"], reference["net"]) != (49, 17, 32):
        raise RuntimeError("BLOCKED_BY_CONTRACT: Direct-C43 reference did not reproduce " + repr(reference))
    if "direct" not in json.dumps(architecture).casefold():
        raise RuntimeError("BLOCKED_BY_CONTRACT: architecture is not Direct-C43")
    print("SVM_DIRECT_C43_BUDGET_FRONTIER_PREFLIGHT_PASS")
    print({"rows": len(rows), "budgets": BUDGETS, "reference": reference, "full_oof_expert_finite": True,
           "future_rows_loaded": 0, "output": str(OUTPUT)}, flush=True)


if __name__ == "__main__": main()
