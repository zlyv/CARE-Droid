#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import (ARMS, MEMBERS, OUTPUT, SOURCE_MEMBER, SOURCE_RF_REFERENCE, arm_probability,
    classification_metrics, dump_json, grouped_top25, mean_opposition, norm, read_csv, write_csv)


def main():
    rows = read_csv(SOURCE_MEMBER); sha = np.asarray([norm(row["sha256"]) for row in rows])
    y = np.asarray([int(float(row["label"])) for row in rows], int)
    fold = np.asarray([int(float(row["fold"])) for row in rows], int)
    member = np.asarray([[float(row[f"p_P31__{name}"]) for name in MEMBERS] for row in rows], float)
    audits = []
    for arm in ARMS:
        base = arm_probability(member, arm); score = mean_opposition(member, base)
        route = grouped_top25(score, sha, fold); target = OUTPUT / "references" / arm / "source_predictions.csv"
        output = []
        for index in range(len(rows)):
            output.append({"sha256": sha[index], "label": int(y[index]), "fold": int(fold[index]),
                "Global24__prob": float(base[index]),
                "Global24__A25__JointS7_ordinary__prob": float(base[index]),
                "A25__score": float(score[index]), "A25__routed": int(route[index]),
                "stage1_arm": arm, "router_anchor": arm})
        write_csv(target, output)
        error = (base >= .5) != y
        audits.append({"arm": arm, **classification_metrics(y, base), "routed": int(route.sum()),
            "captured_errors": int(np.sum(route & error)),
            "error_recall25": float(np.sum(route & error) / max(error.sum(), 1)),
            "route_precision": float(np.sum(route & error) / max(route.sum(), 1))})
    formal = sorted(read_csv(SOURCE_RF_REFERENCE), key=lambda row: norm(row["sha256"]))
    rebuilt = sorted(read_csv(OUTPUT / "references/RF/source_predictions.csv"), key=lambda row: norm(row["sha256"]))
    if [norm(row["sha256"]) for row in formal] != [norm(row["sha256"]) for row in rebuilt]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: rebuilt RF cohort differs from formal RF")
    probability_error = max(abs(float(a["Global24__prob"]) - float(b["Global24__prob"])) for a, b in zip(formal, rebuilt))
    score_error = max(abs(float(a["A25__score"]) - float(b["A25__score"])) for a, b in zip(formal, rebuilt))
    route_equal = all(int(a["A25__routed"]) == int(b["A25__routed"]) for a, b in zip(formal, rebuilt))
    if probability_error > 1e-14 or score_error > 1e-14 or not route_equal:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: RF reference mismatch p={probability_error} score={score_error} route={route_equal}")
    write_csv(OUTPUT / "source_stage1_router_metrics.csv", audits)
    dump_json(OUTPUT / "source_reference_audit.json", {"status": "PASS", "RF_max_probability_error": probability_error,
        "RF_max_router_score_error": score_error, "RF_route_exact": route_equal, "future_rows_loaded": 0})
    print("P31_CLASSIFIER_FAMILY_STAGE2_SOURCE_REFERENCES_PASS RF_exact=true", flush=True)


if __name__ == "__main__":
    main()
