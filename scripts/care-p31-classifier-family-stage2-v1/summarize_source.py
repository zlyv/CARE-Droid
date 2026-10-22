#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json

import numpy as np

from common import (ARMS, OUTPUT, SOURCE_A0, classification_metrics, comparison_metrics, dump_json,
    method_name, norm, read_csv, write_csv)


RF_METHOD = "P31RF__RFRoute__A0_ROLE_ONLY__Bounded"


def arm_paths(arm):
    if arm == "RF":
        return OUTPUT / "references/RF/source_predictions.csv", SOURCE_A0, RF_METHOD
    return OUTPUT / "references" / arm / "source_predictions.csv", OUTPUT / "source" / arm / "source_predictions.csv", method_name(arm)


def main():
    aggregate = []; fold_rows = []; long_predictions = []
    for arm in ARMS:
        reference_path, prediction_path, method = arm_paths(arm)
        if not reference_path.is_file() or not prediction_path.is_file():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: source arm incomplete {arm}")
        reference = sorted(read_csv(reference_path), key=lambda row: norm(row["sha256"]))
        prediction = sorted(read_csv(prediction_path), key=lambda row: norm(row["sha256"]))
        if len(reference) != 799 or [norm(row["sha256"]) for row in reference] != [norm(row["sha256"]) for row in prediction]:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: source alignment failed {arm}")
        sha = np.asarray([norm(row["sha256"]) for row in reference]); y = np.asarray([int(row["label"]) for row in reference], int)
        fold = np.asarray([int(row["fold"]) for row in reference], int)
        base = np.asarray([float(row["Global24__prob"]) for row in reference])
        route = np.asarray([int(row["A25__routed"]) for row in reference], bool)
        key = method + "__prob"; expert_key = method + "__expert_prob"
        if key not in prediction[0] or expert_key not in prediction[0]:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing Stage-II columns arm={arm} key={key}")
        final = np.asarray([float(row[key]) for row in prediction]); expert = np.asarray([float(row[expert_key]) for row in prediction])
        stage1 = classification_metrics(y, base); stage2 = comparison_metrics(y, base, final, route)
        aggregate.append({"protocol": "source_2017_fixed_oof", "scope": "pooled", "arm": arm,
            **{f"stage1_{k}": v for k, v in stage1.items()}, **{f"stage2_{k}": v for k, v in stage2.items()},
            "BA_gain_pp": 100 * (stage2["BA"] - stage1["BA"]), "AUC_gain": stage2["AUC"] - stage1["AUC"],
            "F1_gain": stage2["F1"] - stage1["F1"]})
        for current_fold in range(1, 6):
            selected = fold == current_fold; local_stage1 = classification_metrics(y[selected], base[selected])
            local_stage2 = comparison_metrics(y[selected], base[selected], final[selected], route[selected])
            fold_rows.append({"protocol": "source_2017_fixed_oof", "fold": current_fold, "arm": arm,
                **{f"stage1_{k}": v for k, v in local_stage1.items()}, **{f"stage2_{k}": v for k, v in local_stage2.items()},
                "BA_gain_pp": 100 * (local_stage2["BA"] - local_stage1["BA"])})
        for index in range(len(y)):
            long_predictions.append({"protocol": "source_2017_fixed_oof", "sha256": sha[index],
                "fold": int(fold[index]), "label": int(y[index]), "arm": arm,
                "stage1_prob": float(base[index]), "routed": int(route[index]),
                "expert_prob": float(expert[index]), "stage2_prob": float(final[index])})
    rf = next(row for row in aggregate if row["arm"] == "RF")
    if abs(rf["stage1_BA"] - .9273809523809524) > 1e-12 or abs(rf["stage2_BA"] - .9323840852130325) > 1e-12:
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal RF source full system did not reproduce")
    if (rf["stage2_fixed"], rf["stage2_broken"], rf["stage2_net"]) != (11, 7, 4):
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal RF fixed/broken/net changed")
    write_csv(OUTPUT / "source_stage2_metrics.csv", aggregate); write_csv(OUTPUT / "source_stage2_fold_metrics.csv", fold_rows)
    write_csv(OUTPUT / "source_stage2_predictions.csv", long_predictions)
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    frozen = {**protocol, "source_results_frozen": True,
        "source_ranking_by_stage2_BA": [row["arm"] for row in sorted(aggregate, key=lambda row: -row["stage2_BA"])],
        "future_used_for_source_decision": False}
    text = json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (OUTPUT / "protocol_frozen.json").write_text(text, encoding="utf-8")
    (OUTPUT / "protocol_hash.txt").write_text(hashlib.sha256(text.encode("utf-8")).hexdigest() + "\n", encoding="utf-8")
    dump_json(OUTPUT / "source_summary.json", {"status": "PASS", "formal_RF_reproduced": True,
        "results": aggregate, "future_rows_loaded": 0})
    print("P31_CLASSIFIER_FAMILY_STAGE2_SOURCE_DONE", flush=True)
    for row in aggregate:
        print(f"{row['arm']}: Stage1_BA={row['stage1_BA']:.6f} Stage2_BA={row['stage2_BA']:.6f} "
              f"gain_pp={row['BA_gain_pp']:+.4f} fixed={row['stage2_fixed']} broken={row['stage2_broken']} net={row['stage2_net']}", flush=True)


if __name__ == "__main__":
    main()
