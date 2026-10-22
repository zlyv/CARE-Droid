#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from typing import Any

from common import OUTPUT, discover_assets, dump_json, norm, read_csv, write_csv

PER_CELL = 4


def numeric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def select() -> list[dict[str, Any]]:
    assets = discover_assets(); rows = read_csv(assets["sample_audit"])
    if len(rows) != 2788:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: expected 2788 temporal rows, got {len(rows)}")
    for row in rows:
        row["sha256"] = norm(row["sha256"])
    chosen_errors = []
    cell_counts = {}
    for year in range(2017, 2022):
        for confusion in ("FN", "FP"):
            cell = [row for row in rows if int(row["year"]) == year and row["confusion"] == confusion
                    and row["wrong_confidence_bin"] == "CONFIDENT_WRONG_GE_0.25"]
            cell.sort(key=lambda row: (numeric(row, "expert_true_margin"), row["sha256"]))
            selected = cell[:PER_CELL]
            chosen_errors.extend(selected)
            cell_counts[f"{year}_{confusion}"] = {"available": len(cell), "selected": len(selected)}
    if len(chosen_errors) < 25:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: too few high-confidence errors: {len(chosen_errors)}")

    controls = [row for row in rows if int(row["stage1_error"]) == 0 and int(row["expert_correct"]) == 1]
    used = set(); pairs = []
    for error in chosen_errors:
        year = int(error["year"]); side = int(error["stage1_pred"]); routed = int(error["routed"])
        candidates = [row for row in controls if int(row["year"]) == year and int(row["stage1_pred"]) == side
                      and row["sha256"] not in used]
        if not candidates:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: no control for {error['sha256']}")
        def distance(control):
            route_penalty = 1.0 if int(control["routed"]) != routed else 0.0
            return (abs(float(control["stage1_prob"]) - float(error["stage1_prob"]))
                    + abs(float(control["expert_prob"]) - float(error["expert_prob"]))
                    + route_penalty)
        control = min(candidates, key=lambda row: (distance(row), row["sha256"]))
        used.add(control["sha256"])
        pairs.append({
            "pair_id": len(pairs) + 1, "year": year, "error_type": error["confusion"],
            "stage1_pred": side, "error_sha256": error["sha256"], "control_sha256": control["sha256"],
            "error_stage1_prob": error["stage1_prob"], "control_stage1_prob": control["stage1_prob"],
            "error_expert_prob": error["expert_prob"], "control_expert_prob": control["expert_prob"],
            "error_routed": routed, "control_routed": int(control["routed"]),
            "match_distance": distance(control),
        })
    write_csv(OUTPUT / "matched_selection.csv", pairs)
    dump_json(OUTPUT / "selection_audit.json", {
        "pairs": len(pairs), "unique_apks": len({p[k] for p in pairs for k in ("error_sha256", "control_sha256")}),
        "cell_counts": cell_counts, "selection_uses_labels": True,
        "future_labels_used_for_diagnostic_only": True,
        "selection_is_not_model_training_or_feature_selection": True,
        "matching": "same year and Stage-I prediction side; deterministic nearest Stage-I/Direct-C43 probability with routed mismatch penalty",
    })
    return pairs


def main() -> None:
    pairs = select()
    print(f"HCOA_PAIR_SELECTION_DONE pairs={len(pairs)} unique_apks={len({p[k] for p in pairs for k in ('error_sha256','control_sha256')})}")


if __name__ == "__main__":
    main()
