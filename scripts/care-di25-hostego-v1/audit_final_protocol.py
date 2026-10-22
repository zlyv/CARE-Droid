#!/usr/bin/env python3
"""Read-only paired source statistics, leakage contract, and holdout readiness."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from freeze_final_method import FINAL_FROZEN_METHOD, ROOT, blocked, contract, read_csv, sha256_file


FINAL = ROOT / "final_audits"
ARMS = {
    "M0": ROOT / "source/M0_host_anomaly_bounded/source_predictions.csv",
    "M1": ROOT / "source/M1_hostego_anomaly_bounded/source_predictions.csv",
    "M2": ROOT / "source/M2_host_malben_bounded/source_predictions.csv",
    "M3": ROOT / "source/M3_hostego_malben_bounded/source_predictions.csv",
}
COMPARISONS = (
    ("M0 vs M1", "M0_bounded", "M1_bounded"),
    ("M2 vs M3", "M2_bounded", "M3_bounded"),
    ("M0 Direct vs M0 Bounded", "M0_direct", "M0_bounded"),
    ("M1 Direct vs M1 Bounded", "M1_direct", "M1_bounded"),
    ("M2 Direct vs M2 Bounded", "M2_direct", "M2_bounded"),
    ("M3 Direct vs M3 Bounded", "M3_direct", "M3_bounded"),
)
LOCKED = {
    "M0_bounded": (0.9111058897243107, 0.9410213032581454, 0.9086229086229086, 87, 14),
    "M1_bounded": (0.9148715538847118, 0.9433521303258148, 0.9132653061224489, 90, 14),
    "M2_bounded": (0.9085902255639098, 0.9414786967418547, 0.9050715214564369, 84, 13),
    "M3_bounded": (0.9073402255639098, 0.9451503759398497, 0.9038961038961039, 83, 13),
    "M0_direct": (0.914881, 0.953440, 0.913924, 100, 24),
    "M1_direct": (0.913637, 0.956165, 0.913208, 101, 26),
    "M2_direct": (0.919887, 0.954825, 0.918987, 105, 25),
    "M3_direct": (0.914884, 0.954424, 0.914141, 105, 29),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def exact_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def load_probabilities() -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, list[dict]]]:
    all_rows = {arm: read_csv(path) for arm, path in ARMS.items()}
    for arm, rows in all_rows.items():
        if len(rows) != 799 or len({row["sha256"].strip().lower() for row in rows}) != 799:
            blocked(f"{arm} source prediction cohort mismatch")
    ids = [row["sha256"].strip().lower() for row in all_rows["M0"]]
    aligned = {}
    for arm, rows in all_rows.items():
        index = {row["sha256"].strip().lower(): row for row in rows}
        if set(index) != set(ids):
            blocked(f"{arm} source cohort differs from M0")
        aligned[arm] = [index[value] for value in ids]
    y = np.asarray([int(float(row["label"])) for row in aligned["M0"]], int)
    base = np.asarray([float(row["Global24__prob"]) for row in aligned["M0"]], float)
    route = np.asarray([int(float(row["A25__routed"])) == 1 for row in aligned["M0"]], bool)
    probability: dict[str, np.ndarray] = {}
    for arm, rows in aligned.items():
        columns = [name for name in rows[0] if name.startswith("DI25__") and name.endswith("__prob")]
        if len(columns) != 1:
            blocked(f"{arm} bounded column mismatch: {columns}")
        bounded = np.asarray([float(row[columns[0]]) for row in rows], float)
        direct_column = columns[0][:-6] + "__direct_stage2_probability"
        if direct_column not in rows[0]:
            blocked(f"{arm} direct diagnostic missing")
        direct_expert = np.asarray([float(row[direct_column]) for row in rows], float)
        probability[arm + "_bounded"] = bounded
        probability[arm + "_direct"] = np.where(route, direct_expert, base)
    return ids, y, base, probability, aligned


def metric(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    return float(balanced_accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p)), float(f1_score(y, p >= 0.5))


def bootstrap_ci(y: np.ndarray, left: np.ndarray, right: np.ndarray, count: int) -> dict:
    if count <= 0:
        return {"delta_BA_CI95_low": "not_run", "delta_BA_CI95_high": "not_run",
                "delta_AUC_CI95_low": "not_run", "delta_AUC_CI95_high": "not_run",
                "delta_F1_CI95_low": "not_run", "delta_F1_CI95_high": "not_run"}
    rng = np.random.RandomState(20260919)
    result = np.empty((count, 3), dtype=float)
    n = len(y)
    for i in range(count):
        selected = rng.randint(0, n, n)
        ys = y[selected]
        if len(np.unique(ys)) != 2:
            result[i] = np.nan
            continue
        lm, rm = metric(ys, left[selected]), metric(ys, right[selected])
        result[i] = np.asarray(rm) - np.asarray(lm)
    result = result[np.isfinite(result).all(axis=1)]
    low, high = np.percentile(result, [2.5, 97.5], axis=0)
    return {
        "delta_BA_CI95_low": low[0], "delta_BA_CI95_high": high[0],
        "delta_AUC_CI95_low": low[1], "delta_AUC_CI95_high": high[1],
        "delta_F1_CI95_low": low[2], "delta_F1_CI95_high": high[2],
    }


def source_statistics(bootstrap: int) -> None:
    ids, y, base, probability, aligned = load_probabilities()
    protected_before = {str(path): sha256_file(path) for path in ARMS.values()}
    base_metric = metric(y, base)
    expected_global = (0.8196741854636591, 0.9083709273182957, 0.8038147138964576)
    if not np.allclose(base_metric, expected_global, rtol=0, atol=1e-12):
        blocked(f"Global24 locked metrics changed expected={expected_global} observed={base_metric}")
    route = np.asarray([int(float(row["A25__routed"])) == 1 for row in aligned["M0"]], bool)
    base_ok = (base >= .5) == y
    for name, expected in LOCKED.items():
        p = probability[name]; observed_metric = metric(y, p); final_ok = (p >= .5) == y
        fixed = int(np.sum(route & ~base_ok & final_ok)); broken = int(np.sum(route & base_ok & ~final_ok))
        tolerance = 1e-12 if name.endswith("bounded") else 5.1e-7
        if not np.allclose(observed_metric, expected[:3], rtol=0, atol=tolerance) or (fixed, broken) != expected[3:]:
            blocked(f"{name} locked result changed expected={expected} observed={(*observed_metric, fixed, broken)}")
    rows = []
    for comparison, left_name, right_name in COMPARISONS:
        left, right = probability[left_name], probability[right_name]
        left_correct, right_correct = (left >= 0.5) == y, (right >= 0.5) == y
        n01 = int(np.sum(~left_correct & right_correct))
        n10 = int(np.sum(left_correct & ~right_correct))
        lm, rm = metric(y, left), metric(y, right)
        rows.append({
            "comparison": comparison, "n": len(y),
            "same_correct": int(np.sum(left_correct & right_correct)),
            "same_wrong": int(np.sum(~left_correct & ~right_correct)),
            "n01_left_wrong_right_correct": n01,
            "n10_left_correct_right_wrong": n10,
            "exact_mcnemar_two_sided_p": exact_p(n01, n10),
            "delta_BA_right_minus_left": rm[0] - lm[0],
            "delta_AUC_right_minus_left": rm[1] - lm[1],
            "delta_F1_right_minus_left": rm[2] - lm[2],
            "bootstrap_seed": 20260919 if bootstrap else "not_run",
            "bootstrap_resamples": bootstrap,
            **bootstrap_ci(y, left, right, bootstrap),
        })
    write_csv(FINAL / "source_statistical_analysis.csv", rows)

    m2, m3 = probability["M2_bounded"], probability["M3_bounded"]
    base_correct = (base >= 0.5) == y
    paired = []
    for i, sha in enumerate(ids):
        c2, c3 = bool((m2[i] >= 0.5) == y[i]), bool((m3[i] >= 0.5) == y[i])
        status = "same_correct" if c2 and c3 else "same_wrong" if not c2 and not c3 else "M2_wrong_M3_correct" if not c2 and c3 else "M2_correct_M3_wrong"
        paired.append({
            "sha256": sha, "label": int(y[i]), "Global24_correct": int(base_correct[i]),
            "M2_correct": int(c2), "M3_correct": int(c3),
            "M2_prob": float(m2[i]), "M3_prob": float(m3[i]), "M2_to_M3_status": status,
        })
    write_csv(FINAL / "M2_M3_paired_changes.csv", paired)
    summary = next(row for row in rows if row["comparison"] == "M2 vs M3")
    write_csv(FINAL / "M2_M3_mcnemar.csv", [{
        "comparison": "M2 vs M3", "n": summary["n"],
        "same_correct": summary["same_correct"], "same_wrong": summary["same_wrong"],
        "n01": summary["n01_left_wrong_right_correct"],
        "n10": summary["n10_left_correct_right_wrong"],
        "exact_mcnemar_two_sided_p": summary["exact_mcnemar_two_sided_p"],
    }])
    protected_after = {str(path): sha256_file(path) for path in ARMS.values()}
    if protected_before != protected_after:
        blocked("locked M0/M1/M2/M3 predictions changed during statistical audit")


def leakage_audit() -> None:
    value = {
        "FINAL_FROZEN_METHOD": FINAL_FROZEN_METHOD,
        "2017_labels_used_in_source_training": True,
        "future_labels_used_in_training": False,
        "future_labels_used_in_route": False,
        "future_labels_used_in_scaler": False,
        "future_labels_used_in_anomaly_fitting": False,
        "future_labels_used_in_early_stopping": False,
        "future_labels_used_in_threshold_selection": False,
        "future_labels_used_in_model_selection": False,
        "2018_2021_previously_inspected": True,
        "2018_2021_final_untouched_evaluation": False,
        "2022_2024_used_in_current_design": "unknown",
        "audit_note": "unknown is retained until provenance is explicitly confirmed; no false claim is inferred",
    }
    (FINAL / "final_leakage_audit.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def holdout_readiness() -> None:
    storage = Path(os.environ.get("CARE_STORAGE_ROOT", "/mnt/HDD1/boyuanjie/tasa-droid-storage"))
    rows = []
    for year in (2022, 2023, 2024):
        candidates = {
            "feature_file": [Path(f"artifacts/p1-features-{year}/features.csv"), storage / f"care-final-holdout/{year}/features.csv"],
            "endpoint_cache": [Path(f"artifacts/care-final-holdout/{year}/endpoint_centered_tensor_cache_{year}.jsonl.gz"), storage / f"care-final-holdout/{year}/endpoint_centered_tensor_cache_{year}.jsonl.gz"],
            "host_cache": [Path(f"artifacts/care-final-holdout/{year}/host_regions_{year}.npz"), storage / f"care-final-holdout/{year}/host_regions_{year}.npz"],
            "ego_cache": [Path(f"artifacts/care-final-holdout/{year}/ego_1hop_regions_{year}.npz"), storage / f"care-final-holdout/{year}/ego_1hop_regions_{year}.npz"],
            "global24_input": [Path(f"artifacts/care-final-holdout/{year}/global24_input.csv"), storage / f"care-final-holdout/{year}/global24_input.csv"],
            "label_file": [Path(f"artifacts/care-final-holdout/{year}/labels.csv"), storage / f"care-final-holdout/{year}/labels.csv"],
        }
        found = {key: next((path for path in paths if path.is_file()), None) for key, paths in candidates.items()}
        schema = False
        if found["host_cache"] and found["ego_cache"]:
            try:
                with np.load(found["host_cache"], allow_pickle=False) as host, np.load(found["ego_cache"], allow_pickle=False) as ego:
                    schema = host["host_values"].ndim == 2 and host["host_values"].shape[1] == 74 and "incoming_host_positions" in ego and "outgoing_host_positions" in ego
            except Exception:
                schema = False
        ready = all(found.values()) and schema
        rows.append({
            "year": year, "available": bool(any(found.values())),
            **{key: str(path) if path else "" for key, path in found.items()},
            "schema_compatible": bool(schema), "ready_for_frozen_inference": bool(ready),
            "missing": [key for key, path in found.items() if path is None] + ([] if schema else ["compatible_host_ego_schema"]),
        })
    output = ROOT / "final_holdout/final_holdout_readiness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    output.write_text(text, encoding="utf-8")
    # The protocol names this root-level path explicitly.  It is a byte-for-byte
    # copy of the final_holdout audit, not a second independently computed result.
    (ROOT / "final_holdout_readiness.json").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--readiness-only", action="store_true")
    args = parser.parse_args()
    contract()  # validates the final freeze before any derived audit is written
    FINAL.mkdir(parents=True, exist_ok=True)
    if args.readiness_only:
        holdout_readiness()
        print("FINAL_HOLDOUT_READINESS_AUDIT_PASS")
        return
    source_statistics(args.bootstrap)
    leakage_audit()
    holdout_readiness()
    print("FINAL_PROTOCOL_SOURCE_AUDIT_PASS")


if __name__ == "__main__":
    main()
