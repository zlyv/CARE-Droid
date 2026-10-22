#!/usr/bin/env python3
"""Recombine frozen Manifest-only Stage-I/routes with frozen 2017-only CFG experts.

Reads cached per-APK predictions only, never reads APKs, and writes exclusively to
outputs/care_manifest_cfg_witness_fixed_source_v1.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/boyuanjie/projects/tasa-droid")
HERE = ROOT / "scripts/care-manifest-cfg-witness-fixed-source-v1"
OUT = ROOT / "outputs/care_manifest_cfg_witness_fixed_source_v2"
MANIFEST = ROOT / "outputs/care_manifest_four_member_disagreement_v1/run_20261019_pilot01"
CFG_SOURCE = ROOT / "outputs/care_cfg_witness_fixed_ablation_v1"
CFG_FUTURE = ROOT / "outputs/care_cfg_witness_fixed_source_future_v2"
sys.path.insert(0, str(ROOT / "scripts/care-multiyear-development-v1"))

from metrics import cascade, classification  # noqa: E402

ARMS = (
    "W0_FROZEN_BASELINE",
    "W1_WITNESS3",
    "W2_SIZE3_CONTROL",
    "W3_SIZE_CONDITIONAL_SHUFFLE",
)


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": h.hexdigest()}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=encode) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_csv(path: Path, values):
    values = list(values)
    fields = list(dict.fromkeys(key for row in values for key in row)) or ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    temp.replace(path)


def asset_paths():
    return [
        HERE / "evaluate.py",
        MANIFEST / "protocol.json",
        MANIFEST / "per_apk_predictions.csv",
        MANIFEST / "asset_integrity_check.json",
        CFG_SOURCE / "per_apk_predictions.csv",
        CFG_SOURCE / "baseline_integrity_check.json",
        CFG_FUTURE / "per_apk_predictions.csv",
        CFG_FUTURE / "replay_validation.json",
        CFG_FUTURE / "baseline_integrity_check.json",
    ]


def expert_map():
    result = {}
    for row in rows(CFG_SOURCE / "per_apk_predictions.csv"):
        if not row["split"].startswith("S"):
            continue
        key = (2017, row["arm"], row["sha256"])
        if key in result:
            raise RuntimeError("duplicate source expert prediction")
        result[key] = (int(row["label"]), float(row["expert_probability"]))
    for row in rows(CFG_FUTURE / "per_apk_predictions.csv"):
        key = (int(row["test_year"]), row["arm"], row["sha256"])
        if key in result:
            raise RuntimeError("duplicate future expert prediction")
        result[key] = (int(row["label"]), float(row["expert_probability"]))
    expected = 4 * (799 + 793 + 399 + 397 + 400)
    if len(result) != expected:
        raise RuntimeError(f"expert prediction inventory {len(result)} != {expected}")
    return result


def seed_mean(performance):
    groups = defaultdict(list)
    for row in performance:
        key = (row["year"], row["anchor"], row["strategy"], row["arm"])
        groups[key].append(row)
    output = []
    identifiers = {"seed", "year", "anchor", "strategy", "arm"}
    for key, group in sorted(groups.items()):
        if len(group) != 3:
            raise RuntimeError(f"expected three Manifest seeds for {key}")
        row = {
            "year": key[0],
            "anchor": key[1],
            "strategy": key[2],
            "arm": key[3],
            "seed_count": 3,
        }
        for field in group[0]:
            if field in identifiers:
                continue
            try:
                row[field] = float(np.mean([float(item[field]) for item in group]))
            except (TypeError, ValueError):
                pass
        output.append(row)
    return output


def main():
    if OUT.exists():
        raise RuntimeError("output exists; refuse to overwrite")
    OUT.mkdir(parents=True)
    assets = [digest(path) for path in asset_paths()]
    atomic_json(OUT / "asset_manifest_before.json", assets)
    for path in (MANIFEST, CFG_SOURCE, CFG_FUTURE):
        integrity_name = "asset_integrity_check.json" if path == MANIFEST else "baseline_integrity_check.json"
        integrity = json.loads((path / integrity_name).read_text(encoding="utf-8"))
        if integrity.get("status") != "PASS":
            raise RuntimeError("upstream integrity is not PASS: " + str(path))

    manifest_rows = rows(MANIFEST / "per_apk_predictions.csv")
    experts = expert_map()
    groups = defaultdict(list)
    for row in manifest_rows:
        key = (int(row["year"]), row["seed"], row["anchor"], row["strategy"])
        groups[key].append(row)

    performance = []
    transitions = []
    maximum_w0_replay_difference = 0.0
    for (year, seed, anchor, strategy), group in sorted(groups.items()):
        group.sort(key=lambda item: item["sha256"])
        ids = [item["sha256"] for item in group]
        if len(ids) != len(set(ids)):
            raise RuntimeError("duplicate SHA in Manifest prediction group")
        labels = np.asarray([int(item["label"]) for item in group], int)
        base = np.asarray([float(item["stage1_probability"]) for item in group], float)
        route = np.asarray([int(item["routed"]) == 1 for item in group], bool)
        original_expert = np.asarray([float(item["expert_probability"]) for item in group], float)
        arm_probability = {}
        for arm in ARMS:
            probability = []
            for sid, label in zip(ids, labels):
                key = (year, arm, sid)
                if key not in experts:
                    raise RuntimeError("missing expert prediction " + repr(key))
                expert_label, value = experts[key]
                if expert_label != int(label):
                    raise RuntimeError("label mismatch " + repr(key))
                probability.append(value)
            probability = np.asarray(probability, float)
            arm_probability[arm] = probability
            if arm == ARMS[0]:
                difference = float(np.max(np.abs(probability - original_expert)))
                maximum_w0_replay_difference = max(maximum_w0_replay_difference, difference)
                if difference > 1e-6:
                    raise RuntimeError(f"Manifest W0 replay mismatch {year}/{seed}/{anchor}/{strategy}: {difference}")
            metric = cascade(labels, base, probability, route)
            stage1_metric = classification(labels, base)
            stage1_error = (base >= 0.5) != labels
            routed_error = stage1_error & route
            performance.append(
                {
                    "year": year,
                    "seed": seed,
                    "anchor": anchor,
                    "strategy": strategy,
                    "arm": arm,
                    "test_n": len(group),
                    "route_n": int(route.sum()),
                    "route_fraction": float(route.mean()),
                    "stage1_errors": int(stage1_error.sum()),
                    "routed_stage1_errors": int(routed_error.sum()),
                    "route_error_recall": float(routed_error.sum() / max(stage1_error.sum(), 1)),
                    "stage1_BA": stage1_metric["BA"],
                    "stage1_AUC": stage1_metric["AUC"],
                    **metric,
                }
            )
        w0 = arm_probability[ARMS[0]]
        w1 = arm_probability[ARMS[1]]
        final0 = np.where(route, w0, base)
        final1 = np.where(route, w1, base)
        for i, sid in enumerate(ids):
            transitions.append(
                {
                    "year": year,
                    "seed": seed,
                    "anchor": anchor,
                    "strategy": strategy,
                    "sha256": sid,
                    "label": int(labels[i]),
                    "routed": int(route[i]),
                    "stage1_probability": float(base[i]),
                    "w0_probability": float(w0[i]),
                    "w1_probability": float(w1[i]),
                    "w0_correct": int(int(final0[i] >= 0.5) == int(labels[i])),
                    "w1_correct": int(int(final1[i] >= 0.5) == int(labels[i])),
                }
            )

    means = seed_mean(performance)
    mean_map = {
        (int(row["year"]), row["anchor"], row["strategy"], row["arm"]): row for row in means
    }
    comparisons = []
    years = (2017, 2018, 2019, 2020, 2021)
    anchors = sorted({row["anchor"] for row in means})
    strategies = sorted({row["strategy"] for row in means})
    for anchor in anchors:
        for strategy in strategies:
            for reference in (ARMS[0], ARMS[2], ARMS[3]):
                for year in years:
                    candidate = mean_map[(year, anchor, strategy, ARMS[1])]
                    control = mean_map[(year, anchor, strategy, reference)]
                    comparisons.append(
                        {
                            "scope": str(year),
                            "anchor": anchor,
                            "strategy": strategy,
                            "candidate": ARMS[1],
                            "reference": reference,
                            "BA_gain": candidate["BA"] - control["BA"],
                            "Net_gain": candidate["Net"] - control["Net"],
                            "AUC_gain": candidate["AUC"] - control["AUC"],
                        }
                    )
                future_candidate = [mean_map[(year, anchor, strategy, ARMS[1])] for year in years[1:]]
                future_control = [mean_map[(year, anchor, strategy, reference)] for year in years[1:]]
                comparisons.append(
                    {
                        "scope": "FUTURE_MACRO_2018_2021",
                        "anchor": anchor,
                        "strategy": strategy,
                        "candidate": ARMS[1],
                        "reference": reference,
                        "BA_gain": float(np.mean([a["BA"] - b["BA"] for a, b in zip(future_candidate, future_control)])),
                        "Net_gain": float(np.sum([a["Net"] - b["Net"] for a, b in zip(future_candidate, future_control)])),
                        "AUC_gain": float(np.mean([a["AUC"] - b["AUC"] for a, b in zip(future_candidate, future_control)])),
                        "positive_BA_years": int(np.sum([a["BA"] > b["BA"] for a, b in zip(future_candidate, future_control)])),
                    }
                )

    current = [digest(Path(item["path"])) for item in assets]
    changed = [
        {"before": before, "after": after}
        for before, after in zip(assets, current)
        if before != after
    ]
    integrity = {
        "status": "PASS" if not changed else "FAIL",
        "checked": len(assets),
        "changed": changed,
        "scope": "named frozen Manifest and CFG prediction assets; no APK bytes",
    }
    if changed:
        raise RuntimeError("upstream assets changed during evaluation")

    atomic_csv(OUT / "performance_per_seed.csv", performance)
    atomic_csv(OUT / "performance_seed_mean.csv", means)
    atomic_csv(OUT / "w1_comparisons.csv", comparisons)
    with gzip.open(OUT / "per_apk_w0_w1.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(transitions[0]))
        writer.writeheader()
        writer.writerows(transitions)
    atomic_json(
        OUT / "protocol.json",
        {
            "status": "FROZEN_RECOMBINATION",
            "stage1_features": "Manifest135 only",
            "stage1_anchors": anchors,
            "stage1_training_years": [2017],
            "evaluation_years": list(years),
            "router_strategies": strategies,
            "router_inputs": "Manifest classifier probabilities only",
            "expert_training_years": [2017],
            "expert_arms": list(ARMS),
            "rolling_training_used": False,
            "new_training_performed": False,
            "future_labels_used_for_training_or_routing": False,
            "threshold": 0.5,
            "future_years_are_development": True,
        },
    )
    atomic_json(
        OUT / "replay_validation.json",
        {
            "status": "PASS",
            "maximum_W0_probability_difference": maximum_w0_replay_difference,
            "tolerance": 1e-6,
            "meaning": "Recombined W0 exactly reproduces the prior Manifest cascade expert input.",
        },
    )
    atomic_json(OUT / "baseline_integrity_check.json", integrity)
    atomic_json(
        OUT / "final_decision.json",
        {
            "status": "MANIFEST_STAGE1_CFG_WITNESS_FIXED_SOURCE_EVALUATION_COMPLETE",
            "model_training_performed": False,
            "rolling_training_used": False,
            "future_years_are_development": True,
            "next_action": "Report all frozen Manifest anchors/routes; do not choose a winner from historical future years.",
        },
    )

    lines = [
        "# Manifest第一层 + 2017-only CFG Witness组合评估",
        "",
        "第一层与路由仅使用Manifest135；第二层W0--W3均只使用2017训练。",
        "以下表格报告两个预注册分歧路由在三个Manifest种子上的平均值。",
        "",
        "| 年份 | 第一层 | 路由 | 方案 | 深查占比 | Stage-I BA | Final BA | AUC | Fixed | Broken | Net | 错误覆盖 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for year in years:
        for anchor in anchors:
            for strategy in ("STD25_REF", "OTHER3_OP25_REF"):
                for arm in (ARMS[0], ARMS[1], ARMS[2], ARMS[3]):
                    row = mean_map[(year, anchor, strategy, arm)]
                    lines.append(
                        f"| {year} | {anchor} | {strategy} | {arm} | {row['route_fraction']:.3f} | "
                        f"{row['stage1_BA']:.4f} | {row['BA']:.4f} | {row['AUC']:.4f} | "
                        f"{row['Fixed']:.1f} | {row['Broken']:.1f} | {row['Net']:.1f} | "
                        f"{row['route_error_recall']:.3f} |"
                    )
    lines += [
        "",
        "2018--2021为已反复观察的开发年份；本次不依据未来结果选择第一层、路由或第二层方案。",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("MANIFEST_CFG_WITNESS_FIXED_SOURCE_DONE", flush=True)


if __name__ == "__main__":
    main()
