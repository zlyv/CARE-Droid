#!/usr/bin/env python3
"""Read-only expanded-routing audit for Manifest135 RF plus frozen CFG experts."""
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
HERE = ROOT / "scripts/care-manifest-rf-expanded-routing-v1"
OUT = ROOT / "outputs/care_manifest_rf_expanded_routing_v1"
MANIFEST = ROOT / "outputs/care_manifest_four_member_disagreement_v1/run_20261019_pilot01"
CFG_SOURCE = ROOT / "outputs/care_cfg_witness_fixed_ablation_v1"
CFG_FUTURE = ROOT / "outputs/care_cfg_witness_fixed_source_future_v2"
sys.path.insert(0, str(ROOT / "scripts/care-multiyear-development-v1"))

from metrics import cascade, classification  # noqa: E402

ANCHOR = "RF"
SOURCE_STRATEGY = "OTHER3_OP25_REF"
BUDGETS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.0)
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


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_value) + "\n",
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
        HERE / "run.sh",
        ROOT / "scripts/care-multiyear-development-v1/metrics.py",
        MANIFEST / "protocol.json",
        MANIFEST / "per_apk_predictions.csv",
        MANIFEST / "asset_integrity_check.json",
        CFG_SOURCE / "per_apk_predictions.csv",
        CFG_SOURCE / "baseline_integrity_check.json",
        CFG_FUTURE / "per_apk_predictions.csv",
        CFG_FUTURE / "replay_validation.json",
        CFG_FUTURE / "baseline_integrity_check.json",
    ]


def verify_upstream_status():
    checks = [
        (MANIFEST / "asset_integrity_check.json", "status"),
        (CFG_SOURCE / "baseline_integrity_check.json", "status"),
        (CFG_FUTURE / "baseline_integrity_check.json", "status"),
        (CFG_FUTURE / "replay_validation.json", "status"),
    ]
    for path, field in checks:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get(field) != "PASS":
            raise RuntimeError(f"upstream status not PASS: {path}")


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
        raise RuntimeError(f"expert inventory {len(result)} != {expected}")
    return result


def top_k(ids, score, budget):
    count = max(1, int(round(len(ids) * float(budget))))
    order = np.lexsort((np.asarray(ids), -np.asarray(score, float)))
    mask = np.zeros(len(ids), bool)
    mask[order[:count]] = True
    return mask


def seed_mean(performance):
    groups = defaultdict(list)
    for row in performance:
        key = (row["year"], row["route_mode"], row["budget"], row["arm"])
        groups[key].append(row)
    output = []
    identifiers = {"seed", "year", "route_mode", "budget", "arm"}
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        if len(group) != 3:
            raise RuntimeError(f"expected three Manifest seeds for {key}")
        row = {
            "year": key[0], "route_mode": key[1], "budget": key[2],
            "arm": key[3], "seed_count": 3,
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


def summarize_future(means):
    output = []
    keys = sorted({(r["route_mode"], r["budget"], r["arm"]) for r in means}, key=str)
    for route_mode, budget, arm in keys:
        group = [r for r in means if r["route_mode"] == route_mode and r["budget"] == budget
                 and r["arm"] == arm and int(r["year"]) >= 2018]
        if len(group) != 4:
            raise RuntimeError("future summary does not contain four years")
        stage1_errors = sum(r["stage1_errors"] for r in group)
        routed_errors = sum(r["routed_stage1_errors"] for r in group)
        fixed = sum(r["Fixed"] for r in group)
        broken = sum(r["Broken"] for r in group)
        route_n = sum(r["route_n"] for r in group)
        routed_correct = route_n - routed_errors
        output.append({
            "scope": "FUTURE_MACRO_2018_2021",
            "route_mode": route_mode,
            "budget": budget,
            "arm": arm,
            "mean_route_fraction": float(np.mean([r["route_fraction"] for r in group])),
            "macro_stage1_BA": float(np.mean([r["stage1_BA"] for r in group])),
            "macro_final_BA": float(np.mean([r["BA"] for r in group])),
            "macro_final_AUC": float(np.mean([r["AUC"] for r in group])),
            "stage1_errors": stage1_errors,
            "routed_stage1_errors": routed_errors,
            "route_error_recall": routed_errors / max(stage1_errors, 1),
            "Fixed": fixed,
            "routed_error_fix_rate": fixed / max(routed_errors, 1),
            "routed_errors_not_fixed": routed_errors - fixed,
            "Broken": broken,
            "damage_rate_on_routed_correct": broken / max(routed_correct, 1),
            "Net": fixed - broken,
            "unrouted_stage1_errors": stage1_errors - routed_errors,
            "final_errors": stage1_errors - fixed + broken,
        })
    return output


def marginal_bands(manifest_groups, experts):
    edges = (0.0,) + BUDGETS
    raw = []
    for (year, seed), group in sorted(manifest_groups.items()):
        group = sorted(group, key=lambda item: item["sha256"])
        ids = np.asarray([r["sha256"] for r in group])
        labels = np.asarray([int(r["label"]) for r in group], int)
        base = np.asarray([float(r["stage1_probability"]) for r in group], float)
        score = np.asarray([float(r["other3_opposition"]) for r in group], float)
        order = np.lexsort((ids, -score))
        rank = np.empty(len(ids), int); rank[order] = np.arange(len(ids))
        stage_error = (base >= 0.5) != labels
        for low, high in zip(edges[:-1], edges[1:]):
            begin = int(round(len(ids) * low)); end = int(round(len(ids) * high))
            band = (rank >= begin) & (rank < end)
            for arm in ARMS:
                expert = np.asarray([experts[(year, arm, sid)][1] for sid in ids], float)
                expert_correct = (expert >= 0.5) == labels
                fixed = stage_error & expert_correct & band
                broken = (~stage_error) & (~expert_correct) & band
                raw.append({
                    "year": year, "seed": seed, "band_low": low, "band_high": high,
                    "arm": arm, "n": int(band.sum()),
                    "stage1_errors": int((stage_error & band).sum()),
                    "Fixed": int(fixed.sum()), "Broken": int(broken.sum()),
                    "Net": int(fixed.sum() - broken.sum()),
                })
    groups = defaultdict(list)
    for row in raw:
        scope = "SOURCE_2017" if row["year"] == 2017 else "FUTURE_2018_2021"
        groups[(scope, row["seed"], row["band_low"], row["band_high"], row["arm"])].append(row)
    per_seed = []
    for (scope, seed, low, high, arm), group in groups.items():
        per_seed.append({
            "scope": scope, "seed": seed, "band_low": low, "band_high": high, "arm": arm,
            "n": sum(r["n"] for r in group),
            "stage1_errors": sum(r["stage1_errors"] for r in group),
            "Fixed": sum(r["Fixed"] for r in group),
            "Broken": sum(r["Broken"] for r in group),
            "Net": sum(r["Net"] for r in group),
        })
    groups = defaultdict(list)
    for row in per_seed:
        groups[(row["scope"], row["band_low"], row["band_high"], row["arm"])].append(row)
    output = []
    for (scope, low, high, arm), group in sorted(groups.items(), key=lambda item: str(item[0])):
        if len(group) != 3:
            raise RuntimeError("marginal band lacks three seeds")
        n = float(np.mean([r["n"] for r in group]))
        errors = float(np.mean([r["stage1_errors"] for r in group]))
        fixed = float(np.mean([r["Fixed"] for r in group]))
        broken = float(np.mean([r["Broken"] for r in group]))
        output.append({
            "scope": scope, "band_low": low, "band_high": high, "arm": arm,
            "n": n, "stage1_errors": errors,
            "stage1_error_rate": errors / max(n, 1),
            "Fixed": fixed, "fix_rate_within_band_errors": fixed / max(errors, 1),
            "Broken": broken, "Net": fixed - broken,
        })
    return output


def main():
    if OUT.exists():
        raise RuntimeError("output exists; refuse to overwrite")
    OUT.mkdir(parents=True)
    verify_upstream_status()
    assets = [digest(path) for path in asset_paths()]
    atomic_json(OUT / "asset_manifest_before.json", assets)

    selected = [r for r in rows(MANIFEST / "per_apk_predictions.csv")
                if r["anchor"] == ANCHOR and r["strategy"] == SOURCE_STRATEGY]
    groups = defaultdict(list)
    for row in selected:
        groups[(int(row["year"]), int(row["seed"]))].append(row)
    expected_groups = {(y, s) for y in range(2017, 2022) for s in (20261018, 20261019, 20261020)}
    if set(groups) != expected_groups:
        raise RuntimeError("Manifest group inventory changed")
    experts = expert_map()
    performance = []
    transitions = []
    for (year, seed), group in sorted(groups.items()):
        group.sort(key=lambda item: item["sha256"])
        ids = [r["sha256"] for r in group]
        labels = np.asarray([int(r["label"]) for r in group], int)
        base = np.asarray([float(r["stage1_probability"]) for r in group], float)
        score = np.asarray([float(r["other3_opposition"]) for r in group], float)
        original = np.asarray([int(r["routed"]) == 1 for r in group], bool)
        routes = [("ORIGINAL_REF", "ORIGINAL_REF", original)]
        routes += [("TOPK_OTHER", f"{budget:.2f}", top_k(ids, score, budget)) for budget in BUDGETS]
        stage_metric = classification(labels, base)
        stage_error = (base >= 0.5) != labels
        for route_mode, budget, route in routes:
            for arm in ARMS:
                probability = []
                for sid, label in zip(ids, labels):
                    expert_label, value = experts[(year, arm, sid)]
                    if expert_label != int(label):
                        raise RuntimeError("expert label mismatch")
                    probability.append(value)
                probability = np.asarray(probability, float)
                metric = cascade(labels, base, probability, route)
                performance.append({
                    "year": year, "seed": seed, "anchor": ANCHOR,
                    "score": "other3_opposition", "route_mode": route_mode,
                    "budget": budget, "arm": arm, "test_n": len(ids),
                    "route_n": int(route.sum()), "route_fraction": float(route.mean()),
                    "stage1_errors": int(stage_error.sum()),
                    "routed_stage1_errors": int((stage_error & route).sum()),
                    "route_error_recall": float((stage_error & route).sum() / max(stage_error.sum(), 1)),
                    "stage1_BA": stage_metric["BA"], "stage1_AUC": stage_metric["AUC"],
                    **metric,
                })
                final = np.where(route, probability, base)
                for i, sid in enumerate(ids):
                    transitions.append({
                        "year": year, "seed": seed, "sha256": sid, "label": int(labels[i]),
                        "route_mode": route_mode, "budget": budget, "arm": arm,
                        "routed": int(route[i]), "stage1_probability": float(base[i]),
                        "expert_probability": float(probability[i]), "final_probability": float(final[i]),
                    })

    means = seed_mean(performance)
    future = summarize_future(means)
    bands = marginal_bands(groups, experts)

    after = [digest(Path(item["path"])) for item in assets]
    changed = [{"before": a, "after": b} for a, b in zip(assets, after) if a != b]
    integrity = {
        "status": "PASS" if not changed else "FAIL", "checked": len(assets), "changed": changed,
        "scope": "named frozen Manifest predictions, CFG expert predictions, integrity files and audit code; no APK bytes",
    }
    if changed:
        raise RuntimeError("protected asset changed during evaluation")

    atomic_csv(OUT / "performance_per_seed.csv", performance)
    atomic_csv(OUT / "performance_seed_mean.csv", means)
    atomic_csv(OUT / "future_budget_summary.csv", future)
    atomic_csv(OUT / "marginal_band_summary.csv", bands)
    with gzip.open(OUT / "per_apk_expanded_routes.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(transitions[0]))
        writer.writeheader(); writer.writerows(transitions)
    atomic_json(OUT / "baseline_integrity_check.json", integrity)
    atomic_json(OUT / "protocol.json", {
        "status": "FROZEN_EXPANDED_ROUTING_AUDIT",
        "stage1_features": "Manifest135 only", "stage1_anchor": ANCHOR,
        "routing_score": "other3_opposition from four Manifest classifier probabilities",
        "routing_budgets": list(BUDGETS),
        "routing_selection": "top-k within each target cohort; descending score with SHA tie-break; labels unused",
        "original_route_reference": SOURCE_STRATEGY,
        "stage1_training_years": [2017], "expert_training_years": [2017],
        "evaluation_years": [2017, 2018, 2019, 2020, 2021],
        "expert_arms": list(ARMS), "threshold": 0.5,
        "new_training_performed": False, "rolling_training_used": False,
        "future_labels_used_for_routing_or_selection": False,
        "future_years_are_development": True,
        "selection_rule": "report the complete frozen curve; do not choose a budget from future outcomes",
    })
    atomic_json(OUT / "final_decision.json", {
        "status": "EXPANDED_ROUTING_CAPACITY_AUDIT_COMPLETE",
        "formal_route_changed": False, "formal_model_changed": False,
        "model_training_performed": False, "future_years_are_development": True,
        "next_action": "Report repair/damage saturation across all budgets; do not select a new budget from 2018-2021.",
    })

    lookup = {(r["route_mode"], str(r["budget"]), r["arm"]): r for r in future}
    lines = [
        "# Manifest135 RF 扩大路由预算审计", "",
        "固定Manifest135 RF、other3_opposition排序和2017-only W0--W3专家；不训练模型。",
        "2018--2021仅用于完整预算曲线的开发评价，不从未来结果选择预算。", "",
        "| 路由 | 预算/实际比例 | 方案 | Stage-I错误覆盖 | Fixed | Broken | Net | Final BA | AUC |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = [("ORIGINAL_REF", "ORIGINAL_REF")] + [("TOPK_OTHER", f"{b:.2f}") for b in BUDGETS]
    for mode, budget in order:
        for arm in ARMS:
            row = lookup[(mode, budget, arm)]
            shown = row["mean_route_fraction"]
            lines.append(
                f"| {mode} | {shown:.3f} | {arm} | {row['route_error_recall']:.3f} | "
                f"{row['Fixed']:.1f} | {row['Broken']:.1f} | {row['Net']:.1f} | "
                f"{row['macro_final_BA']:.4f} | {row['macro_final_AUC']:.4f} |"
            )
    lines += ["", "Fixed/Broken/Net为2018--2021四年三个Manifest种子均值的累计。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("MANIFEST_RF_EXPANDED_ROUTING_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
