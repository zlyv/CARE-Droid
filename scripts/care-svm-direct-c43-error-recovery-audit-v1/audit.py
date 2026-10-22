#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(os.environ.get("ERA_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("ERA_OUTPUT", str(PROJECT / "outputs/care-svm-direct-c43-error-recovery-audit-v1"))).resolve()
METHOD = "P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"


def safe(path: Path) -> Path:
    root = (PROJECT / "outputs").resolve(); path = Path(path).resolve()
    if OUTPUT == root or root not in OUTPUT.parents or not OUTPUT.name.startswith("care-svm-direct-c43-error-recovery-audit-v"):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: unsafe output root {OUTPUT}")
    if path != OUTPUT and OUTPUT not in path.parents:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: write outside output {path}")
    path.parent.mkdir(parents=True, exist_ok=True); return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = list(rows); path = safe(path); fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def dump(path: Path, value: Any) -> None:
    path = safe(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


def norm(value: Any) -> str: return str(value or "").strip().casefold()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def record(path: Path, purpose: str) -> dict[str, Any]:
    path = path.resolve(); stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def discover() -> dict[str, Path]:
    values = {
        "source": PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv",
        "architecture": PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/architecture.json",
        "future": PROJECT / "outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv",
        "labels": PROJECT / "artifacts/care-droid-formal-diagnostics-v1/frozen_predictions_with_labels.csv",
    }
    overrides = {"source": "ERA_SOURCE", "architecture": "ERA_ARCHITECTURE", "future": "ERA_FUTURE", "labels": "ERA_LABELS"}
    result = {}
    for key, default in values.items():
        path = Path(os.environ.get(overrides[key], str(default))).resolve()
        if not path.is_file(): raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {key}: {path}")
        result[key] = path
    return result


def auc(y, score) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float); positive = int(y.sum()); negative = len(y) - positive
    if not positive or not negative: return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]; ranks = np.empty(len(y), float); start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ordered[end] == ordered[start]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def metrics(y, base, expert, route) -> dict[str, Any]:
    y = np.asarray(y, int); base = np.asarray(base, float); expert = np.asarray(expert, float); route = np.asarray(route, bool)
    final = np.where(route, expert, base); bp = base >= .5; ep = expert >= .5; fpred = final >= .5
    before = bp == y; after = fpred == y; fixed = (~before) & after; broken = before & (~after)
    tn = int(np.sum((y == 0) & (~fpred))); fp = int(np.sum((y == 0) & fpred)); fn = int(np.sum((y == 1) & (~fpred))); tp = int(np.sum((y == 1) & fpred))
    errors = ~before; expert_fixable = errors & (ep == y); routed_errors = route & errors
    return {"n": len(y), "stage1_errors": int(errors.sum()), "stage1_FN": int(np.sum(errors & (y == 1))),
            "stage1_FP": int(np.sum(errors & (y == 0))), "routed": int(route.sum()),
            "routed_stage1_errors": int(routed_errors.sum()), "route_error_recall": float(routed_errors.sum() / max(errors.sum(), 1)),
            "expert_fixable_all_errors": int(expert_fixable.sum()),
            "expert_ceiling_recall": float(expert_fixable.sum() / max(errors.sum(), 1)),
            "routed_expert_fixable": int(np.sum(route & expert_fixable)),
            "fixable_error_route_recall": float(np.sum(route & expert_fixable) / max(expert_fixable.sum(), 1)),
            "route_missed_fixable": int(np.sum((~route) & expert_fixable)),
            "expert_unfixable_even_if_routed": int(np.sum(errors & (~expert_fixable))),
            "fixed": int(fixed.sum()), "broken": int(broken.sum()), "net": int(fixed.sum() - broken.sum()),
            "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
            "broken_TN": int(np.sum(broken & (y == 0))), "broken_TP": int(np.sum(broken & (y == 1))),
            "fix_rate_among_routed_errors": float(fixed.sum() / max(routed_errors.sum(), 1)),
            "damage_rate_among_routed_correct": float(broken.sum() / max(np.sum(route & before), 1)),
            "BA": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2), "AUC": auc(y, final),
            "F1": float(2 * tp / max(2 * tp + fp + fn, 1)), "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def reason(label: int, base: float, expert: float, routed: bool) -> str:
    bp = base >= .5; ep = expert >= .5
    if bp == label: return "STAGE1_CORRECT"
    if routed and ep == label: return "FIXED"
    if (not routed) and ep == label: return "UNROUTED_EXPERT_COULD_FIX"
    if not routed: return "UNROUTED_EXPERT_ALSO_WRONG"
    sign = 1.0 if label == 1 else -1.0
    movement = sign * (expert - base)
    if abs(movement) <= .01: return "ROUTED_NEAR_ZERO_MOVEMENT"
    if movement > 0: return "ROUTED_CORRECT_DIRECTION_TOO_WEAK"
    return "ROUTED_WRONG_DIRECTION"


def main():
    safe(OUTPUT); OUTPUT.mkdir(parents=True, exist_ok=True); assets = discover()
    protected = [record(assets[key], purpose) for key, purpose in (("source", "frozen source OOF Direct-C43 predictions"),
        ("architecture", "frozen Direct-C43 architecture"), ("future", "frozen future Direct-C43 predictions"),
        ("labels", "frozen future diagnostic labels"))]
    dump(OUTPUT / "baseline_asset_manifest_before.json", protected)
    architecture = json.loads(assets["architecture"].read_text(encoding="utf-8"))
    if "direct" not in json.dumps(architecture).casefold(): raise RuntimeError("BLOCKED_BY_CONTRACT: architecture is not Direct-C43")

    source = read_csv(assets["source"]); future = read_csv(assets["future"])
    labels = {norm(row["sha256"]): int(row["label"]) for row in read_csv(assets["labels"])}
    if len(source) != 799 or len(future) != 1989: raise RuntimeError("BLOCKED_BY_CONTRACT: cohort size changed")
    samples = []
    for row in source:
        samples.append({"sha256": norm(row["sha256"]), "year": 2017, "fold": int(float(row["fold"])),
            "label": int(float(row["label"])), "stage1_prob": float(row["Global24__prob"]),
            "router_score": float(row["A25__score"]), "routed": int(float(row["A25__routed"])),
            "expert_prob": float(row[METHOD + "__expert_prob"])})
    for row in future:
        sid = norm(row["sha256"])
        if sid not in labels: raise RuntimeError("BLOCKED_BY_CONTRACT: future label missing " + sid)
        samples.append({"sha256": sid, "year": int(row["year"]), "fold": "", "label": labels[sid],
            "stage1_prob": float(row["stage1_prob"]), "router_score": float(row["router_score"]),
            "routed": int(row["routed"]), "expert_prob": float(row["direct_c43_prob"])})

    audit = []
    for row in samples:
        y = int(row["label"]); base = float(row["stage1_prob"]); expert = float(row["expert_prob"]); routed = bool(row["routed"])
        bp = int(base >= .5); ep = int(expert >= .5); why = reason(y, base, expert, routed)
        sign = 1.0 if y == 1 else -1.0
        audit.append({**row, "stage1_pred": bp, "stage1_error": int(bp != y), "error_type": "FN" if y == 1 and bp == 0 else ("FP" if y == 0 and bp == 1 else "CORRECT"),
            "expert_pred": ep, "expert_correct": int(ep == y), "actual_final_pred": ep if routed else bp,
            "actual_fixed": int(bp != y and routed and ep == y), "actual_broken": int(bp == y and routed and ep != y),
            "signed_movement_toward_truth": sign * (expert - base), "reason": why})
    write_csv(OUTPUT / "sample_error_audit.csv", audit)

    years = [2017, 2018, 2019, 2020, 2021]
    annual = []; ceiling = []; reason_rows = []; class_reason_rows = []
    for scope, selected in [(str(year), np.asarray([row["year"] == year for row in audit])) for year in years] + [("pooled", np.ones(len(audit), bool))]:
        local = [audit[i] for i in np.flatnonzero(selected)]
        y = np.asarray([row["label"] for row in local]); base = np.asarray([row["stage1_prob"] for row in local]); expert = np.asarray([row["expert_prob"] for row in local]); route = np.asarray([row["routed"] for row in local], bool)
        actual = metrics(y, base, expert, route); all_route = metrics(y, base, expert, np.ones(len(local), bool)); none = metrics(y, base, expert, np.zeros(len(local), bool))
        annual.append({"scope": scope, **actual})
        ceiling.append({"scope": scope, "stage1_errors": none["stage1_errors"], "actual_fixed": actual["fixed"],
            "all_routed_fixed": all_route["fixed"], "additional_fixes_available_by_full_routing": all_route["fixed"] - actual["fixed"],
            "actual_broken": actual["broken"], "all_routed_broken": all_route["broken"],
            "actual_net": actual["net"], "all_routed_net": all_route["net"],
            "routing_deficit": actual["route_missed_fixable"], "expert_deficit": actual["expert_unfixable_even_if_routed"]})
        error_local = [row for row in local if row["stage1_error"]]
        for why in sorted(set(row["reason"] for row in error_local)):
            count = sum(row["reason"] == why for row in error_local)
            reason_rows.append({"scope": scope, "reason": why, "count": count, "fraction_of_stage1_errors": count / max(len(error_local), 1)})
        for error_type in ("FN", "FP"):
            typed = [row for row in error_local if row["error_type"] == error_type]
            for why in sorted(set(row["reason"] for row in typed)):
                count = sum(row["reason"] == why for row in typed)
                class_reason_rows.append({"scope": scope, "error_type": error_type, "reason": why, "count": count,
                    "fraction_within_error_type": count / max(len(typed), 1), "error_type_n": len(typed)})
    write_csv(OUTPUT / "annual_error_recovery.csv", annual); write_csv(OUTPUT / "expert_ceiling.csv", ceiling)
    write_csv(OUTPUT / "error_reason_by_year.csv", reason_rows); write_csv(OUTPUT / "error_reason_by_class_year.csv", class_reason_rows)

    source_ref = next(row for row in annual if row["scope"] == "2017")
    if (source_ref["stage1_errors"], source_ref["fixed"], source_ref["broken"], source_ref["net"]) != (97, 49, 17, 32):
        raise RuntimeError("BLOCKED_BY_CONTRACT: source reference mismatch")
    pooled = next(row for row in annual if row["scope"] == "pooled"); pooled_ceiling = next(row for row in ceiling if row["scope"] == "pooled")
    primary = "EXPERT_DECISION" if pooled_ceiling["expert_deficit"] > pooled_ceiling["routing_deficit"] else "ROUTING_COVERAGE"
    decision = {"status": "ERROR_RECOVERY_AUDIT_COMPLETE", "primary_unrecovered_error_bottleneck": primary,
        "source_reference_reproduced": True, "years": years, "pooled": pooled, "pooled_ceiling": pooled_ceiling,
        "future_used_for_selection": False, "model_changed": False, "router_changed": False,
        "interpretation": "routing_deficit counts Stage-I errors the frozen expert could classify correctly but Top25 did not route; expert_deficit counts errors Direct-C43 misclassifies even under full routing"}
    dump(OUTPUT / "final_decision.json", decision)

    after = [record(Path(row["absolute_path"]), row["purpose"]) for row in protected]
    changed = [old["absolute_path"] for old, new in zip(protected, after) if old["sha256"] != new["sha256"]]
    dump(OUTPUT / "baseline_asset_manifest_after.json", after); dump(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "asset_count": len(after), "changed": changed})
    if changed: raise RuntimeError("BLOCKED_BY_CONTRACT: protected asset changed")

    lines = ["# SVM + Direct-C43 Stage-I error recovery audit", "", "Final status: **ERROR_RECOVERY_AUDIT_COMPLETE**.", "",
        "This audit uses frozen OOF/future predictions. It does not retrain a model, alter routing, or use future data for selection.", "",
        "| Year | Stage-I errors | Entered Stage-II | Route recall | Fixed | Fix/routed-error | Route-missed but fixable | Expert-unfixable | Broken | Net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for year in years:
        row = next(item for item in annual if item["scope"] == str(year))
        lines.append(f"| {year} | {row['stage1_errors']} | {row['routed_stage1_errors']} | {row['route_error_recall']:.4f} | {row['fixed']} | {row['fix_rate_among_routed_errors']:.4f} | {row['route_missed_fixable']} | {row['expert_unfixable_even_if_routed']} | {row['broken']} | {row['net']} |")
    lines += ["", "## Bottleneck decomposition", "", f"- Pooled routing deficit: {pooled_ceiling['routing_deficit']} errors.",
        f"- Pooled expert decision deficit: {pooled_ceiling['expert_deficit']} errors.",
        f"- Primary unrecovered-error bottleneck: **{primary}**.",
        f"- Full routing would change fixed from {pooled_ceiling['actual_fixed']} to {pooled_ceiling['all_routed_fixed']}, but broken from {pooled_ceiling['actual_broken']} to {pooled_ceiling['all_routed_broken']}.", "",
        "Probability-direction categories are diagnostic: correct-direction-too-weak means the Direct-C43 probability moved toward the true side but remained on the wrong side of 0.5; it is not a causal semantic explanation.",
        "Formal asset integrity: PASS.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8"); (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "reproduce_commands.md").write_text(f"# Reproduce\n\n```bash\ncd {PROJECT}\n{os.environ.get('ERA_PYTHON', '/home/boyuanjie/envs/tif/bin/python')} {Path(__file__).resolve()}\n```\n", encoding="utf-8")
    print("DIRECT_C43_ERROR_RECOVERY_AUDIT_DONE primary=" + primary)


if __name__ == "__main__": main()
