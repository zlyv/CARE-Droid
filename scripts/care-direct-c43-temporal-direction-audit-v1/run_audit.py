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

PROJECT = Path(os.environ.get("TDA_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("TDA_OUTPUT", str(PROJECT / "outputs/care-direct-c43-temporal-direction-audit-v1"))).resolve()
METHOD = "P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"
YEARS = (2017, 2018, 2019, 2020, 2021)


def safe(path: Path) -> Path:
    root = (PROJECT / "outputs").resolve(); path = Path(path).resolve()
    if OUTPUT == root or root not in OUTPUT.parents or not OUTPUT.name.startswith("care-direct-c43-temporal-direction-audit-v"):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: unsafe output {OUTPUT}")
    if path != OUTPUT and OUTPUT not in path.parents: raise RuntimeError(f"BLOCKED_BY_CONTRACT: write outside output {path}")
    path.parent.mkdir(parents=True, exist_ok=True); return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = list(rows); path = safe(path); fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def dump(path: Path, value: Any) -> None:
    path = safe(path); temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temp.replace(path)


def norm(value: Any) -> str: return str(value or "").strip().casefold()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def record(path: Path, purpose: str) -> dict[str, Any]:
    path = path.resolve(); stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}


def discover() -> dict[str, Path]:
    defaults = {"source": PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv",
        "future": PROJECT / "outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv",
        "labels": PROJECT / "artifacts/care-droid-formal-diagnostics-v1/frozen_predictions_with_labels.csv",
        "architecture": PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/architecture.json"}
    result = {}
    for key, default in defaults.items():
        path = Path(os.environ.get("TDA_" + key.upper(), str(default))).resolve()
        if not path.is_file(): raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {key}: {path}")
        result[key] = path
    return result


def auc(y, score) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float); p = int(y.sum()); n = len(y) - p
    if not p or not n: return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]; ranks = np.empty(len(y), float); start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ordered[end] == ordered[start]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def bootstrap_auc(y, score, seed: int, repeats: int = 2000) -> tuple[float, float]:
    y = np.asarray(y, int); score = np.asarray(score, float); pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    if not len(pos) or not len(neg): return float("nan"), float("nan")
    rng = np.random.default_rng(seed); values = []
    for _ in range(repeats):
        index = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        values.append(auc(y[index], score[index]))
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def threshold_net(y, score, threshold: float, baseline_pred: int) -> tuple[int, int, int]:
    y = np.asarray(y, int); pred = np.asarray(score, float) >= threshold
    before = np.full(len(y), int(baseline_pred), int) == y; after = pred == y
    fixed = int(np.sum((~before) & after)); broken = int(np.sum(before & (~after)))
    return fixed, broken, fixed - broken


def oracle_threshold(y, score, baseline_pred: int) -> dict[str, Any]:
    y = np.asarray(y, int); score = np.asarray(score, float)
    candidates = np.unique(np.concatenate(([-1e-12], score, [1 + 1e-12])))
    best = None
    for threshold in candidates:
        fixed, broken, net = threshold_net(y, score, float(threshold), baseline_pred)
        key = (net, fixed, -abs(float(threshold) - .5))
        if best is None or key > best[0]: best = (key, float(threshold), fixed, broken, net)
    _, threshold, fixed, broken, net = best
    return {"oracle_threshold": threshold, "oracle_fixed": fixed, "oracle_broken": broken, "oracle_net": net}


def main():
    safe(OUTPUT); OUTPUT.mkdir(parents=True, exist_ok=True); assets = discover()
    protected = [record(assets[k], p) for k, p in (("source", "frozen source Direct-C43 predictions"),
        ("future", "frozen future Direct-C43 predictions"), ("labels", "frozen future labels"),
        ("architecture", "frozen Direct-C43 architecture"))]
    dump(OUTPUT / "baseline_asset_manifest_before.json", protected)
    architecture = json.loads(assets["architecture"].read_text(encoding="utf-8"))
    if "direct" not in json.dumps(architecture).casefold(): raise RuntimeError("BLOCKED_BY_CONTRACT: architecture is not Direct-C43")
    labels = {norm(row["sha256"]): int(row["label"]) for row in read_csv(assets["labels"])}
    rows = []
    source = read_csv(assets["source"]); future = read_csv(assets["future"])
    if len(source) != 799 or len(future) != 1989: raise RuntimeError("BLOCKED_BY_CONTRACT: cohort changed")
    for row in source:
        rows.append({"sha256": norm(row["sha256"]), "year": 2017, "label": int(float(row["label"])),
            "stage1_prob": float(row["Global24__prob"]), "expert_prob": float(row[METHOD + "__expert_prob"]),
            "routed": int(float(row["A25__routed"]))})
    for row in future:
        sid = norm(row["sha256"])
        if sid not in labels: raise RuntimeError("BLOCKED_BY_CONTRACT: missing label " + sid)
        rows.append({"sha256": sid, "year": int(row["year"]), "label": labels[sid],
            "stage1_prob": float(row["stage1_prob"]), "expert_prob": float(row["direct_c43_prob"]), "routed": int(row["routed"])})

    for row in rows:
        y = int(row["label"]); bp = int(row["stage1_prob"] >= .5); ep = int(row["expert_prob"] >= .5)
        row["stage1_pred"] = bp; row["stage1_error"] = int(bp != y); row["expert_pred"] = ep
        row["expert_correct"] = int(ep == y); row["stage1_group"] = "PREDICTED_BENIGN" if bp == 0 else "PREDICTED_MALWARE"
        row["confusion"] = ("FN" if y == 1 else "TN") if bp == 0 else ("TP" if y == 1 else "FP")
        signed_margin = (1 if y == 1 else -1) * (float(row["expert_prob"]) - .5); row["expert_true_margin"] = signed_margin
        if bp != y and ep != y:
            row["wrong_confidence_bin"] = "NEAR_BOUNDARY_0_TO_0.10" if signed_margin >= -.10 else ("MODERATE_WRONG_0.10_TO_0.25" if signed_margin >= -.25 else "CONFIDENT_WRONG_GE_0.25")
        else: row["wrong_confidence_bin"] = "NOT_EXPERT_UNFIXABLE"
    write_csv(OUTPUT / "sample_direction_audit.csv", rows)

    conditional = []; oracle_rows = []; confidence = []
    for year in YEARS:
        for subset_name, subset_rule in (("ALL", lambda r: True), ("ROUTED", lambda r: bool(r["routed"]))):
            for side, side_pred in (("PREDICTED_BENIGN", 0), ("PREDICTED_MALWARE", 1)):
                local = [r for r in rows if r["year"] == year and subset_rule(r) and r["stage1_pred"] == side_pred]
                y = np.asarray([r["label"] for r in local]); score = np.asarray([r["expert_prob"] for r in local])
                value = auc(y, score); low, high = bootstrap_auc(y, score, 7300 + year * 10 + side_pred + (100 if subset_name == "ROUTED" else 0))
                conditional.append({"year": year, "subset": subset_name, "side": side, "n": len(local),
                    "negative_n": int(np.sum(y == 0)), "positive_n": int(np.sum(y == 1)), "AUC": value,
                    "bootstrap_CI_low": low, "bootstrap_CI_high": high})
                if subset_name == "ROUTED" and len(set(y.tolist())) == 2:
                    fixed05, broken05, net05 = threshold_net(y, score, .5, side_pred); oracle = oracle_threshold(y, score, side_pred)
                    oracle_rows.append({"year": year, "side": side, "n": len(local), "threshold_0.5_fixed": fixed05,
                        "threshold_0.5_broken": broken05, "threshold_0.5_net": net05, **oracle,
                        "oracle_delta_net": oracle["oracle_net"] - net05, "deployment_authorized": False})
        failures = [r for r in rows if r["year"] == year and r["stage1_error"] and not r["expert_correct"]]
        for error_type in ("FN", "FP", "ALL"):
            typed = failures if error_type == "ALL" else [r for r in failures if r["confusion"] == error_type]
            for name in ("NEAR_BOUNDARY_0_TO_0.10", "MODERATE_WRONG_0.10_TO_0.25", "CONFIDENT_WRONG_GE_0.25"):
                count = sum(r["wrong_confidence_bin"] == name for r in typed)
                confidence.append({"year": year, "error_type": error_type, "confidence_bin": name, "count": count,
                    "fraction": count / max(len(typed), 1), "expert_unfixable_n": len(typed)})
    write_csv(OUTPUT / "conditional_direction_by_year.csv", conditional)
    write_csv(OUTPUT / "routed_oracle_threshold_diagnostic.csv", oracle_rows)
    write_csv(OUTPUT / "expert_failure_confidence.csv", confidence)

    future_year_rows = [r for r in conditional if r["year"] > 2017 and r["subset"] == "ALL"]
    side_summary = {}
    for side in ("PREDICTED_BENIGN", "PREDICTED_MALWARE"):
        values = [r for r in future_year_rows if r["side"] == side]
        evaluable = [r for r in values if math.isfinite(float(r["AUC"]))]
        side_summary[side] = {"years_evaluable": len(evaluable), "years_auc_at_least_0.65": sum(float(r["AUC"]) >= .65 for r in evaluable),
            "years_ci_excludes_0.5": sum(float(r["bootstrap_CI_low"]) > .5 for r in evaluable),
            "annual_auc": {str(r["year"]): r["AUC"] for r in evaluable}}
    unstable = [side for side, value in side_summary.items() if value["years_auc_at_least_0.65"] < 3 or value["years_ci_excludes_0.5"] < 3]
    oracle_gain = sum(int(r["oracle_delta_net"]) for r in oracle_rows if int(r["year"]) > 2017)
    mechanism = "TEMPORAL_DIRECTION_RANKING_INSTABILITY" if unstable else ("DESCRIPTIVE_THRESHOLD_GAP" if oracle_gain >= 10 else "CONDITIONAL_SCORE_OVERLAP")
    decision = {"status": "DIRECT_C43_TEMPORAL_DIRECTION_AUDIT_COMPLETE", "primary_mechanism": mechanism,
        "unstable_sides": unstable, "side_summary": side_summary, "future_routed_oracle_delta_net_sum_descriptive": oracle_gain,
        "oracle_threshold_is_not_deployable": True, "future_used_for_selection": False, "model_changed": False,
        "next_action": "do not tune thresholds; use the audit to decide whether new static observations are required"}
    dump(OUTPUT / "final_decision.json", decision)
    after = [record(Path(r["absolute_path"]), r["purpose"]) for r in protected]
    changed = [a["absolute_path"] for a, b in zip(protected, after) if a["sha256"] != b["sha256"]]
    dump(OUTPUT / "baseline_asset_manifest_after.json", after); dump(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "asset_count": len(after), "changed": changed})
    if changed: raise RuntimeError("BLOCKED_BY_CONTRACT: protected asset changed")

    lines = ["# Direct-C43 temporal conditional-direction audit", "", f"Final mechanism: **{mechanism}**.", "",
        "This is a frozen diagnostic. Oracle thresholds are descriptive upper bounds and are not deployment candidates.", "",
        "| Year | FN-vs-TN AUC | 95% CI | TP-vs-FP AUC | 95% CI |", "|---|---:|---:|---:|---:|"]
    for year in YEARS:
        fn = next(r for r in conditional if r["year"] == year and r["subset"] == "ALL" and r["side"] == "PREDICTED_BENIGN")
        tp = next(r for r in conditional if r["year"] == year and r["subset"] == "ALL" and r["side"] == "PREDICTED_MALWARE")
        lines.append(f"| {year} | {fn['AUC']:.6f} | [{fn['bootstrap_CI_low']:.6f}, {fn['bootstrap_CI_high']:.6f}] | {tp['AUC']:.6f} | [{tp['bootstrap_CI_low']:.6f}, {tp['bootstrap_CI_high']:.6f}] |")
    lines += ["", f"Unstable sides: {unstable}.", f"Descriptive future routed oracle delta-net sum: {oracle_gain}.",
        "No threshold, prediction-side rule, model, route, or future-selected parameter is authorized.", "Formal asset integrity: PASS.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8"); (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("DIRECT_C43_TEMPORAL_DIRECTION_AUDIT_DONE mechanism=" + mechanism)


if __name__ == "__main__": main()
