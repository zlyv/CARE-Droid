#!/usr/bin/env python3
"""Build the predeclared four-row source-only Stage-II comparison table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict) -> str:
    return str(row.get("sha256", row.get("sample_id", ""))).strip().lower()


def align(rows: list[dict], order: list[str], name: str) -> list[dict]:
    table = {key(row): row for row in rows}
    if len(table) != len(rows) or set(table) != set(order):
        raise RuntimeError(name + " cohort differs from the fixed 799 source APKs")
    return [table[value] for value in order]


def find(rows: list[dict], exact: str, tokens: tuple[str, ...]) -> str:
    if exact in rows[0]:
        return exact
    matches = [name for name in rows[0] if all(token.lower() in name.lower() for token in tokens)]
    if len(matches) != 1:
        raise RuntimeError(f"could not uniquely resolve {exact}: {matches}")
    return matches[0]


def metric(name: str, y, probability, invocation_rate: float) -> dict:
    pred = np.asarray(probability) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "method": name,
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, probability)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "StageII_invocation_rate": float(invocation_rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path(
        "artifacts/global24-a25-route-conditioned-joints7-v1-full/source_predictions.csv"))
    parser.add_argument("--baseline", type=Path, default=Path(
        "artifacts/care-di-budget-bound-selection-v1/retrained_source/max_delta_2p25/source_predictions.csv"))
    parser.add_argument("--malben", type=Path, default=Path(
        "artifacts/care-malben-contrast-v2/source/model/source_predictions.csv"))
    parser.add_argument("--stage2only", type=Path, default=Path(
        "artifacts/care-malben-contrast-v2/source/stage2only/source_predictions.csv"))
    parser.add_argument("--stage2-summary", type=Path, default=Path(
        "artifacts/care-malben-contrast-v2/source/stage2only/source_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-malben-contrast-v2/source/main_comparison"))
    args = parser.parse_args()

    reference = read(args.reference)
    if len(reference) != 799:
        raise RuntimeError("expected fixed 799-row 2017 source cohort")
    order = [key(row) for row in reference]
    if len(set(order)) != 799:
        raise RuntimeError("duplicate source SHA")
    y = np.asarray([int(row["label"]) for row in reference], dtype=np.int64)
    base = np.asarray([float(row["Global24__prob"]) for row in reference])
    route_name = "A25__routed" if "A25__routed" in reference[0] else "route_A25"
    route = np.asarray([int(float(row[route_name])) for row in reference], dtype=bool)
    if int(np.sum(route)) != 200:
        raise RuntimeError("frozen A25 route is not 200/799")

    baseline = align(read(args.baseline), order, "anomaly baseline")
    malben = align(read(args.malben), order, "MalBen")
    stage2 = align(read(args.stage2only), order, "StageIIOnly")
    baseline_col = find(baseline,
        "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep__expert_prob",
        ("RGCondAttn", "Dedup", "expert", "prob"))
    malben_col = find(malben,
        "Global24__A25__CARE_RoleMalBenContrast_FixKeep__expert_prob",
        ("MalBenContrast", "expert", "prob"))
    stage2_col = find(stage2, "StageIIOnly_MalBen__prob", ("StageIIOnly", "prob"))
    baseline_expert = np.asarray([float(row[baseline_col]) for row in baseline])
    malben_expert = np.asarray([float(row[malben_col]) for row in malben])
    stage2_probability = np.asarray([float(row[stage2_col]) for row in stage2])

    rows = [
        metric("Global24-only", y, base, 0.0),
        metric("StageIIOnly-MalBen", y, stage2_probability, 1.0),
        metric("A25 + AnomalyBaseline", y, np.where(route, baseline_expert, base), float(np.mean(route))),
        metric("A25 + MalBenContrast", y, np.where(route, malben_expert, base), float(np.mean(route))),
    ]
    stage2_summary = read(args.stage2_summary)[0]
    for row in rows:
        if row["method"] == "StageIIOnly-MalBen":
            row["valid_region_coverage"] = float(stage2_summary["valid_region_coverage"])
            row["missing_region_count"] = int(float(stage2_summary["missing_region_count"]))
        else:
            row["valid_region_coverage"] = ""
            row["missing_region_count"] = ""
    write(args.output / "main_table.csv", rows)

    lines = [
        "CARE-Droid Stage-II-only source ablation main table", "",
        "Strict fixed 2017 outer-fold OOF; no 2018-2021 row is loaded.",
        "StageIIOnly-MalBen invokes Stage II for all 799 APKs and excludes Global24/p_stage1/router/bounded correction.", "",
    ]
    for row in rows:
        lines.append(
            f"{row['method']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
            f"F1={row['F1']:.6f} TN={row['TN']} FP={row['FP']} "
            f"FN={row['FN']} TP={row['TP']} "
            f"StageII={100.0 * row['StageII_invocation_rate']:.2f}%"
        )
    lines.extend(["", "StageIIOnly coverage:",
        f"valid_region_coverage={stage2_summary['valid_region_coverage']}",
        f"missing_region_count={stage2_summary['missing_region_count']}"])
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
