#!/usr/bin/env python3
"""Audit whether CARE-Droid's A25 gate actually yields selective computation.

This audit is intentionally training-free.  It reads the frozen prediction and
region-audit files, measures the routed workload, and detects whether expert
predictions were nevertheless produced for non-routed APKs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm_sha(value: str) -> str:
    return str(value).strip().lower()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-result", type=Path, required=True)
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expert-column",
        default="Global24__A25__CARE_RGCondAttn_FixKeep__ensemble3__expert_prob",
    )
    parser.add_argument("--parameter-count", type=int, default=35740)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Optional artifact path whose on-disk size should be reported; repeatable.",
    )
    args = parser.parse_args()

    future_predictions = args.future_result / "future_predictions_diagnostic.csv"
    future_regions = args.future_result / "region_bag_audit.csv"
    if not future_predictions.is_file() or not future_regions.is_file():
        raise SystemExit("future result is missing predictions or region_bag_audit.csv")

    predictions = read_csv(future_predictions)
    regions = read_csv(future_regions)
    by_sha = {norm_sha(row["sha256"]): row for row in predictions}
    if len(by_sha) != len(predictions):
        raise RuntimeError("duplicate SHA256 in future predictions")

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    missing_prediction = []
    for row in regions:
        year = str(row.get("metadata_year", "")).strip()
        if year not in {"2018", "2019", "2020", "2021"}:
            continue
        prediction = by_sha.get(norm_sha(row["sha256"]))
        if prediction is None:
            missing_prediction.append(norm_sha(row["sha256"]))
            continue
        routed = int(prediction["A25__routed"])
        region_n = int(row["hostego_total_region_n"])
        grouped[year].append((routed, region_n))
    if missing_prediction:
        raise RuntimeError(f"{len(missing_prediction)} region rows lack predictions")

    detail_rows: list[dict] = []
    all_n = all_routed = all_regions = routed_regions = 0
    for year in sorted(grouped):
        values = grouped[year]
        n = len(values)
        routed_n = sum(route for route, _ in values)
        region_n = sum(count for _, count in values)
        routed_region_n = sum(count for route, count in values if route)
        detail_rows.append({
            "year": year,
            "n": n,
            "routed_n": routed_n,
            "routed_ratio": routed_n / n,
            "all_local_regions": region_n,
            "routed_local_regions": routed_region_n,
            "routed_region_ratio": routed_region_n / region_n,
            "apk_expert_work_reduction": 1.0 - routed_n / n,
            "region_work_reduction": 1.0 - routed_region_n / region_n,
            "mean_regions_all": region_n / n,
            "mean_regions_routed": routed_region_n / routed_n,
        })
        all_n += n
        all_routed += routed_n
        all_regions += region_n
        routed_regions += routed_region_n

    expert_column_present = args.expert_column in (predictions[0] if predictions else {})
    nonrouted_expert_n = 0
    routed_expert_n = 0
    if expert_column_present:
        for row in predictions:
            has_expert = finite(row.get(args.expert_column, ""))
            if int(row["A25__routed"]):
                routed_expert_n += int(has_expert)
            else:
                nonrouted_expert_n += int(has_expert)

    current_pipeline_route_after_expert = bool(
        expert_column_present and nonrouted_expert_n > 0
    )
    parameters = int(args.parameter_count)
    summary = {
        "future_n": all_n,
        "routed_n": all_routed,
        "routed_ratio": all_routed / all_n,
        "nonrouted_n": all_n - all_routed,
        "all_local_regions": all_regions,
        "routed_local_regions": routed_regions,
        "routed_region_ratio": routed_regions / all_regions,
        "route_first_expert_apk_reduction": 1.0 - all_routed / all_n,
        "route_first_local_region_reduction": 1.0 - routed_regions / all_regions,
        "mean_regions_all": all_regions / all_n,
        "mean_regions_routed": routed_regions / all_routed,
        "rgcondattn_parameter_n": parameters,
        "rgcondattn_fp32_parameter_mib": mib(parameters * 4),
        "rgcondattn_fp16_parameter_mib": mib(parameters * 2),
        "expert_column_present": expert_column_present,
        "routed_rows_with_expert_probability": routed_expert_n,
        "nonrouted_rows_with_expert_probability": nonrouted_expert_n,
        "current_pipeline_route_after_expert": current_pipeline_route_after_expert,
        "current_pipeline_realized_expert_reduction": 0.0 if current_pipeline_route_after_expert else None,
        "verdict": (
            "NOT_YET_REALIZED: evaluation computes the expert before routing"
            if current_pipeline_route_after_expert
            else "ROUTE_FIRST_OR_UNDETERMINED"
        ),
    }

    source_summary = None
    if args.source_result:
        source_predictions = args.source_result / "source_predictions.csv"
        source_regions = args.source_result / "region_bag_audit.csv"
        if source_predictions.is_file() and source_regions.is_file():
            sp = read_csv(source_predictions)
            sr = read_csv(source_regions)
            sp_by_sha = {norm_sha(row["sha256"]): row for row in sp}
            source_values = []
            for row in sr:
                pred = sp_by_sha.get(norm_sha(row["sha256"]))
                if pred is not None:
                    source_values.append((
                        int(pred["A25__routed"]),
                        int(row["hostego_total_region_n"]),
                    ))
            if source_values:
                sn = len(source_values)
                sroute = sum(x for x, _ in source_values)
                sregions = sum(x for _, x in source_values)
                srouted_regions = sum(x for routed, x in source_values if routed)
                source_summary = {
                    "source_n": sn,
                    "source_routed_n": sroute,
                    "source_routed_ratio": sroute / sn,
                    "source_all_local_regions": sregions,
                    "source_routed_local_regions": srouted_regions,
                    "source_routed_region_ratio": srouted_regions / sregions,
                }
                summary.update(source_summary)

    artifact_rows = []
    for raw in args.artifact:
        path = Path(raw)
        if path.is_file():
            size = path.stat().st_size
            artifact_rows.append({
                "path": str(path.resolve()),
                "kind": "file",
                "bytes": size,
                "MiB": mib(size),
                "sha256": file_sha256(path),
            })
        elif path.is_dir():
            files = [item for item in path.rglob("*") if item.is_file()]
            size = sum(item.stat().st_size for item in files)
            artifact_rows.append({
                "path": str(path.resolve()),
                "kind": f"directory:{len(files)} files",
                "bytes": size,
                "MiB": mib(size),
                "sha256": "",
            })
        else:
            artifact_rows.append({
                "path": str(path), "kind": "missing", "bytes": 0, "MiB": 0.0, "sha256": ""
            })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cost_by_year.csv", detail_rows)
    write_csv(args.output / "artifact_sizes.csv", artifact_rows)
    (args.output / "cost_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "CARE-Droid selective-computation cost audit",
        "",
        f"Future APKs: {all_n}",
        f"A25 routed APKs: {all_routed} ({100.0 * all_routed / all_n:.2f}%)",
        f"Non-routed APKs: {all_n - all_routed}",
        f"All Host/Ego regions: {all_regions}",
        f"Routed Host/Ego regions: {routed_regions} ({100.0 * routed_regions / all_regions:.2f}%)",
        f"Route-first expert APK reduction: {100.0 * (1.0 - all_routed / all_n):.2f}%",
        f"Route-first local-region workload reduction: {100.0 * (1.0 - routed_regions / all_regions):.2f}%",
        f"Mean regions/APK (all): {all_regions / all_n:.2f}",
        f"Mean regions/APK (routed): {routed_regions / all_routed:.2f}",
        f"RGCondAttn trainable parameters: {parameters}",
        f"FP32 parameter payload: {mib(parameters * 4):.4f} MiB",
        "",
        "Implementation audit:",
        f"Expert probabilities on routed rows: {routed_expert_n}",
        f"Expert probabilities on non-routed rows: {nonrouted_expert_n}",
        f"Route is applied after expert prediction: {current_pipeline_route_after_expert}",
        f"Verdict: {summary['verdict']}",
        "",
        "Interpretation:",
        "The route-first values are achievable workload proxies, not measured wall-clock savings.",
        "Wall time, peak RAM/VRAM, energy, and raw APK feature-extraction cost must be benchmarked separately.",
    ]
    if source_summary:
        report.extend([
            "",
            "Source cohort:",
            f"Source APKs: {source_summary['source_n']}",
            f"Source routed: {source_summary['source_routed_n']} ({100.0 * source_summary['source_routed_ratio']:.2f}%)",
            f"Source routed-region ratio: {100.0 * source_summary['source_routed_region_ratio']:.2f}%",
        ])
    (args.output / "REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print("Done:", args.output)


if __name__ == "__main__":
    main()
