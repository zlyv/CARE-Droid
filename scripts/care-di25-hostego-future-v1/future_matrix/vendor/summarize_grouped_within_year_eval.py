#!/usr/bin/env python3
"""Summarize out-of-fold predictions from grouped within-year evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


YEAR_RE = re.compile(r"^within_(\d{4})_fold_\d+_of_\d+$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--variant", default="locked_probability_blend")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prediction_path = args.predictions.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in read_csv(prediction_path):
        if row.get("variant") != args.variant:
            continue
        match = YEAR_RE.match(row.get("split_name", ""))
        if not match:
            raise SystemExit(f"unexpected split name: {row.get('split_name')!r}")
        grouped[(int(match.group(1)), row["sha256"].casefold())].append(row)
    if not grouped:
        raise SystemExit(f"no predictions for variant {args.variant!r}")

    aggregated = []
    by_year: dict[int, list[dict]] = defaultdict(list)
    for (year, sha), rows in sorted(grouped.items()):
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            raise SystemExit(f"conflicting labels for {sha}")
        seeds = [int(row["seed"]) for row in rows]
        if len(seeds) != len(set(seeds)):
            raise SystemExit(f"APK appears in multiple test folds for a seed: {sha}")
        probability = sum(float(row["probability"]) for row in rows) / len(rows)
        label = labels.pop()
        predicted = int(probability >= 0.5)
        item = {
            "year": year,
            "sha256": sha,
            "label": label,
            "mean_probability": probability,
            "threshold": 0.5,
            "predicted_label": predicted,
            "seed_count": len(rows),
        }
        aggregated.append(item)
        by_year[year].append(item)

    summaries = []
    for year, rows in sorted(by_year.items()):
        tp = sum(row["label"] == 1 and row["predicted_label"] == 1 for row in rows)
        tn = sum(row["label"] == 0 and row["predicted_label"] == 0 for row in rows)
        fp = sum(row["label"] == 0 and row["predicted_label"] == 1 for row in rows)
        fn = sum(row["label"] == 1 and row["predicted_label"] == 0 for row in rows)
        positive = tp + fn
        negative = tn + fp
        tpr = tp / positive
        tnr = tn / negative
        summaries.append({
            "year": year,
            "row_count": len(rows),
            "benign_count": negative,
            "malicious_count": positive,
            "accuracy": (tp + tn) / len(rows),
            "balanced_accuracy": (tpr + tnr) / 2.0,
            "tp_count": tp,
            "tn_count": tn,
            "fp_count": fp,
            "false_positive_rate": fp / negative,
            "fn_count": fn,
            "false_negative_rate": fn / positive,
        })

    summary_path = output / "within_year_summary.csv"
    aggregated_path = output / "within_year_oof_predictions.csv"
    write_csv(summary_path, summaries)
    write_csv(aggregated_path, aggregated)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P41_GROUPED_WITHIN_YEAR_OOF_SUMMARY",
        "formal_training_enabled": False,
        "protocol_status": "RETROSPECTIVE_WITHIN_YEAR_SEPARABILITY_ONLY",
        "variant": args.variant,
        "fixed_threshold": 0.5,
        "year_summaries": summaries,
        "outputs": {
            "summary": str(summary_path),
            "aggregated_predictions": str(aggregated_path),
        },
        "notes": [
            "Seed probabilities are averaged for each APK before applying threshold 0.5.",
            "Each APK is evaluated out of fold exactly once per seed.",
            "These results do not measure future-year generalization.",
        ],
    }
    (output / "within_year_summary_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("year ACC BA FP FPR FN FNR")
    for row in summaries:
        print(
            row["year"],
            f'{row["accuracy"]:.4f}',
            f'{row["balanced_accuracy"]:.4f}',
            row["fp_count"],
            f'{row["false_positive_rate"]:.4f}',
            row["fn_count"],
            f'{row["false_negative_rate"]:.4f}',
        )


if __name__ == "__main__":
    main()
