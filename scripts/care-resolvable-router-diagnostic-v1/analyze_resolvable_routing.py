#!/usr/bin/env python3
"""Measure repairable-error recall and harmful capture for frozen routers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPERT_COLUMNS = {
    "BoundedCorrection": "BoundedCorrection__expert_prob",
    "DirectL2": "DirectL2__expert_prob",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metric_row(
    router: str,
    expert_name: str,
    labels: np.ndarray,
    base_probability: np.ndarray,
    expert_probability: np.ndarray,
    route: np.ndarray,
    *,
    scope: str,
    fold: int | None = None,
) -> dict:
    base_correct = (base_probability >= 0.5) == labels
    expert_correct = (expert_probability >= 0.5) == labels
    base_error = ~base_correct
    repairable = base_error & expert_correct
    unrepairable = base_error & ~expert_correct
    harmful = base_correct & ~expert_correct
    harmless_correct = base_correct & expert_correct

    routed_n = int(np.sum(route))
    fixed = int(np.sum(route & repairable))
    broken = int(np.sum(route & harmful))
    captured_error = int(np.sum(route & base_error))
    captured_unrepairable = int(np.sum(route & unrepairable))
    row = {
        "scope": scope,
        "router": router,
        "second_layer": expert_name,
        "n": int(len(labels)),
        "routed_n": routed_n,
        "routed_ratio": float(np.mean(route)),
        "base_error_n": int(np.sum(base_error)),
        "error_recall": safe_ratio(captured_error, int(np.sum(base_error))),
        "repairable_error_n": int(np.sum(repairable)),
        "routed_repairable_error_n": fixed,
        "repairable_error_recall_at_budget": safe_ratio(
            fixed, int(np.sum(repairable))
        ),
        "unrepairable_error_n": int(np.sum(unrepairable)),
        "routed_unrepairable_error_n": captured_unrepairable,
        "unrepairable_error_capture_rate": safe_ratio(
            captured_unrepairable, int(np.sum(unrepairable))
        ),
        "harmful_sample_n": int(np.sum(harmful)),
        "routed_harmful_sample_n": broken,
        "harmful_capture_rate": safe_ratio(broken, int(np.sum(harmful))),
        "harmless_correct_n": int(np.sum(harmless_correct)),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "routed_repairable_precision": safe_ratio(fixed, routed_n),
        "routed_harmful_fraction": safe_ratio(broken, routed_n),
        "correction_precision": safe_ratio(fixed, fixed + broken),
    }
    if fold is not None:
        row["fold"] = int(fold)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path(
        "artifacts/care-reliability-calibrated-evidential-router-source-v2/"
        "source_predictions.csv"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-resolvable-router-diagnostic-v1"
    ))
    args = parser.parse_args()
    rows = read_csv(args.predictions)
    if not rows:
        raise RuntimeError("empty source prediction file: " + str(args.predictions))
    required = {"label", "fold", "Global24__prob", *EXPERT_COLUMNS.values()}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError("missing prediction columns: " + repr(missing))
    route_columns = sorted(name for name in rows[0] if name.endswith("__routed"))
    if not route_columns:
        raise RuntimeError("no router columns were found")

    labels = np.asarray([int(float(row["label"])) for row in rows], dtype=np.int64)
    folds = np.asarray([int(float(row["fold"])) for row in rows], dtype=np.int64)
    base = np.asarray([float(row["Global24__prob"]) for row in rows], dtype=np.float64)
    experts = {
        name: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        for name, column in EXPERT_COLUMNS.items()
    }
    routes = {
        column.removesuffix("__routed"): np.asarray(
            [int(float(row[column])) for row in rows], dtype=bool
        )
        for column in route_columns
    }
    arrays = [labels, folds, base, *experts.values(), *routes.values()]
    if len(rows) != 799 or any(len(value) != len(rows) for value in arrays):
        raise RuntimeError("expected the fixed 799-row source cohort")
    if not np.isfinite(base).all() or any(
        not np.isfinite(value).all() for value in experts.values()
    ):
        raise RuntimeError("non-finite source probabilities")

    summary: list[dict] = []
    per_fold: list[dict] = []
    for router, route in routes.items():
        for expert_name, expert in experts.items():
            summary.append(metric_row(
                router, expert_name, labels, base, expert, route, scope="source_2017"
            ))
            for fold in sorted(np.unique(folds)):
                selected = folds == fold
                per_fold.append(metric_row(
                    router,
                    expert_name,
                    labels[selected],
                    base[selected],
                    expert[selected],
                    route[selected],
                    scope="source_2017_fold",
                    fold=int(fold),
                ))

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "resolvable_routing_summary.csv", summary)
    write_csv(args.output / "resolvable_routing_per_fold.csv", per_fold)
    definitions = {
        "repairable_error": "Stage I wrong and frozen Stage II correct",
        "repairable_error_recall_at_budget": (
            "routed repairable errors divided by all repairable errors"
        ),
        "harmful_sample": "Stage I correct and frozen Stage II wrong",
        "harmful_capture_rate": (
            "routed harmful samples divided by all harmful samples"
        ),
        "unrepairable_error": "Stage I wrong and frozen Stage II also wrong",
        "fixed": "routed repairable errors",
        "broken": "routed harmful samples",
        "net": "fixed minus broken",
        "important_boundary": (
            "repairability is conditional on the evaluated frozen Stage-II expert"
        ),
    }
    (args.output / "definitions.json").write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "CARE-Droid resolvable-routing diagnostic v1",
        "",
        "RER@budget = routed repairable errors / all repairable errors.",
        "Harmful capture = routed harmful samples / all harmful samples.",
        "Higher RER and lower harmful capture are preferred at equal budget.",
        "Repairability is specific to the frozen second-layer expert.",
        "",
        "Source 2017 aggregate:",
    ]
    for row in summary:
        lines.append(
            f"{row['router']} + {row['second_layer']}: "
            f"ErrorRecall={row['error_recall']:.6f} "
            f"RER@B={row['repairable_error_recall_at_budget']:.6f} "
            f"HarmfulCapture={row['harmful_capture_rate']:.6f} "
            f"UnrepairableCapture={row['unrepairable_error_capture_rate']:.6f} "
            f"fixed={row['fixed']} broken={row['broken']} net={row['net']}"
        )
    bounded = [row for row in summary if row["second_layer"] == "BoundedCorrection"]
    best = max(
        bounded,
        key=lambda row: (
            row["repairable_error_recall_at_budget"],
            -row["harmful_capture_rate"],
            row["net"],
        ),
    )
    lines.extend([
        "",
        "Diagnostic ranking only (not a new model-selection test):",
        f"Highest bounded-correction RER@B: {best['router']} "
        f"({best['repairable_error_recall_at_budget']:.6f}).",
        "Use this table to decide whether a nested resolvable-uncertainty v3 is justified.",
    ])
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
