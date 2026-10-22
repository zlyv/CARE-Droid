#!/usr/bin/env python3
"""Validate and summarize the strict-2017 Need-Gate OOF predictions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from probe_protocol import ARMS
from source_contract import EXPECTED_ROUTE_BY_FOLD, OUTPUT, blocked, norm, write_json

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260920
COMPARISONS = (("S2", "S0"), ("S3", "S0"), ("S3", "S1"), ("S3", "S2"))


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise blocked(f"missing formal OOF predictions: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise blocked(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate(rows: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    required = {
        "sha256", "year", "fold", "label", "p_stage1", "stage1_logit",
        "stage1_pred", "need_correct", "route_DI",
    }
    for arm in ARMS:
        required.update({f"score_{arm}", f"pred_{arm}"})
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise blocked("OOF output missing fields: " + ",".join(sorted(missing)))
    ids = [norm(row["sha256"]) for row in rows]
    years = np.asarray([int(float(row["year"])) for row in rows], dtype=np.int64)
    folds = np.asarray([int(float(row["fold"])) for row in rows], dtype=np.int64)
    labels = np.asarray([int(float(row["need_correct"])) for row in rows], dtype=np.int64)
    routes = np.asarray([int(float(row["route_DI"])) for row in rows], dtype=np.int64)
    if len(rows) != 200 or len(set(ids)) != 200 or "" in ids:
        raise blocked(f"formal OOF must contain 200 rows/unique SHA; rows={len(rows)} unique={len(set(ids))}")
    if set(years.tolist()) != {2017} or set(routes.tolist()) != {1}:
        raise blocked("formal OOF must contain only routed 2017 rows")
    route_by_fold = {fold: int(np.sum(folds == fold)) for fold in range(1, 6)}
    if route_by_fold != EXPECTED_ROUTE_BY_FOLD:
        raise blocked(f"formal OOF fold coverage changed: {route_by_fold}")
    if not set(labels.tolist()).issubset({0, 1}):
        raise blocked("need_correct must be binary")
    scores = {}
    for arm in ARMS:
        score = np.asarray([float(row[f"score_{arm}"]) for row in rows], dtype=np.float64)
        pred = np.asarray([int(float(row[f"pred_{arm}"])) for row in rows], dtype=np.int64)
        if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise blocked(f"{arm} contains invalid scores")
        if not np.array_equal(pred, (score >= 0.5).astype(np.int64)):
            raise blocked(f"{arm} prediction is inconsistent with fixed threshold 0.5")
        scores[arm] = score
    return labels, scores, folds


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    pred = (score >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "need_positive_n": int(np.sum(y == 1)),
        "need_negative_n": int(np.sum(y == 0)),
        "BA": float(balanced_accuracy_score(y, pred)) if both else float("nan"),
        "AUC": float(roc_auc_score(y, score)) if both else float("nan"),
        "PR_AUC": float(average_precision_score(y, score)) if np.any(y == 1) else float("nan"),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Brier": float(brier_score_loss(y, score)),
    }


def statistic(name: str, y: np.ndarray, score: np.ndarray) -> float:
    if name == "AUC":
        return float(roc_auc_score(y, score))
    if name == "PR_AUC":
        return float(average_precision_score(y, score))
    if name == "BA":
        return float(balanced_accuracy_score(y, score >= 0.5))
    if name == "Brier":
        # Higher-is-better convention for paired deltas.
        return -float(brier_score_loss(y, score))
    raise AssertionError(name)


def paired_bootstrap(y: np.ndarray, candidate: np.ndarray, baseline: np.ndarray,
                     metric: str, rng: np.random.Generator) -> dict:
    point = statistic(metric, y, candidate) - statistic(metric, y, baseline)
    values = []
    n = len(y)
    for _ in range(BOOTSTRAP_ITERATIONS):
        index = rng.integers(0, n, n)
        sampled_y = y[index]
        if metric in {"AUC", "PR_AUC", "BA"} and len(np.unique(sampled_y)) < 2:
            continue
        values.append(statistic(metric, sampled_y, candidate[index]) - statistic(metric, sampled_y, baseline[index]))
    if not values:
        raise blocked(f"no valid paired bootstrap samples for {metric}")
    value = np.asarray(values, dtype=np.float64)
    return {
        "delta": float(point), "bootstrap_mean_delta": float(np.mean(value)),
        "ci95_low": float(np.quantile(value, 0.025)),
        "ci95_high": float(np.quantile(value, 0.975)),
        "valid_iterations": int(len(value)),
    }


def main() -> None:
    rows = read_rows(OUTPUT / "oof_predictions.csv")
    y, scores, folds = validate(rows)
    metric_rows = []
    for arm in ARMS:
        for fold in range(1, 6):
            mask = folds == fold
            metric_rows.append({"arm": arm, "scope": f"fold_{fold}", "fold": fold, **metrics(y[mask], scores[arm][mask])})
        metric_rows.append({"arm": arm, "scope": "pooled", "fold": "aggregate", **metrics(y, scores[arm])})
    write_csv(OUTPUT / "metrics.csv", metric_rows)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_rows = []
    for candidate, baseline in COMPARISONS:
        for metric in ("AUC", "PR_AUC", "BA", "Brier"):
            bootstrap_rows.append({
                "candidate": candidate, "baseline": baseline, "metric": metric,
                "iterations_requested": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED,
                **paired_bootstrap(y, scores[candidate], scores[baseline], metric, rng),
            })
    write_csv(OUTPUT / "bootstrap_comparisons.csv", bootstrap_rows)

    pooled = {row["arm"]: row for row in metric_rows if row["scope"] == "pooled"}
    lines = [
        "# CARE-Droid strict-2017 DI25 no-Ego Need-Gate source result", "",
        "- Strict outer five-fold OOF; source year 2017 only.",
        "- Routed cohort: 200/799; exactly 40 validation APKs per outer fold.",
        "- Future rows loaded: 0.", "- Ego consumed: false.",
        "- Fixed probe threshold: 0.5.", "",
        "| Arm | N | Pos | Neg | BA | AUC | PR-AUC | F1 | Precision | Recall | Specificity | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        value = pooled[arm]
        lines.append(
            f"| {arm} | {value['n']} | {value['need_positive_n']} | {value['need_negative_n']} | "
            f"{value['BA']:.6f} | {value['AUC']:.6f} | {value['PR_AUC']:.6f} | "
            f"{value['F1']:.6f} | {value['precision']:.6f} | {value['recall']:.6f} | "
            f"{value['specificity']:.6f} | {value['Brier']:.6f} |"
        )
    lines.extend([
        "", "Paired bootstrap uses 10,000 paired resamples with seed 20260920.",
        "Brier deltas are reported in higher-is-better form (-Brier).",
        "No result in this report authorizes automatic model, route, or threshold modification.", "",
    ])
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUTPUT / "summary_audit.json", {
        "passed": True, "OOF_rows": 200, "OOF_unique_SHA": 200,
        "year": [2017], "route_DI": 200, "route_DI_per_fold": EXPECTED_ROUTE_BY_FOLD,
        "future_rows_loaded": 0, "ego_consumed": False,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED,
    })
    print("SOURCE_NEEDGATE_SUMMARY_PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        if str(error).startswith("BLOCKED_BY_CONTRACT"):
            print(str(error), flush=True)
            raise SystemExit(2)
        raise
