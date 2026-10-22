#!/usr/bin/env python3
"""Evaluate TIF semantic-chain features under the TASA-Droid split protocol.

The JSONL is scanned per split so the full 5,678-dimensional feature table is
never held in memory.  Feature names and nonconstant filtering are derived
from the split's training rows only; test values are used only after the model
is fixed.  The classification threshold is fixed at 0.5.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_graph_pool_diagnostic import read_csv_rows, read_split_rows  # noqa: E402
from run_pilot_compact_eval import metrics  # noqa: E402


THRESHOLD = 0.5
SEEDS = (0, 1, 2, 3, 4)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tif_feature_matrices(
    tif_jsonl: Path,
    train_shas: set[str],
    eval_shas: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str], int, int, set[str]]:
    """Build one split's matrices with two streaming passes over JSONL."""
    total_rows = 0
    duplicate_shas = 0
    feature_names_set: set[str] = set()
    seen_train: set[str] = set()
    seen_eval: set[str] = set()
    with tif_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total_rows += 1
            apk_path = str(row.get("apk_path", ""))
            sha256 = Path(apk_path).stem.strip().casefold()
            if sha256 not in train_shas:
                continue
            if sha256 in seen_train:
                duplicate_shas += 1
                continue
            seen_train.add(sha256)
            features = row.get("features")
            if not isinstance(features, dict):
                raise SystemExit(f"TIF features is not a dict for {sha256}")
            feature_names_set.update(str(name) for name in features)

    feature_names = sorted(feature_names_set)
    feature_index = {name: i for i, name in enumerate(feature_names)}
    train_index = {sha256: i for i, sha256 in enumerate(train_shas)}
    eval_index = {sha256: i for i, sha256 in enumerate(eval_shas)}
    train_matrix = np.zeros(
        (len(train_index), len(feature_names)), dtype=np.float64
    )
    eval_matrix = np.zeros(
        (len(eval_index), len(feature_names)), dtype=np.float64
    )
    seen_train.clear()
    seen_eval.clear()
    with tif_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            apk_path = str(row.get("apk_path", ""))
            sha256 = Path(apk_path).stem.strip().casefold()
            if sha256 in train_index:
                if sha256 in seen_train:
                    duplicate_shas += 1
                    continue
                seen_train.add(sha256)
                matrix = train_matrix
                row_index = train_index[sha256]
            elif sha256 in eval_index:
                if sha256 in seen_eval:
                    duplicate_shas += 1
                    continue
                seen_eval.add(sha256)
                matrix = eval_matrix
                row_index = eval_index[sha256]
            else:
                continue
            features = row.get("features")
            if not isinstance(features, dict):
                raise SystemExit(f"TIF features is not a dict for {sha256}")
            for name, value in features.items():
                column = feature_index.get(str(name))
                if column is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric):
                    matrix[row_index, column] = numeric
    matched = seen_train | seen_eval
    return (
        train_matrix,
        eval_matrix,
        feature_names,
        total_rows,
        duplicate_shas,
        matched,
    )


def prepare_train_eval(
    train_matrix: np.ndarray,
    eval_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    finite = np.isfinite(train_matrix).all(axis=0)
    variable = np.zeros(train_matrix.shape[1], dtype=bool)
    if finite.any():
        variable[finite] = np.ptp(train_matrix[:, finite], axis=0) > 1e-12
    train_matrix = train_matrix[:, variable]
    eval_matrix = eval_matrix[:, variable]
    mean = train_matrix.mean(axis=0)
    scale = train_matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (
        (train_matrix - mean) / scale,
        (eval_matrix - mean) / scale,
        int(variable.sum()),
    )


def score(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    result = metrics(labels, probability, threshold=THRESHOLD)
    result["brier_score"] = float(brier_score_loss(labels, probability))
    result["log_loss"] = float(
        log_loss(labels, np.clip(probability, 1e-7, 1.0 - 1e-7))
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif-jsonl", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    args = parser.parse_args()

    tif_path = args.tif_jsonl.expanduser().resolve(strict=True)
    labels_path = args.labels.expanduser().resolve(strict=True)
    split_path = args.splits.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_map = read_split_rows(split_path)
    if not split_map:
        raise SystemExit("split manifest has no splits")
    label_rows = read_csv_rows(labels_path)
    label_by_sha = {}
    for row in label_rows:
        sha256 = str(row.get("sha256", "")).strip().casefold()
        value = str(row.get("metadata_label", row.get("label", ""))).casefold()
        if not sha256:
            continue
        if value in {"malicious", "malware", "1", "true"}:
            label_by_sha[sha256] = 1
        elif value in {"benign", "0", "false"}:
            label_by_sha[sha256] = 0
        else:
            raise SystemExit(f"unknown label in labels file for {sha256}: {value}")

    split_summaries = []
    total_tif_rows = None
    matched_sha_union: set[str] = set()
    for split_name in sorted(split_map):
        train_rows = split_map[split_name]["train"]
        eval_rows = split_map[split_name]["test"]
        train_shas = [row["sha256"].strip().casefold() for row in train_rows]
        eval_shas = [row["sha256"].strip().casefold() for row in eval_rows]
        (
            train_matrix,
            eval_matrix,
            feature_names,
            tif_rows,
            duplicate_shas,
            matched_shas,
        ) = read_tif_feature_matrices(
            tif_path,
            set(train_shas),
            set(eval_shas),
        )
        if total_tif_rows is None:
            total_tif_rows = tif_rows
        missing = [sha for sha in train_shas + eval_shas if sha not in matched_shas]
        if missing:
            raise SystemExit(
                f"TIF feature join missing {len(missing)} rows in {split_name}; "
                f"first={missing[0]}"
            )
        matched_sha_union.update(matched_shas)
        train_matrix, eval_matrix, variable_count = prepare_train_eval(
            train_matrix, eval_matrix
        )
        y_train = np.asarray(
            [label_by_sha[sha256] for sha256 in train_shas],
            dtype=np.int64,
        )
        y_eval = np.asarray(
            [label_by_sha[sha256] for sha256 in eval_shas],
            dtype=np.int64,
        )
        for c_value in args.c_values:
            seed_rows = []
            for seed in SEEDS:
                classifier = LogisticRegression(
                    C=float(c_value),
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                )
                classifier.fit(train_matrix, y_train)
                probability = classifier.predict_proba(eval_matrix)[:, 1]
                seed_rows.append(score(y_eval, probability))
            split_summaries.append(
                {
                    "split_name": split_name,
                    "C": float(c_value),
                    "train_count": len(train_rows),
                    "eval_count": len(eval_rows),
                    "schema_feature_count_from_train": len(feature_names),
                    "nonconstant_feature_count_from_train": variable_count,
                    "duplicate_tif_sha_count": duplicate_shas,
                    "accuracy_mean": float(np.mean([row["accuracy"] for row in seed_rows])),
                    "balanced_accuracy_mean": float(
                        np.mean([row["balanced_accuracy"] for row in seed_rows])
                    ),
                    "precision_mean": float(np.mean([row["precision"] for row in seed_rows])),
                    "recall_mean": float(np.mean([row["recall"] for row in seed_rows])),
                    "f1_mean": float(np.mean([row["f1"] for row in seed_rows])),
                    "auc_mean": float(np.mean([row["auc"] for row in seed_rows])),
                    "brier_score_mean": float(
                        np.mean([row["brier_score"] for row in seed_rows])
                    ),
                    "log_loss_mean": float(
                        np.mean([row["log_loss"] for row in seed_rows])
                    ),
                    "test_rows_used_for_training_or_selection": False,
                }
            )

    by_c: dict[float, list[dict]] = defaultdict(list)
    for row in split_summaries:
        by_c[row["C"]].append(row)
    aggregate = []
    for c_value, rows in sorted(by_c.items()):
        fold_ba = [row["balanced_accuracy_mean"] for row in rows]
        aggregate.append(
            {
                "C": c_value,
                "fold_count": len(rows),
                "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in rows])),
                "mean_balanced_accuracy": float(np.mean(fold_ba)),
                "minimum_fold_balanced_accuracy": float(np.min(fold_ba)),
                "std_fold_balanced_accuracy": float(np.std(fold_ba)),
                "stability_score_mean_minus_std_ba": float(
                    np.mean(fold_ba) - np.std(fold_ba)
                ),
                "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
                "mean_brier_score": float(
                    np.mean([row["brier_score_mean"] for row in rows])
                ),
                "mean_log_loss": float(
                    np.mean([row["log_loss_mean"] for row in rows])
                ),
            }
        )
    aggregate.sort(
        key=lambda row: (
            -row["stability_score_mean_minus_std_ba"],
            -row["mean_balanced_accuracy"],
            row["C"],
        )
    )

    summary_path = output / "tif_semantic_feature_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P5_TIF_SEMANTIC_FEATURE_PROTOCOL_EVALUATION",
        "formal_training_enabled": False,
        "strict_test_isolation": True,
        "fixed_threshold": THRESHOLD,
        "tif_jsonl": str(tif_path),
        "tif_jsonl_sha256": sha256_file(tif_path),
        "labels_csv": str(labels_path),
        "labels_csv_sha256": sha256_file(labels_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "tif_total_rows_scanned_per_split": total_tif_rows,
        "matched_unique_sha_count": len(matched_sha_union),
        "C_values": list(args.c_values),
        "seeds": list(SEEDS),
        "split_summaries": split_summaries,
        "aggregate_summaries": aggregate,
        "selection_rule": (
            "maximize mean balanced accuracy minus across-split standard deviation"
        ),
        "outputs": {"summary": str(summary_path)},
        "blocking_reasons": [
            "tif_feature_schema_and_preprocessing_equivalence_not_yet_proven",
            "apk_first_seen_utc_not_audited",
        ],
        "notes": [
            "APK join key is the SHA256 encoded in the TIF apk_path basename.",
            "Feature names and nonconstant filtering are derived from training rows only.",
            "TIF labels are not used; labels come from the supplied current feature audit CSV.",
            "No raw APK, DEX, graph, or method signature is persisted.",
        ],
    }
    audit_path = output / "tif_semantic_feature_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("rank C mean_BA min_fold_BA std_BA stability AUC")
    for rank, row in enumerate(aggregate, start=1):
        print(
            rank,
            row["C"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_fold_balanced_accuracy"]:.4f}',
            f'{row["std_fold_balanced_accuracy"]:.4f}',
            f'{row["stability_score_mean_minus_std_ba"]:.4f}',
            f'{row["mean_auc"]:.4f}',
        )


if __name__ == "__main__":
    main()
