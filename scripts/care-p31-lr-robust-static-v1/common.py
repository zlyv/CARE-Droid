#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(os.environ.get("PCFS2_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("PCFS2_OUTPUT", str(PROJECT / "outputs/care-p31-lr-robust-static-v1"))).resolve()
SCRIPTS = PROJECT / "scripts"
FORMAL_DIAG = PROJECT / "artifacts/care-droid-formal-diagnostics-v1"
CLASSIFIER_RESULT = PROJECT / "outputs/care-p31-classifier-family-v1"
PRIOR_STAGE2 = PROJECT / "outputs/care-p31-classifier-family-stage2-v1"
PRIOR_STAGE2_TABLE = PRIOR_STAGE2 / "full_system_comparison.csv"
SOURCE_L0_STAGE2 = PRIOR_STAGE2 / "source/LR/source_predictions.csv"
FUTURE_L0_STAGE2 = PRIOR_STAGE2 / "frozen/LR/predictions_label_free.csv"
WITHIN_L0_STAGE2 = PRIOR_STAGE2 / "within/LR/predictions.csv"
SOURCE_MEMBER = PROJECT / "artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv"
SOURCE_P31 = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
SOURCE_A0 = PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv"
SOURCE_A0_CACHE = PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz"
SOURCE_RF_REFERENCE = PROJECT / "artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv"
ENDPOINT_SOURCE = PROJECT / "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"
FUTURE_REFERENCE = FORMAL_DIAG / "references/future_label_free.csv"
FUTURE_LABELS = FORMAL_DIAG / "frozen_predictions_with_labels.csv"
FUTURE_RF = FORMAL_DIAG / "frozen/future_predictions_label_free.csv"
WITHIN_RF = (FORMAL_DIAG / "within_2018_2019/predictions.csv", FORMAL_DIAG / "within_2020_2021/predictions.csv")

MEMBERS = ("LR", "RF", "XGB", "SVM")
ARMS = ("L0_RAW", "L1_HELLINGER", "L2_LOG", "L3_PRESENCE", "L4_RAW_PRESENCE",
        "L5_CLR_PRESENCE", "L6_HELLINGER_PRESENCE")
NEW_ARMS = ARMS[1:]
_ONLY_ARM = os.environ.get("PLRRS_ONLY_ARM", "").strip()
if _ONLY_ARM and _ONLY_ARM not in NEW_ARMS:
    raise RuntimeError("BLOCKED_BY_CONTRACT: invalid PLRRS_ONLY_ARM=" + _ONLY_ARM)
ACTIVE_ARMS = ("L0_RAW", _ONLY_ARM) if _ONLY_ARM else ARMS
YEARS = (2018, 2019, 2020, 2021)
YEAR_COUNTS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
MAX_DELTA = 2.25
SOURCE_SPLIT_SEED = 20260901
DIAGNOSTIC_SPLIT_SEED = 20260904


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def method_name(arm: str) -> str:
    return f"P31LRROBUST_{arm}__LRRoute__A0_ROLE_C43_RQ__Bounded"


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows) -> None:
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def dump_json(path: Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve(); stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def binary_auc(y, score) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    positive = int(y.sum()); negative = len(y) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]
    ranks = np.empty(len(score), float); start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def classification_metrics(y, probability) -> dict[str, Any]:
    y = np.asarray(y, int); probability = np.asarray(probability, float); pred = probability >= .5
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    return {"n": len(y), "BA": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
            "AUC": binary_auc(y, probability), "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "accuracy": float(np.mean(pred == y)), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "errors": fp + fn}


def comparison_metrics(y, base, final, route) -> dict[str, Any]:
    y = np.asarray(y, int); base = np.asarray(base, float); final = np.asarray(final, float)
    route = np.asarray(route, bool); base_pred = base >= .5; final_pred = final >= .5
    base_ok = base_pred == y; final_ok = final_pred == y
    fixed = (~base_ok) & final_ok; broken = base_ok & (~final_ok); error = ~base_ok
    return {**classification_metrics(y, final), "fixed": int(fixed.sum()), "broken": int(broken.sum()),
            "net": int(fixed.sum() - broken.sum()), "fixed_FN": int(np.sum(fixed & (y == 1))),
            "fixed_FP": int(np.sum(fixed & (y == 0))), "broken_TN": int(np.sum(broken & (y == 0))),
            "broken_TP": int(np.sum(broken & (y == 1))), "routed": int(route.sum()),
            "captured_stage1_errors": int(np.sum(route & error)),
            "error_recall25": float(np.sum(route & error) / max(error.sum(), 1)),
            "route_precision": float(np.sum(route & error) / max(route.sum(), 1)),
            "class_flips": int(np.sum(base_pred != final_pred))}


def arm_probability(member_probability, arm: str):
    member_probability = np.asarray(member_probability, float)
    if member_probability.ndim != 2 or member_probability.shape[1] != 4:
        raise RuntimeError("member probability must be N x 4")
    if arm not in ARMS:
        raise KeyError(arm)
    return member_probability[:, 0]


def mean_opposition(member_probability, anchor_probability):
    member_probability = np.asarray(member_probability, float)
    decision = np.asarray(anchor_probability, float) >= .5
    return np.where(decision[:, None], 1 - member_probability, member_probability).mean(axis=1)


def grouped_top25(score, sha, groups):
    score = np.asarray(score, float); sha = np.asarray(sha).astype(str); groups = np.asarray(groups)
    route = np.zeros(len(score), bool)
    for group in sorted(set(groups.tolist())):
        index = np.flatnonzero(groups == group); count = int(round(len(index) * .25))
        order = np.lexsort((sha[index], -score[index])); route[index[order[:count]]] = True
    return route
