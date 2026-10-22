#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(os.environ.get("CESA_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("CESA_OUTPUT", str(PROJECT / "outputs/care-c43-equivalence-sensitivity-audit-v2"))).resolve()
SCRIPTS = PROJECT / "scripts"
FAMILY_SCRIPT = SCRIPTS / "care-p31-classifier-family-stage2-v1"
DIRECT_SCRIPT = SCRIPTS / "care-svm-c43-remove-bounded-correction-v1"
FAMILY_OUTPUT = PROJECT / "outputs/care-p31-classifier-family-stage2-v1"
DIRECT_OUTPUT = PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1"
FORMAL = FAMILY_OUTPUT / "source_stage2_predictions.csv"
DIRECT = DIRECT_OUTPUT / "source/U2_DIRECT_STAGE2/source_predictions.csv"
ENDPOINT = PROJECT / "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"
A0_CACHE = PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz"
METHOD = "P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"
REPLAY_METHOD = METHOD + "__EquivalenceProxyAudit"
SEED = 20261012
TRANSFORMS = (
    "T0_ORDER",
    "T1_WRAPPER_SENSITIVE",
    "T2_DEAD_SUBGRAPH",
    "T3_DEAD_EDGE_NEAR_HOST",
    "N1_WRAPPER_FAR",
)


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def ensure(path: Path) -> Path:
    path = Path(path).resolve()
    try:
        path.relative_to(OUTPUT)
    except ValueError as exc:
        raise RuntimeError("BLOCKED_BY_CONTRACT: output outside new experiment: " + str(path)) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows) -> None:
    rows = list(rows); path = ensure(path)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def dump_json(path: Path, value: Any) -> None:
    path = ensure(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def asset(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve(); stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def auc(y, score) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    positive = int(y.sum()); negative = len(y) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]
    ranks = np.empty(len(y), float); start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def spearman(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    def ranks(v):
        order = np.argsort(v, kind="mergesort"); out = np.empty(len(v), float); a = 0
        while a < len(v):
            b = a + 1
            while b < len(v) and v[order[b]] == v[order[a]]: b += 1
            out[order[a:b]] = (a + 1 + b) / 2; a = b
        return out
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def load_reference():
    formal = [row for row in read_csv(FORMAL) if norm(row.get("arm")) == "svm"]
    direct_rows = read_csv(DIRECT); direct = {norm(row["sha256"]): row for row in direct_rows}
    if len(formal) != 799 or len(direct) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen source reference changed")
    ids = np.asarray([norm(row["sha256"]) for row in formal])
    y = np.asarray([int(float(row["label"])) for row in formal])
    fold = np.asarray([int(float(row["fold"])) for row in formal])
    base = np.asarray([float(row["stage1_prob"]) for row in formal])
    route = np.asarray([int(float(row["routed"])) for row in formal], bool)
    field = METHOD + "__expert_prob" if METHOD + "__expert_prob" in direct_rows[0] else METHOD + "__prob"
    expert = np.asarray([float(direct[sid][field]) for sid in ids])
    return ids, y, fold, base, route, expert


def categories(y, base, expert):
    y = np.asarray(y, int); before = (np.asarray(base) >= .5) == y; after = (np.asarray(expert) >= .5) == y
    return np.asarray(["FIXED" if (not b and a) else "BROKEN" if (b and not a) else "STABLE_CORRECT" if b else "PERSISTENT_ERROR" for b, a in zip(before, after)])
