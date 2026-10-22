#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

PROJECT = Path(os.environ.get("CSRC_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("CSRC_OUTPUT", str(PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1"))).resolve()
PRIOR_SCRIPT = PROJECT / "scripts/care-p31-classifier-family-stage2-v1"
PRIOR_OUTPUT = PROJECT / "outputs/care-p31-classifier-family-stage2-v1"
FORMAL = PRIOR_OUTPUT / "source_stage2_predictions.csv"
ARMS = ("U1_UNBOUNDED_RESIDUAL", "U2_DIRECT_STAGE2")
METHOD = {
    "U1_UNBOUNDED_RESIDUAL": "P31SVM__SVMRoute__A0_ROLE_C43_RQ__UnboundedResidual",
    "U2_DIRECT_STAGE2": "P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2",
}


def norm(value): return str(value or "").strip().casefold()


def ensure_output(path):
    path = Path(path).resolve()
    try: path.relative_to(OUTPUT)
    except ValueError as exc: raise RuntimeError("BLOCKED_BY_CONTRACT: output outside new experiment") from exc
    path.parent.mkdir(parents=True, exist_ok=True); return path


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows); path = ensure_output(path)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def dump_json(path, value):
    path = ensure_output(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def asset(path, purpose):
    path = Path(path).resolve(); stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def auc(y, score):
    y = np.asarray(y, int); score = np.asarray(score, float); p = int(y.sum()); n = len(y) - p
    if p == 0 or n == 0: return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]; ranks = np.empty(len(y), float); start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ordered[end] == ordered[start]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def metrics(y, base, final, route):
    y = np.asarray(y, int); base = np.asarray(base, float); final = np.asarray(final, float); route = np.asarray(route, bool)
    bp = base >= .5; pred = final >= .5; before = bp == y; after = pred == y; fixed = (~before) & after; broken = before & (~after)
    tn = int(np.sum((y == 0) & (~pred))); fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & (~pred))); tp = int(np.sum((y == 1) & pred))
    return {"n": len(y), "BA": float((tn/max(tn+fp,1)+tp/max(tp+fn,1))/2), "AUC": auc(y, final),
            "F1": float(2*tp/max(2*tp+fp+fn,1)), "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp+fn,
            "fixed": int(fixed.sum()), "broken": int(broken.sum()), "net": int(fixed.sum()-broken.sum()),
            "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
            "broken_TN": int(np.sum(broken & (y == 0))), "broken_TP": int(np.sum(broken & (y == 1))),
            "routed": int(route.sum()), "class_flips": int(np.sum(bp != pred))}


def mcnemar_exact(candidate_ok, baseline_ok):
    candidate_ok = np.asarray(candidate_ok, bool); baseline_ok = np.asarray(baseline_ok, bool)
    a = int(np.sum(candidate_ok & (~baseline_ok))); b = int(np.sum((~candidate_ok) & baseline_ok)); n = a+b
    if n == 0: return a, b, 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(a,b)+1)) / (2**n)
    return a, b, float(min(1.0, 2*tail))


def formal_rows():
    rows = sorted([row for row in read_csv(FORMAL) if norm(row.get("arm")) == "svm"], key=lambda row: norm(row["sha256"]))
    if len(rows) != 799: raise RuntimeError("BLOCKED_BY_CONTRACT: formal SVM rows changed")
    return rows
