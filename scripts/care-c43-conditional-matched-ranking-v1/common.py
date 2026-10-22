#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, json, math, os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(os.environ.get("CCMR_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("CCMR_OUTPUT", str(PROJECT / "outputs/care-c43-conditional-matched-ranking-v1"))).resolve()
SOURCE_MEMBER = PROJECT / "artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv"
SOURCE_P31 = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
SOURCE_BAGS = PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz"
DIRECT_ROOT = PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1"
DIRECT_PREDICTIONS = DIRECT_ROOT / "source/U2_DIRECT_STAGE2/source_predictions.csv"
SEED = 20261012
FOLDS = (1, 2, 3, 4, 5)
ARMS = ("C43_MATCHED_REAL", "C43_RANDOM_PAIR", "C43_UNMATCHED", "P31_MATCHED", "COUNT_ONLY_MATCHED")


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def ensure_output(path: Path) -> Path:
    path = Path(path).resolve()
    try:
        path.relative_to(OUTPUT)
    except ValueError as exc:
        raise RuntimeError("BLOCKED_BY_CONTRACT: output outside new experiment") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows) -> None:
    rows = list(rows); path = ensure_output(path)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def dump_json(path: Path, value: Any) -> None:
    path = ensure_output(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve(); stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def binary_auc(y, score) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float); positive = int(y.sum()); negative = len(y) - positive
    if positive == 0 or negative == 0: return float("nan")
    order = np.argsort(score, kind="mergesort"); ordered = score[order]; ranks = np.empty(len(y), float); start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and ordered[end] == ordered[start]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def finite(value: Any) -> bool:
    try: return math.isfinite(float(value))
    except Exception: return False


def load_source():
    rows = read_csv(SOURCE_MEMBER)
    if len(rows) != 799: raise RuntimeError("BLOCKED_BY_CONTRACT: source rows changed")
    rows = sorted(rows, key=lambda row: norm(row["sha256"]))
    ids = np.asarray([norm(row["sha256"]) for row in rows])
    y = np.asarray([int(float(row["label"])) for row in rows], int)
    fold = np.asarray([int(float(row["fold"])) for row in rows], int)
    member = np.asarray([[float(row[f"p_P31__{name}"]) for name in ("LR", "RF", "XGB", "SVM")] for row in rows], float)
    base = member[:, 3]; stage1_pred = base >= .5
    route_score = np.where(stage1_pred[:, None], 1.0 - member, member).mean(axis=1)
    route = np.zeros(len(rows), bool)
    for current in FOLDS:
        index = np.flatnonzero(fold == current); count = int(round(len(index) * .25))
        order = np.lexsort((ids[index], -route_score[index])); route[index[order[:count]]] = True
    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        p_ids = [norm(value) for value in archive["sha256"]]; p_values = np.asarray(archive["role_prevalence"], float)
    p_map = {sid: p_values[i] for i, sid in enumerate(p_ids)}
    if any(sid not in p_map for sid in ids): raise RuntimeError("BLOCKED_BY_CONTRACT: P31 coverage changed")
    p31 = np.asarray([p_map[sid] for sid in ids], float)
    with np.load(SOURCE_BAGS, allow_pickle=False) as archive:
        b_ids = [norm(value) for value in archive["sha256"]]
        values = np.asarray(archive["host_values"], np.float32); offsets = np.asarray(archive["host_offsets"], int)
    if values.ndim != 2 or values.shape[1] != 169 or offsets.shape != (len(b_ids) + 1,):
        raise RuntimeError("BLOCKED_BY_CONTRACT: A0 Host cache schema changed")
    b_map = {sid: i for i, sid in enumerate(b_ids)}; bags = []
    for sid in ids:
        if sid not in b_map: raise RuntimeError("BLOCKED_BY_CONTRACT: A0 Host cache missing " + sid)
        i = b_map[sid]; bags.append(values[offsets[i]:offsets[i + 1]])
    return rows, ids, y, fold, member, base, route_score, route, p31, bags


def host_summary(bags) -> tuple[np.ndarray, np.ndarray]:
    output = []; counts = []
    for bag in bags:
        x = np.asarray(bag, float); counts.append(len(x))
        if len(x) == 0:
            output.append(np.zeros(169 * 3, float)); continue
        output.append(np.concatenate([np.mean(x, axis=0), np.quantile(x, .90, axis=0), np.max(x, axis=0)]))
    result = np.asarray(output, float)
    if not np.isfinite(result).all(): raise RuntimeError("BLOCKED_BY_CONTRACT: non-finite Host summary")
    return result, np.asarray(counts, int)


def match_pairs(indices, y, pred, match_x, random_pair=False, seed=SEED):
    """Return malware/benign index pairs within each frozen Stage-I prediction side."""
    rng = np.random.RandomState(int(seed)); pairs = []
    indices = np.asarray(indices, int)
    for side in (0, 1):
        local = indices[pred[indices] == side]
        malware = local[y[local] == 1]; benign = local[y[local] == 0]
        if not len(malware) or not len(benign): continue
        if random_pair:
            left = malware.copy(); right = benign.copy(); rng.shuffle(left); rng.shuffle(right)
            count = min(len(left), len(right)); chosen = zip(left[:count], right[:count])
            for m, b in chosen: pairs.append((int(m), int(b), side, float("nan")))
            continue
        distance = np.sqrt(np.maximum(((match_x[malware, None, :] - match_x[None, benign, :]) ** 2).sum(axis=2), 0.0))
        # Deterministic global-nearest greedy assignment avoids a new SciPy dependency.
        candidates = sorted((float(distance[r,c]), int(malware[r]), int(benign[c]))
                            for r in range(len(malware)) for c in range(len(benign)))
        used_m=set(); used_b=set()
        for value, m, b in candidates:
            if m in used_m or b in used_b: continue
            pairs.append((m,b,side,value)); used_m.add(m); used_b.add(b)
            if len(used_m)==len(malware) or len(used_b)==len(benign): break
    return pairs


def bootstrap_auc(y, score, seed=SEED, repeats=2000):
    y = np.asarray(y, int); score = np.asarray(score, float); rng = np.random.RandomState(int(seed)); values = []
    for _ in range(int(repeats)):
        index = rng.randint(0, len(y), len(y))
        value = binary_auc(y[index], score[index])
        if finite(value): values.append(value)
    return {"auc": binary_auc(y, score), "ci95_low": float(np.quantile(values, .025)),
            "ci95_high": float(np.quantile(values, .975)), "bootstrap_repeats": len(values)}
