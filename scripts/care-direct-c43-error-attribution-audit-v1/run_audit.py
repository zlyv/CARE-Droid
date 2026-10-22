#!/usr/bin/env python3
"""Source-only, read-only error-phenotype and evidence attribution audit for frozen Direct-C43.

This script NEVER trains, selects a model or threshold, reads future-year labels,
or infers a static root cause from model errors alone. Optional same-model
embeddings and source-only static descriptors are descriptive evidence only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(os.environ.get("ECA_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("ECA_OUTPUT", str(PROJECT / "outputs/care-direct-c43-error-attribution-audit-v1"))).resolve()
METHOD = "P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"
DEFAULT_SOURCE = PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv"
DEFAULT_ARCH = PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/architecture.json"
SRC_PROB = os.environ.get("ECA_SOURCE_PROB_COL", "Global24__prob")
ROUTED_COL = os.environ.get("ECA_SOURCE_ROUTED_COL", "A25__routed")
EXPERT_PROB = os.environ.get("ECA_EXPERT_PROB_COL", METHOD + "__expert_prob")
TARGET_GROUPS = {"ROUTED_UNFIXED", "ROUTED_BROKEN"}
EXPECTED_COUNTS = {"ROUTED_UNFIXED": 35, "ROUTED_BROKEN": 17,
                   "ROUTED_FIXED": 49, "ROUTED_STABLE_CORRECT": 99}
SCORE_K = (5, 10)


def fail(message: str) -> None:
    raise RuntimeError("BLOCKED_BY_CONTRACT: " + message)


def safe_path(path: Path) -> Path:
    path = Path(path).resolve()
    root = (PROJECT / "outputs").resolve()
    if root not in OUTPUT.parents or not OUTPUT.name.startswith("care-direct-c43-error-attribution-audit-v"):
        fail("unsafe output directory: " + str(OUTPUT))
    if path != OUTPUT and OUTPUT not in path.parents:
        fail("write outside output directory: " + str(path))
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            fail("CSV has no header: " + str(path))
        return list(reader)


def write_csv(name: str, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    dest = safe_path(OUTPUT / name)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = dest.with_suffix(dest.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})
    temp.replace(dest)


def dump(name: str, obj: Any) -> None:
    dest = safe_path(OUTPUT / name)
    temp = dest.with_suffix(dest.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(dest)


def norm_sha(s: Any) -> str:
    result = str(s or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        fail("invalid 64-character SHA256")
    return result


def finite_prob(v: Any, context: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        fail("invalid probability in " + context)
    if not math.isfinite(x) or not 0 <= x <= 1:
        fail("probability outside [0,1] in " + context)
    return x


def to_bit(v: Any, context: str) -> int:
    try:
        x = float(v)
    except (TypeError, ValueError):
        fail("invalid binary value: " + context)
    if x not in (0.0, 1.0):
        fail("non-binary value: " + context)
    return int(x)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path, purpose: str) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "purpose": purpose, "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(path)}


def discover() -> dict[str, Path]:
    defaults = {"source": DEFAULT_SOURCE, "architecture": DEFAULT_ARCH}
    inputs: dict[str, Path] = {}
    for name, default in defaults.items():
        path = Path(os.environ.get("ECA_" + name.upper(), str(default))).resolve()
        if not path.is_file():
            fail("missing required " + name + " input: " + str(path))
        inputs[name] = path
    for name in ("folds", "embeddings", "static_features", "stage1_reference", "apk_index"):
        value = os.environ.get("ECA_" + name.upper(), "").strip()
        if value:
            path = Path(value).resolve()
            if not path.is_file():
                fail("missing optional " + name + " input: " + str(path))
            inputs[name] = path
    for name, path in inputs.items():
        if path == OUTPUT or OUTPUT in path.parents:
            fail("protected input is inside output directory: " + name)
    return inputs


def verify_year(row: dict[str, str], description: str) -> None:
    if "year" in row and str(row["year"]).strip() != "":
        try:
            year = int(float(row["year"]))
        except ValueError:
            fail("malformed year in " + description)
        if year != 2017:
            fail("non-source year in " + description + ": " + str(year))


def category(row: dict[str, Any]) -> str:
    if not row["routed"]:
        return "UNROUTED_S1_ERROR" if row["stage1_error"] else "UNROUTED_S1_CORRECT"
    if row["stage1_error"]:
        return "ROUTED_FIXED" if row["expert_correct"] else "ROUTED_UNFIXED"
    return "ROUTED_STABLE_CORRECT" if row["expert_correct"] else "ROUTED_BROKEN"


def build_source_rows(source_path: Path) -> list[dict[str, Any]]:
    raw = read_csv(source_path)
    if len(raw) != 799:
        fail(f"source cohort must have 799 rows; got {len(raw)}")
    required = ("sha256", "label", SRC_PROB, EXPERT_PROB, ROUTED_COL)
    missing = [c for c in required if c not in raw[0]]
    if missing:
        fail("source prediction columns missing: " + str(missing))
    result = []
    sha_seen = set()
    for item in raw:
        verify_year(item, "source predictions")
        sid = norm_sha(item["sha256"])
        if sid in sha_seen:
            fail("duplicate SHA in source predictions")
        sha_seen.add(sid)
        y = to_bit(item["label"], "source label")
        p1 = finite_prob(item[SRC_PROB], "source Stage-I " + sid)
        p2 = finite_prob(item[EXPERT_PROB], "source Direct-C43 " + sid)
        route = to_bit(item[ROUTED_COL], "source route")
        s1, e = int(p1 >= 0.5), int(p2 >= 0.5)
        r = {"sha256": sid, "year": 2017, "label": y, "stage1_prob": p1,
             "expert_prob": p2, "routed": route, "stage1_pred": s1, "expert_pred": e,
             "stage1_error": int(s1 != y), "expert_correct": int(e == y),
             "stage1_side": "PREDICTED_BENIGN" if s1 == 0 else "PREDICTED_MALWARE",
             "stage1_confusion": ("FN" if y else "TN") if s1 == 0 else ("TP" if y else "FP"),
             "expert_true_margin": (1 if y == 1 else -1) * (p2 - 0.5),
             "expert_wrong_abs_margin": abs(p2 - .5) if e != y else None,
             "outer_fold": str(item.get("fold", item.get("outer_fold", item.get("fold_id", "")))).strip()}
        r["group"] = category(r)
        r["pairwise_inversion"] = None
        r["nearest_opposite_score_gap"] = None
        result.append(r)
    return result


def source_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n0 = sum(r["label"] == 0 for r in rows)
    n1 = len(rows) - n0
    ct = Counter(r["group"] for r in rows)
    stage1_conf = Counter(r["stage1_confusion"] for r in rows)
    routed = sum(r["routed"] for r in rows)
    routed_wrong = sum(r["routed"] and r["stage1_error"] for r in rows)
    if (n0, n1) != (400, 399):
        fail(f"source labels differ from frozen 2017: {(n0, n1)}")
    if (stage1_conf["TN"], stage1_conf["FP"], stage1_conf["FN"], stage1_conf["TP"]) != (356, 44, 53, 346):
        fail("Stage-I source confusion differs from frozen P31-SVM; check Global24__prob alias")
    if routed != 200 or routed_wrong != 84:
        fail(f"router differs from frozen Top25: routed={routed}, routed_wrong={routed_wrong}")
    for k, v in EXPECTED_COUNTS.items():
        if ct[k] != v:
            fail(f"Direct-C43 group mismatch {k}: actual={ct[k]} expected={v}")
    final = {"TN": 0, "FP": 0, "FN": 0, "TP": 0}
    for r in rows:
        pred = r["expert_pred"] if r["routed"] else r["stage1_pred"]
        y = r["label"]
        name = ("FN" if y else "TN") if pred == 0 else ("TP" if y else "FP")
        final[name] += 1
    if tuple(final[k] for k in ("TN", "FP", "FN", "TP")) != (378, 22, 43, 356):
        fail("Direct-C43 source final confusion mismatch; source asset/protocol mismatch")
    return {"source_rows": 799, "benign": n0, "malware": n1, "routed": routed,
            "routed_stage1_errors": routed_wrong, "group_counts": dict(sorted(ct.items())),
            "stage1_confusion": dict(stage1_conf), "final_confusion": final}


def auc_binary(y: list[int], score: list[float]) -> float:
    pos = [s for yi, s in zip(y, score) if yi == 1]
    neg = [s for yi, s in zip(y, score) if yi == 0]
    if not pos or not neg:
        return float("nan")
    win = sum(float(p > n) + 0.5 * float(p == n) for p in pos for n in neg)
    return win / (len(pos) * len(neg))


def add_pairwise_overlap(rows: list[dict[str, Any]]) -> dict[str, float]:
    results = {}
    for stage1_pred, name in ((0, "FN_vs_TN"), (1, "TP_vs_FP")):
        local = [r for r in rows if r["routed"] and r["stage1_pred"] == stage1_pred]
        pos = [r["expert_prob"] for r in local if r["label"] == 1]
        neg = [r["expert_prob"] for r in local if r["label"] == 0]
        if not pos or not neg:
            fail("missing either class in routed side " + name)
        results[name] = auc_binary([r["label"] for r in local], [r["expert_prob"] for r in local])
        for r in local:
            score = r["expert_prob"]
            opp = neg if r["label"] == 1 else pos
            if r["label"] == 1:
                inversions = sum(float(s > score) + .5 * float(s == score) for s in opp)
            else:
                inversions = sum(float(s < score) + .5 * float(s == score) for s in opp)
            r["pairwise_inversion"] = inversions / len(opp)
            r["nearest_opposite_score_gap"] = min(abs(score - s) for s in opp)
    # Frozen routed source diagnostic from previous audit; tolerate roundoff only.
    if abs(results["FN_vs_TN"] - 0.7556910569105691) > 1e-8:
        fail("routed FN-vs-TN AUC does not match formal source direct audit")
    if abs(results["TP_vs_FP"] - 0.8450996677740864) > 1e-8:
        fail("routed TP-vs-FP AUC does not match formal source direct audit")
    return results


def merge_source_folds(rows: list[dict[str, Any]], path: Path | None) -> None:
    if path is None:
        return
    raw = read_csv(path)
    if not raw or "sha256" not in raw[0]:
        fail("fold map requires sha256")
    fold_key = next((x for x in ("fold", "outer_fold", "fold_id") if x in raw[0]), None)
    if fold_key is None:
        fail("fold map requires fold / outer_fold / fold_id")
    mapping = {}
    for item in raw:
        verify_year(item, "fold map")
        sid = norm_sha(item["sha256"])
        if sid in mapping:
            fail("duplicate SHA in fold map")
        mapping[sid] = str(item[fold_key]).strip()
    source_keys = {r["sha256"] for r in rows}
    if set(mapping) != source_keys:
        fail("fold map must contain precisely 799 source SHAs")
    for r in rows:
        fold = mapping[r["sha256"]]
        if r["outer_fold"] and r["outer_fold"] != fold:
            fail("fold map conflicts with source prediction fold")
        r["outer_fold"] = fold


def compare_stage1_reference(rows: list[dict[str, Any]], path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "UNVERIFIED_LEGACY_COLUMN_ALIAS",
                "note": "Global24__prob / A25__routed are old field names; confusion counts alone do not prove SVM provenance."}
    raw = read_csv(path)
    if not raw or "sha256" not in raw[0] or "stage1_prob" not in raw[0] or "routed" not in raw[0]:
        fail("stage1 reference needs sha256,stage1_prob,routed")
    reference = {}
    for item in raw:
        verify_year(item, "stage1 reference")
        sid = norm_sha(item["sha256"])
        if sid in reference:
            fail("duplicate Stage-I reference SHA")
        reference[sid] = item
    if set(reference) != {r["sha256"] for r in rows}:
        fail("stage1 reference must match all source SHAs")
    max_diff = 0.0
    for row in rows:
        ref = reference[row["sha256"]]
        p = finite_prob(ref["stage1_prob"], "Stage-I source reference")
        rr = to_bit(ref["routed"], "Stage-I source route reference")
        max_diff = max(max_diff, abs(row["stage1_prob"] - p))
        if rr != row["routed"]:
            fail("Stage-I route differs from canonical reference")
    if max_diff > 1e-12:
        fail(f"Stage-I source probability differs from reference max_abs={max_diff}")
    return {"status": "EXACT_SOURCE_PROB_AND_ROUTE_MATCH", "max_abs_probability_diff": max_diff,
            "note": "Comparison checks values, not training-code provenance."}


def read_embeddings(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as pack:
            names = set(pack.files)
            sha_key = next((x for x in ("sha256", "sha") if x in names), None)
            fold_key = next((x for x in ("fold", "outer_fold", "fold_id") if x in names), None)
            vec_key = next((x for x in ("embedding", "embeddings", "hidden", "z") if x in names), None)
            if not all((sha_key, fold_key, vec_key)):
                fail("NPZ embeddings need sha256, fold, embedding (or documented aliases)")
            sha = pack[sha_key]
            fold = pack[fold_key]
            arr = np.asarray(pack[vec_key], float)
    elif path.suffix.lower() == ".csv":
        raw = read_csv(path)
        if not raw or "sha256" not in raw[0]:
            fail("embedding CSV requires sha256")
        fold_key = next((x for x in ("fold", "outer_fold", "fold_id") if x in raw[0]), None)
        if not fold_key:
            fail("embedding CSV requires fold / outer_fold / fold_id")
        names = [x for x in raw[0] if re.fullmatch(r"(z|emb|embedding|hidden)_[0-9]+", x)]
        if not names:
            fail("embedding CSV requires z_0,z_1,... (or emb_*/embedding_*/hidden_*)")
        names.sort(key=lambda k: int(k.rsplit("_", 1)[-1]))
        sha = [r["sha256"] for r in raw]
        fold = [r[fold_key] for r in raw]
        try:
            arr = np.asarray([[float(r[col]) for col in names] for r in raw], float)
        except (KeyError, ValueError):
            fail("embedding CSV has non-numeric dimensions")
    else:
        fail("embeddings must be an explicit .csv or .npz file")
    if arr.ndim != 2 or arr.shape[0] != len(sha) or arr.shape[0] != len(fold) or arr.shape[1] < 2:
        fail("embedding matrix shape does not match SHA and fold arrays")
    if not np.all(np.isfinite(arr)):
        fail("embeddings contain NaN/inf")
    output = {}
    for sid, fd, v in zip(sha, fold, arr):
        sid = norm_sha(sid.decode() if isinstance(sid, bytes) else sid)
        if sid in output:
            fail("duplicate embedding SHA")
        fold_id = str(fd.decode() if isinstance(fd, bytes) else fd).strip()
        if not fold_id:
            fail("missing embedding fold")
        output[sid] = (fold_id, np.asarray(v, float))
    return output


def attach_geometry(rows: list[dict[str, Any]], path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "NOT_AVAILABLE", "reason": "No source-only frozen hidden states provided; no geometry claim allowed."}
    embeddings = read_embeddings(path)
    source = {r["sha256"]: r for r in rows}
    if not set(embeddings).issubset(source):
        fail("embedding input contains non-source SHA")
    for sid, (fd, _) in embeddings.items():
        if source[sid]["outer_fold"] and source[sid]["outer_fold"] != fd:
            fail("embedding fold conflicts with source fold")
        source[sid]["outer_fold"] = fd
    routed = [r for r in rows if r["routed"]]
    covered = [r for r in routed if r["sha256"] in embeddings]
    missing = len(routed) - len(covered)
    fold_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in covered:
        fold_groups[r["outer_fold"]].append(r)
    if len(fold_groups) < 2:
        return {"status": "INSUFFICIENT_FOLD_COVERAGE", "covered_routed": len(covered), "missing_routed": missing,
                "note": "Never compare hidden vectors across different outer-fold models."}
    geom_rows = []
    for fold, group in sorted(fold_groups.items()):
        vecs = np.stack([embeddings[r["sha256"]][1] for r in group])
        norms = np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        normed = vecs / norms
        distance = 1 - normed @ normed.T
        np.fill_diagonal(distance, np.inf)
        for i, r in enumerate(group):
            others = np.argsort(distance[i])
            if len(others) < 1:
                continue
            avail = len(others) - 1
            result = {"sha256": r["sha256"], "fold": fold, "group": r["group"],
                      "label": r["label"], "neighbor_pool": "SAME_FOLD_ROUTED_OOF_ONLY",
                      "neighbors_available": avail}
            for k in SCORE_K:
                if avail >= k:
                    idx = [j for j in others if j != i][:k]
                    result[f"same_label_purity_k{k}"] = sum(group[j]["label"] == r["label"] for j in idx) / k
                else:
                    result[f"same_label_purity_k{k}"] = None
            opposite = [j for j in range(len(group)) if group[j]["label"] != r["label"]]
            same = [j for j in range(len(group)) if group[j]["label"] == r["label"] and j != i]
            result["nearest_opposite_cosine_distance"] = float(min(distance[i, j] for j in opposite)) if opposite else None
            result["nearest_same_cosine_distance"] = float(min(distance[i, j] for j in same)) if same else None
            result["nearest_neighbor_is_opposite"] = int(group[others[0]]["label"] != r["label"]) if avail else None
            geom_rows.append(result)
    write_csv("same_fold_geometry.csv", geom_rows)
    by_group = defaultdict(list)
    for r in geom_rows:
        if r.get("same_label_purity_k10") is not None:
            by_group[r["group"]].append(float(r["same_label_purity_k10"]))
    status = "DESCRIPTIVE_ONLY_FOLD_LOCAL_OOF"
    return {"status": status, "covered_routed": len(covered), "missing_routed": missing,
            "folds_with_embeddings": len(fold_groups), "cosine_k10_purity_by_group":
                {g: sum(a)/len(a) for g, a in sorted(by_group.items()) if a},
            "note": "Same-fold only; not comparable to the previous train-neighbor kNN protocol, if different. No causal conclusion."}


def attach_static_features(rows: list[dict[str, Any]], path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "NOT_AVAILABLE", "reason": "No verified source-only static-descriptor export supplied."}
    names = [c.strip() for c in os.environ.get("ECA_FEATURE_COLUMNS", "").split(",") if c.strip()]
    if not names:
        fail("ECA_STATIC_FEATURES provided without explicit ECA_FEATURE_COLUMNS allowlist")
    forbidden = re.compile(r"(label|pred|correct|error|fix|broken|routed|prob|score|future|year|fold|sha)", re.I)
    if len(set(names)) != len(names) or any(forbidden.search(n) for n in names):
        fail("feature allowlist includes duplicate/label/prediction/outcome-like columns")
    raw = read_csv(path)
    if not raw or "sha256" not in raw[0] or any(n not in raw[0] for n in names):
        fail("source feature export does not contain requested descriptors")
    inputs = {}
    for item in raw:
        verify_year(item, "source feature export")
        sid = norm_sha(item["sha256"])
        if sid in inputs:
            fail("duplicate static feature SHA")
        nums = {}
        for name in names:
            try:
                v = float(item[name]) if str(item[name]).strip() else None
            except ValueError:
                fail("nonnumeric static descriptor: " + name)
            if v is not None and not math.isfinite(v):
                fail("nonfinite static descriptor: " + name)
            nums[name] = v
        inputs[sid] = nums
    if not set(inputs).issubset({r["sha256"] for r in rows}):
        fail("static descriptor export contains non-source samples")
    joined = []
    for r in rows:
        if r["routed"] and r["sha256"] in inputs:
            joined.append({"sha256": r["sha256"], "group": r["group"], "stage1_side": r["stage1_side"], **inputs[r["sha256"]]})
    write_csv("static_descriptor_source_rows.csv", joined, ["sha256", "group", "stage1_side"] + names)
    summaries = []
    for name in names:
        for side in ("PREDICTED_BENIGN", "PREDICTED_MALWARE"):
            for group in ("ROUTED_UNFIXED", "ROUTED_BROKEN", "ROUTED_FIXED", "ROUTED_STABLE_CORRECT"):
                data = [float(r[name]) for r in joined if r["group"] == group and r["stage1_side"] == side and r[name] is not None]
                summaries.append({"descriptor": name, "stage1_side": side, "group": group, "n": len(data),
                                  "mean": statistics.mean(data) if data else None,
                                  "median": statistics.median(data) if data else None,
                                  "min": min(data) if data else None, "max": max(data) if data else None})
    write_csv("static_descriptor_group_summary.csv", summaries)
    return {"status": "DESCRIPTIVE_ONLY", "feature_columns": names, "covered_routed": len(joined),
            "note": "Descriptor associations are not missing-evidence proofs; no feature selection or model training."}


def scan_apk_archives(rows: list[dict[str, Any]], path: Path | None) -> dict[str, Any]:
    """Inspect optional source APK ZIP metadata; never treat string hits as executed calls."""
    if path is None:
        return {"status": "NOT_AVAILABLE", "reason": "No explicit source-2017 SHA-to-APK index provided."}
    raw = read_csv(path)
    if not raw or "sha256" not in raw[0] or "apk_path" not in raw[0]:
        fail("APK index requires sha256,apk_path")
    source = {r["sha256"]: r for r in rows}
    index = {}
    for item in raw:
        verify_year(item, "APK index")
        sid = norm_sha(item["sha256"])
        if sid not in source:
            fail("APK index includes SHA outside the 2017 source cohort")
        if sid in index:
            fail("duplicate SHA in APK index")
        index[sid] = Path(str(item["apk_path"]).strip()).resolve()
    routed = [r for r in rows if r["routed"]]
    selected = [r for r in routed if r["sha256"] in index]
    results = []
    for r in selected:
        path_apk = index[r["sha256"]]
        if not path_apk.is_file():
            fail("source APK does not exist: " + str(path_apk))
        # Verify complete APK content hash, not filename or asserted metadata.
        if sha256_file(path_apk) != r["sha256"]:
            fail("source APK hash differs from indexed SHA")
        try:
            with zipfile.ZipFile(path_apk, "r") as z:
                entries = z.infolist()
                if len(entries) > 50000:
                    fail("APK archive entry count too large")
                names = [info.filename.replace("\\", "/").lower() for info in entries]
                dex_indices = [i for i, name in enumerate(names) if re.fullmatch(r"classes([2-9][0-9]*)?\.dex", name)]
                native = [name for name in names if name.startswith("lib/") and name.endswith(".so")]
                embedded = [name for name in names if (name.endswith((".dex", ".jar", ".apk", ".zip"))
                             and not re.fullmatch(r"classes([2-9][0-9]*)?\.dex", name))]
                blob = bytearray()
                dex_bytes_total = 0
                for i in dex_indices:
                    info = entries[i]
                    if info.file_size > 128 * 1024 * 1024:
                        fail("individual classes.dex too large for bounded static hint scan")
                    dex_bytes_total += info.file_size
                    if dex_bytes_total > 256 * 1024 * 1024:
                        fail("total DEX bytes too large for bounded static hint scan")
                    blob.extend(z.read(info))
                b = bytes(blob)
                results.append({"sha256": r["sha256"], "group": r["group"],
                                "stage1_side": r["stage1_side"], "apk_verified_sha256": True,
                                "has_manifest_archive_entry": int("androidmanifest.xml" in names),
                                "dex_count": len(dex_indices), "native_library_count": len(native),
                                "embedded_code_archive_count": len(embedded),
                                "has_native_library": int(bool(native)), "has_secondary_dex": int(len(dex_indices) > 1),
                                "dex_contains_loader_class_string": int(any(token in b for token in (
                                    b"DexClassLoader", b"InMemoryDexClassLoader", b"PathClassLoader"))),
                                "dex_contains_reflection_type_string": int(b"java/lang/reflect" in b),
                                "dex_contains_loadlibrary_string": int(b"loadLibrary" in b),
                                "interpretation": "ZIP_METADATA_AND_DEX_STRING_HINTS_NOT_EXECUTION_EVIDENCE"})
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            fail("cannot inspect verified source APK " + r["sha256"] + ": " + str(exc))
    write_csv("source_apk_archive_hints.csv", results)
    by_group = {}
    for group in ("ROUTED_UNFIXED", "ROUTED_BROKEN", "ROUTED_FIXED", "ROUTED_STABLE_CORRECT"):
        part = [r for r in results if r["group"] == group]
        by_group[group] = {"n": len(part),
                           "native_library_rate": (sum(x["has_native_library"] for x in part) / len(part)) if part else None,
                           "multidex_rate": (sum(x["has_secondary_dex"] for x in part) / len(part)) if part else None}
    return {"status": "APK_ARCHIVE_OBSERVABILITY_HINTS_ONLY", "scanned_routed": len(results),
            "total_routed": len(routed), "full_routed_coverage": len(results) == 200,
            "by_group": by_group,
            "note": "APK ZIP facts and raw DEX type-name strings; no invocation paths, dataflow, or omitted-C43 facts are inferred."}


def make_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Purely descriptive score proximity; NOT a proposed training match or an embedding similarity.
    groups = defaultdict(list)
    for r in rows:
        if r["routed"]:
            groups[(r["group"], r["stage1_side"])].append(r)
    pairs = []
    for side in ("PREDICTED_BENIGN", "PREDICTED_MALWARE"):
        broken = groups[("ROUTED_BROKEN", side)]
        for wrong in groups[("ROUTED_UNFIXED", side)]:
            if not broken:
                continue
            nearest = min(broken, key=lambda r: (abs(r["expert_prob"] - wrong["expert_prob"]), r["sha256"]))
            pairs.append({"side": side, "unfixed_sha256": wrong["sha256"],
                          "broken_sha256": nearest["sha256"], "unfixed_score": wrong["expert_prob"],
                          "broken_score": nearest["expert_prob"],
                          "absolute_score_gap": abs(wrong["expert_prob"] - nearest["expert_prob"]),
                          "meaning": "SCALAR_SCORE_PROXIMITY_NOT_CAUSAL_EVIDENCE"})
    return sorted(pairs, key=lambda r: (r["absolute_score_gap"], r["side"]))


def make_manual_review(rows: list[dict[str, Any]]) -> None:
    errors = [r for r in rows if r["group"] in TARGET_GROUPS]
    # Review packets do not disclose error group or ground-truth label to the reviewer.
    mapping = []
    blank = []
    for r in sorted(errors, key=lambda x: hashlib.sha256(("ECA-R1-" + x["sha256"]).encode()).hexdigest()):
        case_id = "CASE-" + hashlib.sha256(("ECA-R1-" + r["sha256"]).encode()).hexdigest()[:12]
        mapping.append({"case_id": case_id, "sha256": r["sha256"], "group": r["group"],
                        "label": r["label"], "stage1_side": r["stage1_side"],
                        "expert_prob": r["expert_prob"], "expert_wrong_abs_margin": r["expert_wrong_abs_margin"],
                        "pairwise_inversion": r["pairwise_inversion"], "outer_fold": r["outer_fold"]})
        blank.append({"case_id": case_id, "sha256": r["sha256"], "reviewer": "",
                      "source_artifact_path": "", "source_artifact_sha256": "", "method_or_instruction": "",
                      "observed_static_fact": "", "c43_field_or_extraction_mapping_checked": "",
                      "independent_validation": "", "assessment": "UNREVIEWED", "notes": ""})
    if len(mapping) != 52:
        fail("manual review packet must have exactly 52 source routed error samples")
    write_csv("manual_review_blinded_template.csv", blank)
    write_csv("review_case_key_SOURCE_ONLY.csv", mapping)


def summary_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for group in ("ROUTED_UNFIXED", "ROUTED_BROKEN", "ROUTED_FIXED", "ROUTED_STABLE_CORRECT"):
        for side in ("ALL", "PREDICTED_BENIGN", "PREDICTED_MALWARE"):
            part = [r for r in rows if r["group"] == group and (side == "ALL" or r["stage1_side"] == side)]
            wrong = [r["expert_wrong_abs_margin"] for r in part if r["expert_wrong_abs_margin"] is not None]
            inversion = [r["pairwise_inversion"] for r in part if r["pairwise_inversion"] is not None]
            summary.append({"group": group, "stage1_side": side, "n": len(part),
                            "expert_prob_median": statistics.median(r["expert_prob"] for r in part) if part else None,
                            "wrong_margin_median": statistics.median(wrong) if wrong else None,
                            "wrong_confident_ge_0.25": sum(v >= .25 for v in wrong),
                            "pairwise_inversion_mean": statistics.mean(inversion) if inversion else None,
                            "pairwise_inversion_median": statistics.median(inversion) if inversion else None})
    return summary


def main() -> None:
    safe_path(OUTPUT)
    if (OUTPUT / "final_decision.json").exists():
        fail("final result exists; refusing to overwrite an existing completed audit")
    inputs = discover()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior = [record(path, name) for name, path in sorted(inputs.items())]
    dump("baseline_asset_manifest_before.json", prior)
    architecture_text = inputs["architecture"].read_text(encoding="utf-8").lower()
    if "direct" not in architecture_text:
        fail("architecture does not declare Direct-C43")
    rows = build_source_rows(inputs["source"])
    merge_source_folds(rows, inputs.get("folds"))
    provenance = compare_stage1_reference(rows, inputs.get("stage1_reference"))
    contract = source_contract(rows)
    auc_sides = add_pairwise_overlap(rows)
    geometry = attach_geometry(rows, inputs.get("embeddings"))
    static = attach_static_features(rows, inputs.get("static_features"))
    apk_hints = scan_apk_archives(rows, inputs.get("apk_index"))
    write_csv("source_2017_sample_phenotypes.csv", rows)
    write_csv("source_2017_group_summary.csv", summary_stats(rows))
    write_csv("source_2017_score_proximity_pairs.csv", make_pairs(rows))
    make_manual_review(rows)
    after = [record(Path(r["path"]), r["purpose"]) for r in prior]
    changed = [a["path"] for a, b in zip(prior, after) if a["sha256"] != b["sha256"] or a["size"] != b["size"]]
    dump("baseline_asset_manifest_after.json", after)
    integrity = {"status": "PASS" if not changed else "FAIL", "asset_count": len(prior), "changed": changed}
    dump("baseline_integrity_check.json", integrity)
    if changed:
        fail("read-only inputs modified")
    decision = {"status": "SOURCE_ERROR_PHENOTYPE_AUDIT_COMPLETE_CAUSE_UNRESOLVED",
                "source_2017_only": True, "future_rows_loaded": 0, "future_used_for_selection": False,
                "model_changed": False, "threshold_changed": False, "route_changed": False,
                "model_training_authorized": False, "causal_attribution_established": False,
                "source_contract": contract, "routed_auc": auc_sides,
                "stage1_reference_identity": provenance, "geometry": geometry, "static_descriptors": static, "apk_archive_hints": apk_hints,
                "review_cases": 52, "formal_integrity": integrity["status"],
                "next_action": "Independent blinded DEX/static-evidence review of 35 unfixed and 17 broken; verify omitted fact in C43 before proposing modules."}
    report = ["# CARE-Droid source-only error cause attribution audit v1", "",
              "**Status:** SOURCE_ERROR_PHENOTYPE_AUDIT_COMPLETE_CAUSE_UNRESOLVED", "",
              "This is a hypothesis-generation and evidence collection audit, NOT a proof of root cause.",
              "No new model, threshold, routing decision, or future-year asset was used.", "",
              "## Frozen source verification", "",
              "- 2017 APKs: 799 (400 benign / 399 malware).", "- Routed: 200; routed Stage-I errors: 84.",
              "- Routed unfixed: 35; broken: 17; fixed: 49; stable-correct: 99.",
              "- Direct-C43 source final TN=378, FP=22, FN=43, TP=356.",
              f"- Frozen routed FN-vs-TN AUC: {auc_sides['FN_vs_TN']:.6f}.",
              f"- Frozen routed TP-vs-FP AUC: {auc_sides['TP_vs_FP']:.6f}.", "",
              "## Evidence boundaries", "",
              "- Scalar pairwise inversions and wrong-score confidence are observable error phenotypes only.",
              "- A low same-label neighbor purity supports descriptive local overlap, NOT a provable static semantic limit.",
              "- Hidden vectors from different outer folds must never be compared directly.",
              "- A missing static feature can be established only by source-code/DEX evidence AND feature mapping inspection.",
              "- A confirmed feature omission is not automatically the cause of a prediction error.",
              "- No diagnosis is assigned to individual cases by automated score rules.", "",
              "## Optional assets", "",
              f"- Embedding geometry: {geometry['status']}.",
              f"- Source-only descriptors: {static['status']}.",
              f"- Optional APK archive hints: {apk_hints['status']}.",
              f"- Stage-I alias verification: {provenance['status']}.", "",
              "## Outputs", "",
              "- source_2017_sample_phenotypes.csv: all 799 source phenotypes, error score margins and pairwise inversions.",
              "- source_2017_group_summary.csv: four groups, two Stage-I sides, no model tuning.",
              "- source_2017_score_proximity_pairs.csv: descriptive unfixed-vs-broken score proximity.",
              "- review_case_key_SOURCE_ONLY.csv: 52 errors with group and ground truth; keep hidden during blinded review.",
              "- manual_review_blinded_template.csv: case-by-case DEX/static evidence annotation sheet.",
              "- same_fold_geometry.csv: available only if valid frozen source embeddings are supplied.",
              "- static_descriptor_group_summary.csv: available only if allowlisted source features are supplied.",
              "- source_apk_archive_hints.csv: available only with explicit hash-verified source APK index.", "",
              "## Prohibitions", "",
              "No future rows, no refitting, no threshold/oracle selection, no causal claims from scalar AUC,",
              "no reuse of unsupported benefit gates, and no tuning after seeing future diagnostics.", "",
              "Formal asset integrity: PASS.", ""]
    safe_path(OUTPUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    dump("final_decision.json", decision)
    print("CARE_DIRECT_C43_ERROR_ATTRIBUTION_FINAL_STATUS=" + decision["status"])
    print("SOURCE_ROUTED_UNFIXED=35 BROKEN=17 FIXED=49 STABLE_CORRECT=99")
    print("FORMAL_ASSET_INTEGRITY=PASS FUTURE_ROWS_LOADED=0")


if __name__ == "__main__":
    main()
