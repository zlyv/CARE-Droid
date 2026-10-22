#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT = Path("/home/boyuanjie/projects/tasa-droid")
SCRIPTS = PROJECT / "scripts"
ROOT = PROJECT / "artifacts/care-droid-formal-diagnostics-v1"
YEARS = (2018, 2019, 2020, 2021)
YEAR_COUNTS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
SOURCE_N = 799
FUTURE_N = 1989
MAX_DELTA = 2.25
METHOD = "P31RF__RFRoute__A0_ROLE_ONLY__Bounded"

SOURCE_P31 = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
SOURCE_REFERENCE = PROJECT / "artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv"
SOURCE_A0_CACHE = PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz"
FORMAL_FREEZE = PROJECT / "artifacts/care-droid-formal-v1/formal_freeze_manifest.json"
FUTURE_INTER = PROJECT / "artifacts/s7-future-interz-cache-v1/future_interz.npz"
ENDPOINT_CACHE_LEGACY = PROJECT / "artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz"
FOLD_ASSIGNMENT_CANDIDATES = (
    Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/baseline-results/same-year-baselines-v1/fold_assignments.csv"),
    PROJECT / "artifacts/same-year-baselines-v1/fold_assignments.csv",
    PROJECT / "same-year-baselines-v1/fold_assignments.csv",
    PROJECT / "scripts/same-year-baselines-v1/fold_assignments.csv",
)
FOLD_ASSIGNMENTS = next((p for p in FOLD_ASSIGNMENT_CANDIDATES if p.exists()), FOLD_ASSIGNMENT_CANDIDATES[0])

LABEL_FREE_COHORT_CANDIDATES = (
    PROJECT / "artifacts/care-di25-host-m0-future-replay-v1/preflight/future_cohort_label_free.csv",
    PROJECT / "artifacts/care-route-stage2-future-matrix-v1/preflight/future_cohort_label_free.csv",
    PROJECT / "artifacts/care-future-local-assets-v2/future_lineage_2018_2021.csv",
)
SEALED_FEATURE_CANDIDATES = (
    PROJECT / "artifacts/care-di25-host-m0-future-replay-v1/preflight/compact_features_2017_train_2018_2021_label_sealed.csv",
    PROJECT / "artifacts/care-route-stage2-future-matrix-v1/preflight/compact_features_2017_train_2018_2021_label_sealed.csv",
    PROJECT / "artifacts/care-route-stage2-within-year-matrix-v1/preflight/compact_features_2017_train_2018_2021_label_sealed.csv",
)
FUTURE_LABEL_CANDIDATES = (
    PROJECT / "artifacts/care-di25-host-m0-future-replay-v1/replay_predictions_with_labels.csv",
    PROJECT / "artifacts/care-litelocal-p31-c43-complete-diagnostics-v1/frozen_future_predictions_with_labels.csv",
)


def norm(value):
    return str(value or "").strip().casefold()


def read_csv(path: Path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def first_existing(paths: Iterable[Path], label: str):
    for path in paths:
        if Path(path).is_file() and Path(path).stat().st_size:
            return Path(path)
    raise RuntimeError("BLOCKED_BY_CONTRACT: no " + label + " found; checked:\n- " + "\n- ".join(map(str, paths)))


def infer_year(row):
    for key in ("year", "metadata_year"):
        value = str(row.get(key, "")).strip()
        if value:
            return int(float(value))
    raise RuntimeError("BLOCKED_BY_CONTRACT: row has no year")


def load_future_cohort():
    path = first_existing(LABEL_FREE_COHORT_CANDIDATES, "label-free future cohort")
    rows = []
    for row in read_csv(path):
        sid = norm(row.get("sha256"))
        if sid and infer_year(row) in YEARS:
            rows.append((sid, infer_year(row)))
    counts = {year: sum(y == year for _, y in rows) for year in YEARS}
    if len(rows) != FUTURE_N or len({x for x, _ in rows}) != FUTURE_N or counts != YEAR_COUNTS:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: future cohort changed n={len(rows)} counts={counts}")
    return path, rows


def _fold_rows_from_csv(path: Path, cohort_ids):
    """Read fold-bearing future rows from one CSV; return None if unsuitable."""
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            id_key = "sample_id" if "sample_id" in fields else "sha256" if "sha256" in fields else None
            fold_key = "fold" if "fold" in fields else "outer_fold" if "outer_fold" in fields else None
            if id_key is None or fold_key is None:
                return None
            rows = []
            seen = set()
            for row in reader:
                sid = norm(row.get(id_key))
                if sid not in cohort_ids:
                    continue
                raw_fold = str(row.get(fold_key, "")).strip()
                if not raw_fold:
                    return None
                fold = int(float(raw_fold))
                if fold not in (1, 2, 3, 4, 5) or sid in seen:
                    return None
                raw_year = str(row.get("year", row.get("metadata_year", ""))).strip()
                rows.append({"sample_id": sid, "fold": fold, "year": raw_year})
                seen.add(sid)
            return rows or None
    except Exception:
        return None


def discover_fold_assignment_rows():
    """Discover a prior fixed-fold asset that exactly covers the formal cohort."""
    _, cohort = load_future_cohort()
    wanted = {sid for sid, _ in cohort}
    candidates = list(FOLD_ASSIGNMENT_CANDIDATES)
    artifact_root = PROJECT / "artifacts"
    if artifact_root.is_dir():
        for path in artifact_root.glob("**/*.csv"):
            name = path.name.casefold()
            if any(token in name for token in ("fold", "prediction", "oof")):
                candidates.append(path)
    unique = []
    seen_paths = set()
    for path in candidates:
        key = str(path)
        if key not in seen_paths and path.is_file() and path.stat().st_size:
            seen_paths.add(key); unique.append(path)
    parsed = []
    for path in unique:
        rows = _fold_rows_from_csv(path, wanted)
        if not rows:
            continue
        ids = {r["sample_id"] for r in rows}
        text = str(path).casefold()
        score = (
            16 * int("care-litelocal-p31-diagnostics-v1" in text)
            + 12 * int("complete-diagnostics" in text)
            + 8 * int("within" in text)
            + 4 * int("prediction" in text)
            + 2 * int("fold_assignment" in text)
        )
        parsed.append((score, path, rows, ids))
    exact = [item for item in parsed if item[3] == wanted]
    if exact:
        exact.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
        best = exact[0]
        return [best[1]], best[2]

    # Some prior diagnostics saved disjoint year groups in sibling directories.
    # Combine only non-overlapping subsets from the same top-level experiment.
    grouped = {}
    for item in parsed:
        try:
            relative = item[1].relative_to(artifact_root)
            experiment = relative.parts[0]
        except Exception:
            continue
        grouped.setdefault(experiment, []).append(item)
    combinations = []
    for experiment, items in grouped.items():
        items.sort(key=lambda item: (len(item[3]), item[0]), reverse=True)
        merged_rows = []; merged_ids = set(); sources = []; score = 0
        for item in items:
            if merged_ids.intersection(item[3]):
                continue
            merged_rows.extend(item[2]); merged_ids.update(item[3]); sources.append(item[1]); score += item[0]
            if merged_ids == wanted:
                combinations.append((score, experiment, sources, merged_rows))
                break
    if combinations:
        combinations.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best = combinations[0]
        return best[2], best[3]

    inventory = sorted(
        ((len(item[3]), str(item[1])) for item in parsed), reverse=True
    )[:12]
    raise RuntimeError(
        "BLOCKED_BY_CONTRACT: no existing fixed-fold asset exactly covers the formal "
        f"1989-SHA cohort; best candidates={inventory}"
    )


def canonicalize_fold_assignments(path=None, raw_rows=None):
    """Attach the authoritative label-free cohort year to fixed fold IDs.

    The historical fold table contains one stale year annotation, while its
    SHA set and fold identifiers cover the complete 1,989-APK cohort.  Year is
    therefore taken from the frozen cohort contract; the existing fold number
    is preserved verbatim.  Any SHA/fold discrepancy remains fail-closed.
    """
    _, cohort = load_future_cohort()
    year_by_sha = {sid: year for sid, year in cohort}
    if raw_rows is None:
        raw = read_csv(Path(path or FOLD_ASSIGNMENTS))
    else:
        raw = list(raw_rows)
    if len(raw) != FUTURE_N:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: fold rows={len(raw)} expected={FUTURE_N}")
    normalized = []
    seen = set()
    mismatches = []
    for row in raw:
        sid = norm(row.get("sample_id", row.get("sha256", "")))
        if not sid or sid in seen:
            raise RuntimeError("BLOCKED_BY_CONTRACT: missing/duplicate SHA in fold assignments: " + sid)
        if sid not in year_by_sha:
            raise RuntimeError("BLOCKED_BY_CONTRACT: fold SHA outside formal future cohort: " + sid)
        seen.add(sid)
        try:
            fold = int(row["fold"])
            raw_source_year = str(row.get("year", "")).strip()
            source_year = int(float(raw_source_year)) if raw_source_year else int(year_by_sha[sid])
        except Exception as exc:
            raise RuntimeError("BLOCKED_BY_CONTRACT: invalid year/fold for " + sid) from exc
        if fold not in (1, 2, 3, 4, 5):
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: invalid fold={fold} for {sid}")
        canonical_year = int(year_by_sha[sid])
        item = {
            "sample_id": sid,
            "year": canonical_year,
            "fold": fold,
            "source_year": source_year,
            "year_mismatch": int(source_year != canonical_year),
        }
        normalized.append(item)
        if source_year != canonical_year:
            mismatches.append(item.copy())
    missing = sorted(set(year_by_sha).difference(seen))
    if missing:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: fold assignments miss {len(missing)} formal SHA")
    counts = {year: sum(r["year"] == year for r in normalized) for year in YEARS}
    if counts != YEAR_COUNTS:
        raise RuntimeError("BLOCKED_BY_CONTRACT: canonical fold year counts changed " + repr(counts))
    for year in YEARS:
        observed = {r["fold"] for r in normalized if r["year"] == year}
        if observed != {1, 2, 3, 4, 5}:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: year {year} folds={sorted(observed)}")
    return normalized, mismatches


def discover_p31_cache(explicit=None):
    _, cohort = load_future_cohort()
    wanted = {sid for sid, _ in cohort}
    candidates = [Path(explicit)] if explicit else sorted((PROJECT / "artifacts").glob("**/*.npz"))
    matches = []
    for path in candidates:
        try:
            with np.load(path, allow_pickle=False) as z:
                if "sha256" not in z or "role_prevalence" not in z:
                    continue
                ids = [norm(x) for x in z["sha256"]]
                x = np.asarray(z["role_prevalence"])
                if x.ndim != 2 or x.shape[1] != 31 or len(ids) != len(x):
                    continue
                coverage = len(wanted.intersection(ids))
                if coverage == FUTURE_N:
                    score = int("future" in str(path).casefold()) + int("semantic" in str(path).casefold())
                    matches.append((score, -len(ids), str(path), path))
        except Exception:
            continue
    if not matches:
        raise RuntimeError("BLOCKED_BY_CONTRACT: no sha256+role_prevalence[*,31] cache covers all 1989 future APKs")
    matches.sort(reverse=True)
    return matches[0][3]


def discover_base_relation_cache(explicit=None):
    _, cohort = load_future_cohort()
    wanted = {sid for sid, _ in cohort}
    candidates = [Path(explicit)] if explicit else sorted((PROJECT / "artifacts").glob("**/*.npz"))
    matches = []
    for path in candidates:
        if path.resolve() == SOURCE_A0_CACHE.resolve():
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                required = {"sha256", "host_values", "host_offsets", "real_host_count", "host_was_empty"}
                if not required.issubset(z.files):
                    continue
                ids = [norm(x) for x in z["sha256"]]
                values = np.asarray(z["host_values"])
                offsets = np.asarray(z["host_offsets"])
                if values.ndim != 2 or values.shape[1] != 105 or offsets.shape != (len(ids) + 1,):
                    continue
                coverage = len(wanted.intersection(ids))
                if coverage == FUTURE_N:
                    text = str(path).casefold()
                    score = (8 * int("care-di25-compact43-relationquery-diagnostics-v1" in text)
                             + 4 * int("relation" in text) + 2 * int("future" in text)
                             + int("compact43" in text))
                    matches.append((score, -len(ids), str(path), path))
        except Exception:
            continue
    if not matches:
        raise RuntimeError("BLOCKED_BY_CONTRACT: no Compact43+Relation31 cache covers all 1989 future APKs")
    matches.sort(reverse=True)
    best = matches[0]
    tied = [m for m in matches if m[:2] == best[:2]]
    if len(tied) > 1 and explicit is None:
        raise RuntimeError("BLOCKED_BY_CONTRACT: ambiguous future relation caches; set C43_FUTURE_REGION_CACHE:\n- " + "\n- ".join(str(m[3]) for m in tied))
    return best[3]


def discover_future_inter_cache(explicit=None):
    """Find an order-compatible, label-free 2018--2021 InterZ cache.

    Historical experiments did not use one stable output directory, so the
    formal diagnostic must identify this asset by its schema and SHA contract,
    not by a guessed pathname.  Returning None is intentional: the caller may
    then build the same label-free cache from the already extracted endpoint
    cache.  A supplied explicit path is strict and never silently ignored.
    """
    _, cohort = load_future_cohort()
    expected = [sid for sid, _ in cohort]
    required = {
        "sha256", "pair_raw", "inter_z_clipped", "null_samples",
        "swaps_per_edge", "null_seed", "z_clip",
    }
    if explicit:
        candidates = [Path(explicit)]
        if not candidates[0].is_file():
            raise RuntimeError(
                "BLOCKED_BY_CONTRACT: FUTURE_INTER_CACHE does not exist: "
                + str(candidates[0])
            )
    else:
        seen = set()
        candidates = []
        for path in [FUTURE_INTER, *sorted((PROJECT / "artifacts").glob("**/*.npz"))]:
            key = str(path)
            if key not in seen:
                seen.add(key)
                candidates.append(path)
    matches = []
    for path in candidates:
        if not path.is_file() or not path.stat().st_size:
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                if not required.issubset(z.files):
                    continue
                ids = [norm(x) for x in z["sha256"]]
                pair = np.asarray(z["pair_raw"])
                inter = np.asarray(z["inter_z_clipped"])
                params_ok = (
                    int(z["null_samples"]) == 64
                    and int(z["swaps_per_edge"]) == 10
                    and int(z["null_seed"]) == 20260827
                    and float(z["z_clip"]) == 5.0
                )
                if (
                    ids != expected
                    or pair.shape != (FUTURE_N, 961)
                    or inter.shape != (FUTURE_N, 961)
                    or not params_ok
                    or not np.isfinite(pair).all()
                    or not np.isfinite(inter).all()
                ):
                    continue
                text = str(path).casefold()
                score = 4 * int("future" in text) + 2 * int("interz" in text) + int("inter" in text)
                matches.append((score, str(path), path))
        except Exception:
            continue
    if explicit and not matches:
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: explicit FUTURE_INTER_CACHE does not satisfy "
            "the 1989-row ordered SHA/schema/parameter contract: " + str(explicit)
        )
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][2]


def discover_endpoint_cache(explicit=None):
    """Find a successful endpoint JSONL cache covering source and future SHA."""
    _, cohort = load_future_cohort()
    wanted = {sid for sid, _ in cohort}
    wanted.update(norm(row["sha256"]) for row in read_csv(SOURCE_REFERENCE))
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        candidates.extend([
            ENDPOINT_CACHE_LEGACY,
            PROJECT / "artifacts/p54-new-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz/endpoint_centered_tensor_cache.jsonl.gz",
        ])
        candidates.extend(sorted((PROJECT / "artifacts").glob("**/*endpoint*cache*.jsonl.gz")))
        storage = Path("/mnt/HDD1/boyuanjie/tasa-droid-storage")
        if storage.is_dir():
            candidates.extend(sorted(storage.glob("**/*endpoint*cache*.jsonl.gz")))
    unique = []
    seen_paths = set()
    for path in candidates:
        key = str(path)
        if key not in seen_paths:
            seen_paths.add(key); unique.append(path)
    sha_pattern = re.compile(r'"sha256"\s*:\s*"([0-9A-Fa-f]{64})"')
    status_pattern = re.compile(r'"status"\s*:\s*"OK"')
    matches = []; inventory = []; coverage_by_path = []
    for path in unique:
        if not path.is_file() or not path.stat().st_size:
            continue
        covered = set()
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not status_pattern.search(line):
                        continue
                    match = sha_pattern.search(line)
                    if match:
                        sid = match.group(1).casefold()
                        if sid in wanted:
                            covered.add(sid)
                            if len(covered) == len(wanted):
                                break
        except Exception:
            continue
        inventory.append((len(covered), str(path)))
        if covered:
            coverage_by_path.append((path, covered))
        if covered == wanted:
            text = str(path).casefold()
            score = 8 * int("p54-new" in text) + 4 * int("full-cap256" in text) + int("storage" in text)
            matches.append((score, str(path), path))
    if explicit and not matches:
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: ENDPOINT_CACHE does not cover all 2788 formal SHA: "
            + str(explicit)
        )
    if not matches and not explicit:
        # Permit a source cache plus a disjoint future cache.  Merge verbatim
        # successful JSONL rows; no feature is recomputed and no APK is read.
        selected = []; accumulated = set()
        remaining = list(coverage_by_path)
        while remaining and accumulated != wanted:
            remaining.sort(key=lambda item: len(item[1].difference(accumulated)), reverse=True)
            path, coverage = remaining.pop(0)
            gain = coverage.difference(accumulated)
            if not gain:
                break
            selected.append(path); accumulated.update(coverage)
        if accumulated == wanted:
            target = ROOT / "cache/endpoint_cache_2788.jsonl.gz"
            target.parent.mkdir(parents=True, exist_ok=True)
            written = set()
            with gzip.open(target, "wt", encoding="utf-8") as output:
                for path in selected:
                    with gzip.open(path, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            if not status_pattern.search(line):
                                continue
                            match = sha_pattern.search(line)
                            if not match:
                                continue
                            sid = match.group(1).casefold()
                            if sid in wanted and sid not in written:
                                output.write(line if line.endswith("\n") else line + "\n")
                                written.add(sid)
            if written != wanted:
                raise RuntimeError("BLOCKED_BY_CONTRACT: merged endpoint cache lost SHA rows")
            return target
    if not matches:
        inventory.sort(reverse=True)
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: no endpoint cache or cache union covers all 2788 formal SHA; "
            f"best candidates={inventory[:8]}"
        )
    matches.sort(reverse=True)
    return matches[0][2]


def binary_auc(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos or not neg:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def metrics(y, base, probability, route):
    y = np.asarray(y, int)
    base = np.asarray(base, float)
    probability = np.asarray(probability, float)
    route = np.asarray(route, bool)
    bp = base >= .5
    pp = probability >= .5
    base_ok = bp == y
    ok = pp == y
    tn = int(np.sum((y == 0) & ~pp)); fp = int(np.sum((y == 0) & pp))
    fn = int(np.sum((y == 1) & ~pp)); tp = int(np.sum((y == 1) & pp))
    fixed = (~base_ok) & ok
    broken = base_ok & ~ok
    return {
        "BA": float((tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2),
        "AUC": binary_auc(y, probability),
        "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp + fn,
        "fixed": int(fixed.sum()), "broken": int(broken.sum()),
        "net": int(fixed.sum() - broken.sum()), "routed": int(route.sum()),
        "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
        "broken_TN_to_FP": int(np.sum(broken & (y == 0))), "broken_TP_to_FN": int(np.sum(broken & (y == 1))),
    }


def load_future_labels():
    path = first_existing(FUTURE_LABEL_CANDIDATES, "future diagnostic labels")
    labels = {}
    for row in read_csv(path):
        sid = norm(row.get("sha256"))
        raw = row.get("y", row.get("label", ""))
        if sid and str(raw).strip() != "":
            labels[sid] = int(float(raw))
    _, cohort = load_future_cohort()
    missing = [sid for sid, _ in cohort if sid not in labels]
    if missing:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: future label table missing {len(missing)} SHA")
    return path, labels
