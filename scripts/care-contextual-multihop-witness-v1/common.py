#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT = Path(os.environ.get("CMW_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("CMW_OUTPUT", str(PROJECT / "outputs/care-contextual-multihop-witness-v1"))).resolve()
SEED = 20261007
MAX_DELTA = 2.25
WS = re.compile(r"\s+")


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def canonical(value: Any) -> str:
    return WS.sub("", str(value or "")).casefold()


def stable_id(value: str) -> int:
    return int.from_bytes(hashlib.sha256(canonical(value).encode("utf-8")).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def first_existing(paths: Iterable[Path], label: str) -> Path:
    checked = []
    for path in paths:
        path = Path(path)
        checked.append(str(path))
        if path.is_file():
            return path.resolve()
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {label}; checked={checked!r}")


def optional_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        path = Path(path)
        if path.is_file():
            return path.resolve()
    return None


def env_path(name: str) -> list[Path]:
    return [Path(os.environ[name])] if os.environ.get(name) else []


def file_record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def source_assets() -> dict[str, Path]:
    return {
        "formal": first_existing(env_path("CMW_SOURCE_FORMAL") + [
            PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv",
            PROJECT / "artifacts/care-droid-formal-v1/formal_predictions.csv",
        ], "source formal predictions"),
        "endpoint": first_existing(env_path("CMW_SOURCE_ENDPOINT") + [
            PROJECT / "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
            PROJECT / "artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz",
        ], "source endpoint cache"),
        "attention": first_existing(env_path("CMW_SOURCE_ATTENTION") + [
            PROJECT / "outputs/care-ccbc-source-v3/cache/formal_attention.npz",
            PROJECT / "outputs/care-ccbc-source-v2/cache/formal_attention.npz",
            PROJECT / "outputs/care-ccbc-source-v1/cache/formal_attention.npz",
        ], "source C43 attention"),
    }


def diagnostic_assets(required: bool = True) -> dict[str, Path] | None:
    frozen = optional_existing(env_path("CMW_FUTURE_FROZEN") + [
        PROJECT / "artifacts/care-droid-formal-diagnostics-v1/frozen_predictions_with_labels.csv",
        PROJECT / "outputs/care-droid-formal-diagnostics-v1/frozen_predictions_with_labels.csv",
    ])
    within_a = optional_existing(env_path("CMW_WITHIN_2018_2019") + [
        PROJECT / "artifacts/care-droid-formal-diagnostics-v1/within_2018_2019/predictions.csv",
        PROJECT / "outputs/care-droid-formal-diagnostics-v1/within_2018_2019/predictions.csv",
    ])
    within_b = optional_existing(env_path("CMW_WITHIN_2020_2021") + [
        PROJECT / "artifacts/care-droid-formal-diagnostics-v1/within_2020_2021/predictions.csv",
        PROJECT / "outputs/care-droid-formal-diagnostics-v1/within_2020_2021/predictions.csv",
    ])
    frozen_attention = optional_existing(env_path("CMW_FUTURE_FROZEN_ATTENTION") + [
        PROJECT / "outputs/care-ccbc-diagnostics-v1/cache/future_attention_frozen.npz",
    ])
    within_attention = optional_existing(env_path("CMW_FUTURE_WITHIN_ATTENTION") + [
        PROJECT / "outputs/care-ccbc-diagnostics-v1/cache/future_attention_within.npz",
    ])
    endpoint = resolve_diagnostic_endpoint()
    values = {"frozen": frozen, "within_a": within_a, "within_b": within_b,
              "frozen_attention": frozen_attention, "within_attention": within_attention,
              "endpoint": endpoint}
    if any(value is None for value in values.values()):
        if required:
            raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete frozen/within-year diagnostics assets " + repr(values))
        return None
    return {key: value for key, value in values.items() if value is not None}


def resolve_diagnostic_endpoint() -> Path | None:
    explicit = optional_existing(env_path("CMW_FUTURE_ENDPOINT"))
    if explicit:
        return explicit
    for preflight in [
        PROJECT / "artifacts/care-droid-formal-diagnostics-v1/resolved_inputs.json",
        PROJECT / "outputs/care-ccbc-diagnostics-v1/preflight.json",
        PROJECT / "artifacts/care-droid-formal-diagnostics-v1/preflight.json",
        PROJECT / "outputs/care-droid-formal-diagnostics-v1/preflight.json",
    ]:
        if not preflight.is_file():
            continue
        try:
            data = json.loads(preflight.read_text(encoding="utf-8"))
            for container in (data, data.get("resolved_inputs", {}), data.get("assets", {})):
                raw = container.get("endpoint_cache") if isinstance(container, dict) else None
                if raw and Path(raw).is_file():
                    return Path(raw).resolve()
        except Exception:
            pass
    return optional_existing([
        PROJECT / "outputs/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz",
        PROJECT / "outputs/care-ccbc-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz",
    ])


def prediction_column(rows: list[dict[str, str]], suffix: str) -> str:
    matches = [name for name in rows[0] if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"prediction column {suffix!r} ambiguous: {matches!r}")
    return matches[0]


def parse_source_formal(path: Path) -> list[dict[str, Any]]:
    raw = read_csv(path)
    if len(raw) != 799:
        raise RuntimeError(f"source rows changed: {len(raw)}")
    base_field = "Global24__prob" if "Global24__prob" in raw[0] else "stage1_prob"
    route_field = "A25__routed" if "A25__routed" in raw[0] else "routed"
    score_field = "A25__score" if "A25__score" in raw[0] else "router_score"
    final_field = "P31RF__RFRoute__A0_ROLE_ONLY__Bounded__prob" if "P31RF__RFRoute__A0_ROLE_ONLY__Bounded__prob" in raw[0] else "final_prob"
    delta_field = "P31RF__RFRoute__A0_ROLE_ONLY__Bounded__bounded_delta"
    rows = []
    for row in raw:
        base = float(row[base_field])
        final = float(row[final_field])
        rows.append({"sha256": norm(row["sha256"]), "fold": int(float(row["fold"])),
                     "year": 2017, "label": int(float(row["label"])), "stage1_prob": base,
                     "stage1_pred": int(base >= .5), "router_score": float(row[score_field]),
                     "routed": int(float(row[route_field])), "formal_prob": final,
                     "formal_delta": float(row.get(delta_field, 0.0) or 0.0)})
    return rows


def parse_diagnostic_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path):
        base = float(row.get("stage1_prob", row.get("base_prob", 0.0)))
        final = float(row.get("final_prob", row.get("di25_final_prob",
                      row.get("bounded_prob", row.get("candidate_prob", base)))))
        rows.append({"sha256": norm(row["sha256"]), "fold": int(float(row.get("fold", 0) or 0)),
                     "year": int(float(row["year"])), "label": int(float(row["label"])),
                     "stage1_prob": base, "stage1_pred": int(base >= .5),
                     "router_score": float(row.get("router_score", 0.0) or 0.0),
                     "routed": int(float(row.get("routed", 0) or 0)), "formal_prob": final,
                     "formal_delta": float(row.get("bounded_delta", 0.0) or 0.0)})
    return rows


def read_endpoint_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = norm(row.get("sha256", row.get("sample_id")))
            status = str(row.get("status", "OK")).upper()
            if sid in wanted and status in {"OK", "SUCCESS"}:
                payload = row.get("payload", row)
                if isinstance(payload, dict):
                    result[sid] = payload
                    if len(result) == len(wanted):
                        break
    return result


def endpoint_hosts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ids = list(payload.get("node_ids") or [])
    roles = list(payload.get("node_role_ids") or [])
    base = list(payload.get("node_base_features") or [])
    edge_index = payload.get("edge_index") or [[], []]
    if not ids:
        return []
    if len(ids) != len(base) or len(ids) != len(roles) or len(edge_index) != 2:
        raise RuntimeError("endpoint payload schema mismatch")
    endpoints = {index for index, values in enumerate(base) if len(values) > 4 and float(values[4]) > .5}
    by_host: dict[int, set[int]] = {}
    for source, target in zip(edge_index[0], edge_index[1]):
        source, target = int(source), int(target)
        if target in endpoints:
            by_host.setdefault(source, set()).add(target)
    output = []
    for host_index in sorted(by_host):
        merged = sorted({int(role) for target in by_host[host_index] for role in (roles[target] or []) if 0 <= int(role) < 31})
        if merged:
            output.append({"old_node_index": host_index, "node_id": int(ids[host_index]), "roles": merged})
    return output


def load_attention(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        ids = [norm(value) for value in archive["sha256"]]
        offsets = np.asarray(archive["offsets"], int)
        values = np.asarray(archive["attention"], float)
        critical = np.asarray(archive["critical_index"], int)
        if len(offsets) != len(ids) + 1:
            raise RuntimeError("attention offsets mismatch")
        for index, sid in enumerate(ids):
            result[sid] = {"attention": values[offsets[index]:offsets[index + 1]], "critical": int(critical[index])}
    return result


def load_apk_paths(wanted: set[str], future: bool = False) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    candidates = []
    env = "CMW_FUTURE_APK_INDEX" if future else "CMW_SOURCE_APK_INDEX"
    candidates += env_path(env)
    if future:
        candidates += [PROJECT / "outputs/care-ccbc-diagnostics-v1/apk_paths.json",
                       PROJECT / "outputs/care-ccbc-diagnostics-v1/future_apk_paths.json"]
    else:
        candidates += [PROJECT / "outputs/care-ccbc-source-v3/apk_paths.json",
                       PROJECT / "outputs/care-ccbc-source-v2/apk_paths.json",
                       PROJECT / "outputs/care-icde-source-v1/apk_asset_audit.json"]
    result: dict[str, Path] = {}
    audit = []
    for path in candidates:
        if not path.is_file():
            continue
        matched = 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if path.name.endswith("asset_audit.json"):
                pairs = ((row.get("sha256"), row.get("path")) for row in data.get("records", []))
            elif isinstance(data, dict):
                pairs = data.items()
            else:
                pairs = []
            for sid, raw in pairs:
                sid = norm(sid)
                apk = Path(str(raw or ""))
                if sid in wanted and apk.is_file():
                    result.setdefault(sid, apk.resolve()); matched += 1
            audit.append({"path": str(path.resolve()), "matched": matched})
        except Exception as exc:
            audit.append({"path": str(path.resolve()), "error": repr(exc)})
    roots_env = "CMW_FUTURE_APK_ROOTS" if future else "CMW_SOURCE_APK_ROOTS"
    roots = [Path(value) for value in os.environ.get(roots_env, "").split(os.pathsep) if value]
    for root in roots:
        matched = 0
        if root.is_dir():
            for apk in root.rglob("*.apk"):
                sid = apk.stem.casefold()
                if sid in wanted and sid not in result:
                    result[sid] = apk.resolve(); matched += 1
        audit.append({"path": str(root), "root_scan_matched": matched})
    return result, audit


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    pos, neg = int(y.sum()), len(y) - int(y.sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float); sorted_score = score[order]; start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def metrics(y: np.ndarray, base: np.ndarray, probability: np.ndarray, route: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, int); base = np.asarray(base, float); probability = np.asarray(probability, float); route = np.asarray(route, bool)
    base_pred = base >= .5; pred = probability >= .5; base_ok = base_pred == y; ok = pred == y
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    fixed = ~base_ok & ok; broken = base_ok & ~ok
    return {"BA": float(.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))), "AUC": binary_auc(y, probability),
            "F1": float(2 * tp / max(2 * tp + fp + fn, 1)), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "fixed": int(fixed.sum()), "broken": int(broken.sum()), "net": int(fixed.sum() - broken.sum()),
            "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
            "broken_TN": int(np.sum(broken & (y == 0))), "broken_TP": int(np.sum(broken & (y == 1))),
            "routed": int(route.sum()), "class_flips": int(np.sum(base_pred != pred))}


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, float), -50, 50)
    return 1 / (1 + np.exp(-values))
