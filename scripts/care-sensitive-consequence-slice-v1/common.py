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

PROJECT = Path(os.environ.get("SCS_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("SCS_OUTPUT", str(PROJECT / "outputs/care-sensitive-consequence-slice-v1"))).resolve()
SEED = 20261008
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
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def env_path(name: str) -> list[Path]:
    return [Path(os.environ[name])] if os.environ.get(name) else []


def first_existing(paths: Iterable[Path], label: str, directory: bool = False) -> Path:
    checked = []
    for path in paths:
        path = Path(path); checked.append(str(path))
        if path.is_dir() if directory else path.is_file():
            return path.resolve()
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {label}; checked={checked!r}")


def assets() -> dict[str, Path]:
    return {
        "formal": first_existing(env_path("SCS_FORMAL") + [
            PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv",
            PROJECT / "artifacts/care-droid-formal-v1/formal_predictions.csv",
        ], "formal source predictions"),
        "endpoint": first_existing(env_path("SCS_ENDPOINT") + [
            Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
            PROJECT / "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
            PROJECT / "artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz",
        ], "2017 endpoint cache"),
        "attention": first_existing(env_path("SCS_ATTENTION") + [
            PROJECT / "outputs/care-ccbc-source-v3/cache/formal_attention.npz",
            PROJECT / "outputs/care-ccbc-source-v2/cache/formal_attention.npz",
        ], "formal C43 attention"),
        "graphs": first_existing(env_path("SCS_GRAPH_DIR") + [
            PROJECT / "outputs/care-contextual-multihop-witness-v1/cache/callsite_graphs",
            PROJECT / "artifacts/care-contextual-multihop-witness-v1/cache/callsite_graphs",
        ], "CMW callsite-aware graph cache", directory=True),
    }


def parse_formal(path: Path) -> list[dict[str, Any]]:
    raw = read_csv(path)
    if len(raw) != 799:
        raise RuntimeError(f"source rows changed: {len(raw)}")
    base_field = "Global24__prob" if "Global24__prob" in raw[0] else "stage1_prob"
    route_field = "A25__routed" if "A25__routed" in raw[0] else "routed"
    final_field = "P31RF__RFRoute__A0_ROLE_ONLY__Bounded__prob" if "P31RF__RFRoute__A0_ROLE_ONLY__Bounded__prob" in raw[0] else "final_prob"
    rows = []
    for item in raw:
        probability = float(item[base_field])
        rows.append({"sha256": norm(item["sha256"]), "fold": int(float(item["fold"])),
                     "label": int(float(item["label"])), "stage1_prob": probability,
                     "stage1_pred": int(probability >= .5), "routed": int(float(item[route_field])),
                     "formal_prob": float(item[final_field])})
    return rows


def read_endpoint_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line); sid = norm(row.get("sha256", row.get("sample_id")))
            if sid in wanted and str(row.get("status", "OK")).upper() in {"OK", "SUCCESS"}:
                payload = row.get("payload", row)
                if isinstance(payload, dict):
                    result[sid] = payload
                    if len(result) == len(wanted):
                        break
    return result


def endpoint_hosts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ids = list(payload.get("node_ids") or []); roles = list(payload.get("node_role_ids") or [])
    base = list(payload.get("node_base_features") or []); edges = payload.get("edge_index") or [[], []]
    if not ids:
        return []
    if len(ids) != len(base) or len(ids) != len(roles) or len(edges) != 2:
        raise RuntimeError("endpoint payload schema mismatch")
    endpoints = {index for index, values in enumerate(base) if len(values) > 4 and float(values[4]) > .5}
    by_host: dict[int, set[int]] = {}
    for source, target in zip(edges[0], edges[1]):
        source, target = int(source), int(target)
        if target in endpoints:
            by_host.setdefault(source, set()).add(target)
    output = []
    for host_index in sorted(by_host):
        merged = sorted({int(role) for target in by_host[host_index] for role in (roles[target] or []) if 0 <= int(role) < 31})
        if merged:
            output.append({"node_id": int(ids[host_index]), "roles": merged})
    return output


def load_attention(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        ids = [norm(value) for value in archive["sha256"]]
        offsets = np.asarray(archive["offsets"], int); values = np.asarray(archive["attention"], float)
        critical = np.asarray(archive["critical_index"], int)
        if len(offsets) != len(ids) + 1:
            raise RuntimeError("attention offsets mismatch")
        for index, sid in enumerate(ids):
            result[sid] = {"attention": values[offsets[index]:offsets[index + 1]], "critical": int(critical[index])}
    return result


def file_record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve(); status = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": status.st_size,
            "mtime_ns": status.st_mtime_ns, "sha256": sha256_file(path)}


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    positive, negative = int(y.sum()), len(y) - int(y.sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort"); ranks = np.empty(len(score), float)
    sorted_score = score[order]; start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False
