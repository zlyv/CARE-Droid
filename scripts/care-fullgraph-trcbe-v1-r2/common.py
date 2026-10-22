#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(os.environ.get("FULLGRAPH_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("FULLGRAPH_OUTPUT", str(PROJECT / "outputs/care-fullgraph-trcbe-v2"))).resolve()
WS = re.compile(r"\s+")


def canonical(value: Any) -> str:
    return WS.sub("", str(value or "")).casefold()


def stable_id(signature: str) -> int:
    return int.from_bytes(hashlib.sha256(canonical(signature).encode("utf-8")).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first_existing(paths: Iterable[Path], label: str) -> Path:
    checked = []
    for path in paths:
        checked.append(str(path))
        if path.is_file():
            return path.resolve()
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {label}; checked={checked!r}")


def file_record(path: Path, purpose: str) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "absolute_path": str(resolved),
        "purpose": purpose,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def explicit(name: str) -> list[Path]:
    return [Path(os.environ[name])] if os.environ.get(name) else []


def discover_assets() -> dict[str, Path]:
    return {
        "formal": first_existing(explicit("FULLGRAPH_FORMAL") + [
            PROJECT / "artifacts/care-droid-formal-v1/formal_predictions.csv",
            PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv",
        ], "formal predictions"),
        "endpoint": first_existing(explicit("FULLGRAPH_ENDPOINT_CACHE") + [
            PROJECT / "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
            PROJECT / "artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz",
        ], "frozen endpoint cache"),
        "attention": first_existing(explicit("FULLGRAPH_ATTENTION") + [
            PROJECT / "outputs/care-ccbc-source-v3/cache/formal_attention.npz",
            PROJECT / "outputs/care-ccbc-source-v2/cache/formal_attention.npz",
            PROJECT / "outputs/care-ccbc-source-v1/cache/formal_attention.npz",
        ], "frozen C43 attention"),
        "trcbe_schema": first_existing(explicit("FULLGRAPH_TRCBE_SCHEMA") + [
            PROJECT / "outputs/care-trcbe-source-v1/trigger_root_schema.json",
        ], "frozen TRCBE root taxonomy"),
        "trcbe_coverage": first_existing(explicit("FULLGRAPH_TRCBE_COVERAGE") + [
            PROJECT / "outputs/care-trcbe-source-v1/coverage_audit.json",
        ], "previous TRCBE coverage audit"),
    }


def formal_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(path):
        sid = str(raw.get("sha256", "")).strip().casefold()
        fold = int(float(raw["fold"]))
        routed = int(float(raw.get("routed", raw.get("A25__routed", 0))))
        if len(sid) != 64:
            raise RuntimeError("formal SHA invalid")
        rows.append({"sha256": sid, "fold": fold, "routed": routed})
    if len(rows) != 799 or len({row["sha256"] for row in rows}) != 799:
        raise RuntimeError(f"formal lineage mismatch rows={len(rows)} unique={len({r['sha256'] for r in rows})}")
    return rows


def load_apk_map(wanted: set[str]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    candidates: list[Path] = []
    if os.environ.get("FULLGRAPH_APK_INDEX"):
        candidates.append(Path(os.environ["FULLGRAPH_APK_INDEX"]))
    candidates += [
        PROJECT / "outputs/care-icde-source-v1/apk_asset_audit.json",
        PROJECT / "outputs/care-trcbe-source-v1/apk_paths.json",
        PROJECT / "outputs/care-ccbc-source-v3/apk_paths.json",
        PROJECT / "outputs/care-ccbc-source-v2/apk_paths.json",
    ]
    result: dict[str, Path] = {}
    audit = []
    for path in candidates:
        if not path.is_file():
            continue
        matched = 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "apk_asset_audit.json":
                pairs = ((row.get("sha256"), row.get("path")) for row in data.get("records", []))
            elif isinstance(data, dict):
                pairs = data.items()
            else:
                pairs = []
            for sid, raw_path in pairs:
                sid = str(sid or "").strip().casefold()
                apk = Path(str(raw_path or ""))
                if sid in wanted and apk.is_file():
                    result.setdefault(sid, apk.resolve())
                    matched += 1
            audit.append({"path": str(path.resolve()), "matched": matched})
        except Exception as exc:
            audit.append({"path": str(path.resolve()), "error": repr(exc)})
    roots = [Path(value) for value in os.environ.get("FULLGRAPH_APK_ROOTS", "").split(os.pathsep) if value]
    for root in roots:
        matched = 0
        if root.is_dir():
            for apk in root.rglob("*.apk"):
                sid = apk.stem.casefold()
                if sid in wanted and sid not in result:
                    result[sid] = apk.resolve()
                    matched += 1
        audit.append({"path": str(root), "root_scan_matched": matched})
    return result, audit


def read_endpoint_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("sha256", "")).strip().casefold()
            if sid in wanted and row.get("status") == "OK" and isinstance(row.get("payload"), dict):
                result[sid] = row["payload"]
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
    endpoints = {index for index, row in enumerate(base) if len(row) > 4 and float(row[4]) > 0.5}
    by_host: dict[int, set[int]] = {}
    for source, target in zip(edge_index[0], edge_index[1]):
        source, target = int(source), int(target)
        if target in endpoints:
            by_host.setdefault(source, set()).add(target)
    result = []
    for host_index in sorted(by_host):
        merged_roles = sorted({int(role) for target in by_host[host_index] for role in (roles[target] or []) if 0 <= int(role) < 31})
        if merged_roles:
            result.append({"old_node_index": host_index, "node_id": int(ids[host_index]), "roles": merged_roles})
    return result


def load_attention(path: Path) -> dict[str, dict[str, Any]]:
    import numpy as np

    result = {}
    with np.load(path, allow_pickle=False) as archive:
        shas = [str(value).strip().casefold() for value in archive["sha256"]]
        offsets = np.asarray(archive["offsets"], int)
        attention = np.asarray(archive["attention"], float)
        critical = np.asarray(archive["critical_index"], int)
        if len(offsets) != len(shas) + 1:
            raise RuntimeError("attention offsets mismatch")
        for index, sid in enumerate(shas):
            result[sid] = {
                "attention": attention[offsets[index]:offsets[index + 1]],
                "critical": int(critical[index]),
            }
    return result


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)
