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

PROJECT = Path(os.environ.get("SGCA_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("SGCA_OUTPUT", str(PROJECT / "outputs/care-sensitive-guard-context-audit-v1"))).resolve()
WS = re.compile(r"\s+")


def ensure_output_boundary() -> None:
    allowed = (PROJECT / "outputs").resolve()
    if OUTPUT == allowed or allowed not in OUTPUT.parents:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: output outside project outputs: {OUTPUT}")
    if not OUTPUT.name.startswith("care-sensitive-guard-context-audit-v"):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: unexpected output directory: {OUTPUT}")


def canonical(value: Any) -> str:
    return WS.sub("", str(value or "")).casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(signature: str) -> int:
    return int.from_bytes(hashlib.sha256(canonical(signature).encode()).digest()[:8], "big")


def dump_json(path: Path, value: Any) -> None:
    ensure_output_boundary()
    path = Path(path).resolve()
    if OUTPUT != path.parent and OUTPUT not in path.parents:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: write outside output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    ensure_output_boundary()
    path = Path(path).resolve()
    if OUTPUT != path.parent and OUTPUT not in path.parents:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: write outside output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_gzip_json(path: Path, value: Any) -> None:
    ensure_output_boundary()
    path = Path(path).resolve()
    if OUTPUT != path.parent and OUTPUT not in path.parents:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: write outside output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def first_existing(paths: Iterable[Path], label: str) -> Path:
    checked = []
    for path in paths:
        checked.append(str(path))
        if path.is_file():
            return path.resolve()
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {label}; checked={checked!r}")


def discover_assets() -> dict[str, Path]:
    return {
        "source_member": first_existing([
            PROJECT / "artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv",
        ], "source member OOF predictions"),
        "endpoint": first_existing([
            Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
            PROJECT / "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
        ], "frozen endpoint cache"),
        "mapping": first_existing([
            PROJECT / "scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml",
            PROJECT / "configs/protocol/semantic_mapping_v1.yaml",
        ], "frozen Role31 mapping"),
    }


def svm_route_rows(path: Path) -> list[dict[str, Any]]:
    raw = read_csv(path)
    if len(raw) != 799:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: source rows changed: {len(raw)}")
    rows = []
    for row in raw:
        sid = str(row.get("sha256", "")).strip().casefold()
        if len(sid) != 64:
            raise RuntimeError("BLOCKED_BY_CONTRACT: invalid source SHA")
        probabilities = [float(row[f"p_P31__{name}"]) for name in ("LR", "RF", "XGB", "SVM")]
        svm = float(probabilities[3])
        pred = int(svm >= .5)
        opposition = sum((1.0 - value) if pred else value for value in probabilities) / len(probabilities)
        rows.append({"sha256": sid, "fold": int(float(row["fold"])), "stage1_prob": svm,
                     "stage1_pred": pred, "route_score": opposition, "routed": 0})
    for fold in range(1, 6):
        index = [position for position, row in enumerate(rows) if row["fold"] == fold]
        order = sorted(index, key=lambda position: (-rows[position]["route_score"], rows[position]["sha256"]))
        for position in order[:40]:
            rows[position]["routed"] = 1
    if sum(row["routed"] for row in rows) != 200:
        raise RuntimeError("BLOCKED_BY_CONTRACT: SVM route is not 200")
    return rows


def load_apk_map(wanted: set[str]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    candidates = []
    if os.environ.get("SGCA_APK_INDEX"):
        candidates.append(Path(os.environ["SGCA_APK_INDEX"]))
    candidates += [
        PROJECT / "outputs/care-ccbc-source-v3/apk_paths.json",
        PROJECT / "outputs/care-icde-source-v1/apk_asset_audit.json",
        PROJECT / "outputs/care-trcbe-source-v1/apk_paths.json",
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
    return result, audit


def load_endpoint_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
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
    edges = payload.get("edge_index") or [[], []]
    if len(ids) != len(roles) or len(ids) != len(base) or len(edges) != 2:
        raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint payload schema mismatch")
    endpoints = {i for i, row in enumerate(base) if len(row) > 4 and float(row[4]) > .5}
    by_host: dict[int, set[int]] = {}
    for source, target in zip(edges[0], edges[1]):
        if int(target) in endpoints:
            by_host.setdefault(int(source), set()).add(int(target))
    output = []
    for host in sorted(by_host):
        merged = sorted({int(role) for target in by_host[host] for role in (roles[target] or []) if 0 <= int(role) < 31})
        if merged:
            output.append({"node_id": int(ids[host]), "roles": merged})
    return output


def file_record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}
