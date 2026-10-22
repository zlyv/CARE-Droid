#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT = Path(os.environ.get("CARE_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("CARE_LINEAGE_OUTPUT", str(PROJECT / "outputs/care-p31-lineage-host-audit-v1"))).resolve()
SEED = 20261008
BIT_COUNT = 262144
BIT_BYTES = BIT_COUNT // 8

ROLE_NAMES = (
    "command_execution", "dynamic_code_loading", "native_library_loading", "reflective_invocation",
    "package_installation", "package_uninstallation", "device_administration", "accessibility_control",
    "sms_send_receive", "call_control_or_recording", "account_or_credential_access",
    "contacts_or_calendar_access", "device_identifier_access", "location_access", "microphone_access",
    "camera_access", "clipboard_access", "network_transmission_or_upload",
    "sensitive_file_or_content_write", "boot_persistence", "task_or_alarm_scheduling",
    "background_service_control", "overlay_or_ui_deception", "icon_or_component_hiding",
    "process_or_wakelock_control", "cryptography", "encoding_or_compression", "url_or_http_setup",
    "file_read", "database_access", "other",
)


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
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


def first_existing_dir(paths: Iterable[Path], label: str) -> Path:
    checked = []
    for path in paths:
        checked.append(str(path))
        if path.is_dir():
            return path.resolve()
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {label}; checked={checked!r}")


def discover_assets() -> dict[str, Path]:
    explicit = lambda name: [Path(os.environ[name])] if os.environ.get(name) else []
    return {
        "formal": first_existing(explicit("CARE_FORMAL") + [
            PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv",
            PROJECT / "artifacts/care-droid-formal-v1/formal_predictions.csv",
        ], "formal predictions"),
        "p31": first_existing(explicit("CARE_P31") + [
            PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz",
        ], "P31 cache"),
        "group_assignments": first_existing(explicit("CARE_GROUP_ASSIGNMENTS") + [
            PROJECT / "outputs/care-group-calibration-compact43-validation-v1/p31_group_assignments.csv",
        ], "strict P31 group assignments"),
        "group_structure": first_existing(explicit("CARE_GROUP_STRUCTURE") + [
            PROJECT / "outputs/care-group-calibration-compact43-validation-v1/p31_group_structure_audit.csv",
        ], "strict P31 group structure"),
        "group_predictions": first_existing(explicit("CARE_GROUP_PREDICTIONS") + [
            PROJECT / "outputs/care-group-calibration-compact43-validation-v1/group_stage1_predictions.csv",
        ], "group-safe Stage-I predictions"),
        "endpoint": first_existing(explicit("CARE_ENDPOINT") + [
            PROJECT / "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
            PROJECT / "artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz",
        ], "endpoint cache"),
        "attention": first_existing(explicit("CARE_ATTENTION") + [
            PROJECT / "outputs/care-ccbc-source-v3/cache/formal_attention.npz",
            PROJECT / "outputs/care-ccbc-source-v2/cache/formal_attention.npz",
        ], "frozen C43 attention"),
        "full_graph_dir": first_existing_dir(explicit("CARE_FULL_GRAPH_DIR") + [
            PROJECT / "outputs/care-fullgraph-trcbe-v2/full_graph_cache",
        ], "routed full graph cache"),
    }


def formal_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(path):
        sid = norm(raw.get("sha256"))
        if len(sid) != 64:
            raise RuntimeError("formal SHA invalid")
        stage1_key = "Global24__prob" if "Global24__prob" in raw else "stage1_probability"
        route_key = "A25__routed" if "A25__routed" in raw else "routed"
        score_key = "A25__score" if "A25__score" in raw else "router_score"
        method = "P31RF__RFRoute__A0_ROLE_ONLY__Bounded"
        final_key = method + "__prob"
        delta_key = method + "__bounded_delta"
        rows.append({
            "sha256": sid,
            "fold": int(float(raw["fold"])),
            "label": int(float(raw.get("label", raw.get("y", 0)))),
            "stage1_probability": float(raw[stage1_key]),
            "routed": int(float(raw.get(route_key, 0))),
            "router_score": float(raw.get(score_key, 0.0)),
            "bounded_delta": float(raw.get(delta_key, raw.get("bounded_delta", 0.0))),
            "final_probability": float(raw.get(final_key, raw.get("final_prob", raw.get("final_probability", raw[stage1_key])))),
        })
    if len(rows) != 799 or len({row["sha256"] for row in rows}) != 799:
        raise RuntimeError(f"formal lineage mismatch rows={len(rows)} unique={len({r['sha256'] for r in rows})}")
    return rows


def load_apk_map(wanted: set[str]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    candidates = [Path(os.environ["CARE_APK_INDEX"])] if os.environ.get("CARE_APK_INDEX") else []
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
                sid = norm(sid)
                apk = Path(str(raw_path or ""))
                if sid in wanted and apk.is_file():
                    result.setdefault(sid, apk.resolve())
                    matched += 1
            audit.append({"path": str(path.resolve()), "matched": matched})
        except Exception as exc:
            audit.append({"path": str(path.resolve()), "error": repr(exc)})
    roots = [Path(value) for value in os.environ.get("CARE_APK_ROOTS", "").split(os.pathsep) if value]
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
            sid = norm(row.get("sha256"))
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
    endpoints = {index for index, row in enumerate(base) if len(row) > 4 and float(row[4]) > .5}
    by_host: dict[int, set[int]] = {}
    for source, target in zip(edge_index[0], edge_index[1]):
        source, target = int(source), int(target)
        if target in endpoints:
            by_host.setdefault(source, set()).add(target)
    result = []
    for host_index in sorted(by_host):
        merged = sorted({int(role) for target in by_host[host_index] for role in (roles[target] or []) if 0 <= int(role) < 31})
        if merged:
            result.append({"old_node_index": host_index, "node_id": int(ids[host_index]), "roles": merged})
    return result


def load_attention(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        shas = [norm(value) for value in archive["sha256"]]
        offsets = np.asarray(archive["offsets"], int)
        attention = np.asarray(archive["attention"], float)
        critical = np.asarray(archive["critical_index"], int)
        if len(offsets) != len(shas) + 1:
            raise RuntimeError("attention offsets mismatch")
        for index, sid in enumerate(shas):
            result[sid] = {"attention": attention[offsets[index]: offsets[index + 1]], "critical": int(critical[index])}
    return result


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def file_record(path: Path, purpose: str) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    return {"absolute_path": str(path), "purpose": purpose, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def token_hash(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8", errors="replace"), digest_size=8).digest(), "big")


def sketch(tokens: Iterable[str]) -> dict[str, Any]:
    values = sorted({token_hash(token) for token in tokens})
    bits = np.zeros(BIT_COUNT, dtype=np.uint8)
    if values:
        bits[np.asarray(values, dtype=np.uint64) % BIT_COUNT] = 1
    packed = np.packbits(bits, bitorder="little")
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "big"))
    return {"cardinality": len(values), "set_sha256": digest.hexdigest(),
            "bit_count": int(bits.sum()), "bits_b64": base64.b64encode(packed.tobytes()).decode("ascii")}


POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


def sketch_jaccard(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = np.frombuffer(base64.b64decode(left["bits_b64"]), dtype=np.uint8)
    b = np.frombuffer(base64.b64decode(right["bits_b64"]), dtype=np.uint8)
    if len(a) != BIT_BYTES or len(b) != BIT_BYTES:
        raise RuntimeError("invalid lineage sketch width")
    intersection = int(POPCOUNT[np.bitwise_and(a, b)].sum())
    union = int(POPCOUNT[np.bitwise_or(a, b)].sum())
    return float(intersection / union) if union else 1.0
