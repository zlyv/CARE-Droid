#!/usr/bin/env python3
from __future__ import annotations

import csv, gzip, hashlib, json, os, re
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(os.environ.get("TRCBE_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("TRCBE_OUTPUT", str(PROJECT / "outputs/care-trcbe-source-v1"))).resolve()
ROLE_NAMES = (
    "command_execution","dynamic_code_loading","native_library_loading","reflective_invocation",
    "package_installation","package_uninstallation","device_administration","accessibility_control",
    "sms_send_receive","call_control_or_recording","account_or_credential_access","contacts_or_calendar_access",
    "device_identifier_access","location_access","microphone_access","camera_access","clipboard_access",
    "network_transmission_or_upload","sensitive_file_or_content_write","boot_persistence",
    "task_or_alarm_scheduling","background_service_control","overlay_or_ui_deception",
    "icon_or_component_hiding","process_or_wakelock_control","cryptography","encoding_or_compression",
    "url_or_http_setup","file_read","database_access","other",
)
WS = re.compile(r"\s+")

def canonical(value: Any) -> str:
    return WS.sub("", str(value or "")).casefold()

def stable_id(signature: str) -> int:
    return int.from_bytes(hashlib.sha256(canonical(signature).encode("utf-8")).digest()[:8], "big", signed=False)

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    tmp.replace(path)

def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def first_existing(paths: Iterable[Path], what: str) -> Path:
    for path in paths:
        if path.is_file():return path.resolve()
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {what}; checked="+repr([str(p) for p in paths]))

def file_record(path: Path,purpose: str) -> dict[str,Any]:
    p=path.resolve();s=p.stat()
    return {"absolute_path":str(p),"purpose":purpose,"size_bytes":s.st_size,"mtime_ns":s.st_mtime_ns,"sha256":sha256_file(p)}

def read_endpoint_rows(path: Path, wanted: set[str]) -> dict[str,dict[str,Any]]:
    out={}
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line);sid=str(row.get("sha256","")).strip().casefold()
            if sid in wanted and row.get("status")=="OK" and isinstance(row.get("payload"),dict):out[sid]=row["payload"]
    return out

def discover_assets() -> dict[str,Path]:
    explicit=lambda key:[Path(os.environ[key])] if os.environ.get(key) else []
    return {
      "formal":first_existing(explicit("TRCBE_FORMAL")+[
        PROJECT/"artifacts/care-droid-formal-v1/formal_predictions.csv",
        PROJECT/"artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv"],"formal predictions"),
      "p31":first_existing(explicit("TRCBE_P31")+[PROJECT/"artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"],"P31"),
      "endpoint":first_existing(explicit("TRCBE_ENDPOINT_CACHE")+[
        PROJECT/"artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
        PROJECT/"artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz"],"endpoint cache"),
      "compact":first_existing(explicit("TRCBE_COMPACT_CACHE")+[
        PROJECT/"artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz"],"Compact43 cache"),
      "role_manifest":first_existing(explicit("TRCBE_ROLE_MANIFEST")+[
        PROJECT/"artifacts/care-source-local-assets-v2/local_asset_manifest.json",
        Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json")],"Role31 mapping manifest"),
      "attention":first_existing(explicit("TRCBE_ATTENTION")+[
        PROJECT/"outputs/care-ccbc-source-v3/cache/formal_attention.npz",
        PROJECT/"outputs/care-ccbc-source-v2/cache/formal_attention.npz",
        PROJECT/"outputs/care-ccbc-source-v1/cache/formal_attention.npz"],"frozen formal C43 attention"),
    }

def load_apk_map(wanted:set[str]) -> tuple[dict[str,Path],list[dict[str,Any]]]:
    candidates=[]
    if os.environ.get("TRCBE_APK_INDEX"):candidates.append(Path(os.environ["TRCBE_APK_INDEX"]))
    candidates += [PROJECT/"outputs/care-icde-source-v1/apk_asset_audit.json",PROJECT/"outputs/care-ccbc-source-v3/apk_paths.json",PROJECT/"outputs/care-ccbc-source-v2/apk_paths.json"]
    out={};audit=[]
    for path in candidates:
        if not path.is_file():continue
        try:
            data=json.loads(path.read_text(encoding="utf-8"));found=0
            if path.name=="apk_asset_audit.json":
                pairs=((r.get("sha256"),r.get("path")) for r in data.get("records",[]))
            elif isinstance(data,dict):pairs=data.items()
            else:pairs=[]
            for sid,raw in pairs:
                sid=str(sid or "").strip().casefold();p=Path(str(raw or ""))
                if sid in wanted and p.is_file():out.setdefault(sid,p.resolve());found+=1
            audit.append({"path":str(path.resolve()),"matched":found})
        except Exception as exc:audit.append({"path":str(path.resolve()),"error":repr(exc)})
    roots=[Path(v) for v in os.environ.get("TRCBE_APK_ROOTS","").split(os.pathsep) if v]
    for root in roots:
        found=0
        if root.is_dir():
            for p in root.rglob("*.apk"):
                sid=p.stem.casefold()
                if sid in wanted and sid not in out:out[sid]=p.resolve();found+=1
        audit.append({"path":str(root),"root_scan_matched":found})
    return out,audit

def endpoint_hosts(payload:dict[str,Any]) -> list[dict[str,Any]]:
    ids=list(payload.get("node_ids") or []);roles=list(payload.get("node_role_ids") or []);base=list(payload.get("node_base_features") or []);edges=payload.get("edge_index") or [[],[]]
    if not ids:return []
    if len(ids)!=len(base) or len(ids)!=len(roles) or len(edges)!=2:raise RuntimeError("endpoint payload schema mismatch")
    endpoints={i for i,row in enumerate(base) if len(row)>4 and float(row[4])>.5};by={}
    for s,t in zip(edges[0],edges[1]):
        s=int(s);t=int(t)
        if t in endpoints:by.setdefault(s,set()).add(t)
    out=[]
    for host in sorted(by):
        merged=sorted({int(r) for target in by[host] for r in (roles[target] or []) if 0<=int(r)<31})
        if merged:out.append({"node_index":host,"node_id":int(ids[host]),"roles":merged})
    return out

