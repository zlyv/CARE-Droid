#!/usr/bin/env python3
"""Build label-free future OriginalHost74/HostRel74 caches with role sidecars."""
from __future__ import annotations
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hostrel_features import build_hostrel74, reconstruct_host_relations

ROOT = Path("artifacts/care-di25-hostrel-semantic-background-diagnostics-v1")
REGIONS = (Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz"), Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/regions_2018_2021.npz"))
ENDPOINTS = (Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"), Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"))
MANIFESTS = (Path("artifacts/care-source-local-assets-v2/local_asset_manifest.json"), Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json"))
ARMS = ("B1_HARD_ROLE", "R1_HOSTREL74_HARD_ROLE")


def locate(candidates, name):
    for path in candidates:
        if path.is_file() and path.stat().st_size > 100:
            return path
    raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty " + name)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    existing=[ROOT/"caches"/(arm+"_regions_2018_2021.npz") for arm in ARMS]
    audit_path=ROOT/"future_cache_build_audit.json"
    if audit_path.is_file() and all(path.is_file() and path.stat().st_size>100 for path in existing):
        valid=True
        for path in existing:
            try:
                with np.load(path,allow_pickle=False) as saved:
                    valid = valid and saved["host_values"].shape==(263602,105) and saved["host_offsets"].shape==(1990,) and len(saved["sha256"])==1989
            except Exception:
                valid=False
        if valid:
            print("HOSTREL_SEMANTIC_FUTURE_CACHES_PASS reused=true")
            print("future_rows=1989 real_hosts=263600 labels_loaded=false ego_consumed=false")
            return
    region = locate(REGIONS, "future regions")
    endpoint = locate(ENDPOINTS, "future endpoint cache")
    manifest = locate(MANIFESTS, "role manifest")
    mapping = np.asarray(json.loads(manifest.read_text())["payload_to_model_role"], np.int64)
    with np.load(region, allow_pickle=False) as z:
        ids = [str(x).lower() for x in z["sha256"]]
        old = np.asarray(z["host_values"], np.float32)
        offsets = np.asarray(z["host_offsets"], np.int64)
        real = np.asarray(z["real_host_count"], np.int64)
        empty = np.asarray(z["host_was_empty"], np.int8)
    if len(ids) != 1989 or len(set(ids)) != 1989 or old.shape != (263602, 74) or int(real.sum()) != 263600:
        raise RuntimeError("BLOCKED_BY_CONTRACT: future region contract changed")
    payloads = {}
    with gzip.open(endpoint, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid = str(row.get("sha256") or row.get("sample_id") or "").lower()
            if sid in payloads:
                raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate future endpoint row " + sid)
            payloads[sid] = row["payload"]
    if set(payloads) != set(ids):
        raise RuntimeError("BLOCKED_BY_CONTRACT: future endpoint lineage changed")
    output = {arm: [] for arm in ARMS}
    any_neighbor = 0; nonzero_relation = 0
    for i, sid in enumerate(ids):
        records = reconstruct_host_relations(payloads[sid], mapping)
        begin, end, count = int(offsets[i]), int(offsets[i+1]), int(real[i])
        if len(records) != count:
            raise RuntimeError("BLOCKED_BY_CONTRACT: Host mismatch " + sid)
        if count:
            raw = old[begin:begin+count]
            roles = np.stack([r["endpoint_counts"] for r in records]).astype(np.float32)
            output["B1_HARD_ROLE"].append(np.concatenate([raw, roles], axis=1))
            output["R1_HOSTREL74_HARD_ROLE"].append(np.concatenate([build_hostrel74(raw, records), roles], axis=1))
            any_neighbor += sum((r["incoming_neighbor_count"] + r["outgoing_neighbor_count"]) > 0 for r in records)
            nonzero_relation += sum(np.any(r["directed_neighbor_role"] != 0) for r in records)
        else:
            if end-begin != 1 or int(empty[i]) != 1:
                raise RuntimeError("BLOCKED_BY_CONTRACT: placeholder changed " + sid)
            sidecar = np.zeros((1, 31), np.float32)
            output["B1_HARD_ROLE"].append(np.concatenate([old[begin:end], sidecar], axis=1))
            rel = np.concatenate([old[begin:end, :31], np.zeros((1,31),np.float32), old[begin:end,62:74]], axis=1)
            output["R1_HOSTREL74_HARD_ROLE"].append(np.concatenate([rel, sidecar], axis=1))
        if (i+1) % 200 == 0 or i+1 == len(ids):
            print(f"future HostRel semantic reconstruction {i+1}/{len(ids)}", flush=True)
    caches = {}
    for arm in ARMS:
        values = np.concatenate(output[arm]).astype(np.float32)
        if values.shape != (263602, 105):
            raise RuntimeError("BLOCKED_BY_CONTRACT: cache shape " + arm + " " + str(values.shape))
        path = ROOT/"caches"/(arm + "_regions_2018_2021.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, sha256=np.asarray(ids), host_values=values, host_offsets=offsets, real_host_count=real, host_was_empty=empty)
        caches[arm] = {"path": str(path), "sha256": digest(path), "shape": list(values.shape)}
    audit = {"future_rows":1989,"real_hosts":263600,"placeholder_rows":2,"stored_width":105,"model_host_width":74,"role_sidecar_width":31,"hosts_with_any_host_neighbor":int(any_neighbor),"hosts_with_nonzero_directed_relation":int(nonzero_relation),"region_source":str(region),"endpoint_source":str(endpoint),"labels_loaded":False,"ego_consumed":False,"caches":caches}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/"future_cache_build_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print("HOSTREL_SEMANTIC_FUTURE_CACHES_PASS")
    print(f"future_rows=1989 real_hosts=263600 relation_nontrivial_hosts={any_neighbor} relation_nonzero_hosts={nonzero_relation}")
    print("labels_loaded=false ego_consumed=false")


if __name__ == "__main__": main()
