#!/usr/bin/env python3
"""Build source-only HostRel74 plus a non-neural 31-role selection sidecar."""
from __future__ import annotations
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hostrel_features import build_hostrel74, reconstruct_host_relations

REGION = Path("artifacts/care-source-local-assets-v2/regions_2017.npz")
ENDPOINT = Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
OUT = Path("artifacts/care-di25-hostrel-semantic-background-source-v1/cache/regions_2017_hostrel_role_sidecar.npz")
MANIFESTS = (
    Path("artifacts/care-source-local-assets-v2/local_asset_manifest.json"),
    Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json"),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = next((p for p in MANIFESTS if p.is_file()), None)
    if manifest is None:
        raise RuntimeError("BLOCKED_BY_CONTRACT: role manifest missing")
    mapping = np.asarray(json.loads(manifest.read_text())["payload_to_model_role"], np.int64)
    with np.load(REGION, allow_pickle=False) as z:
        ids = [str(x).lower() for x in z["sha256"]]
        old = np.asarray(z["host_values"], np.float32)
        offsets = np.asarray(z["host_offsets"], np.int64)
        real = np.asarray(z["real_host_count"], np.int64)
        empty = np.asarray(z["host_was_empty"], np.int8)
    payloads = {}
    with gzip.open(ENDPOINT, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            payloads[str(row.get("sha256") or row.get("sample_id")).lower()] = row["payload"]
    if len(ids) != 799 or old.shape != (95316, 74) or set(ids) != set(payloads):
        raise RuntimeError("BLOCKED_BY_CONTRACT: source cache lineage changed")

    rows = []
    nontrivial_hosts = 0
    directed_nonzero = 0
    for i, sample_id in enumerate(ids):
        records = reconstruct_host_relations(payloads[sample_id], mapping)
        begin, end, count = int(offsets[i]), int(offsets[i + 1]), int(real[i])
        if len(records) != count:
            raise RuntimeError("BLOCKED_BY_CONTRACT: Host order/count mismatch " + sample_id)
        if count:
            hostrel = build_hostrel74(old[begin:begin + count], records)
            role_sidecar = np.stack([r["endpoint_counts"] for r in records]).astype(np.float32)
            rows.append(np.concatenate([hostrel, role_sidecar], axis=1))
            nontrivial_hosts += sum((r["incoming_neighbor_count"] + r["outgoing_neighbor_count"]) > 0 for r in records)
            directed_nonzero += sum(np.any(r["directed_neighbor_role"] != 0) for r in records)
        else:
            if end - begin != 1:
                raise RuntimeError("BLOCKED_BY_CONTRACT: placeholder contract changed")
            placeholder = np.concatenate([old[begin:end, :31], np.zeros((1, 31), np.float32), old[begin:end, 62:74]], axis=1)
            rows.append(np.concatenate([placeholder, np.zeros((1, 31), np.float32)], axis=1))
    values = np.concatenate(rows)
    if values.shape != (95316, 105):
        raise RuntimeError("BLOCKED_BY_CONTRACT: cache shape " + str(values.shape))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, sha256=np.asarray(ids), host_values=values, host_offsets=offsets, real_host_count=real, host_was_empty=empty)
    audit = {
        "source_rows": 799,
        "real_hosts": 95314,
        "placeholder_rows": 2,
        "stored_width": 105,
        "model_host_width": 74,
        "self_role_width": 31,
        "directed_neighbor_role_width": 31,
        "preserved_tail_width": 12,
        "semantic_selection_sidecar_width": 31,
        "hosts_with_any_host_neighbor": int(nontrivial_hosts),
        "hosts_with_nonzero_directed_relation": int(directed_nonzero),
        "relation_definition": "tanh(log1p(outgoing normalized-neighbor role mass)-log1p(incoming normalized-neighbor role mass))",
        "path_contraction": False,
        "labels_loaded": False,
        "future_rows_loaded": 0,
        "ego_tensor_consumed": False,
        "ego_encoder_instantiated": False,
        "region_sha256": sha(REGION),
        "endpoint_sha256": sha(ENDPOINT),
        "output_sha256": sha(OUT),
    }
    Path("artifacts/care-di25-hostrel-semantic-background-source-v1/cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("HOSTREL_SEMANTIC_BACKGROUND_CACHE_PASS")
    print(f"source_rows=799 real_hosts=95314 relation_nontrivial_hosts={nontrivial_hosts} relation_nonzero_hosts={directed_nonzero}")
    print("model_host_width=74 future_rows_loaded=0 ego_consumed=false")


if __name__ == "__main__":
    main()
