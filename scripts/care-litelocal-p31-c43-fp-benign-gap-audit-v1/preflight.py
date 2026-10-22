#!/usr/bin/env python3
from __future__ import annotations

import csv, gzip, json
from pathlib import Path

P31 = Path("artifacts/care-litelocal-p31-c43-source-v1/preflight/p31_reference/source_predictions.csv")
ENDPOINT = Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
MANIFESTS = (
    Path("artifacts/care-source-local-assets-v2/local_asset_manifest.json"),
    Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json"),
)
ROOT = Path("artifacts/care-litelocal-p31-c43-fp-benign-gap-audit-v1")


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    required = [P31, ENDPOINT]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size < 100]
    manifest = next((path for path in MANIFESTS if path.is_file() and path.stat().st_size > 100), None)
    if manifest is None: missing.append("one local_asset_manifest.json candidate")
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- " + "\n- ".join(missing))
    rows = read(P31)
    if len(rows) != 799 or len({row["sha256"].strip().casefold() for row in rows}) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source lineage changed")
    routed = [row for row in rows if int(float(row["P31_DI25__routed"])) == 1]
    labels = [int(float(row["label"])) for row in routed]
    predictions = [int(float(row["P31_MEAN_COMMITTEE__prob"]) >= .5) for row in routed]
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    if (len(routed), fp, tp, tn, fn) != (200, 20, 71, 83, 26):
        raise RuntimeError("BLOCKED_BY_CONTRACT: routed P31 cohorts changed")
    contract = json.loads(manifest.read_text(encoding="utf-8"))
    remap = contract.get("payload_to_model_role", [])
    names = contract.get("model_role_order", [])
    if len(remap) != 31 or sorted(map(int, remap)) != list(range(31)) or len(names) != 31:
        raise RuntimeError("BLOCKED_BY_CONTRACT: role taxonomy unavailable")
    ids = set()
    sample = None
    with gzip.open(ENDPOINT, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            row = json.loads(line); ids.add(str(row.get("sha256", "")).strip().casefold())
            if sample is None: sample = row.get("payload", {})
    if len(ids) != 799 or ids != {row["sha256"].strip().casefold() for row in rows}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint cache lineage changed")
    needed = {"node_ids", "node_role_ids", "node_base_features", "edge_index", "edge_features"}
    if sample is None or not needed.issubset(sample):
        raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint payload schema insufficient")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "resolved_inputs.json").write_text(json.dumps({
        "p31": str(P31.resolve()), "endpoint": str(ENDPOINT.resolve()),
        "manifest": str(manifest.resolve()), "role_names": names,
        "payload_to_model_role": list(map(int, remap)),
    }, indent=2), encoding="utf-8")
    print("P31_C43_FP_BENIGN_GAP_PREFLIGHT_PASS")
    print("source_rows=799 routed_rows=200 FP=20 TP=71 TN=83 FN=26")
    print("endpoint_cache_rows=799 future_rows_loaded=0 ego_consumed=false")
    print("training_invoked=false neural_inference_invoked=false apk_reanalysis_invoked=false")


if __name__ == "__main__": main()
