#!/usr/bin/env python3
"""Read-only real-asset preflight; it does not create prompts or call an LLM."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prompt_v2_core import MODEL_ROLE_ORDER, PAYLOAD_TO_MODEL_ROLE, norm_sha

V1 = Path("artifacts/care-llm-semantic-correction-probe-v1/prompts_unlabeled.jsonl")
SHARDS = Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/shards")
MANIFEST = Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json")
CACHE = Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")


def load_v1():
    rows = [json.loads(line) for line in V1.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"case_id", "sha256", "stage1_pred", "candidate_pred"}
    if len(rows) != 109 or any(not required.issubset(row) for row in rows):
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen V1 candidate manifest contract failed")
    ids = [norm_sha(row["sha256"]) for row in rows]
    if "" in ids or len(set(ids)) != 109:
        raise RuntimeError("BLOCKED_BY_CONTRACT: V1 candidate SHA contract failed")
    for row in rows:
        if int(row["stage1_pred"]) not in (0, 1) or int(row["candidate_pred"]) not in (0, 1):
            raise RuntimeError("BLOCKED_BY_CONTRACT: nonbinary frozen prediction")
        if int(row["stage1_pred"]) == int(row["candidate_pred"]):
            raise RuntimeError("BLOCKED_BY_CONTRACT: candidate is not a prediction flip")
    return rows


def main():
    for path in (V1, SHARDS, MANIFEST, CACHE):
        if not path.exists():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {path}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("model_role_order") != MODEL_ROLE_ORDER:
        raise RuntimeError("BLOCKED_BY_CONTRACT: model_role_order differs from frozen contract")
    if [int(x) for x in manifest.get("payload_to_model_role", [])] != PAYLOAD_TO_MODEL_ROLE:
        raise RuntimeError("BLOCKED_BY_CONTRACT: payload/model role permutation differs")
    rows = load_v1()
    required_keys = {
        "sha256", "host_primary_role", "outgoing_offsets", "outgoing_host_positions",
        "role_node_counts", "role_edge_counts", "same_role_edge_count", "cross_role_edge_count",
    }
    for row in rows:
        sha = norm_sha(row["sha256"])
        path = SHARDS / f"{sha}.npz"
        if not path.is_file():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing source shard {sha}")
        with np.load(path, allow_pickle=False) as saved:
            if not required_keys.issubset(saved.files):
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: shard keys missing for {sha}")
            if tuple(saved["role_node_counts"].shape) != (31,):
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: role-node shape failed for {sha}")
            if tuple(saved["role_edge_counts"].shape) != (31, 31):
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: saved role-edge shape failed for {sha}")
    print("LLM_PROMPT_V2_PREFLIGHT_PASS")
    print("candidate_rows = 109")
    print("unique_sha = 109")
    print("source_year = 2017")
    print("future_rows_loaded = 0")
    print("ground_truth_labels_loaded = 0")
    print("llm_calls_performed = 0")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), flush=True)
        raise SystemExit(2)
