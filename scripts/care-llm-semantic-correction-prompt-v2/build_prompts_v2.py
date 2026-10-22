#!/usr/bin/env python3
"""Build all 109 label-free Prompt V2 records and a deterministic five-case pilot."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from preflight import CACHE, MANIFEST, SHARDS, V1, load_v1
from prompt_v2_core import (
    MODEL_ROLE_ORDER, SYSTEM_PROMPT, norm_sha, prompt_hash,
    reconstruct_primary_graph, render_user_prompt, summarize_evidence,
)

OUTPUT = Path("artifacts/care-llm-semantic-correction-prompt-v2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def load_endpoint_records(needed):
    result = {}
    with gzip.open(CACHE, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sha = norm_sha(row.get("sha256"))
            if sha in needed:
                result[sha] = row
    if set(result) != set(needed):
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: endpoint cache candidate mismatch "
            f"missing={len(set(needed) - set(result))}"
        )
    return result


def select_pilot(rows):
    selected = []

    def add(row, reason):
        if row is not None and row["sha256"] not in {item["sha256"] for item in selected}:
            selected.append({**row, "pilot_selection_reason": reason})

    for stage1, candidate, label in ((0, 1, "benign_to_malicious"), (1, 0, "malicious_to_benign")):
        group = sorted(
            [row for row in rows if row["stage1_pred"] == stage1 and row["candidate_pred"] == candidate],
            key=lambda row: (row["semantic_cross_relation_entropy"], row["sha256"]),
        )
        if group:
            add(group[-1], f"{label}_maximum_cross_relation_entropy")
            add(group[(len(group) - 1) // 2], f"{label}_median_cross_relation_entropy")

    sparse = sorted(
        rows,
        key=lambda row: (
            int(row["semantic_cross_relation_available"]),
            row["audit_semantic_cross_edge_total"],
            row["semantic_cross_relation_entropy"],
            row["sha256"],
        ),
    )
    add(sparse[0], "minimum_or_unavailable_semantic_cross_relation")

    for row in sorted(rows, key=lambda value: value["sha256"]):
        if len(selected) == 5:
            break
        add(row, "deterministic_sha_fill")
    if len(selected) != 5:
        raise RuntimeError("BLOCKED_BY_CONTRACT: pilot selection did not yield five unique cases")
    return selected


def main():
    candidates = load_v1()
    ids = [norm_sha(row["sha256"]) for row in candidates]
    endpoint = load_endpoint_records(set(ids))
    all_prompts = []
    audit_rows = []
    pilot_basis = []

    for row in candidates:
        sha = norm_sha(row["sha256"])
        shard_path = SHARDS / f"{sha}.npz"
        with np.load(shard_path, allow_pickle=False) as saved:
            shard_sha = norm_sha(saved["sha256"].item())
            if shard_sha != sha:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: shard identity mismatch {sha}")
            matrix, host_edges = reconstruct_primary_graph(
                saved["host_primary_role"], saved["outgoing_offsets"],
                saved["outgoing_host_positions"],
            )
            saved_same = int(saved["same_role_edge_count"])
            saved_cross = int(saved["cross_role_edge_count"])
            saved_role_matrix_total = int(np.asarray(saved["role_edge_counts"], dtype=np.int64).sum())
            if int(np.trace(matrix)) != saved_same or int(matrix.sum() - np.trace(matrix)) != saved_cross:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host relation reconstruction mismatch {sha}")
            role_nodes = np.asarray(saved["role_node_counts"], dtype=np.int64)

        evidence = summarize_evidence(role_nodes, matrix, endpoint[sha].get("summary", {}))
        stage1 = int(row["stage1_pred"]); candidate = int(row["candidate_pred"])
        user_prompt = render_user_prompt(stage1, candidate, evidence)
        record = {
            "case_id": str(row["case_id"]), "sha256": sha,
            "stage1_pred": stage1, "candidate_pred": candidate,
            "system_prompt": SYSTEM_PROMPT, "user_prompt": user_prompt,
            "prompt_sha256": prompt_hash(SYSTEM_PROMPT, user_prompt),
        }
        all_prompts.append(record)
        audit = {
            "case_id": record["case_id"], "sha256": sha,
            "stage1_pred": stage1, "candidate_pred": candidate,
            "direct_relation_available": evidence["direct_relation_available"],
            "semantic_cross_relation_available": evidence["semantic_cross_relation_available"],
            "semantic_cross_relation_entropy": evidence["semantic_cross_relation_entropy"],
            "entry_available": evidence["entry_available"],
            "provenance_available": evidence["provenance_available"],
            "audit_host_edge_total": len(host_edges),
            "audit_role_mass_total": evidence["audit_role_mass_total"],
            "audit_semantic_cross_edge_total": evidence["audit_semantic_cross_edge_total"],
            "audit_relation_type_n": evidence["audit_relation_type_n"],
            "saved_role_edge_matrix_total": saved_role_matrix_total,
            "ground_truth_label_loaded": 0,
            "fixed_broken_loaded": 0,
        }
        audit_rows.append(audit)
        pilot_basis.append({**record, **audit})

    if len(all_prompts) != 109 or len({row["sha256"] for row in all_prompts}) != 109:
        raise RuntimeError("BLOCKED_BY_CONTRACT: Prompt V2 output cohort changed")
    pilot = select_pilot(pilot_basis)
    pilot_records = [{
        key: row[key] for key in (
            "case_id", "sha256", "stage1_pred", "candidate_pred",
            "system_prompt", "user_prompt", "prompt_sha256",
        )
    } for row in pilot]

    frozen_manifest = [{
        "case_id": str(row["case_id"]), "sha256": norm_sha(row["sha256"]),
        "stage1_pred": int(row["stage1_pred"]), "candidate_pred": int(row["candidate_pred"]),
    } for row in candidates]
    write_jsonl(OUTPUT / "candidate_manifest_frozen.jsonl", frozen_manifest)
    write_jsonl(OUTPUT / "prompts_v2_all_unlabeled.jsonl", all_prompts)
    write_jsonl(OUTPUT / "prompts_v2_pilot5_unlabeled.jsonl", pilot_records)
    write_csv(OUTPUT / "semantic_relation_audit.csv", audit_rows)
    write_csv(OUTPUT / "pilot_selection_audit.csv", [{
        "case_id": row["case_id"], "sha256": row["sha256"],
        "stage1_pred": row["stage1_pred"], "candidate_pred": row["candidate_pred"],
        "selection_reason": row["pilot_selection_reason"],
        "semantic_cross_relation_available": row["semantic_cross_relation_available"],
        "semantic_cross_relation_entropy": row["semantic_cross_relation_entropy"],
        "audit_semantic_cross_edge_total": row["audit_semantic_cross_edge_total"],
    } for row in pilot])
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    write_json(OUTPUT / "role_mapping_contract.json", {
        "model_role_order": MODEL_ROLE_ORDER,
        "payload_to_model_role": manifest["payload_to_model_role"],
        "shard_role_coordinate": "model_role_order",
        "mapping_applied_again": False,
    })
    contract = {
        "status": "PROMPT_V2_BUILD_ONLY_NO_LLM_CALL",
        "candidate_rows": 109, "candidate_unique_sha": 109,
        "source_year": 2017, "future_rows_loaded": 0,
        "ground_truth_labels_loaded": 0, "fixed_broken_loaded": 0,
        "llm_calls_performed": 0, "pilot_prompt_rows": 5,
        "relation_definition": "reconstructed direct Host-primary-role graph",
        "path_contraction": False, "ordinary_method_hops": 0,
        "raw_size_counts_rendered": False,
        "v1_candidate_manifest_sha256": sha256_file(V1),
        "source_manifest_sha256": sha256_file(MANIFEST),
    }
    write_json(OUTPUT / "prompt_v2_contract.json", contract)
    report = [
        "CARE-Droid LLM semantic-correction Prompt V2 build", "",
        "Build-only; no LLM/API call was performed.",
        "Frozen source-2017 candidate flips: 109 unique SHA.",
        "Pilot prompts: 5, selected without labels or FIXED/BROKEN outcomes.",
        "Relations: reconstructed direct Host-primary-role graph; no path contraction.",
        "All role, entry-reachability and provenance values rendered to the LLM are within-APK ratios.",
        "The original all-zero saved role_edge_counts are not used or overwritten.",
        "Future rows loaded: 0.", "Ground-truth labels loaded: 0.",
        "LLM calls performed: 0.",
    ]
    atomic_text(OUTPUT / "REPORT.txt", "\n".join(report) + "\n")
    print("LLM_PROMPT_V2_BUILD_DONE")
    print("candidate_rows = 109")
    print("pilot_prompts = 5")
    print("future_rows_loaded = 0")
    print("ground_truth_labels_loaded = 0")
    print("llm_calls_performed = 0")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), flush=True)
        raise SystemExit(2)
