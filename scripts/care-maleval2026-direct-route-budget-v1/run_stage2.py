#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from common import (FORMAL_DIAG, MAX_DELTA, OUTPUT, PROJECT, SCRIPTS, SEED, SOURCE_P31,
                    dump_json, norm, read_csv, read_json, write_csv)


def args_object(resolved):
    return SimpleNamespace(
        features=Path(resolved["sealed_features"]), cache=Path(resolved["endpoint_cache"]),
        future_inter_cache=Path(resolved["future_inter_cache"]),
        region_cache=FORMAL_DIAG / "cache/a0_role_2017_2021.npz",
        rebuild_region_cache=False, rebuild_future_inter_cache=False,
        schema=Path("artifacts/p1-features/feature_schema.json"),
        p72_oof=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),
        source_multiscale_features=Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"),
        inner_folds=4, split_seed=20261010, max_epochs=30, early_stop_fraction=.18,
        early_stop_patience=5, early_stop_min_delta=1e-4, batch_size=16, hidden_dim=48,
        embedding_dim=24, graph_dim=32, graph_heads=4, graph_layers=2, dropout=.20,
        learning_rate=8e-4, weight_decay=5e-4, grad_clip=5., max_delta=MAX_DELTA,
        keep_weight=2., fix_weight=1., keep_consistency=.5, shared_prototypes=32,
        benign_prototypes=24, prototype_regions_per_apk=16, internal_background_coverage=.70,
        null_samples=64, swaps_per_edge=10, null_seed=20260827, z_clip=5.,
        progress_every=20, device="cuda", jobs=4,
    )


def load_endpoint_rows(path):
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "OK": result[norm(row.get("sha256"))] = row
    return result


def payload_mapping():
    candidates = [
        PROJECT / "artifacts/care-source-local-assets-v2/local_asset_manifest.json",
        Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json"),
    ]
    for path in candidates:
        if path.is_file():
            value = read_json(path).get("payload_to_model_role")
            if value is not None: return np.asarray(value, np.int64)
    raise RuntimeError("BLOCKED_BY_CONTRACT: payload_to_model_role mapping missing")


def validate_host_alignment(base, payload_records, model_records, sid):
    """Validate Host identity/order in payload space, then return model-space rows.

    The legacy 74-wide region builder deliberately keeps Role columns in the
    endpoint payload's native order.  Relation31, by contrast, is defined in
    the frozen model Role order.  Comparing the former directly with mapped
    Relation31 presence is therefore invalid whenever the frozen permutation
    is non-identity.  Host identity/order is checked against an identity-map
    reconstruction; the mapped reconstruction is used only after that check.
    """
    if len(payload_records) != len(model_records):
        raise RuntimeError(
            f"BLOCKED_BY_CONTRACT: Host reconstruction count mismatch {sid} "
            f"payload={len(payload_records)} model={len(model_records)}"
        )
    payload_nodes = [int(record["host_node"]) for record in payload_records]
    model_nodes = [int(record["host_node"]) for record in model_records]
    if payload_nodes != model_nodes:
        raise RuntimeError("BLOCKED_BY_CONTRACT: Host identity/order mismatch " + sid)
    payload_presence = np.stack(
        [record["presence"] for record in payload_records]
    ).astype(np.float32)
    if not np.allclose(base[:, :31], payload_presence, atol=1e-6, rtol=0):
        raise RuntimeError("BLOCKED_BY_CONTRACT: payload-space Host/Role mismatch " + sid)
    return model_records


def build_external_bags(v4, args, rows, endpoint_cache, endpoint_rows):
    raw_cache = OUTPUT / "cache/external_raw_regions.npz"
    args.region_cache = raw_cache; args.rebuild_region_cache = False
    raw_bags, _, audits, _ = v4.care_v1.get_region_bags(rows, endpoint_cache, args)
    relation_dir = SCRIPTS / "care-di25-compact43-relationquery-source-v1"
    if str(relation_dir) not in sys.path: sys.path.insert(0, str(relation_dir))
    from hostrel_features import reconstruct_host_relations
    mapping = payload_mapping(); output = []; audit_rows = []
    for row, raw, audit in zip(rows, raw_bags, audits):
        sid = norm(row["sha256"]); raw = np.asarray(raw, np.float32)
        payload = endpoint_rows[sid]["payload"]
        payload_records = reconstruct_host_relations(payload, np.arange(31, dtype=np.int64))
        records = reconstruct_host_relations(payload, mapping)
        real_value = audit.get("real_host_region_n", len(records))
        try: real_count = int(real_value)
        except Exception: real_count = len(records)
        if len(records) != real_count:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host relation mismatch {sid} records={len(records)} real={real_count}")
        if real_count:
            base = raw[:real_count]
            if base.shape[1] != 74: raise RuntimeError("BLOCKED_BY_CONTRACT: raw Host width changed")
            records = validate_host_alignment(base, payload_records, records, sid)
            presence = np.stack([record["presence"] for record in records]).astype(np.float32)
            compact = np.concatenate([presence, base[:, 62:74]], axis=1)
            role_sidecar = np.stack([record["endpoint_counts"] for record in records]).astype(np.float32)
            relation = np.stack([record["directed_neighbor_role"] for record in records]).astype(np.float32)
            bag105 = np.concatenate([compact, role_sidecar, relation], axis=1)
        else:
            if raw.shape[0] != 1 or raw.shape[1] != 74:
                raise RuntimeError("BLOCKED_BY_CONTRACT: empty Host placeholder changed " + sid)
            compact = np.concatenate([raw[:, :31], raw[:, 62:74]], axis=1)
            bag105 = np.concatenate([compact, np.zeros((1, 62), np.float32)], axis=1)
        output.append(np.concatenate([bag105, np.zeros((len(bag105), 64), np.float32)], axis=1))
        audit_rows.append({"sha256": sid, "real_hosts": real_count, "stored_hosts": len(bag105),
                           "host_alignment": "PAYLOAD_IDENTITY_VERIFIED_MODEL_ROLE_MAPPED"})
    write_csv(OUTPUT / "external_host_bag_audit.csv", audit_rows)
    return output


def main():
    # Reuse the exact frozen implementation; local modules are added without modifying them.
    package_dir = SCRIPTS / "care-p31-classifier-family-stage2-v1"
    train_dir = SCRIPTS / "care-svm-train-window-expansion-v1"
    for path in (package_dir, train_dir):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    from model_adapter import install_a0, install_fold_safe_compact76, load_bags, prepare_imports
    from stage1_core import crossfit_svm_probability
    prepare_imports(); import run_care_rolegraph_conditioned_dualstream_v4 as v4
    direct_dir = SCRIPTS / "care-svm-c43-remove-bounded-correction-v1"
    if str(direct_dir) not in sys.path: sys.path.insert(0, str(direct_dir))
    from model_variants import install_variant

    resolved = read_json(FORMAL_DIAG / "resolved_inputs.json"); args = args_object(resolved)
    install_fold_safe_compact76(v4.audit)
    manifest, source_rows, clean_rows, compact_by_sha, compact_schema, original_names = v4.audit.load_inputs(args)
    source_y = np.asarray([row["label"] for row in manifest], int)
    future_2018 = [row for row in clean_rows if str(row.get("metadata_year", "")).strip() == "2018"]
    labels18 = {norm(row["sha256"]): int(float(row["label"])) for row in read_csv(FORMAL_DIAG / "within_2018_2019/predictions.csv") if int(float(row["year"])) == 2018}
    if len(future_2018) != 793 or any(norm(row["sha256"]) not in labels18 for row in future_2018):
        raise RuntimeError("BLOCKED_BY_CONTRACT: 2018 training alignment changed")
    train_rows = list(source_rows) + list(future_2018)
    train_y = np.concatenate([source_y, np.asarray([labels18[norm(row["sha256"])] for row in future_2018], int)])

    stage1 = read_csv(OUTPUT / "stage1_predictions_label_free.csv")
    stage1_by_sha = {norm(row["sha256"]): row for row in stage1}
    compact_rows = [row for row in read_csv(OUTPUT / "cache/compact/features.csv") if row.get("status") == "OK"]
    endpoint_path = OUTPUT / "cache/endpoint/endpoint_centered_tensor_cache.jsonl.gz"
    endpoint_rows = load_endpoint_rows(endpoint_path)
    target_rows = [row for row in compact_rows if norm(row["sha256"]) in endpoint_rows and int(stage1_by_sha[norm(row["sha256"])]["routed"])]
    target_rows.sort(key=lambda row: norm(row["sha256"]))
    if len(target_rows) < 96:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: complete Stage-II routed assets={len(target_rows)}")
    for row in compact_rows: compact_by_sha[norm(row["sha256"])] = row

    x_train, x_target, names = v4.audit.compact_matrix_variant(v4.VARIANT, train_rows, target_rows,
        compact_by_sha, compact_schema, original_names)
    global_index, _ = v4.evidence.split_view_indices(names)
    global_train = np.asarray(x_train[:, global_index], np.float32)
    global_target = np.asarray(x_target[:, global_index], np.float32)

    formal_endpoint = v4.audit.exp.read_cache(args.cache)
    source_s7, _ = v4.v3.load_source_s7(source_rows, formal_endpoint, args)
    with np.load(FORMAL_DIAG / "cache/future_s7.npz", allow_pickle=False) as archive:
        s7_ids = [norm(x) for x in archive["sha256"]]; s7_values = np.asarray(archive["s7"], np.float32)
    s7_map = {sid: s7_values[i] for i, sid in enumerate(s7_ids)}
    train_s7 = np.concatenate([source_s7, np.asarray([s7_map[norm(row["sha256"])] for row in future_2018], np.float32)], axis=0)
    external_endpoint = v4.audit.exp.read_cache(endpoint_path)
    s3, _ = v4.audit.future.reconstruct_structure(target_rows, external_endpoint, int(v4.audit.exp.ROLE_VOCABULARY_SIZE))
    args.future_inter_cache = OUTPUT / "cache/external_interz.npz"; args.rebuild_future_inter_cache = False
    inter, _, _ = v4.v3.future_s7.load_or_build_future_inter(target_rows, s3, external_endpoint, args)
    target_s7 = np.concatenate([s3, inter], axis=1).astype(np.float32)
    if target_s7.shape[1] != 1546: raise RuntimeError("BLOCKED_BY_CONTRACT: external S7 width changed")

    train_bags = load_bags(FORMAL_DIAG / "cache/a0_role_2017_2021.npz", train_rows)
    target_bags = build_external_bags(v4, args, target_rows, external_endpoint, endpoint_rows)

    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        source_p31_ids = [norm(x) for x in archive["sha256"]]; source_p31_values = np.asarray(archive["role_prevalence"], float)
    with np.load(resolved["future_p31_cache"], allow_pickle=False) as archive:
        future_p31_ids = [norm(x) for x in archive["sha256"]]; future_p31_values = np.asarray(archive["role_prevalence"], float)
    p31_map = {sid: source_p31_values[i] for i, sid in enumerate(source_p31_ids)}
    p31_map.update({sid: future_p31_values[i] for i, sid in enumerate(future_p31_ids)})
    train_p31 = np.asarray([p31_map[norm(row["sha256"])] for row in train_rows], float)
    base_train = crossfit_svm_probability(train_p31, train_y, SEED + 1701, jobs=4)
    target_base = np.asarray([float(stage1_by_sha[norm(row["sha256"])]["stage1_prob"]) for row in target_rows], float)

    install_a0(v4, "P31SVM__SVMRoute__A0_C43__W1718__MalEval2026__DirectStage2")
    install_variant(v4, "U2_DIRECT_STAGE2")
    args.keep_weight = 1.0; args.fix_weight = 1.0; args.keep_consistency = 0.0
    device = v4.resolve_device(args.device); model_seeds = [SEED * 100 + offset for offset in (0, 1, 2)]
    expert, diagnostics, audits = v4.fit_predict_variant(global_train, global_target, train_s7, target_s7,
        train_bags, target_bags, base_train, target_base, train_y, "RGCondAttn", model_seeds, args, device)
    if not np.isfinite(expert).all(): raise RuntimeError("BLOCKED_BY_CONTRACT: non-finite external expert")
    expert_map = {norm(row["sha256"]): float(expert[i]) for i, row in enumerate(target_rows)}
    output_rows = []
    for row in stage1:
        sid = norm(row["sha256"]); base = float(row["stage1_prob"]); available = sid in expert_map
        final = expert_map[sid] if available and int(row["routed"]) else base
        output_rows.append({**row, "stage2_asset_available": int(available),
                            "expert_prob": expert_map.get(sid, base), "stage2_prob": final})
    write_csv(OUTPUT / "stage2_predictions_label_free.csv", output_rows)
    dump_json(OUTPUT / "stage2_fit_audit.json", {"status": "PASS", "train_rows": len(train_rows),
        "train_years": [2017, 2018], "target_rows": len(stage1), "routed": 128,
        "stage2_asset_complete": len(target_rows), "fallback_to_stage1": 128 - len(target_rows),
        "target_labels_loaded": False, "model_seeds": model_seeds, "fit_audit": audits,
        "bounded_correction_removed": True, "stage2_semantics": "direct Stage-II logit without Stage-I-logit input",
        "max_delta": "NOT_APPLICABLE_DIRECT_STAGE2"})
    print(f"MALEVAL_DIRECT_STAGE2_TOP50_DONE complete={len(target_rows)} fallback={128-len(target_rows)} labels_loaded=false", flush=True)


if __name__ == "__main__": main()

