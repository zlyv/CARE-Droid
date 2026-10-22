#!/usr/bin/env python3
"""Formal strict-2017 outer-OOF Need-Gate experiment (explicit invocation only)."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
for path in (
    HERE, PROJECT / "scripts/care-rolegraph-conditioned-v4-script",
    PROJECT / "scripts/care-rgcondattn-dedup-bounded-source-v1",
    PROJECT / "scripts/care-stage2only-anomaly-v1/vendor",
):
    sys.path.insert(0, str(path))

from fold_protocol import inner_oof_stage1, outer_fold_indices
from noego_representation import fit_seed
from probe_protocol import ARMS, need_correct, stage1_logit
from source_contract import ENDPOINT_CACHE, OUTPUT, blocked, norm, write_json
from source_loader import load_source_contract

import run_care_rolegraph_conditioned_dualstream_v4 as v4
import run_care_rgcondattn_dedup_bounded_source_v1 as anomaly

SEEDS = (0, 1, 2)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise blocked(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--split-seed", type=int, default=20260901)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--early-stop-fraction", type=float, default=0.18)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--embedding-dim", type=int, default=24)
    p.add_argument("--graph-dim", type=int, default=32)
    p.add_argument("--graph-heads", type=int, default=4)
    p.add_argument("--graph-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--learning-rate", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--max-delta", type=float, default=2.25)
    p.add_argument("--keep-weight", type=float, default=2.0)
    p.add_argument("--fix-weight", type=float, default=1.0)
    p.add_argument("--keep-consistency", type=float, default=0.5)
    p.add_argument("--shared-prototypes", type=int, default=32)
    p.add_argument("--benign-prototypes", type=int, default=24)
    p.add_argument("--prototype-regions-per-apk", type=int, default=16)
    p.add_argument("--internal-background-coverage", type=float, default=0.70)
    p.add_argument("--null-samples", type=int, default=64)
    p.add_argument("--swaps-per-edge", type=int, default=10)
    p.add_argument("--null-seed", type=int, default=20260827)
    p.add_argument("--z-clip", type=float, default=5.0)
    p.add_argument("--progress-every", type=int, default=100)
    return p.parse_args()


def assert_frozen(args) -> None:
    expected = {
        "seeds": "0,1,2", "inner_folds": 4, "split_seed": 20260901,
        "max_epochs": 30, "learning_rate": 8e-4, "weight_decay": 5e-4,
        "max_delta": 2.25, "keep_weight": 2.0, "fix_weight": 1.0,
        "keep_consistency": 0.5, "null_samples": 64, "swaps_per_edge": 10,
        "null_seed": 20260827, "z_clip": 5.0,
    }
    changed = {key: (getattr(args, key), value) for key, value in expected.items() if getattr(args, key) != value}
    if changed:
        raise blocked("historical frozen parameter changed: " + repr(changed))


def compact_context(data):
    compact_by_sha = {norm(row["sha256"]): row for row in data.feature_rows}
    try:
        _, _, names = v4.audit.allratio.original_compact_matrix_named(
            data.feature_rows, data.feature_rows[:1], compact_by_sha, data.feature_schema
        )
    except Exception as error:
        raise blocked("cannot reconstruct canonical Compact76 from source_loader rows: " + repr(error)) from error
    names = list(names)
    if len(names) != 76:
        raise blocked(f"canonical Compact width={len(names)} expected=76")
    return compact_by_sha, names


def fold_global24(data, compact_by_sha, names, train, validation):
    x_train, x_validation, feature_names = v4.audit.compact_matrix_variant(
        v4.VARIANT, [data.feature_rows[i] for i in train], [data.feature_rows[i] for i in validation],
        compact_by_sha, data.feature_schema, names,
    )
    global_index, _ = v4.evidence.split_view_indices(feature_names)
    x_train = np.asarray(x_train[:, global_index], dtype=np.float32)
    x_validation = np.asarray(x_validation[:, global_index], dtype=np.float32)
    if x_train.shape[1] != 24 or x_validation.shape[1] != 24:
        raise blocked("fold-local Global24 width changed")
    return x_train, x_validation


def source_s7_in_memory(data, args):
    # The cache path is fixed by source_contract; no cache is created or resolved.
    endpoint = v4.audit.exp.read_cache(ENDPOINT_CACHE)
    if set(endpoint) != set(data.ids) or len(endpoint) != 799:
        raise blocked("endpoint payload cohort differs from canonical source loader")
    raw, _ = v4.audit.future.reconstruct_structure(data.feature_rows, endpoint, 31)
    # This is a pure 2017 in-memory computation. No future_interz file is read/written.
    _, _, inter_z, _ = v4.v3.future_s7.clipped.build_multiscale_features(
        data.feature_rows, endpoint, raw[:, 120:], args
    )
    result = np.concatenate([raw, np.clip(inter_z, -float(args.z_clip), float(args.z_clip))], axis=1)
    if result.shape != (799, v4.S7_DIM) or not np.isfinite(result).all():
        raise blocked(f"source-only S7 contract failed: {result.shape}")
    return np.asarray(result, dtype=np.float32)


def fit_probe(x_train, y_train, x_validation):
    if set(np.unique(y_train).tolist()) != {0, 1}:
        raise blocked("routed outer-train need target has one class")
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear", max_iter=5000,
        random_state=20260920,
    )
    model.fit(scaler.transform(x_train), y_train)
    return model.predict_proba(scaler.transform(x_validation))[:, 1]


def main() -> None:
    args = arguments(); assert_frozen(args)
    data = load_source_contract()
    if data.future_rows_loaded != 0 or data.ego_consumed or len(data.ids) != 799 or int(data.route.sum()) != 200:
        raise blocked("formal source-only hard gate failed")
    if tuple(v4.parse_ints(args.seeds)) != SEEDS:
        raise blocked("historical source seed ensemble must be 0,1,2")
    device = v4.resolve_device(args.device)
    protocol_path = OUTPUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    protocol.update({
        "formal_training_started": True,
        "formal_training_complete": False,
        "future_rows_loaded": 0,
        "ego_consumed": False,
    })
    write_json(protocol_path, protocol)
    compact_by_sha, compact_names = compact_context(data)
    s7 = source_s7_in_memory(data, args)
    v4.AUGMENTED_REGION_DIM = v4.v3.v2.BASE_REGION_DIM + 4
    v4.RoleGraphConditionedEvidence = anomaly.DedupBoundedRGCondAttn

    rows_by_sha = {}; assigned = set(); fold_audits = []; representation_audits = []
    for fold in range(1, 6):
        train, validation = outer_fold_indices(data.folds, fold)
        train_route = data.route[train]; validation_route = data.route[validation]
        if int(np.sum(validation_route)) != 40:
            raise blocked(f"outer fold {fold} routed validation count changed")
        global_train, global_validation = fold_global24(data, compact_by_sha, compact_names, train, validation)

        def inner_predict(fit_global, validation_global, model_seed):
            x_fit, x_val = fold_global24(data, compact_by_sha, compact_names, fit_global, validation_global)
            return v4.g24.fit_view(
                x_fit, x_val, data.labels[fit_global], np.arange(24), int(model_seed)
            )

        base_train = inner_oof_stage1(
            train, data.labels, inner_predict, folds=args.inner_folds,
            split_seed=args.split_seed + 1000 * fold,
            model_seed=args.split_seed + 10000 * fold,
        )
        base_validation = data.p_stage1[validation]
        target_train = need_correct(data.labels[train], base_train)
        target_validation = need_correct(data.labels[validation], base_validation)
        logit_train = stage1_logit(base_train); logit_validation = stage1_logit(base_validation)

        simple = {
            "S0": (logit_train[:, None], logit_validation[:, None]),
            "S1": (
                np.concatenate([global_train, logit_train[:, None]], axis=1),
                np.concatenate([global_validation, logit_validation[:, None]], axis=1),
            ),
        }
        probabilities = {
            arm: [fit_probe(x[0][train_route], target_train[train_route], x[1][validation_route])]
            for arm, x in simple.items()
        }
        probabilities.update({"S2": [], "S3": []})
        fold_seeds = tuple(100000 * fold + seed for seed in SEEDS)
        for seed in fold_seeds:
            train_rep, validation_rep, audit = fit_seed(
                v4, anomaly, global_train, s7[train], s7[validation],
                [data.host_bags[i] for i in train], [data.host_bags[i] for i in validation],
                base_train, data.labels[train], args, device, seed,
            )
            representation_audits.append({"fold": fold, **audit})
            probabilities["S2"].append(fit_probe(
                train_rep[train_route], target_train[train_route], validation_rep[validation_route]))
            probabilities["S3"].append(fit_probe(
                np.concatenate([train_rep, logit_train[:, None]], axis=1)[train_route],
                target_train[train_route],
                np.concatenate([validation_rep, logit_validation[:, None]], axis=1)[validation_route]))

        routed_indices = validation[validation_route]
        for local, index in enumerate(routed_indices):
            sha = data.ids[index]
            if sha not in rows_by_sha:
                pred = int(data.p_stage1[index] >= 0.5)
                rows_by_sha[sha] = {
                    "sha256": sha, "year": 2017, "fold": fold,
                    "label": int(data.labels[index]), "p_stage1": float(data.p_stage1[index]),
                    "stage1_logit": float(stage1_logit([data.p_stage1[index]])[0]),
                    "stage1_pred": pred, "need_correct": int(pred != int(data.labels[index])), "route_DI": 1,
                }
            for arm in ("S0", "S1", "S2", "S3"):
                value = float(np.mean(np.asarray(probabilities[arm]), axis=0)[local])
                key = (sha, arm)
                if key in assigned or not np.isfinite(value):
                    raise blocked(f"invalid/duplicate OOF assignment: {key}")
                assigned.add(key); rows_by_sha[sha][f"score_{arm}"] = value; rows_by_sha[sha][f"pred_{arm}"] = int(value >= 0.5)
        fold_audits.append({
            "fold": fold, "outer_train_n": len(train), "outer_validation_n": len(validation),
            "routed_train_n": int(np.sum(train_route)), "routed_validation_n": int(np.sum(validation_route)),
            "inner_oof_stage1_complete": True, "validation_label_used_for_training": False,
            "future_rows_loaded": 0, "ego_consumed": False,
        })
        write_csv(OUTPUT / "oof_predictions.partial.csv", sorted(rows_by_sha.values(), key=lambda row: (row["fold"], row["sha256"])))
        print(f"source Need-Gate outer fold {fold}/5 done", flush=True)

    output_rows = sorted(rows_by_sha.values(), key=lambda row: (row["fold"], row["sha256"]))
    if len(output_rows) != 200 or len({row["sha256"] for row in output_rows}) != 200 or len(assigned) != 800:
        raise blocked("final routed OOF coverage must be 200 SHA x 4 arms")
    write_csv(OUTPUT / "oof_predictions.csv", output_rows)
    write_csv(OUTPUT / "fold_audit.csv", fold_audits)
    write_csv(OUTPUT / "representation_training_audit.csv", representation_audits)
    write_json(OUTPUT / "cohort_audit.json", {
        **data.router_audit, "OOF_rows": 200, "OOF_unique_SHA": 200,
        "future_rows_loaded": 0, "ego_consumed": False,
    })
    protocol.update({
        "formal_training_started": True, "formal_training_complete": True,
        "representation_learner": "historical Host-only DedupBoundedRGCondAttn",
        "representation_training_scope": "all outer-train APKs",
        "probe_training_scope": "routed outer-train APKs only",
        "S2_probe_input": "graph32+critical_host24+attended_host24+summary8; excludes Global24/logit/Ego/delta",
        "representation_training_may_use": "outer-train Global24, fold-safe inner-OOF Stage-I probability, outer-train labels",
        "base_seeds": list(SEEDS),
        "fold_seed_rule": "100000 * outer_fold + base_seed",
        "inner_split_seed_rule": "20260901 + 1000 * outer_fold",
        "inner_model_seed_rule": "20260901 + 10000 * outer_fold + inner_fold",
        "max_delta": 2.25, "learning_rate": 8e-4,
        "weight_decay": 5e-4, "epoch_selection": "historical outer-train-only early stopping",
        "future_rows_loaded": 0, "ego_consumed": False,
    })
    write_json(protocol_path, protocol)
    print("SOURCE_NEEDGATE_FORMAL_OOF_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        if str(error).startswith("BLOCKED_BY_CONTRACT"):
            print(str(error), flush=True); raise SystemExit(2)
        raise
