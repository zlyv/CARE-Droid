#!/usr/bin/env python3
"""Run one strict within-year five-fold OOF Stage-II expert arm."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
for path in (HERE, HERE / "vendor"):
    sys.path.insert(0, str(path))

import run_future_arm as future
import run_care_rolegraph_conditioned_dualstream_v4 as v4
import run_care_rgcondattn_dedup_bounded_source_v1 as anomaly_bounded
import run_care_malben_role_source_v2 as malben_bounded
import run_malben_host_v2 as malben_host
import run_stage2only_malben_v1 as direct_malben
import run_stage2only_anomaly_v1 as direct_anomaly
from shared_role_prototypes import evidence_columns, fit_role_prototypes
from source_label_contract import install_source_label_contract
from source_frozen_compact_contract import (
    compact_contract_audit,
    install_source_frozen_compact_contract,
)


FEATURE_CONTRACT_VERSION = "source-frozen-compact76-foldsafe-v1"


def norm(value) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def subset_bags(bags: list[np.ndarray], index: np.ndarray) -> list[np.ndarray]:
    return [bags[int(i)] for i in index]


def load_data(args):
    install_source_label_contract(v4)
    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    load_args = SimpleNamespace(
        features=Path(resolved["features"]), schema=Path(resolved["schema"]),
        p72_oof=Path(resolved["source_manifest"]),
    )
    _, _, _, compact_by_sha, schema, names = v4.audit.load_inputs(load_args)
    install_source_frozen_compact_contract(v4)
    if v4.VARIANT != "allratio76_ratio52":
        raise RuntimeError("unexpected frozen Compact variant: " + str(v4.VARIANT))
    router_rows = read_csv(args.router_inputs)
    if len(router_rows) != 1989:
        raise RuntimeError("within-year router input must contain 1989 rows")
    ids = [norm(row["sha256"]) for row in router_rows]
    target_rows = [compact_by_sha[value] for value in ids]
    labels = np.asarray([int(row["label"]) for row in router_rows], dtype=np.int64)
    years = np.asarray([int(row["year"]) for row in router_rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in router_rows], dtype=np.int64)
    base = np.asarray([float(row["p_stage1"]) for row in router_rows], dtype=np.float64)

    endpoint = future.merge_endpoint_caches(args.source_cache, args.future_cache)
    target_s3, _ = v4.audit.future.reconstruct_structure(target_rows, endpoint, 31)
    target_inter, _, inter_audit = v4.v3.future_s7.load_or_build_future_inter(
        target_rows, target_s3, endpoint, args)
    target_s7 = np.concatenate([target_s3, target_inter], axis=1)
    if target_s7.shape != (1989, v4.S7_DIM):
        raise RuntimeError("within-year S7 contract failed")
    target_bags, target_count = future.load_bags(args.future_regions, target_rows)
    return SimpleNamespace(
        rows=target_rows, ids=ids, labels=labels, years=years, folds=folds,
        base=base, s7=target_s7, bags=target_bags, counts=target_count,
        compact_by_sha=compact_by_sha, compact_schema=schema,
        canonical_names=names, inter_audit=inter_audit,
    )


def fold_global24(data, train: np.ndarray, validation: np.ndarray):
    x_train, x_validation, feature_names = v4.audit.compact_matrix_variant(
        v4.VARIANT,
        [data.rows[int(i)] for i in train],
        [data.rows[int(i)] for i in validation],
        data.compact_by_sha,
        data.compact_schema,
        data.canonical_names,
    )
    global_index, _ = v4.evidence.split_view_indices(feature_names)
    x_train = np.asarray(x_train[:, global_index], dtype=np.float32)
    x_validation = np.asarray(x_validation[:, global_index], dtype=np.float32)
    if x_train.shape[1] != 24 or x_validation.shape[1] != 24:
        raise RuntimeError("fold-local Global24 feature contract failed")
    return x_train, x_validation


def inner_oof_base(data, outer_train: np.ndarray, seed: int) -> np.ndarray:
    y = data.labels[outer_train]
    result = np.full(len(outer_train), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    for fold, (train_local, validation_local) in enumerate(
        splitter.split(np.zeros(len(y)), y), start=1
    ):
        train = outer_train[train_local]
        validation = outer_train[validation_local]
        x_train, x_validation = fold_global24(data, train, validation)
        result[validation_local] = v4.audit.p72.fit_predict(
            x_train, data.labels[train], x_validation, seed=seed + fold
        )[0]
    if not np.isfinite(result).all():
        raise RuntimeError("incomplete inner OOF Stage-I probabilities")
    return result


def direct_fold(data, train, validation, args, seeds, device):
    if args.representation == "anomaly":
        direct_malben.StageIIOnlyModel = direct_anomaly.StageIIOnlyAnomalyModel
        direct_malben.fit_scaled_shared_role_bags = direct_anomaly.fit_scaled_anomaly_bags
    else:
        malben_bounded.fit_role_prototypes = fit_role_prototypes
        malben_bounded.evidence_columns = evidence_columns
    train_feature, validation_feature, audits = direct_malben.fit_fold_features(
        data.s7[train], data.s7[validation],
        subset_bags(data.bags, train), subset_bags(data.bags, validation),
        data.labels[train], seeds, args, device)
    if train_feature.shape[1] != 81 or validation_feature.shape[1] != 81:
        raise RuntimeError("direct within-year feature dimension changed")
    scaler = StandardScaler().fit(train_feature)
    classifier = LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear", max_iter=2000,
        random_state=int(args.split_seed))
    classifier.fit(scaler.transform(train_feature), data.labels[train])
    probability = classifier.predict_proba(scaler.transform(validation_feature))[:, 1]
    return probability, audits


def bounded_fold(data, train, validation, args, seeds, device, fold_seed):
    x_train, x_validation = fold_global24(data, train, validation)
    base_train = inner_oof_base(data, train, fold_seed)
    base_validation = data.base[validation]
    if args.representation == "anomaly":
        v4.AUGMENTED_REGION_DIM = future.ORIGINAL_AUGMENTED_DIM
        v4.v3.fit_scaled_dualproto_bags = future.ORIGINAL_ANOMALY_FIT
        v4.RoleGraphConditionedEvidence = anomaly_bounded.DedupBoundedRGCondAttn
        v4.predict_model = future.ORIGINAL_V4_PREDICT
        fit = v4.fit_predict_variant
        variant = "RGCondAttn"
    else:
        malben_bounded.fit_role_prototypes = fit_role_prototypes
        malben_bounded.evidence_columns = evidence_columns
        v4.v3.fit_scaled_dualproto_bags = malben_host.fit_scaled_shared_role_bags
        v4.AUGMENTED_REGION_DIM = malben_bounded.FULL_REGION_DIM
        v4.RoleGraphConditionedEvidence = malben_bounded.RoleConditionedEvidenceModel
        v4.predict_model = future.ORIGINAL_V4_PREDICT
        fit = malben_bounded.grouped_fit_predict
        variant = "MalBenContrast"
    probability, _, audits = fit(
        x_train, x_validation,
        data.s7[train], data.s7[validation],
        subset_bags(data.bags, train), subset_bags(data.bags, validation),
        base_train, base_validation, data.labels[train],
        variant, seeds, args, device)
    return probability, audits


def main() -> None:
    parser = future.parser()
    parser.description = __doc__
    parser.set_defaults(
        resolved=Path("artifacts/care-route-stage2-within-year-matrix-v1/preflight/resolved_inputs.json"),
        future_inter_cache=Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"),
    )
    parser.add_argument("--router-inputs", type=Path, required=True)
    args = parser.parse_args()
    if float(args.max_delta) != 2.25:
        raise SystemExit("within-year protocol keeps source-selected max-delta=2.25")
    seeds = v4.parse_ints(args.seeds)
    if seeds != [900000, 900001, 900002]:
        raise SystemExit("within-year protocol requires frozen three-seed ensemble")
    device = v4.resolve_device(args.device)
    data = load_data(args)
    args.output.mkdir(parents=True, exist_ok=True)
    output = np.full(1989, np.nan, dtype=np.float64)
    audit_checkpoint = args.output / "fit_audit_checkpoint.csv"
    audit_rows = read_csv(audit_checkpoint) if audit_checkpoint.is_file() else []
    checkpoint = args.output / "within_year_expert_checkpoint.csv"
    if checkpoint.is_file():
        saved = read_csv(checkpoint)
        by_id = {norm(row["sha256"]): row for row in saved}
        if len(saved) != len(by_id):
            raise RuntimeError("duplicate SHA in within-year arm checkpoint")
        versions = {row.get("feature_contract_version", "") for row in saved}
        if versions != {FEATURE_CONTRACT_VERSION}:
            print("[invalidate] checkpoint predates fold-safe Compact76 contract", flush=True)
        else:
            for i, value in enumerate(data.ids):
                if value not in by_id:
                    continue
                row = by_id[value]
                if row.get("representation") != args.representation or row.get("head") != args.head:
                    raise RuntimeError("checkpoint belongs to a different expert arm")
                output[i] = float(row["p_expert"])
            print(f"[resume] restored {int(np.sum(np.isfinite(output)))}/1989 expert predictions", flush=True)
    for year in sorted(set(data.years.tolist())):
        for fold in range(1, 6):
            validation = np.flatnonzero((data.years == year) & (data.folds == fold))
            train = np.flatnonzero((data.years == year) & (data.folds != fold))
            if len(validation) == 0 or len(np.unique(data.labels[train])) != 2:
                raise RuntimeError(f"invalid within-year split: {year} fold {fold}")
            if np.isfinite(output[validation]).all():
                print(
                    f"[resume] within-year arm {args.representation}/{args.head}: "
                    f"{year} fold {fold}/5 already complete", flush=True)
                continue
            if args.head == "direct":
                probability, audits = direct_fold(data, train, validation, args, seeds, device)
            else:
                probability, audits = bounded_fold(
                    data, train, validation, args, seeds, device,
                    int(args.split_seed) + year * 10 + fold)
            output[validation] = np.asarray(probability, dtype=np.float64)
            audit_rows.extend({"year": year, "outer_fold": fold, **item} for item in audits)
            completed = np.flatnonzero(np.isfinite(output))
            checkpoint_rows = [{
                "sha256": data.ids[i], "year": int(data.years[i]),
                "fold": int(data.folds[i]), "representation": args.representation,
                "head": args.head, "p_expert": float(output[i]),
                "feature_contract_version": FEATURE_CONTRACT_VERSION,
            } for i in completed]
            temporary = checkpoint.with_suffix(".tmp")
            write_csv(temporary, checkpoint_rows)
            temporary.replace(checkpoint)
            audit_temporary = audit_checkpoint.with_suffix(".tmp")
            write_csv(audit_temporary, audit_rows)
            audit_temporary.replace(audit_checkpoint)
            print(
                f"within-year arm {args.representation}/{args.head}: "
                f"{year} fold {fold}/5 done", flush=True)
    if not np.isfinite(output).all():
        raise RuntimeError("incomplete within-year expert OOF predictions")
    rows = [{
        "sha256": data.ids[i], "year": int(data.years[i]),
        "label": int(data.labels[i]), "fold": int(data.folds[i]),
        "representation": args.representation, "head": args.head,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "p_expert": float(output[i]), "real_host_count": int(data.counts[i]),
        "same_year_oof": True,
    } for i in range(1989)]
    write_csv(args.output / "within_year_expert_predictions.csv", rows)
    if audit_rows:
        write_csv(args.output / "fit_audit.csv", audit_rows)
    protocol = {
        "representation": args.representation, "head": args.head,
        "protocol": "independent per-year five-fold OOF",
        "temporal_generalization_result": False,
        "future_year_labels_used_for_same_year_training": True,
        "max_delta_logit": 2.25 if args.head == "bounded" else None,
        "seeds": seeds, "host_ego_assets_reused": True,
        "apk_reextraction": False, "future_inter_audit": data.inter_audit,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (args.output / "compact76_contract_audit.json").write_text(
        json.dumps(compact_contract_audit(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("WITHIN_YEAR_ARM_DONE", args.representation, args.head, flush=True)


if __name__ == "__main__":
    main()
