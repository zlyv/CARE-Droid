#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (DIAGNOSTIC_SPLIT_SEED, FORMAL_DIAG, FUTURE_REFERENCE, NEW_ARMS, OUTPUT,
    SOURCE_P31, YEARS, arm_probability, grouped_top25, mean_opposition, method_name, norm, read_csv, write_csv, dump_json)
from model_adapter import install_a0, install_fold_safe_compact76, load_bags, prepare_imports
from stage1_core import crossfit_arm_probability


def build_parser():
    parser = argparse.ArgumentParser(); parser.add_argument("--arm", choices=NEW_ARMS, required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--p72-oof", type=Path, default=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"))
    parser.add_argument("--source-multiscale-features", type=Path, default=Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"))
    parser.add_argument("--inner-folds", type=int, default=4); parser.add_argument("--split-seed", type=int, default=DIAGNOSTIC_SPLIT_SEED)
    parser.add_argument("--max-epochs", type=int, default=30); parser.add_argument("--early-stop-fraction", type=float, default=.18)
    parser.add_argument("--early-stop-patience", type=int, default=5); parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=24); parser.add_argument("--graph-dim", type=int, default=32)
    parser.add_argument("--graph-heads", type=int, default=4); parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=.20); parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4); parser.add_argument("--grad-clip", type=float, default=5.)
    parser.add_argument("--max-delta", type=float, default=2.25); parser.add_argument("--keep-weight", type=float, default=2.)
    parser.add_argument("--fix-weight", type=float, default=1.); parser.add_argument("--keep-consistency", type=float, default=.5)
    parser.add_argument("--shared-prototypes", type=int, default=32); parser.add_argument("--benign-prototypes", type=int, default=24)
    parser.add_argument("--prototype-regions-per-apk", type=int, default=16); parser.add_argument("--internal-background-coverage", type=float, default=.70)
    parser.add_argument("--null-samples", type=int, default=64); parser.add_argument("--swaps-per-edge", type=int, default=10)
    parser.add_argument("--null-seed", type=int, default=20260827); parser.add_argument("--z-clip", type=float, default=5.)
    parser.add_argument("--progress-every", type=int, default=200); parser.add_argument("--rebuild-future-inter-cache", action="store_true")
    return parser


def main():
    args = build_parser().parse_args(); output = OUTPUT / "frozen" / args.arm / "predictions_label_free.csv"
    if output.is_file():
        print(f"P31_FAMILY_STAGE2_FROZEN_RESUME arm={args.arm}", flush=True); return
    resolved = json.loads((FORMAL_DIAG / "resolved_inputs.json").read_text(encoding="utf-8"))
    args.features = Path(resolved["sealed_features"]); args.cache = Path(resolved["endpoint_cache"])
    args.future_inter_cache = Path(resolved["future_inter_cache"])
    args.region_cache = FORMAL_DIAG / "cache/a0_role_2017_2021.npz"
    prepare_imports(); import run_care_rolegraph_conditioned_dualstream_v4 as v4
    install_fold_safe_compact76(v4.audit); install_a0(v4, method_name(args.arm)); device = v4.resolve_device(args.device)
    manifest, source_rows, clean_rows, compact_by_sha, compact_schema, original_names = v4.audit.load_inputs(args)
    source_y = np.asarray([row["label"] for row in manifest], int)
    target_rows = [row for row in clean_rows if str(row.get("metadata_year", "")).strip() in {str(year) for year in YEARS}]
    if len(target_rows) != 1989:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: future rows={len(target_rows)}")
    reference = {norm(row["sha256"]): row for row in read_csv(FUTURE_REFERENCE)}
    target_ids = np.asarray([norm(row["sha256"]) for row in target_rows])
    member = np.asarray([[float(reference[sid][f"p_P31__{name}"]) for name in ("LR", "RF", "XGB", "SVM")] for sid in target_ids])
    base_target = arm_probability(member, args.arm); router_score = mean_opposition(member, base_target)
    target_year = np.asarray([int(row["metadata_year"]) for row in target_rows], int)
    route = grouped_top25(router_score, target_ids, target_year)
    endpoint_cache = v4.audit.exp.read_cache(args.cache)
    print(f"P31_FAMILY_STAGE2_FROZEN_PREPARE arm={args.arm}", flush=True)
    source_s7, _ = v4.v3.load_source_s7(source_rows, endpoint_cache, args)
    with np.load(FORMAL_DIAG / "cache/future_s7.npz", allow_pickle=False) as archive:
        s7_ids = [norm(value) for value in archive["sha256"]]; s7_values = np.asarray(archive["s7"], np.float32)
    s7_position = {sid: index for index, sid in enumerate(s7_ids)}
    target_s7 = s7_values[[s7_position[sid] for sid in target_ids]]
    all_rows = list(source_rows) + list(target_rows); all_bags = load_bags(args.region_cache, all_rows)
    source_bags = all_bags[:len(source_rows)]; target_bags = all_bags[len(source_rows):]
    x_source, x_target, names = v4.audit.compact_matrix_variant(v4.VARIANT, source_rows, target_rows,
        compact_by_sha, compact_schema, original_names)
    global_index, _ = v4.evidence.split_view_indices(names)
    global_source = np.asarray(x_source[:, global_index], np.float32); global_target = np.asarray(x_target[:, global_index], np.float32)
    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        p31_ids = [norm(value) for value in archive["sha256"]]; p31 = np.asarray(archive["role_prevalence"], float)
    p31_position = {sid: index for index, sid in enumerate(p31_ids)}
    source_p31 = p31[[p31_position[norm(row["sha256"])] for row in source_rows]]
    base_train = crossfit_arm_probability(source_p31, source_y, args.split_seed + 1701, args.arm, jobs=args.jobs)
    model_seeds = [900000, 900001, 900002]
    expert, diagnostics, audits = v4.fit_predict_variant(global_source, global_target, source_s7, target_s7,
        source_bags, target_bags, base_train, base_target, source_y, "RGCondAttn", model_seeds, args, device)
    final = v4.acn.routed_probability(base_target, expert, route)
    if not all(np.isfinite(value).all() for value in (expert, diagnostics, final)):
        raise RuntimeError("BLOCKED_BY_CONTRACT: non-finite frozen predictions")
    rows = []
    for index, sid in enumerate(target_ids):
        rows.append({"sha256": sid, "year": int(target_year[index]), "arm": args.arm,
            "stage1_prob": float(base_target[index]), "router_score": float(router_score[index]),
            "routed": int(route[index]), "expert_prob": float(expert[index]), "stage2_prob": float(final[index])})
    write_csv(output, rows)
    dump_json(output.parent / "fit_audit.json", {"arm": args.arm, "train_rows": 799, "future_rows": 1989,
        "model_seeds": model_seeds, "fit_audit": audits, "future_labels_loaded": False,
        "stage1_train_probability": f"strict nested {args.arm} OOF",
        "router": f"{args.arm}-anchored MeanOpposition Top25% independently by future year"})
    print(f"P31_FAMILY_STAGE2_FROZEN_ARM_DONE arm={args.arm} rows=1989 labels_loaded=false", flush=True)


if __name__ == "__main__":
    main()
