#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (ENDPOINT_SOURCE, FORMAL_DIAG, OUTPUT, SOURCE_A0_CACHE, SOURCE_P31, TARGET_YEARS,
    WINDOWS, WITHIN_META, dump_json, grouped_top25, mean_opposition, norm, read_csv, write_csv)
from model_adapter import install_a0, install_fold_safe_compact76, load_bags, prepare_imports
from stage1_core import crossfit_svm_probability, fit_calibrated_committee


def parser():
    p = argparse.ArgumentParser(); p.add_argument("--window", choices=WINDOWS, required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--split-seed", type=int, default=20261010)
    p.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    p.add_argument("--p72-oof", type=Path, default=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"))
    p.add_argument("--source-multiscale-features", type=Path, default=Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"))
    p.add_argument("--max-epochs", type=int, default=30); p.add_argument("--early-stop-fraction", type=float, default=.18)
    p.add_argument("--early-stop-patience", type=int, default=5); p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--embedding-dim", type=int, default=24); p.add_argument("--graph-dim", type=int, default=32)
    p.add_argument("--graph-heads", type=int, default=4); p.add_argument("--graph-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=.20); p.add_argument("--learning-rate", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4); p.add_argument("--grad-clip", type=float, default=5.)
    p.add_argument("--max-delta", type=float, default=2.25); p.add_argument("--keep-weight", type=float, default=2.)
    p.add_argument("--fix-weight", type=float, default=1.); p.add_argument("--keep-consistency", type=float, default=.5)
    p.add_argument("--shared-prototypes", type=int, default=32); p.add_argument("--benign-prototypes", type=int, default=24)
    p.add_argument("--prototype-regions-per-apk", type=int, default=16); p.add_argument("--internal-background-coverage", type=float, default=.70)
    p.add_argument("--null-samples", type=int, default=64); p.add_argument("--swaps-per-edge", type=int, default=10)
    p.add_argument("--null-seed", type=int, default=20260827); p.add_argument("--z-clip", type=float, default=5.)
    p.add_argument("--progress-every", type=int, default=200); p.add_argument("--rebuild-future-inter-cache", action="store_true")
    return p


def main():
    args = parser().parse_args(); output = OUTPUT / args.window / "predictions_label_free.csv"
    if output.is_file():
        print(f"SVM_TRAIN_WINDOW_RESUME window={args.window}", flush=True); return
    resolved = json.loads((FORMAL_DIAG / "resolved_inputs.json").read_text(encoding="utf-8"))
    args.features = Path(resolved["sealed_features"]); args.cache = Path(resolved["endpoint_cache"])
    args.future_inter_cache = Path(resolved["future_inter_cache"]); args.region_cache = FORMAL_DIAG / "cache/a0_role_2017_2021.npz"
    prepare_imports(); import run_care_rolegraph_conditioned_dualstream_v4 as v4
    install_fold_safe_compact76(v4.audit); install_a0(v4, f"P31SVM__SVMRoute__A0_C43__{args.window}")
    device = v4.resolve_device(args.device)
    manifest, source_rows, clean_rows, compact_by_sha, compact_schema, original_names = v4.audit.load_inputs(args)
    source_y = np.asarray([row["label"] for row in manifest], int)
    future_rows = [row for row in clean_rows if str(row.get("metadata_year", "")).strip() in {"2018", "2019", "2020", "2021"}]
    by_year = {year: [row for row in future_rows if int(row["metadata_year"]) == year] for year in (2018,) + TARGET_YEARS}
    label18 = {norm(row["sha256"]): int(row["label"]) for row in read_csv(WITHIN_META) if int(row["year"]) == 2018}
    ids18 = [norm(row["sha256"]) for row in by_year[2018]]
    if len(label18) != 793 or set(ids18) != set(label18):
        raise RuntimeError("BLOCKED_BY_CONTRACT: 2018 train-label alignment failed")
    train_rows = list(by_year[2018]) if args.window == "W18" else list(source_rows) + list(by_year[2018])
    train_y = np.asarray([label18[norm(row["sha256"])] for row in by_year[2018]], int)
    if args.window == "W1718":
        train_y = np.concatenate([source_y, train_y])
    target_rows = [row for year in TARGET_YEARS for row in by_year[year]]
    target_ids = np.asarray([norm(row["sha256"]) for row in target_rows])
    target_year = np.asarray([int(row["metadata_year"]) for row in target_rows], int)
    if len(target_rows) != 1196 or len(train_rows) != (793 if args.window == "W18" else 1592):
        raise RuntimeError("BLOCKED_BY_CONTRACT: train/target cohort changed")

    endpoint_cache = v4.audit.exp.read_cache(args.cache)
    source_s7, _ = v4.v3.load_source_s7(source_rows, endpoint_cache, args)
    with np.load(FORMAL_DIAG / "cache/future_s7.npz", allow_pickle=False) as archive:
        future_s7_ids = [norm(value) for value in archive["sha256"]]; future_s7_values = np.asarray(archive["s7"], np.float32)
    future_s7 = {sid: future_s7_values[i] for i, sid in enumerate(future_s7_ids)}
    source_s7_map = {norm(row["sha256"]): source_s7[i] for i, row in enumerate(source_rows)}
    train_s7 = np.asarray([source_s7_map.get(norm(row["sha256"]), future_s7.get(norm(row["sha256"]))) for row in train_rows], np.float32)
    target_s7 = np.asarray([future_s7[norm(row["sha256"])] for row in target_rows], np.float32)

    all_rows = list(source_rows) + list(future_rows); all_bags = load_bags(args.region_cache, all_rows)
    bag_map = {norm(row["sha256"]): all_bags[i] for i, row in enumerate(all_rows)}
    train_bags = [bag_map[norm(row["sha256"])] for row in train_rows]; target_bags = [bag_map[sid] for sid in target_ids]
    x_train, x_target, names = v4.audit.compact_matrix_variant(v4.VARIANT, train_rows, target_rows,
        compact_by_sha, compact_schema, original_names)
    global_index, _ = v4.evidence.split_view_indices(names)
    global_train = np.asarray(x_train[:, global_index], np.float32); global_target = np.asarray(x_target[:, global_index], np.float32)

    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        source_p31_ids = [norm(value) for value in archive["sha256"]]; source_p31_values = np.asarray(archive["role_prevalence"], float)
    with np.load(resolved["future_p31_cache"], allow_pickle=False) as archive:
        future_p31_ids = [norm(value) for value in archive["sha256"]]; future_p31_values = np.asarray(archive["role_prevalence"], float)
    p31_map = {sid: source_p31_values[i] for i, sid in enumerate(source_p31_ids)}
    p31_map.update({sid: future_p31_values[i] for i, sid in enumerate(future_p31_ids)})
    train_p31 = np.asarray([p31_map[norm(row["sha256"])] for row in train_rows], float)
    target_p31 = np.asarray([p31_map[sid] for sid in target_ids], float)
    seed = 20261010 + (18 if args.window == "W18" else 1718)
    member_target = fit_calibrated_committee(train_p31, train_y, target_p31, seed, inner_folds=3, jobs=args.jobs)
    base_target = member_target[:, 3]; router_score = mean_opposition(member_target, base_target)
    route = grouped_top25(router_score, target_ids, target_year)
    base_train = crossfit_svm_probability(train_p31, train_y, seed + 1701, jobs=args.jobs)
    model_seeds = [seed * 100 + offset for offset in (0, 1, 2)]
    expert, diagnostics, audits = v4.fit_predict_variant(global_train, global_target, train_s7, target_s7,
        train_bags, target_bags, base_train, base_target, train_y, "RGCondAttn", model_seeds, args, device)
    final = v4.acn.routed_probability(base_target, expert, route)
    if not all(np.isfinite(value).all() for value in (member_target, base_train, expert, diagnostics, final)):
        raise RuntimeError("BLOCKED_BY_CONTRACT: non-finite predictions")
    rows = []
    for i, sid in enumerate(target_ids):
        rows.append({"sha256": sid, "year": int(target_year[i]), "window": args.window,
            "p_LR": float(member_target[i, 0]), "p_RF": float(member_target[i, 1]),
            "p_XGB": float(member_target[i, 2]), "p_SVM": float(member_target[i, 3]),
            "stage1_prob": float(base_target[i]), "router_score": float(router_score[i]), "routed": int(route[i]),
            "expert_prob": float(expert[i]), "stage2_prob": float(final[i])})
    write_csv(output, rows)
    dump_json(output.parent / "fit_audit.json", {"window": args.window, "train_rows": len(train_rows),
        "train_years": [2018] if args.window == "W18" else [2017, 2018], "target_years": list(TARGET_YEARS),
        "target_rows": len(target_rows), "model_seeds": model_seeds, "fit_audit": audits,
        "target_labels_used_for_fit": False, "stage1_train_probability": "strict nested calibrated SVM OOF",
        "router": "SVM-anchored MeanOpposition Top25% independently per target year", "max_delta": 2.25})
    print(f"SVM_TRAIN_WINDOW_DONE window={args.window} train={len(train_rows)} target={len(target_rows)} labels_used_for_fit=2017_2018_only", flush=True)


if __name__ == "__main__":
    main()
