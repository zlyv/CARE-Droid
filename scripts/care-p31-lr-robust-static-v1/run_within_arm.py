#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import (FORMAL_DIAG, OUTPUT, YEARS, arm_probability,
    grouped_top25, mean_opposition, method_name, norm, read_csv, write_csv, dump_json)
from model_adapter import install_a0, install_fold_safe_compact76, load_bags, prepare_imports
from run_frozen_arm import build_parser
from stage1_core import crossfit_arm_probability, fit_calibrated_committee


def load_or_fit_member(ids, x, y, folds, year, fold, seed, jobs, arm):
    test = np.flatnonzero(folds == fold)
    train = np.flatnonzero(folds != fold)
    return fit_calibrated_committee(x[train], y[train], x[test], seed, arm,
        inner_folds=3, jobs=jobs), "rebuilt_arm_specific"


def main():
    args = build_parser().parse_args(); output = OUTPUT / "within" / args.arm / "predictions.csv"
    if output.is_file():
        print(f"P31_FAMILY_STAGE2_WITHIN_RESUME arm={args.arm}", flush=True); return
    resolved = json.loads((FORMAL_DIAG / "resolved_inputs.json").read_text(encoding="utf-8"))
    args.features = Path(resolved["sealed_features"]); args.cache = Path(resolved["endpoint_cache"])
    args.future_inter_cache = Path(resolved["future_inter_cache"])
    args.region_cache = FORMAL_DIAG / "cache/a0_role_2017_2021.npz"
    prepare_imports(); import run_care_rolegraph_conditioned_dualstream_v4 as v4
    install_fold_safe_compact76(v4.audit); install_a0(v4, method_name(args.arm)); device = v4.resolve_device(args.device)
    _, source_rows, clean_rows, compact_by_sha, compact_schema, original_names = v4.audit.load_inputs(args)
    target_rows = [row for row in clean_rows if str(row.get("metadata_year", "")).strip() in {str(year) for year in YEARS}]
    target_position = {norm(row["sha256"]): index for index, row in enumerate(target_rows)}
    metadata_rows = read_csv(FORMAL_DIAG / "within_2018_2019/predictions.csv") + read_csv(FORMAL_DIAG / "within_2020_2021/predictions.csv")
    metadata = {norm(row["sha256"]): {"year": int(row["year"]), "fold": int(row["fold"]), "label": int(row["label"])} for row in metadata_rows}
    if len(metadata) != 1989 or set(metadata) != set(target_position):
        raise RuntimeError("BLOCKED_BY_CONTRACT: same-year metadata cohort changed")
    with np.load(resolved["future_p31_cache"], allow_pickle=False) as archive:
        p31_ids = [norm(value) for value in archive["sha256"]]; p31_values = np.asarray(archive["role_prevalence"], float)
    p31_position = {sid: index for index, sid in enumerate(p31_ids)}
    with np.load(FORMAL_DIAG / "cache/future_s7.npz", allow_pickle=False) as archive:
        s7_ids = [norm(value) for value in archive["sha256"]]; s7_values = np.asarray(archive["s7"], np.float32)
    s7_position = {sid: index for index, sid in enumerate(s7_ids)}
    all_bags = load_bags(args.region_cache, list(source_rows) + list(target_rows))[len(source_rows):]
    prediction_rows = []; audit_rows = []; checkpoint_root = OUTPUT / "checkpoints" / "within" / args.arm
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        ids = np.asarray(sorted(sid for sid, value in metadata.items() if value["year"] == year))
        labels = np.asarray([metadata[sid]["label"] for sid in ids], int)
        folds = np.asarray([metadata[sid]["fold"] for sid in ids], int)
        p31 = p31_values[[p31_position[sid] for sid in ids]]
        s7 = s7_values[[s7_position[sid] for sid in ids]]
        rows = [target_rows[target_position[sid]] for sid in ids]
        bags = [all_bags[target_position[sid]] for sid in ids]
        stage1_oof = np.full(len(ids), np.nan); expert_oof = np.full(len(ids), np.nan)
        final_oof = np.full(len(ids), np.nan); score_oof = np.full(len(ids), np.nan); route_oof = np.zeros(len(ids), bool)
        for fold in range(1, 6):
            train = np.flatnonzero(folds != fold); test = np.flatnonzero(folds == fold)
            checkpoint = checkpoint_root / f"{year}_fold_{fold}.npz"
            if checkpoint.is_file():
                with np.load(checkpoint, allow_pickle=False) as archive:
                    stored_ids = [norm(value) for value in archive["sha256"]]
                    base_test = np.asarray(archive["stage1_prob"], float); router_score = np.asarray(archive["router_score"], float)
                    route = np.asarray(archive["route"], bool); expert = np.asarray(archive["expert_prob"], float)
                    final = np.asarray(archive["stage2_prob"], float)
                if stored_ids != ids[test].tolist() or not all(np.isfinite(value).all() for value in (base_test, router_score, expert, final)):
                    raise RuntimeError("BLOCKED_BY_CONTRACT: incompatible checkpoint " + str(checkpoint))
                stage1_source = "checkpoint"; selected_epochs = "checkpoint"
            else:
                stage_seed = args.split_seed + year * 10000 + fold * 100
                member_test, stage1_source = load_or_fit_member(ids, p31, labels, folds, year, fold,
                    stage_seed, args.jobs, args.arm)
                base_test = arm_probability(member_test, args.arm); router_score = mean_opposition(member_test, base_test)
                route = grouped_top25(router_score, ids[test], np.zeros(len(test), int))
                x_train, x_test, names = v4.audit.compact_matrix_variant(v4.VARIANT,
                    [rows[index] for index in train], [rows[index] for index in test],
                    compact_by_sha, compact_schema, original_names)
                global_index, _ = v4.evidence.split_view_indices(names)
                global_train = np.asarray(x_train[:, global_index], np.float32)
                global_test = np.asarray(x_test[:, global_index], np.float32)
                base_train = crossfit_arm_probability(p31[train], labels[train], stage_seed + 1701, args.arm, jobs=args.jobs)
                model_seeds = [year * 1000000 + fold * 1000 + seed for seed in (0, 1, 2)]
                expert, _, audits = v4.fit_predict_variant(global_train, global_test, s7[train], s7[test],
                    [bags[index] for index in train], [bags[index] for index in test], base_train, base_test,
                    labels[train], "RGCondAttn", model_seeds, args, device)
                final = v4.acn.routed_probability(base_test, expert, route)
                selected_epochs = ";".join(str(value["selected_epochs"]) for value in audits)
                np.savez_compressed(checkpoint, sha256=ids[test], stage1_prob=base_test, router_score=router_score,
                    route=route.astype(np.uint8), expert_prob=expert, stage2_prob=final)
            stage1_oof[test] = base_test; score_oof[test] = router_score; route_oof[test] = route
            expert_oof[test] = expert; final_oof[test] = final
            audit_rows.append({"arm": args.arm, "year": year, "fold": fold, "train_n": len(train),
                "validation_n": len(test), "routed_n": int(route.sum()), "stage1_source": stage1_source,
                "strict_nested_stage1_train": True, "selected_epochs": selected_epochs})
            print(f"P31_FAMILY_STAGE2_WITHIN arm={args.arm} year={year} fold={fold}/5", flush=True)
        if not all(np.isfinite(value).all() for value in (stage1_oof, score_oof, expert_oof, final_oof)):
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: incomplete OOF arm={args.arm} year={year}")
        for index, sid in enumerate(ids):
            prediction_rows.append({"sha256": sid, "year": year, "fold": int(folds[index]),
                "label": int(labels[index]), "arm": args.arm, "stage1_prob": float(stage1_oof[index]),
                "router_score": float(score_oof[index]), "routed": int(route_oof[index]),
                "expert_prob": float(expert_oof[index]), "stage2_prob": float(final_oof[index])})
    write_csv(output, prediction_rows); write_csv(output.parent / "fit_audit.csv", audit_rows)
    dump_json(output.parent / "audit.json", {"arm": args.arm, "rows": len(prediction_rows),
        "protocol": "independent same-year fixed five-fold nested OOF; not temporal generalization",
        "future_used_for_method_selection": False})
    print(f"P31_FAMILY_STAGE2_WITHIN_ARM_DONE arm={args.arm} rows={len(prediction_rows)}", flush=True)


if __name__ == "__main__":
    main()
