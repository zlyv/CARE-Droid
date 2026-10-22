#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import FUTURE_INTER, FOLD_ASSIGNMENTS, ROOT, YEARS, METHOD, load_future_labels, norm, read_csv, write_csv, dump
from stage1_core import MEMBERS, crossfit_rf_probability, fit_calibrated_committee, mean_opposition, top25
from a0_adapter import install_a0, install_fold_safe_compact76, load_bags, prepare_imports
from run_frozen_future import parser as base_parser


def main():
    p = base_parser()
    p.add_argument("--years", nargs="+", type=int, required=True)
    args = p.parse_args()
    if any(year not in YEARS for year in args.years):
        raise RuntimeError("BLOCKED_BY_CONTRACT: invalid year")
    resolved = json.loads((ROOT / "resolved_inputs.json").read_text())
    if args.features is None:
        args.features = Path(resolved["sealed_features"])
    args.cache = Path(resolved["endpoint_cache"])
    prepare_imports()
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    install_fold_safe_compact76(v4.audit); install_a0(v4)
    device = v4.resolve_device(args.device)
    manifest, source_rows, clean_rows, compact_by_sha, compact_schema, original_names = v4.audit.load_inputs(args)
    requested = {str(y) for y in args.years}
    target_rows = [r for r in clean_rows if str(r.get("metadata_year", "")).strip() in {str(y) for y in YEARS}]
    label_path, label_map = load_future_labels()
    print("Loading shared label-free future RoleGraph/S7...", flush=True)
    with np.load(ROOT / "cache/future_s7.npz", allow_pickle=False) as z:
        s7_ids = [norm(x) for x in z["sha256"]]; s7_values = np.asarray(z["s7"], np.float32)
    s7_pos = {sid: i for i, sid in enumerate(s7_ids)}
    target_s7 = s7_values[[s7_pos[norm(r["sha256"])] for r in target_rows]]
    all_rows = list(source_rows) + list(target_rows)
    bags = load_bags(args.region_cache, all_rows)[len(source_rows):]
    target_pos = {norm(row["sha256"]): i for i, row in enumerate(target_rows)}
    with np.load(resolved["future_p31_cache"], allow_pickle=False) as z:
        p31_ids = [norm(x) for x in z["sha256"]]; p31_x = np.asarray(z["role_prevalence"], float)
    p31_pos = {sid: i for i, sid in enumerate(p31_ids)}
    assignments = read_csv(Path(resolved["fold_assignments"]))
    all_predictions = []; fold_audit = []
    for year in args.years:
        assigned = [r for r in assignments if int(r["year"]) == year]
        ids = np.asarray([norm(r["sample_id"]) for r in assigned])
        if any(sid not in target_pos or sid not in p31_pos or sid not in label_map for sid in ids):
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: {year} SHA alignment failed")
        idx = np.asarray([target_pos[sid] for sid in ids], int)
        rows = [target_rows[i] for i in idx]
        labels = np.asarray([label_map[sid] for sid in ids], int)
        folds = np.asarray([int(r["fold"]) for r in assigned], int)
        p31 = p31_x[[p31_pos[sid] for sid in ids]]
        s7 = target_s7[idx]; year_bags = [bags[i] for i in idx]
        stage1_oof = np.full(len(ids), np.nan); expert_oof = np.full(len(ids), np.nan)
        final_oof = np.full(len(ids), np.nan); route_oof = np.zeros(len(ids), bool); score_oof = np.full(len(ids), np.nan)
        for fold in sorted(np.unique(folds)):
            train = np.flatnonzero(folds != fold); test = np.flatnonzero(folds == fold)
            train_rows = [rows[i] for i in train]; test_rows = [rows[i] for i in test]
            x_train, x_test, names = v4.audit.compact_matrix_variant(v4.VARIANT, train_rows, test_rows, compact_by_sha, compact_schema, original_names)
            global_index, _ = v4.evidence.split_view_indices(names)
            global_train = np.asarray(x_train[:, global_index], np.float32); global_test = np.asarray(x_test[:, global_index], np.float32)
            train_y = labels[train]
            stage_seed = args.split_seed + year * 10000 + int(fold) * 100
            member_test = fit_calibrated_committee(p31[train], train_y, p31[test], stage_seed, inner_folds=3, jobs=args.jobs)
            base_test = member_test[:, MEMBERS.index("RF")]
            router_score = mean_opposition(member_test, base_test)
            route = top25(router_score, ids[test])
            base_train = crossfit_rf_probability(p31[train], train_y, stage_seed + 1701, jobs=args.jobs)
            model_seeds = [year * 1000000 + int(fold) * 1000 + seed for seed in (0, 1, 2)]
            expert, _, audits = v4.fit_predict_variant(global_train, global_test, s7[train], s7[test],
                [year_bags[i] for i in train], [year_bags[i] for i in test], base_train, base_test, train_y,
                "RGCondAttn", model_seeds, args, device)
            final = v4.acn.routed_probability(base_test, expert, route)
            stage1_oof[test] = base_test; expert_oof[test] = expert; final_oof[test] = final
            route_oof[test] = route; score_oof[test] = router_score
            fold_audit.append({"year": year, "fold": int(fold), "train_n": len(train), "validation_n": len(test),
                "routed_n": int(route.sum()), "stage1": "P31 calibrated RF", "committee_role": "routing_only",
                "strict_nested_stage1_train": True, "model_seeds": ";".join(map(str, model_seeds)),
                "selected_epochs": ";".join(str(x["selected_epochs"]) for x in audits)})
            print(f"FORMAL same-year {year} fold {fold}/5 done", flush=True)
        if not all(np.isfinite(x).all() for x in (stage1_oof, expert_oof, final_oof, score_oof)):
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: incomplete {year} OOF")
        for i, sid in enumerate(ids):
            all_predictions.append({"sha256": sid, "year": year, "fold": int(folds[i]), "label": int(labels[i]),
                "stage1_prob": float(stage1_oof[i]), "router_score": float(score_oof[i]), "routed": int(route_oof[i]),
                "expert_prob": float(expert_oof[i]), "di25_final_prob": float(final_oof[i]), "full100_prob": float(expert_oof[i])})
    suffix = "_".join(map(str, args.years))
    out = ROOT / ("within_" + suffix)
    write_csv(out / "predictions.csv", all_predictions); write_csv(out / "fold_audit.csv", fold_audit)
    dump(out / "audit.json", {"years": args.years, "rows": len(all_predictions), "future_labels_used_for_training": True,
        "protocol": "independent same-year fixed five-fold OOF; not temporal generalization", "label_source": str(label_path),
        "future_used_for_method_selection": False})
    print(f"FORMAL_WITHIN_YEAR_GROUP_DONE years={suffix} rows={len(all_predictions)}")


if __name__ == "__main__":
    main()
