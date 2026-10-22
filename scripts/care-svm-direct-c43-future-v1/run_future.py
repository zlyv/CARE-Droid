#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from future_common import (DIRECT_METHOD, DIRECT_SCRIPT, FAMILY_SCRIPT, FORMAL_DIAG, OUTPUT, YEARS,
    dump_json, norm, write_csv)


def main():
    cli = argparse.ArgumentParser(); cli.add_argument("--device", default="cuda"); cli.add_argument("--jobs", type=int, default=4)
    options = cli.parse_args(); target_path = OUTPUT / "predictions_label_free.csv"
    if target_path.is_file() and (OUTPUT / "fit_audit.json").is_file():
        print("SVM_DIRECT_C43_FUTURE_RESUME rows=1989"); return

    # Import the already-tested family future runner and exact Direct-C43 source variant.
    sys.path.insert(0, str(FAMILY_SCRIPT))
    from common import (DIAGNOSTIC_SPLIT_SEED, FUTURE_REFERENCE, SOURCE_P31,
        arm_probability, grouped_top25, mean_opposition, read_csv)
    from model_adapter import install_a0, install_fold_safe_compact76, load_bags, prepare_imports
    from run_frozen_arm import build_parser
    from stage1_core import crossfit_arm_probability
    sys.path.insert(0, str(DIRECT_SCRIPT))
    from model_variants import install_variant

    args = build_parser().parse_args(["--arm", "SVM", "--device", options.device, "--jobs", str(options.jobs)])
    args.keep_weight = 1.0; args.fix_weight = 1.0; args.keep_consistency = 0.0
    resolved = json.loads((FORMAL_DIAG / "resolved_inputs.json").read_text(encoding="utf-8"))
    args.features = Path(resolved["sealed_features"])
    args.cache = Path(resolved["endpoint_cache"])
    args.future_inter_cache = Path(resolved["future_inter_cache"])
    args.region_cache = FORMAL_DIAG / "cache/a0_role_2017_2021.npz"

    prepare_imports(); import run_care_rolegraph_conditioned_dualstream_v4 as v4
    install_fold_safe_compact76(v4.audit); install_a0(v4, DIRECT_METHOD); install_variant(v4, "U2_DIRECT_STAGE2")
    device = v4.resolve_device(args.device)
    manifest, source_rows, clean_rows, compact_by_sha, compact_schema, original_names = v4.audit.load_inputs(args)
    source_y = np.asarray([row["label"] for row in manifest], int)
    target_rows = [row for row in clean_rows if str(row.get("metadata_year", "")).strip() in {str(year) for year in YEARS}]
    if len(source_rows) != 799 or len(target_rows) != 1989:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: cohort changed source={len(source_rows)} target={len(target_rows)}")
    reference = {norm(row["sha256"]): row for row in read_csv(FUTURE_REFERENCE)}
    target_ids = np.asarray([norm(row["sha256"]) for row in target_rows])
    members = np.asarray([[float(reference[sid][f"p_P31__{name}"]) for name in ("LR", "RF", "XGB", "SVM")]
                          for sid in target_ids])
    base_target = arm_probability(members, "SVM"); router_score = mean_opposition(members, base_target)
    target_year = np.asarray([int(row["metadata_year"]) for row in target_rows], int)
    route = grouped_top25(router_score, target_ids, target_year)
    expected_route = {year: int(round(int(np.sum(target_year == year)) * .25)) for year in YEARS}
    actual_route = {year: int(np.sum(route & (target_year == year))) for year in YEARS}
    if actual_route != expected_route:
        raise RuntimeError("BLOCKED_BY_CONTRACT: Top25 route changed " + repr(actual_route))

    endpoint_cache = v4.audit.exp.read_cache(args.cache)
    print("SVM_DIRECT_C43_FUTURE_PREPARE_S7", flush=True)
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
    global_source = np.asarray(x_source[:, global_index], np.float32)
    global_target = np.asarray(x_target[:, global_index], np.float32)
    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        p31_ids = [norm(value) for value in archive["sha256"]]
        p31 = np.asarray(archive["role_prevalence"], float)
    p31_position = {sid: index for index, sid in enumerate(p31_ids)}
    source_p31 = p31[[p31_position[norm(row["sha256"])] for row in source_rows]]
    base_train = crossfit_arm_probability(source_p31, source_y, DIAGNOSTIC_SPLIT_SEED + 1701, "SVM", jobs=args.jobs)
    model_seeds = [900000, 900001, 900002]
    print("SVM_DIRECT_C43_FUTURE_FIT train=799 target=1989 labels_target=false", flush=True)
    expert, diagnostics, audits = v4.fit_predict_variant(global_source, global_target, source_s7, target_s7,
        source_bags, target_bags, base_train, base_target, source_y, "RGCondAttn", model_seeds, args, device)
    final = v4.acn.routed_probability(base_target, expert, route)
    if not all(np.isfinite(value).all() for value in (expert, diagnostics, final)):
        raise RuntimeError("BLOCKED_BY_CONTRACT: non-finite frozen predictions")
    rows = []
    for index, sid in enumerate(target_ids):
        rows.append({"sha256": sid, "year": int(target_year[index]), "stage1_prob": float(base_target[index]),
            "router_score": float(router_score[index]), "routed": int(route[index]),
            "direct_c43_prob": float(expert[index]), "final_prob": float(final[index])})
    write_csv(target_path, rows)
    dump_json(OUTPUT / "fit_audit.json", {"status": "PASS", "train_rows": 799, "future_rows": 1989,
        "future_labels_loaded": False, "model_seeds": model_seeds, "route_by_year": actual_route,
        "stage1_train_probability": "strict cross-fit P31-SVM OOF", "fit_audit": audits,
        "direct_variant_source": str(DIRECT_SCRIPT / "model_variants.py")})
    print("SVM_DIRECT_C43_FUTURE_PREDICTIONS_DONE rows=1989 labels_loaded=false", flush=True)


if __name__ == "__main__":
    main()
