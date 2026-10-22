#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import FUTURE_INTER, ROOT, SOURCE_P31, YEARS, METHOD, norm, read_csv, write_csv, dump
from stage1_core import crossfit_rf_probability
from a0_adapter import install_a0, install_fold_safe_compact76, load_bags, prepare_imports


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--features", type=Path)
    p.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz"))
    p.add_argument("--p72-oof", type=Path, default=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"))
    p.add_argument("--source-multiscale-features", type=Path, default=Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"))
    p.add_argument("--future-inter-cache", type=Path, default=FUTURE_INTER)
    p.add_argument("--region-cache", type=Path, default=ROOT / "cache/a0_role_2017_2021.npz")
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--split-seed", type=int, default=20260904)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--early-stop-fraction", type=float, default=.18)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--embedding-dim", type=int, default=24)
    p.add_argument("--graph-dim", type=int, default=32)
    p.add_argument("--graph-heads", type=int, default=4)
    p.add_argument("--graph-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=.20)
    p.add_argument("--learning-rate", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=5.)
    p.add_argument("--max-delta", type=float, default=2.25)
    p.add_argument("--keep-weight", type=float, default=2.)
    p.add_argument("--fix-weight", type=float, default=1.)
    p.add_argument("--keep-consistency", type=float, default=.5)
    p.add_argument("--shared-prototypes", type=int, default=32)
    p.add_argument("--benign-prototypes", type=int, default=24)
    p.add_argument("--prototype-regions-per-apk", type=int, default=16)
    p.add_argument("--internal-background-coverage", type=float, default=.70)
    p.add_argument("--null-samples", type=int, default=64)
    p.add_argument("--swaps-per-edge", type=int, default=10)
    p.add_argument("--null-seed", type=int, default=20260827)
    p.add_argument("--z-clip", type=float, default=5.)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--rebuild-future-inter-cache", action="store_true")
    return p


def main():
    a = parser().parse_args()
    resolved = json.loads((ROOT / "resolved_inputs.json").read_text())
    if a.features is None:
        a.features = Path(resolved["sealed_features"])
    a.cache = Path(resolved["endpoint_cache"])
    prepare_imports()
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    install_fold_safe_compact76(v4.audit); install_a0(v4)
    device = v4.resolve_device(a.device)
    manifest, source_rows, clean_rows, compact_by_sha, compact_schema, original_names = v4.audit.load_inputs(a)
    source_y = np.asarray([row["label"] for row in manifest], int)
    target_rows = [row for row in clean_rows if str(row.get("metadata_year", "")).strip() in {str(y) for y in YEARS}]
    if len(target_rows) != 1989:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: future rows={len(target_rows)}")
    ref = {norm(r["sha256"]): r for r in read_csv(ROOT / "references/future_label_free.csv")}
    base_target = np.asarray([float(ref[norm(r["sha256"])]["stage1_prob"]) for r in target_rows])
    route = np.asarray([int(ref[norm(r["sha256"])]["routed"]) for r in target_rows], bool)
    router_score = np.asarray([float(ref[norm(r["sha256"])]["router_score"]) for r in target_rows])
    target_year = np.asarray([int(r["metadata_year"]) for r in target_rows])
    endpoint_cache = v4.audit.exp.read_cache(a.cache)
    print("Reconstructing frozen source RoleGraph/S7 and loading shared future S7...", flush=True)
    source_s7, _ = v4.v3.load_source_s7(source_rows, endpoint_cache, a)
    with np.load(ROOT / "cache/future_s7.npz", allow_pickle=False) as z:
        s7_ids = [norm(x) for x in z["sha256"]]; s7_values = np.asarray(z["s7"], np.float32)
    s7_pos = {sid: i for i, sid in enumerate(s7_ids)}
    target_s7 = s7_values[[s7_pos[norm(r["sha256"])] for r in target_rows]]
    all_rows = list(source_rows) + list(target_rows)
    all_bags = load_bags(a.region_cache, all_rows)
    source_bags = all_bags[:len(source_rows)]; target_bags = all_bags[len(source_rows):]
    x_source, x_target, names = v4.audit.compact_matrix_variant(v4.VARIANT, source_rows, target_rows, compact_by_sha, compact_schema, original_names)
    global_index, _ = v4.evidence.split_view_indices(names)
    global_source = np.asarray(x_source[:, global_index], np.float32); global_target = np.asarray(x_target[:, global_index], np.float32)
    with np.load(SOURCE_P31, allow_pickle=False) as z:
        p31_ids = [norm(x) for x in z["sha256"]]; p31 = np.asarray(z["role_prevalence"], float)
    p31_pos = {sid: i for i, sid in enumerate(p31_ids)}
    source_p31 = p31[[p31_pos[norm(row["sha256"])] for row in source_rows]]
    base_train = crossfit_rf_probability(source_p31, source_y, a.split_seed + 1701, jobs=a.jobs)
    model_seeds = [900000, 900001, 900002]
    expert, diagnostics, audits = v4.fit_predict_variant(global_source, global_target, source_s7, target_s7,
        source_bags, target_bags, base_train, base_target, source_y, "RGCondAttn", model_seeds, a, device)
    final = v4.acn.routed_probability(base_target, expert, route)
    if not all(np.isfinite(x).all() for x in (expert, final, diagnostics)):
        raise RuntimeError("BLOCKED_BY_CONTRACT: non-finite frozen predictions")
    rows = []
    for i, row in enumerate(target_rows):
        rows.append({"sha256": norm(row["sha256"]), "year": int(target_year[i]),
            "stage1_prob": float(base_target[i]), "router_score": float(router_score[i]), "routed": int(route[i]),
            "expert_prob": float(expert[i]), "di25_final_prob": float(final[i]), "full100_prob": float(expert[i])})
    write_csv(ROOT / "frozen/future_predictions_label_free.csv", rows)
    dump(ROOT / "frozen/fit_audit.json", {"train_rows": 799, "future_rows": 1989,
        "model_seeds": model_seeds, "fit_audit": audits, "future_labels_loaded": False,
        "stage1_train_probability": "strict nested RF OOF", "stage1_future_probability": "full-2017 calibrated RF",
        "router": "full-2017 four-member committee, RF-anchored MeanOpposition, year-wise Top25%"})
    print("FORMAL_FROZEN_FUTURE_LABEL_FREE_DONE")
    print(f"future_rows=1989 routed={int(route.sum())} labels_loaded=false")


if __name__ == "__main__":
    main()
