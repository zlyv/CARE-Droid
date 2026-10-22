#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (METHOD, ROOT, aligned_rows, install_c43, install_paths,
                    load_bags, norm, resolved, write_csv)


install_paths()
import run_future_arm as future
import run_care_rolegraph_conditioned_dualstream_v4 as v4


def main():
    parser = future.parser()
    parser.description = "Frozen 2017-trained P31-DI25 + C43-RQ future diagnostic"
    parser.set_defaults(
        representation="anomaly", head="bounded",
        resolved=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json"),
        output=ROOT / "frozen_future/model",
    )
    args = parser.parse_args()
    if args.representation != "anomaly" or args.head != "bounded":
        raise SystemExit("this runner is fixed to C43-RQ bounded correction")
    if float(args.max_delta) != 2.25:
        raise SystemExit("frozen max-delta must remain 2.25")
    seeds = v4.parse_ints(args.seeds)
    if seeds != [900000, 900001, 900002]:
        raise SystemExit("frozen seeds must be 900000,900001,900002")
    device = v4.resolve_device(args.device)
    cfg = resolved()
    args.resolved = Path(cfg["future_matrix_resolved"])
    data = future.load_common(args)

    source_p31 = aligned_rows(data.source_rows, Path(cfg["source_p31_reference"]))
    target_p31 = aligned_rows(data.target_rows, Path(cfg["frozen_p31_predictions"]))
    data.base_source = np.asarray([float(row["P31_MEAN_COMMITTEE__prob"]) for row in source_p31])
    data.base_target = np.asarray([float(row["p_stage1"]) for row in target_p31])
    route = np.asarray([int(float(row["route_DI"])) for row in target_p31], dtype=bool)
    if int(route.sum()) != 497:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen P31 route changed")
    data.source_bags, data.source_count = load_bags(
        Path(cfg["source_c43_cache"]), data.source_rows)
    data.target_bags, data.target_count = load_bags(
        Path(cfg["future_c43_cache"]), data.target_rows)

    x_source, x_target, names = v4.audit.compact_matrix_variant(
        v4.VARIANT, data.source_rows, data.target_rows,
        data.compact_by_sha, data.schema, data.names)
    global_index, _ = v4.evidence.split_view_indices(names)
    global_source = np.asarray(x_source[:, global_index], dtype=np.float32)
    global_target = np.asarray(x_target[:, global_index], dtype=np.float32)
    if global_source.shape != (799, 24) or global_target.shape != (1989, 24):
        raise RuntimeError("BLOCKED_BY_CONTRACT: Global24 expert-context shape changed")
    install_c43(v4)
    expert, diagnostic, audits = v4.fit_predict_variant(
        global_source, global_target, data.source_s7, data.target_s7,
        data.source_bags, data.target_bags, data.base_source, data.base_target,
        data.source_y, "RGCondAttn", seeds, args, device)
    expert = np.asarray(expert, float)
    final = np.where(route, expert, data.base_target)
    if expert.shape != (1989,) or not np.isfinite(expert).all():
        raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete frozen C43 expert probabilities")
    rows = []
    for index, target in enumerate(data.target_rows):
        rows.append({
            "sha256": norm(target["sha256"]),
            "year": int(target_p31[index]["year"]),
            "p_stage1": float(data.base_target[index]),
            "U_DI": float(target_p31[index]["U_DI"]),
            "route_DI": int(route[index]),
            "p_expert": float(expert[index]),
            "p_final": float(final[index]),
            "method": METHOD,
        })
    output = ROOT / "frozen_future"
    write_csv(output / "label_free_predictions.csv", rows)
    if audits:
        write_csv(output / "fit_audit.csv", audits)
    protocol = {
        "protocol": "frozen 2017 train -> 2018-2021 diagnostic",
        "source_train_n": 799, "future_n": 1989,
        "stage1": "P31 calibrated four-member mean",
        "router": "P31 MeanOpposition year-wise Top25%",
        "stage2": "C43_RELATION_QUERY bounded correction",
        "stage2_train_scope": "all 799 source APKs",
        "future_labels_loaded_by_model": False,
        "future_route_n": 497,
        "max_delta": 2.25, "seeds": seeds,
        "ego_consumed": False,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print("P31_C43_FROZEN_FUTURE_LABEL_FREE_DONE")
    print("future_rows=1989 routed=497 labels_loaded=false")


if __name__ == "__main__":
    main()
