#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (METHOD, ROOT, aligned_rows, align_fold_global24, install_c43,
                    install_paths, load_bags, norm, read_csv, resolved,
                    subset_bags, write_csv)


install_paths()
import run_future_arm as future
import run_care_rolegraph_conditioned_dualstream_v4 as v4
from p31_core import crossfit_mean_probability


def fold_global(data, train, validation):
    # Build the fold-standardized raw Compact view, then align Global24 only.
    # Fold-only semantic columns are irrelevant to C43's Global24 context and
    # must not make a valid annual fold fail its fixed source schema contract.
    x_train, x_validation, names = v4.audit.allratio.original_compact_matrix_named(
        [data.target_rows[int(index)] for index in train],
        [data.target_rows[int(index)] for index in validation],
        data.compact_by_sha, data.schema)
    return align_fold_global24(x_train, x_validation, names, data.names)


def main():
    parser = future.parser()
    parser.description = "Independent same-year P31-DI25 + C43-RQ five-fold OOF"
    parser.set_defaults(
        representation="anomaly", head="bounded",
        resolved=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json"),
    )
    parser.add_argument("--years", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    years_requested = [int(value) for value in args.years.split(",") if value.strip()]
    if not years_requested or not set(years_requested).issubset({2018, 2019, 2020, 2021}):
        raise SystemExit("--years must be a comma-separated subset of 2018,2019,2020,2021")
    if float(args.max_delta) != 2.25:
        raise SystemExit("same-year max-delta must remain 2.25")
    seeds = v4.parse_ints(args.seeds)
    if seeds != [900000, 900001, 900002]:
        raise SystemExit("same-year seeds must be 900000,900001,900002")
    device = v4.resolve_device(args.device)
    cfg = resolved()
    args.resolved = Path(cfg["future_matrix_resolved"])
    data = future.load_common(args)
    contract = aligned_rows(data.target_rows, Path(cfg["within_p31_predictions"]))
    labels = np.asarray([int(float(row["label"])) for row in contract])
    years = np.asarray([int(float(row["year"])) for row in contract])
    folds = np.asarray([int(float(row["fold"])) for row in contract])
    base = np.asarray([float(row["p_stage1"]) for row in contract])
    route = np.asarray([int(float(row["route_DI"])) for row in contract], dtype=bool)
    with np.load(cfg["future_p31_features"], allow_pickle=False) as saved:
        raw_ids = [norm(value) for value in saved["sha256"].astype(str)]
        raw_values = np.asarray(saved["role_prevalence"], float)
    raw_lookup = {value: index for index, value in enumerate(raw_ids)}
    p31 = np.asarray([raw_values[raw_lookup[norm(row["sha256"])]]
                      for row in data.target_rows], float)
    bags, counts = load_bags(Path(cfg["future_c43_cache"]), data.target_rows)
    install_c43(v4)

    args.output.mkdir(parents=True, exist_ok=True)
    output = np.full(1989, np.nan, dtype=float)
    fit_audit = []
    checkpoint = args.output / "checkpoint.csv"
    if checkpoint.is_file():
        saved = {norm(row["sha256"]): row for row in read_csv(checkpoint)}
        for index, target in enumerate(data.target_rows):
            row = saved.get(norm(target["sha256"]))
            if row is not None:
                output[index] = float(row["p_expert"])
        print(f"[resume] restored {int(np.isfinite(output).sum())}/1989 rows", flush=True)
    for year in years_requested:
        for fold in range(1, 6):
            validation = np.flatnonzero((years == year) & (folds == fold))
            train = np.flatnonzero((years == year) & (folds != fold))
            if np.isfinite(output[validation]).all():
                print(f"[resume] P31-C43 same-year {year} fold {fold}/5", flush=True)
                continue
            global_train, global_validation, alignment = fold_global(data, train, validation)
            ignored = alignment["ignored_fold_only_role"]
            missing = alignment["missing_canonical_global"]
            print(f"Global24 alignment {year} fold {fold}: fold_width={alignment['fold_compact_width']} "
                  f"ignored_role={ignored} missing_global={missing}", flush=True)
            base_train = crossfit_mean_probability(
                p31[train], labels[train],
                int(args.split_seed) + year * 10 + fold, jobs=args.jobs)
            probability, _, audits = v4.fit_predict_variant(
                global_train, global_validation,
                data.target_s7[train], data.target_s7[validation],
                subset_bags(bags, train), subset_bags(bags, validation),
                base_train, base[validation], labels[train], "RGCondAttn",
                seeds, args, device)
            output[validation] = np.asarray(probability, float)
            fit_audit.extend({"year": year, "outer_fold": fold, **row} for row in audits)
            fit_audit.append({
                "year": year, "outer_fold": fold, "audit_type": "Global24_schema_alignment",
                "fold_compact_width": alignment["fold_compact_width"],
                "ignored_fold_only_role": "|".join(ignored),
                "missing_canonical_global": "|".join(missing),
                "global24_width": 24,
            })
            completed = np.flatnonzero(np.isfinite(output))
            write_csv(checkpoint, [{
                "sha256": norm(data.target_rows[index]["sha256"]),
                "year": int(years[index]), "fold": int(folds[index]),
                "p_expert": float(output[index]),
            } for index in completed])
            print(f"P31-C43 same-year {year} fold {fold}/5 done", flush=True)

    selected = np.flatnonzero(np.isin(years, years_requested))
    if not np.isfinite(output[selected]).all():
        raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete requested same-year OOF")
    rows = []
    for index in selected:
        final = output[index] if route[index] else base[index]
        rows.append({
            "sha256": norm(data.target_rows[index]["sha256"]),
            "year": int(years[index]), "fold": int(folds[index]),
            "label": int(labels[index]), "p_stage1": float(base[index]),
            "U_DI": float(contract[index]["U_DI"]),
            "route_DI": int(route[index]), "p_expert": float(output[index]),
            "p_final": float(final), "method": METHOD,
        })
    write_csv(args.output / "oof_predictions.csv", rows)
    if fit_audit:
        write_csv(args.output / "fit_audit.csv", fit_audit)
    protocol = {
        "protocol": "independent same-year five-fold OOF",
        "years": years_requested,
        "stage1": "P31 Mean Committee fitted inside each annual outer fold",
        "outer_train_stage1": "nested inner-OOF P31 Mean Committee",
        "router": "P31 MeanOpposition annual fold-wise Top25%",
        "stage2": "C43_RELATION_QUERY bounded correction",
        "stage2_training_scope": "all outer-train APKs in the same year",
        "temporal_generalization": False,
        "max_delta": 2.25, "seeds": seeds, "ego_consumed": False,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print("P31_C43_WITHIN_YEAR_GROUP_DONE", ",".join(map(str, years_requested)))


if __name__ == "__main__":
    main()
