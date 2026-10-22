#!/usr/bin/env python3
"""Fixed 2018--2021 diagnostic for retained S7 and RSSR side control.

The source-selected main candidate is frozen as C76 + A20-routed JointLogistic
on Stats120 + PairRaw465 + InterZ961 clipped to [-5,5].  This script trains on
all fixed 2017 rows once and reports 2018--2021 diagnostic results.  It also
reports S3 (without Inter-Z), ordinary anchored residual, fixed RSSR, and ACN20
controls.  No future label is used for fitting, routing, tuning, or selection.

The future years have already been viewed elsewhere in this project, so every
result here is diagnostic only; formal temporal claims still need untouched
2022+ or an independent dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_acn_routing_ablation_2017 as acn
import run_allratio70_vs_nodrop76_experiment as audit
import run_anchored_route_conditioned_residual_2017 as anchored
import run_clipped_intra_inter_structure_ablation_2017 as clipped
import run_pairz_structure_ablation_2017 as pairz_exp
import run_rssr_interz_2017 as rssr


VARIANT = "allratio76_ratio52"
THRESHOLD = 0.5


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_s7_from_reference(path, cohort_rows, raw_source):
    path = Path(path)
    if not path.exists():
        raise RuntimeError("source multiscale feature reference not found: " + str(path))
    reference = np.load(path)
    expected_sha = np.asarray([pairz_exp.norm_sha(row["sha256"]) for row in cohort_rows])
    result = {
        "path": str(path),
        "sha_order_equal": bool(np.array_equal(expected_sha, reference["sha256"].astype(str))),
        "pair_raw_max_abs_difference": float(np.max(np.abs(
            raw_source[:, 120:] - reference["pair_raw"]
        ))),
        "inter_clip_finite": bool(np.isfinite(reference["inter_z_clipped"]).all()),
    }
    if (
        not result["sha_order_equal"]
        or result["pair_raw_max_abs_difference"] > 0.0
        or not result["inter_clip_finite"]
    ):
        raise RuntimeError("invalid source multiscale reference: " + repr(result))
    s7 = np.concatenate([
        raw_source,
        np.asarray(reference["inter_z_clipped"], dtype=np.float64),
    ], axis=1)
    if s7.shape != (799, 1546):
        raise RuntimeError("bad source S7 shape: " + repr(s7.shape))
    return s7, result


def load_or_build_future_inter(target_rows, raw_target, cache, args):
    path = Path(args.future_inter_cache)
    expected_sha = np.asarray([pairz_exp.norm_sha(row["sha256"]) for row in target_rows])
    pair_raw = raw_target[:, 120:]
    if path.exists() and not args.rebuild_future_inter_cache:
        saved = np.load(path)
        result = {
            "path": str(path),
            "loaded": True,
            "sha_order_equal": bool(np.array_equal(expected_sha, saved["sha256"].astype(str))),
            "pair_raw_max_abs_difference": float(np.max(np.abs(pair_raw - saved["pair_raw"]))),
            "null_samples": int(saved["null_samples"]),
            "swaps_per_edge": int(saved["swaps_per_edge"]),
            "null_seed": int(saved["null_seed"]),
            "z_clip": float(saved["z_clip"]),
        }
        parameter_equal = (
            result["null_samples"] == int(args.null_samples)
            and result["swaps_per_edge"] == int(args.swaps_per_edge)
            and result["null_seed"] == int(args.null_seed)
            and result["z_clip"] == float(args.z_clip)
        )
        result["parameter_equal"] = bool(parameter_equal)
        if (
            not result["sha_order_equal"]
            or result["pair_raw_max_abs_difference"] > 0.0
            or not parameter_equal
        ):
            raise RuntimeError("future Inter-Z cache mismatch: " + repr(result))
        inter_clip = np.asarray(saved["inter_z_clipped"], dtype=np.float64)
        if inter_clip.shape != (len(target_rows), 961) or not np.isfinite(inter_clip).all():
            raise RuntimeError("bad cached future Inter-Z")
        return inter_clip, [], result

    print("Constructing label-free future directed Inter-Z...", flush=True)
    _, _, inter_z, construction_rows = clipped.build_multiscale_features(
        target_rows, cache, pair_raw, args
    )
    inter_clip = np.clip(inter_z, -float(args.z_clip), float(args.z_clip))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sha256=expected_sha,
        pair_raw=pair_raw,
        inter_z_clipped=inter_clip,
        null_samples=np.asarray(int(args.null_samples)),
        swaps_per_edge=np.asarray(int(args.swaps_per_edge)),
        null_seed=np.asarray(int(args.null_seed)),
        z_clip=np.asarray(float(args.z_clip)),
    )
    return inter_clip, construction_rows, {
        "path": str(path),
        "loaded": False,
        "sha_order_equal": True,
        "pair_raw_max_abs_difference": 0.0,
        "null_samples": int(args.null_samples),
        "swaps_per_edge": int(args.swaps_per_edge),
        "null_seed": int(args.null_seed),
        "z_clip": float(args.z_clip),
        "parameter_equal": True,
    }


def metric_rows_by_year(methods, years, labels, year_array, coarse, probability, routes):
    rows = []
    for year in years:
        selected = year_array == year
        for method in methods:
            route_key = method.split("__", 1)[0] if "__" in method else None
            route = routes[route_key][selected] if route_key in routes else np.zeros(np.sum(selected), dtype=bool)
            rows.append({
                **pairz_exp.metric_row(
                    method,
                    labels[selected],
                    probability[method][selected],
                    coarse[selected],
                    route,
                ),
                "year": int(year),
                "diagnostic_only": True,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--cache", type=Path, default=Path(
        "artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz"
    ))
    parser.add_argument("--p72-oof", type=Path, default=Path(
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"
    ))
    parser.add_argument("--source-multiscale-features", type=Path, default=Path(
        "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"
    ))
    parser.add_argument("--future-inter-cache", type=Path, default=Path(
        "artifacts/s7-future-interz-cache-v1/future_interz.npz"
    ))
    parser.add_argument("--rebuild-future-inter-cache", action="store_true")
    parser.add_argument("--years", default="2018,2019,2020,2021")
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/s7-interz-future-diagnostic-v1"
    ))
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--source-crossfit-folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260827)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--ordinary-l2", type=float, default=0.03)
    parser.add_argument("--rssr-gamma", type=float, default=1.0)
    parser.add_argument("--rssr-l2", type=float, default=0.03)
    parser.add_argument("--stability-resamples", type=int, default=24)
    parser.add_argument("--subsample-fraction", type=float, default=0.75)
    parser.add_argument("--stability-seed", type=int, default=20260827)
    parser.add_argument("--stability-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-8)
    parser.add_argument("--null-samples", type=int, default=64)
    parser.add_argument("--swaps-per-edge", type=int, default=10)
    parser.add_argument("--null-seed", type=int, default=20260827)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    years = sorted({int(value.strip()) for value in args.years.split(",") if value.strip()})
    if not years:
        raise SystemExit("--years must contain at least one year")
    if not 0.0 < args.budget < 1.0:
        raise SystemExit("--budget must be in (0,1)")
    if not 0.0 <= args.eta < 1.0:
        raise SystemExit("--eta must be in [0,1)")
    if args.source_crossfit_folds < 2 or args.stability_resamples < 4:
        raise SystemExit("invalid crossfit/resample count")
    if not 0.5 <= args.subsample_fraction < 1.0:
        raise SystemExit("--subsample-fraction must be in [0.5,1)")
    if args.stability_epsilon <= 0.0:
        raise SystemExit("--stability-epsilon must be positive")
    if args.null_samples < 2 or args.swaps_per_edge < 1 or args.z_clip <= 0.0:
        raise SystemExit("invalid null-model parameters")
    args.output.mkdir(parents=True, exist_ok=True)

    (
        manifest,
        source_rows,
        clean_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    ) = audit.load_inputs(args)
    requested = {str(year) for year in years}
    target_rows = [
        row for row in clean_rows
        if str(row.get("metadata_year", "")).strip() in requested
    ]
    observed_years = sorted({int(row["metadata_year"]) for row in target_rows})
    if observed_years != years:
        raise RuntimeError("requested/observed future years differ: " + repr((years, observed_years)))

    source_y = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    target_y = np.asarray([audit.future.label_of(row) for row in target_rows], dtype=np.int64)
    target_year = np.asarray([int(row["metadata_year"]) for row in target_rows], dtype=np.int64)
    cache = audit.exp.read_cache(args.cache)
    print("Reconstructing source and future Stats120+PairRaw465...", flush=True)
    source_s3, _ = audit.future.reconstruct_structure(
        source_rows, cache, int(audit.exp.ROLE_VOCABULARY_SIZE)
    )
    target_s3, _ = audit.future.reconstruct_structure(
        target_rows, cache, int(audit.exp.ROLE_VOCABULARY_SIZE)
    )
    if source_s3.shape != (799, 585) or target_s3.shape[1] != 585:
        raise RuntimeError("bad S3 shape: " + repr((source_s3.shape, target_s3.shape)))
    source_s7, source_reference = source_s7_from_reference(
        args.source_multiscale_features, source_rows, source_s3
    )
    target_inter, construction_rows, future_cache_audit = load_or_build_future_inter(
        target_rows, target_s3, cache, args
    )
    target_s7 = np.concatenate([target_s3, target_inter], axis=1)
    if target_s7.shape != (len(target_rows), 1546):
        raise RuntimeError("bad target S7 shape: " + repr(target_s7.shape))

    x_source, x_target, compact_names = audit.compact_matrix_variant(
        VARIANT,
        source_rows,
        target_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    )
    coarse, _, _ = audit.p72.fit_predict(x_source, source_y, x_target, seed=0)
    joint_s3 = audit.fit_config(x_source, x_target, source_s3, target_s3, source_y, seed=0)
    joint_s7 = audit.fit_config(x_source, x_target, source_s7, target_s7, source_y, seed=0)

    (
        source_oof,
        source_ambiguity,
        source_collision,
        source_novelty,
    ) = anchored.crossfit_base_and_risks(
        source_rows,
        source_y,
        compact_by_sha,
        compact_schema,
        original_names,
        args.source_crossfit_folds,
        args.split_seed,
        args.split_seed + 10000,
        args.knn_k,
    )
    source_route_data = anchored.calibrated_routes(
        source_ambiguity,
        source_collision,
        source_novelty,
        source_ambiguity,
        source_collision,
        source_novelty,
        args.budget,
    )
    target_collision, target_novelty, _, _ = acn.neighborhood_risks(
        x_source, source_y, x_target, args.knn_k
    )
    target_route_data = anchored.calibrated_routes(
        source_ambiguity,
        source_collision,
        source_novelty,
        acn.ambiguity(coarse),
        target_collision,
        target_novelty,
        args.budget,
    )
    budget_tag = str(int(round(100 * args.budget)))
    route_tags = ["A" + budget_tag, "ACN" + budget_tag]
    source_routes = {
        route_tags[0]: source_route_data["ambiguity_route"],
        route_tags[1]: source_route_data["acn_route"],
    }
    target_routes = {
        route_tags[0]: target_route_data["ambiguity_route"],
        route_tags[1]: target_route_data["acn_route"],
    }

    feature_names = (
        ["Stats120__" + format(i, "03d") for i in range(120)]
        + ["PairRaw465__" + format(i, "03d") for i in range(465)]
        + ["InterZ961__" + format(i, "03d") for i in range(961)]
    )
    standardized_source, standardized_target, structure_audit = audit.p72.standardize_added(
        source_s7, target_s7, feature_names
    )
    usable = int(sum(not bool(row["constant_in_train"]) for row in structure_audit))
    probability = {"C76": coarse}
    fit_rows = []
    stability_rows = []
    for route_tag in route_tags:
        source_route = source_routes[route_tag]
        target_route = target_routes[route_tag]
        probability[route_tag + "__Joint_S7"] = acn.routed_probability(
            coarse, joint_s7, target_route
        )
        if route_tag == route_tags[0]:
            probability[route_tag + "__Joint_S3"] = acn.routed_probability(
                coarse, joint_s3, target_route
            )
        stability = rssr.stability_scores(
            standardized_source,
            source_y,
            source_oof,
            source_route,
            args.eta,
            args.stability_resamples,
            args.subsample_fraction,
            args.stability_seed + (0 if route_tag == route_tags[0] else 1),
            args.stability_epsilon,
        )
        weights = np.where(source_route, 1.0, float(args.eta))
        for model, gamma, l2 in (
            ("Ordinary_S7", 0.0, args.ordinary_l2),
            ("RSSR_fixed_S7", args.rssr_gamma, args.rssr_l2),
        ):
            beta, diagnostic = rssr.fit_adaptive_residual(
                standardized_source,
                source_y,
                source_oof,
                weights,
                stability["stability"],
                stability["epsilon"],
                gamma,
                l2,
                args.max_iterations,
                args.optimizer_tolerance,
            )
            delta = standardized_target @ beta
            enhanced = anchored.sigmoid(anchored.logit(coarse) + delta)
            probability[route_tag + "__" + model] = acn.routed_probability(
                coarse, enhanced, target_route
            )
            fit_rows.append({
                "route": route_tag,
                "model": model,
                "gamma": float(gamma),
                "l2": float(l2),
                "source_routed_n": int(np.sum(source_route)),
                "future_routed_n": int(np.sum(target_route)),
                "future_routed_ratio": float(np.mean(target_route)),
                "future_delta_mean": float(np.mean(delta)),
                "future_delta_std": float(np.std(delta)),
                "future_delta_max_abs": float(np.max(np.abs(delta))),
                **diagnostic,
            })
        for index, name in enumerate(feature_names):
            stability_rows.append({
                "route": route_tag,
                "feature_index": int(index),
                "feature_name": name,
                "feature_family": name.split("__", 1)[0],
                "consistency": float(stability["consistency"][index]),
                "strength": float(stability["strength"][index]),
                "mad": float(stability["mad"][index]),
                "stability": float(stability["stability"][index]),
                "constant_in_source": int(bool(structure_audit[index]["constant_in_train"])),
            })

    methods = list(probability)
    summary_rows = metric_rows_by_year(
        methods, years, target_y, target_year, coarse, probability, target_routes
    )
    pooled_rows = []
    for method in methods:
        route_key = method.split("__", 1)[0] if "__" in method else None
        route = target_routes[route_key] if route_key in target_routes else np.zeros(len(target_y), dtype=bool)
        pooled_rows.append({
            **pairz_exp.metric_row(method, target_y, probability[method], coarse, route),
            "years": "|".join(map(str, years)),
            "diagnostic_only": True,
        })
    mean_rows = []
    for method in methods:
        rows = [row for row in summary_rows if row["method"] == method]
        mean_rows.append({
            "method": method,
            "mean_BA": float(np.mean([row["BA"] for row in rows])),
            "min_BA": float(np.min([row["BA"] for row in rows])),
            "std_BA": float(np.std([row["BA"] for row in rows])),
            "mean_AUC": float(np.mean([row["AUC"] for row in rows])),
            "mean_F1": float(np.mean([row["F1"] for row in rows])),
            "mean_fixed": float(np.mean([row["fixed"] for row in rows])),
            "mean_broken": float(np.mean([row["broken"] for row in rows])),
            "mean_net": float(np.mean([row["net"] for row in rows])),
            "diagnostic_only": True,
        })

    comparison_pairs = [
        (route_tags[0] + "__Joint_S3", route_tags[0] + "__Joint_S7"),
        (route_tags[0] + "__Joint_S7", route_tags[0] + "__Ordinary_S7"),
        (route_tags[0] + "__Ordinary_S7", route_tags[0] + "__RSSR_fixed_S7"),
        (route_tags[0] + "__Joint_S7", route_tags[0] + "__RSSR_fixed_S7"),
        (route_tags[0] + "__Joint_S7", route_tags[1] + "__Joint_S7"),
    ]
    pairwise_rows = []
    for year in years:
        selected = target_year == year
        for left, right in comparison_pairs:
            left_correct = (probability[left][selected] >= THRESHOLD) == target_y[selected]
            right_correct = (probability[right][selected] >= THRESHOLD) == target_y[selected]
            left_only, right_only, p_value = pairz_exp.exact_mcnemar(left_correct, right_correct)
            pairwise_rows.append({
                "year": int(year),
                "left": left,
                "right": right,
                "left_only_correct": int(left_only),
                "right_only_correct": int(right_only),
                "net_right_minus_left": int(right_only - left_only),
                "discordant": int(left_only + right_only),
                "mcnemar_exact_p": float(p_value),
                "diagnostic_only": True,
            })

    prediction_rows = []
    for index, row in enumerate(target_rows):
        output = {
            "sha256": row["sha256"],
            "year": int(target_year[index]),
            "label": int(target_y[index]),
            "ambiguity_routed": int(target_routes[route_tags[0]][index]),
            "acn_routed": int(target_routes[route_tags[1]][index]),
            "ambiguity_percentile": float(target_route_data["ambiguity_percentile"][index]),
            "collision_percentile": float(target_route_data["collision_percentile"][index]),
            "novelty_percentile": float(target_route_data["novelty_percentile"][index]),
            "acn_score": float(target_route_data["acn_score"][index]),
            "diagnostic_only": True,
        }
        for method in methods:
            output[method + "__prob"] = float(probability[method][index])
        prediction_rows.append(output)

    write_csv(args.output / "future_summary_diagnostic.csv", summary_rows)
    write_csv(args.output / "future_mean_summary_diagnostic.csv", mean_rows)
    write_csv(args.output / "future_pooled_summary_diagnostic.csv", pooled_rows)
    write_csv(args.output / "future_pairwise_tests_diagnostic.csv", pairwise_rows)
    write_csv(args.output / "future_predictions_diagnostic.csv", prediction_rows)
    write_csv(args.output / "fit_audit.csv", fit_rows)
    write_csv(args.output / "source_stability_feature_audit.csv", stability_rows)
    write_csv(args.output / "future_inter_construction_audit.csv", construction_rows)
    metadata = {
        "experiment": "s7-interz-future-diagnostic-v1",
        "diagnostic_only": True,
        "diagnostic_reason": "2018-2021 labels were viewed previously elsewhere in the project",
        "source_year": 2017,
        "source_apks": 799,
        "future_years": years,
        "future_apks": int(len(target_rows)),
        "future_labels_used_for_training_routing_or_selection": False,
        "compact": "C76_allratio_ratio52",
        "compact_dim": int(len(compact_names)),
        "source_selected_main_candidate": route_tags[0] + "__Joint_S7",
        "S3": "Stats120+PairRaw465",
        "S7": "Stats120+PairRaw465+InterZClip5",
        "S7_dim": 1546,
        "usable_S7_dim": usable,
        "source_reference": source_reference,
        "future_inter_cache": future_cache_audit,
        "routing": {
            "budget": float(args.budget),
            "calibration": (
                "risks normalized against fixed source OOF references, then one top-budget "
                "selection over the entire requested unlabeled future batch; no per-year reranking"
            ),
            "eta": float(args.eta),
            "knn_k": int(args.knn_k),
        },
        "rssr_side_control": {
            "gamma": float(args.rssr_gamma),
            "l2": float(args.rssr_l2),
            "resamples": int(args.stability_resamples),
            "subsample_fraction": float(args.subsample_fraction),
            "not_used_for_selection": True,
        },
        "selection_statement": "No model or hyperparameter is selected using future labels.",
        "formal_claim_statement": "An untouched 2022+ or independent dataset is still required.",
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Done:", args.output)


if __name__ == "__main__":
    main()
