#!/usr/bin/env python3
"""Strict 2017 clipped Intra-Z / directed Inter-Z structure ablation.

This is the final source-only falsification test for the degree-preserving
null-model structure idea.  C76, five outer folds, Ambiguity20/ACN20 routes,
JointLogistic, the anchored route-conditioned residual (eta=0.25, L2=0.03),
and the Monte Carlo budget are fixed.  Only the second-layer representation is
changed:

  S3: Stats120 + PairRaw465                         (current reference)
  S4: Stats120 + clip(IntraZ465, -5, 5)
  S5: Stats120 + clip(IntraZ465) + clip(InterZ961)
  S7: Stats120 + PairRaw465 + clip(InterZ961)

For InterZ, the retained directed Host-to-Host graph E_HH is fixed exactly.
Only the Host--Role incidence E_HR is degree-preservingly rewired, so every
Host degree, Role degree, and Host-call in/out degree remains unchanged.  No
future rows or labels are loaded.

Copy this file plus the two helper files named in README into ``scripts/`` and
run from the tasa-droid repository root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_host_role_null_model_assets as host_asset
import run_acn_routing_ablation_2017 as acn
import run_allratio70_vs_nodrop76_experiment as audit
import run_anchored_route_conditioned_residual_2017 as anchored
import run_pairz_structure_ablation_2017 as pairz_exp


THRESHOLD = 0.5
VARIANT = "allratio76_ratio52"
RAW_TAG = "S3_Stats120_PairRaw465"
INTRA_TAG = "S4_Stats120_IntraZClip5"
INTRA_INTER_TAG = "S5_Stats120_IntraZClip5_InterZClip5"
RAW_INTER_TAG = "S7_Stats120_PairRaw465_InterZClip5"
ROLE_COUNT = 31
INTER_DIM = ROLE_COUNT * ROLE_COUNT


def inter_counts(host_roles, host_host_edges):
    result = np.zeros(INTER_DIM, dtype=np.float64)
    for source, target in host_host_edges:
        source_roles = host_roles.get(int(source), ())
        target_roles = host_roles.get(int(target), ())
        for role_a in source_roles:
            for role_b in target_roles:
                result[int(role_a) * ROLE_COUNT + int(role_b)] += 1.0
    return result


def z_score(observed, samples):
    if samples:
        matrix = np.asarray(samples, dtype=np.float64)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=0)
    else:
        mean = np.zeros_like(observed, dtype=np.float64)
        std = np.zeros_like(observed, dtype=np.float64)
    z = np.divide(
        observed - mean,
        std,
        out=np.zeros_like(observed, dtype=np.float64),
        where=std > 0.0,
    )
    return z, mean, std


def multiscale_null(host_roles, host_host_edges, pair_index, args, seed):
    observed_edges = host_asset.incidence_edges(host_roles)
    original_degrees = host_asset.degrees(observed_edges)
    observed_intra = host_asset.pair_counts(host_roles, pair_index)
    observed_inter = inter_counts(host_roles, host_host_edges)
    rng = np.random.RandomState(int(seed))
    intra_samples = []
    inter_samples = []
    target_successes = max(1, int(args.swaps_per_edge) * len(observed_edges))
    total_successes = 0
    total_attempts = 0
    degree_preserved = True
    for _ in range(int(args.null_samples)):
        randomized_edges, successes, attempts = host_asset.swapped_incidence(
            observed_edges,
            rng,
            target_successes,
            max_attempt_factor=40,
        )
        total_successes += int(successes)
        total_attempts += int(attempts)
        degree_preserved = (
            degree_preserved
            and host_asset.degrees(randomized_edges) == original_degrees
        )
        randomized_roles = host_asset.edge_set_to_host_roles(randomized_edges)
        intra_samples.append(host_asset.pair_counts(randomized_roles, pair_index))
        inter_samples.append(inter_counts(randomized_roles, host_host_edges))
    intra_z, intra_mean, intra_std = z_score(observed_intra, intra_samples)
    inter_z, inter_mean, inter_std = z_score(observed_inter, inter_samples)
    return {
        "observed_intra": observed_intra,
        "intra_mean": intra_mean,
        "intra_std": intra_std,
        "intra_z": intra_z,
        "observed_inter": observed_inter,
        "inter_mean": inter_mean,
        "inter_std": inter_std,
        "inter_z": inter_z,
        "degree_preserved": bool(degree_preserved),
        "successful_swaps": int(total_successes),
        "attempted_swaps": int(total_attempts),
    }


def build_multiscale_features(cohort_rows, cache, pair_raw_reference, args):
    _, pair_index = host_asset.pair_schema()
    n = len(cohort_rows)
    intra_z = np.zeros((n, 465), dtype=np.float64)
    inter_raw = np.zeros((n, INTER_DIM), dtype=np.float64)
    inter_z = np.zeros((n, INTER_DIM), dtype=np.float64)
    rows = []
    for index, row in enumerate(cohort_rows, 1):
        sha = pairz_exp.norm_sha(row["sha256"])
        if sha not in cache:
            raise RuntimeError("structure cache missing: " + sha)
        asset = host_asset.reconstruct_host_roles(cache[sha]["payload"])
        result = multiscale_null(
            asset["host_roles"],
            asset["host_host_edges"],
            pair_index,
            args,
            pairz_exp.deterministic_apk_seed(args.null_seed, sha),
        )
        if not np.array_equal(result["observed_intra"], pair_raw_reference[index - 1]):
            raise RuntimeError("PairRaw reconstruction mismatch: " + sha)
        if not result["degree_preserved"]:
            raise RuntimeError("degree-preserving null failed: " + sha)
        intra_z[index - 1] = result["intra_z"]
        inter_raw[index - 1] = result["observed_inter"]
        inter_z[index - 1] = result["inter_z"]
        attempts = result["attempted_swaps"]
        successes = result["successful_swaps"]
        rows.append({
            "sha256": sha,
            "behavior_host_count": int(len(asset["host_roles"])),
            "host_role_incidence_count": int(
                len(host_asset.incidence_edges(asset["host_roles"]))
            ),
            "host_host_directed_edge_count": int(len(asset["host_host_edges"])),
            "null_samples": int(args.null_samples),
            "null_successful_swaps": successes,
            "null_attempted_swaps": attempts,
            "null_swap_success_rate": float(successes / attempts) if attempts else 0.0,
            "degree_preserved": 1,
            "host_graph_fixed": 1,
            "intra_variable_dims": int(np.count_nonzero(result["intra_std"] > 0.0)),
            "intra_nonzero_z_dims": int(np.count_nonzero(result["intra_z"])),
            "intra_clipped_dims": int(np.sum(np.abs(result["intra_z"]) > args.z_clip)),
            "intra_max_abs_z": float(np.max(np.abs(result["intra_z"]))),
            "inter_raw_nonzero_dims": int(np.count_nonzero(result["observed_inter"])),
            "inter_variable_dims": int(np.count_nonzero(result["inter_std"] > 0.0)),
            "inter_nonzero_z_dims": int(np.count_nonzero(result["inter_z"])),
            "inter_clipped_dims": int(np.sum(np.abs(result["inter_z"]) > args.z_clip)),
            "inter_max_abs_z": float(np.max(np.abs(result["inter_z"]))),
        })
        if index % args.progress_every == 0 or index == n:
            print("  Intra/Inter Z:", index, "/", n, flush=True)
    if np.any(~np.isfinite(intra_z)) or np.any(~np.isfinite(inter_z)):
        raise RuntimeError("non-finite multiscale Z")
    return intra_z, inter_raw, inter_z, rows


def check_pairz_feature_reference(path, cohort_rows, pair_raw, intra_z):
    path = Path(path)
    if not path.exists():
        return {"available": False, "path": str(path)}
    reference = np.load(path)
    reference_sha = reference["sha256"].astype(str)
    current_sha = np.asarray([
        pairz_exp.norm_sha(row["sha256"]) for row in cohort_rows
    ])
    result = {
        "available": True,
        "path": str(path),
        "rows": int(len(current_sha)),
        "sha_order_equal": bool(np.array_equal(reference_sha, current_sha)),
        "pair_raw_max_abs_difference": float(np.max(np.abs(reference["pair_raw"] - pair_raw))),
        "intra_z_max_abs_difference": float(np.max(np.abs(reference["pair_z"] - intra_z))),
    }
    if (
        not result["sha_order_equal"]
        or result["pair_raw_max_abs_difference"] > 0.0
        or result["intra_z_max_abs_difference"] > 1e-12
    ):
        raise RuntimeError("multiscale null does not reproduce prior PairZ: " + repr(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=Path("artifacts/p1-features/features.csv"),
    )
    parser.add_argument(
        "--schema", type=Path,
        default=Path("artifacts/p1-features/feature_schema.json"),
    )
    parser.add_argument(
        "--cache", type=Path,
        default=Path(
            "artifacts/p25-endpoint-cache-full-cap256/"
            "endpoint_centered_tensor_cache.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--p72-oof", type=Path,
        default=Path(
            "artifacts/p72-verified-dependency-source-ablation/"
            "p72_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--anchored-reference", type=Path,
        default=Path(
            "artifacts/anchored-route-conditioned-residual-2017-v1/"
            "source_predictions.csv"
        ),
    )
    parser.add_argument(
        "--pairz-feature-reference", type=Path,
        default=Path(
            "artifacts/pairz-structure-ablation-2017-v1-full/"
            "pairz_features.npz"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1"),
    )
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260827)
    parser.add_argument("--residual-l2", type=float, default=0.03)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-8)
    parser.add_argument("--null-samples", type=int, default=64)
    parser.add_argument("--swaps-per-edge", type=int, default=10)
    parser.add_argument("--null-seed", type=int, default=20260827)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if not 0.0 < args.budget < 1.0:
        raise SystemExit("--budget must be in (0,1)")
    if args.knn_k < 2 or args.inner_folds < 2:
        raise SystemExit("invalid k/fold setting")
    if args.residual_l2 <= 0.0:
        raise SystemExit("--residual-l2 must be positive")
    if not 0.0 <= args.eta < 1.0:
        raise SystemExit("--eta must be in [0,1)")
    if args.null_samples < 2 or args.swaps_per_edge < 1:
        raise SystemExit("invalid null-model setting")
    if args.z_clip <= 0.0:
        raise SystemExit("--z-clip must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    (
        manifest,
        cohort_rows,
        _,
        compact_by_sha,
        compact_schema,
        original_names,
    ) = audit.load_inputs(args)
    cache = audit.exp.read_cache(args.cache)
    print("Reconstructing source Stats120+PairRaw465...")
    raw_structure, _ = audit.future.reconstruct_structure(
        cohort_rows, cache, int(audit.exp.ROLE_VOCABULARY_SIZE)
    )
    if raw_structure.shape != (799, 585):
        raise RuntimeError("expected source structure (799,585), got " + repr(raw_structure.shape))
    stats = raw_structure[:, :120]
    pair_raw = raw_structure[:, 120:]
    print("Constructing label-free Intra-Z and directed Inter-Z...")
    intra_z, inter_raw, inter_z, construction_rows = build_multiscale_features(
        cohort_rows, cache, pair_raw, args
    )
    pairz_reference = check_pairz_feature_reference(
        args.pairz_feature_reference, cohort_rows, pair_raw, intra_z
    )
    intra_clip = np.clip(intra_z, -args.z_clip, args.z_clip)
    inter_clip = np.clip(inter_z, -args.z_clip, args.z_clip)
    structures = {
        RAW_TAG: raw_structure,
        INTRA_TAG: np.concatenate([stats, intra_clip], axis=1),
        INTRA_INTER_TAG: np.concatenate([stats, intra_clip, inter_clip], axis=1),
        RAW_INTER_TAG: np.concatenate([stats, pair_raw, inter_clip], axis=1),
    }
    expected_dims = {
        RAW_TAG: 585,
        INTRA_TAG: 585,
        INTRA_INTER_TAG: 1546,
        RAW_INTER_TAG: 1546,
    }
    for tag, matrix in structures.items():
        if matrix.shape != (799, expected_dims[tag]):
            raise RuntimeError("bad structure shape for " + tag + ": " + repr(matrix.shape))

    y = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    unique_folds = sorted(np.unique(folds))
    n = len(y)
    budget_tag = str(int(round(100.0 * args.budget)))
    route_tags = ["A" + budget_tag, "ACN" + budget_tag]
    residual_tag = "RouteResidual_eta" + format(args.eta, "g")
    methods = ["C76"]
    for structure_tag in structures:
        for route_tag in route_tags:
            methods.extend([
                pairz_exp.method_name(route_tag, "Joint", structure_tag),
                pairz_exp.method_name(route_tag, residual_tag, structure_tag),
            ])
    probabilities = {method: np.full(n, np.nan) for method in methods}
    ambiguity_routes = np.zeros(n, dtype=bool)
    acn_routes = np.zeros(n, dtype=bool)
    risk_outputs = {
        "ambiguity_percentile": np.full(n, np.nan),
        "collision_percentile": np.full(n, np.nan),
        "novelty_percentile": np.full(n, np.nan),
        "acn_score": np.full(n, np.nan),
    }
    fit_rows = []
    fold_rows = []

    for fold in unique_folds:
        train_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        train_rows = [cohort_rows[i] for i in train_idx]
        val_rows = [cohort_rows[i] for i in val_idx]
        x_train, x_val, compact_names = audit.compact_matrix_variant(
            VARIANT,
            train_rows,
            val_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        coarse, _, _ = audit.p72.fit_predict(
            x_train, y[train_idx], x_val, seed=int(fold) - 1
        )
        (
            train_oof_probability,
            train_ambiguity_raw,
            train_collision_raw,
            train_novelty_raw,
        ) = anchored.crossfit_base_and_risks(
            train_rows,
            y[train_idx],
            compact_by_sha,
            compact_schema,
            original_names,
            args.inner_folds,
            args.split_seed + 1000 * int(fold),
            args.split_seed + 10000 * int(fold),
            args.knn_k,
        )
        train_route_data = anchored.calibrated_routes(
            train_ambiguity_raw,
            train_collision_raw,
            train_novelty_raw,
            train_ambiguity_raw,
            train_collision_raw,
            train_novelty_raw,
            args.budget,
        )
        outer_collision, outer_novelty, _, _ = acn.neighborhood_risks(
            x_train, y[train_idx], x_val, args.knn_k
        )
        val_route_data = anchored.calibrated_routes(
            train_ambiguity_raw,
            train_collision_raw,
            train_novelty_raw,
            acn.ambiguity(coarse),
            outer_collision,
            outer_novelty,
            args.budget,
        )
        train_routes = {
            route_tags[0]: train_route_data["ambiguity_route"],
            route_tags[1]: train_route_data["acn_route"],
        }
        val_routes = {
            route_tags[0]: val_route_data["ambiguity_route"],
            route_tags[1]: val_route_data["acn_route"],
        }
        probabilities["C76"][val_idx] = coarse

        for structure_tag, structure in structures.items():
            joint = audit.fit_config(
                x_train,
                x_val,
                structure[train_idx],
                structure[val_idx],
                y[train_idx],
                seed=int(fold) - 1,
            )
            names = [
                structure_tag + "__" + format(i, "04d")
                for i in range(structure.shape[1])
            ]
            s_train, s_val, structure_audit = audit.p72.standardize_added(
                structure[train_idx], structure[val_idx], names
            )
            usable = int(sum(
                not bool(item["constant_in_train"]) for item in structure_audit
            ))
            for route_tag in route_tags:
                route_train = train_routes[route_tag]
                route_val = val_routes[route_tag]
                joint_name = pairz_exp.method_name(route_tag, "Joint", structure_tag)
                probabilities[joint_name][val_idx] = acn.routed_probability(
                    coarse, joint, route_val
                )
                weights = np.where(route_train, 1.0, float(args.eta))
                beta, diagnostic = anchored.fit_offset_residual(
                    s_train,
                    y[train_idx],
                    train_oof_probability,
                    weights,
                    args.residual_l2,
                    args.max_iterations,
                    args.optimizer_tolerance,
                )
                delta = s_val @ beta
                enhanced = anchored.sigmoid(anchored.logit(coarse) + delta)
                residual_name = pairz_exp.method_name(
                    route_tag, residual_tag, structure_tag
                )
                probabilities[residual_name][val_idx] = acn.routed_probability(
                    coarse, enhanced, route_val
                )
                fit_rows.append({
                    "fold": int(fold),
                    "route": route_tag,
                    "model": residual_tag,
                    "structure": structure_tag,
                    "structure_dim": int(structure.shape[1]),
                    "usable_structure_dim": usable,
                    "eta": float(args.eta),
                    "l2": float(args.residual_l2),
                    "train_routed_n": int(np.sum(route_train)),
                    "train_weight_sum": float(np.sum(weights)),
                    "eval_delta_mean": float(np.mean(delta)),
                    "eval_delta_std": float(np.std(delta)),
                    "eval_delta_max_abs": float(np.max(np.abs(delta))),
                    **diagnostic,
                })
            fold_rows.append({
                "fold": int(fold),
                "structure": structure_tag,
                "structure_dim": int(structure.shape[1]),
                "usable_structure_dim": usable,
                "compact_dim": int(len(compact_names)),
                "outer_train_n": int(len(train_idx)),
                "outer_val_n": int(len(val_idx)),
                "ambiguity_train_route_n": int(np.sum(train_routes[route_tags[0]])),
                "acn_train_route_n": int(np.sum(train_routes[route_tags[1]])),
                "ambiguity_val_route_n": int(np.sum(val_routes[route_tags[0]])),
                "acn_val_route_n": int(np.sum(val_routes[route_tags[1]])),
            })
        ambiguity_routes[val_idx] = val_routes[route_tags[0]]
        acn_routes[val_idx] = val_routes[route_tags[1]]
        for key in risk_outputs:
            risk_outputs[key][val_idx] = val_route_data[key]
        print("fold", int(fold), "done", flush=True)

    if any(np.any(~np.isfinite(value)) for value in probabilities.values()):
        raise RuntimeError("incomplete OOF probability")
    if any(np.any(~np.isfinite(value)) for value in risk_outputs.values()):
        raise RuntimeError("incomplete routing risk")

    route_map = {route_tags[0]: ambiguity_routes, route_tags[1]: acn_routes}
    zero_route = np.zeros(n, dtype=bool)
    summary_rows = [
        pairz_exp.metric_row(
            "C76", y, probabilities["C76"], probabilities["C76"], zero_route
        )
    ]
    per_fold_rows = []
    for fold in unique_folds:
        idx = folds == fold
        per_fold_rows.append({
            **pairz_exp.metric_row(
                "C76", y[idx], probabilities["C76"][idx],
                probabilities["C76"][idx], zero_route[idx]
            ),
            "fold": int(fold),
        })
    for method in methods[1:]:
        route_tag = method.split("__", 1)[0]
        route = route_map[route_tag]
        summary_rows.append(pairz_exp.metric_row(
            method, y, probabilities[method], probabilities["C76"], route
        ))
        for fold in unique_folds:
            idx = folds == fold
            per_fold_rows.append({
                **pairz_exp.metric_row(
                    method,
                    y[idx],
                    probabilities[method][idx],
                    probabilities["C76"][idx],
                    route[idx],
                ),
                "fold": int(fold),
            })

    comparison_pairs = []
    for route_tag in route_tags:
        for model in ("Joint", residual_tag):
            raw_name = pairz_exp.method_name(route_tag, model, RAW_TAG)
            intra_name = pairz_exp.method_name(route_tag, model, INTRA_TAG)
            intra_inter_name = pairz_exp.method_name(
                route_tag, model, INTRA_INTER_TAG
            )
            raw_inter_name = pairz_exp.method_name(route_tag, model, RAW_INTER_TAG)
            comparison_pairs.extend([
                (raw_name, intra_name),
                (intra_name, intra_inter_name),
                (raw_name, raw_inter_name),
                (intra_inter_name, raw_inter_name),
            ])
    for structure_tag in structures:
        comparison_pairs.extend([
            (
                pairz_exp.method_name(route_tags[0], "Joint", structure_tag),
                pairz_exp.method_name(route_tags[0], residual_tag, structure_tag),
            ),
            (
                pairz_exp.method_name(route_tags[1], "Joint", structure_tag),
                pairz_exp.method_name(route_tags[1], residual_tag, structure_tag),
            ),
            (
                pairz_exp.method_name(route_tags[0], residual_tag, structure_tag),
                pairz_exp.method_name(route_tags[1], residual_tag, structure_tag),
            ),
        ])
    pairwise_rows = []
    seen = set()
    for left, right in comparison_pairs:
        if (left, right) in seen:
            continue
        seen.add((left, right))
        left_correct = (probabilities[left] >= THRESHOLD) == y
        right_correct = (probabilities[right] >= THRESHOLD) == y
        left_only, right_only, p_value = pairz_exp.exact_mcnemar(
            left_correct, right_correct
        )
        pairwise_rows.append({
            "left": left,
            "right": right,
            "left_only_correct": int(left_only),
            "right_only_correct": int(right_only),
            "net_right_minus_left": int(right_only - left_only),
            "discordant": int(left_only + right_only),
            "mcnemar_exact_p": float(p_value),
        })

    prediction_rows = []
    for i, row in enumerate(manifest):
        output = {
            "sha256": row["sha256"],
            "label": int(y[i]),
            "fold": int(folds[i]),
            "ambiguity_routed": int(ambiguity_routes[i]),
            "acn_routed": int(acn_routes[i]),
            **{key: float(value[i]) for key, value in risk_outputs.items()},
        }
        for method in methods:
            output[method + "__prob"] = float(probabilities[method][i])
        prediction_rows.append(output)

    old_raw_tag = "Stats120_PairRaw465"
    raw_probability_aliases = {"C76": probabilities["C76"]}
    for route_tag in route_tags:
        for model in ("Joint", residual_tag):
            raw_probability_aliases[
                pairz_exp.method_name(route_tag, model, old_raw_tag)
            ] = probabilities[pairz_exp.method_name(route_tag, model, RAW_TAG)]
    anchored_reference = pairz_exp.check_previous_reference(
        args.anchored_reference,
        manifest,
        raw_probability_aliases,
        {"ambiguity": ambiguity_routes, "acn": acn_routes},
        budget_tag,
        args.eta,
    )
    pairz_exp.write_csv(args.output / "source_summary.csv", summary_rows)
    pairz_exp.write_csv(args.output / "source_per_fold.csv", per_fold_rows)
    pairz_exp.write_csv(args.output / "source_predictions.csv", prediction_rows)
    pairz_exp.write_csv(args.output / "pairwise_tests.csv", pairwise_rows)
    pairz_exp.write_csv(args.output / "fit_audit.csv", fit_rows)
    pairz_exp.write_csv(args.output / "fold_audit.csv", fold_rows)
    pairz_exp.write_csv(args.output / "multiscale_construction_audit.csv", construction_rows)
    np.savez_compressed(
        args.output / "multiscale_features.npz",
        sha256=np.asarray([pairz_exp.norm_sha(row["sha256"]) for row in cohort_rows]),
        pair_raw=pair_raw,
        intra_z=intra_z,
        intra_z_clipped=intra_clip,
        inter_raw=inter_raw,
        inter_z=inter_z,
        inter_z_clipped=inter_clip,
    )
    reference = {
        "anchored_raw_branch": anchored_reference,
        "prior_pairz_features": pairz_reference,
    }
    (args.output / "reference_check.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "experiment": "clipped-intra-inter-structure-ablation-2017-v1",
        "source_only": True,
        "source_apks": int(n),
        "future_rows_loaded": False,
        "test_label_used_by_structure": False,
        "test_label_used_by_router": False,
        "coarse": "C76 allratio76_ratio52",
        "structures": {tag: int(matrix.shape[1]) for tag, matrix in structures.items()},
        "null_model": {
            "randomized": "binary Host-Role incidence E_HR",
            "fixed": "directed Host-to-Host graph E_HH",
            "preserved_exactly": [
                "each Host role-degree",
                "each Role host-degree",
                "each Host-call edge",
                "Host-call in/out degree",
            ],
            "null_samples_per_apk": int(args.null_samples),
            "successful_swaps_target_per_sample": (
                str(args.swaps_per_edge) + " x incidence_edge_count"
            ),
            "zero_variance_policy": "Z=0",
            "z_clip": [-float(args.z_clip), float(args.z_clip)],
        },
        "intra_definition": (
            "sum_h I[a in R(h) and b in R(h)], unordered a<b, dimension 465"
        ),
        "inter_definition": (
            "sum_(hi->hj) I[a in R(hi) and b in R(hj)], directed a,b, dimension 961"
        ),
        "routes": route_tags,
        "budget": float(args.budget),
        "knn_k": int(args.knn_k),
        "inner_folds": int(args.inner_folds),
        "residual_eta": float(args.eta),
        "residual_l2": float(args.residual_l2),
        "decision_rule": (
            "Retain clipped zero-model structure only if it improves global BA/net without "
            "fold-local collapse. Retain Inter only if it adds stable benefit over its matched "
            "Intra or PairRaw base. Ambiguity and ACN remain separate interaction tests."
        ),
        "reference_reproduction": reference,
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Done:", args.output)


if __name__ == "__main__":
    main()
