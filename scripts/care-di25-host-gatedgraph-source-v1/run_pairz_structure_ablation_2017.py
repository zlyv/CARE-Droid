#!/usr/bin/env python3
"""Strict source-only PairRaw/PairZ structure ablation for the two-layer model.

The first layer, outer folds, Ambiguity20/ACN20 routes, residual objective,
eta=0.25, and L2=0.03 are frozen from the previous experiments.  This script
changes only the second-layer structure representation:

  * Stats120 + PairRaw465 (the current reference);
  * Stats120 + PairZ465 (degree-preserving null corrected);
  * Stats120 + PairRaw465 + PairZ465 (incremental corrected evidence).

PairZ is constructed independently per APK without labels.  The binary
Host--Role double-edge-swap null preserves every Host degree and Role degree.
No future rows or labels are loaded.

Copy this file and ``audit_host_role_null_model_assets.py`` into ``scripts/``
and run from the tasa-droid repository root.
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

import audit_host_role_null_model_assets as host_asset
import run_acn_routing_ablation_2017 as acn
import run_allratio70_vs_nodrop76_experiment as audit
import run_anchored_route_conditioned_residual_2017 as anchored


THRESHOLD = 0.5
VARIANT = "allratio76_ratio52"
RAW_TAG = "Stats120_PairRaw465"
Z_TAG = "Stats120_PairZ465"
RAW_Z_TAG = "Stats120_PairRaw465_PairZ465"


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


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm_sha(value):
    return str(value or "").strip().casefold()


def deterministic_apk_seed(base_seed, sha):
    return int(base_seed) ^ int(norm_sha(sha)[:8], 16)


def build_pairz(cohort_rows, cache, pair_raw_reference, args):
    pairs, pair_index = host_asset.pair_schema()
    pair_z = np.zeros_like(pair_raw_reference, dtype=np.float64)
    construction_rows = []
    for index, row in enumerate(cohort_rows, 1):
        sha = norm_sha(row["sha256"])
        if sha not in cache:
            raise RuntimeError("structure cache missing: " + sha)
        asset = host_asset.reconstruct_host_roles(cache[sha]["payload"])
        independent_raw = host_asset.pair_counts(asset["host_roles"], pair_index)
        if not np.array_equal(independent_raw, pair_raw_reference[index - 1]):
            raise RuntimeError("PairRaw reconstruction mismatch: " + sha)
        null = host_asset.null_pair_z(
            asset["host_roles"],
            pair_index,
            args.null_samples,
            args.swaps_per_edge,
            deterministic_apk_seed(args.null_seed, sha),
        )
        if not null["degree_preserved"]:
            raise RuntimeError("degree-preserving null failed: " + sha)
        pair_z[index - 1] = null["z"]
        attempts = int(null["attempted_swaps"])
        successes = int(null["successful_swaps"])
        construction_rows.append({
            "sha256": sha,
            "behavior_host_count": int(len(asset["host_roles"])),
            "host_role_incidence_count": int(
                len(host_asset.incidence_edges(asset["host_roles"]))
            ),
            "null_samples": int(args.null_samples),
            "null_successful_swaps": successes,
            "null_attempted_swaps": attempts,
            "null_swap_success_rate": float(successes / attempts) if attempts else 0.0,
            "null_degree_preserved": 1,
            "pairraw_nonzero_dims": int(np.count_nonzero(independent_raw)),
            "pairz_variable_dims": int(np.count_nonzero(null["std"] > 0.0)),
            "pairz_nonzero_dims": int(np.count_nonzero(null["z"])),
            "pairz_max_abs": float(np.max(np.abs(null["z"]))),
            "host_host_directed_edge_count": int(len(asset["host_host_edges"])),
        })
        if index % args.progress_every == 0 or index == len(cohort_rows):
            print("  PairZ:", index, "/", len(cohort_rows), flush=True)
    if np.any(~np.isfinite(pair_z)):
        raise RuntimeError("non-finite PairZ")
    return pair_z, construction_rows


def method_name(route, model, structure_tag):
    return route + "__" + model + "__" + structure_tag


def metric_row(method, y, probability, coarse, route):
    return acn.metric_row(method, y, probability, coarse, route)


def exact_mcnemar(left_correct, right_correct):
    return audit.exact_mcnemar_p(
        np.asarray(left_correct, dtype=bool),
        np.asarray(right_correct, dtype=bool),
    )


def check_previous_reference(path, manifest, probabilities, routes, budget_tag, eta):
    path = Path(path)
    if not path.exists():
        return {"available": False, "path": str(path)}
    rows = read_csv(path)
    by_sha = {norm_sha(row["sha256"]): row for row in rows}
    missing = [row["sha256"] for row in manifest if norm_sha(row["sha256"]) not in by_sha]
    if missing:
        raise RuntimeError("previous reference missing SHAs: " + repr(missing[:5]))
    mapping = {
        "C76": "C76__prob",
        method_name("A" + budget_tag, "Joint", RAW_TAG): (
            "A" + budget_tag + "_JointLogistic__prob"
        ),
        method_name("ACN" + budget_tag, "Joint", RAW_TAG): (
            "ACN" + budget_tag + "_JointLogistic__prob"
        ),
        method_name("A" + budget_tag, "RouteResidual_eta" + format(eta, "g"), RAW_TAG): (
            "A" + budget_tag + "_RouteResidual_eta" + format(eta, "g") + "__prob"
        ),
        method_name("ACN" + budget_tag, "RouteResidual_eta" + format(eta, "g"), RAW_TAG): (
            "ACN" + budget_tag + "_RouteResidual_eta" + format(eta, "g") + "__prob"
        ),
    }
    result = {"available": True, "path": str(path), "rows": int(len(rows))}
    for new_name, old_column in mapping.items():
        differences = [
            abs(float(by_sha[norm_sha(row["sha256"])][old_column]) - probabilities[new_name][i])
            for i, row in enumerate(manifest)
        ]
        result[new_name + "__max_abs_probability_difference"] = float(max(differences))
    ambiguity_column = "ambiguity_routed"
    acn_column = "acn_routed"
    result["ambiguity_route_mismatches"] = int(sum(
        int(by_sha[norm_sha(row["sha256"])][ambiguity_column]) != int(routes["ambiguity"][i])
        for i, row in enumerate(manifest)
    ))
    result["acn_route_mismatches"] = int(sum(
        int(by_sha[norm_sha(row["sha256"])][acn_column]) != int(routes["acn"][i])
        for i, row in enumerate(manifest)
    ))
    probability_diffs = [
        value for key, value in result.items()
        if key.endswith("__max_abs_probability_difference")
    ]
    if max(probability_diffs, default=0.0) > 1e-10:
        raise RuntimeError("PairRaw branch does not reproduce previous residual experiment")
    if result["ambiguity_route_mismatches"] or result["acn_route_mismatches"]:
        raise RuntimeError("routing does not reproduce previous residual experiment")
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
        "--previous-reference", type=Path,
        default=Path(
            "artifacts/anchored-route-conditioned-residual-2017-v1/"
            "source_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/pairz-structure-ablation-2017-v1"),
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
    print("Constructing label-free degree-preserving PairZ465...")
    pair_z, pairz_audit_rows = build_pairz(cohort_rows, cache, pair_raw, args)

    structures = {
        RAW_TAG: raw_structure,
        Z_TAG: np.concatenate([stats, pair_z], axis=1),
        RAW_Z_TAG: np.concatenate([stats, pair_raw, pair_z], axis=1),
    }
    expected_dims = {RAW_TAG: 585, Z_TAG: 585, RAW_Z_TAG: 1050}
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
        methods.extend([
            method_name(route_tags[0], "Joint", structure_tag),
            method_name(route_tags[1], "Joint", structure_tag),
            method_name(route_tags[0], residual_tag, structure_tag),
            method_name(route_tags[1], residual_tag, structure_tag),
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
                joint_name = method_name(route_tag, "Joint", structure_tag)
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
                residual_name = method_name(route_tag, residual_tag, structure_tag)
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
    summary_rows = [metric_row("C76", y, probabilities["C76"], probabilities["C76"], zero_route)]
    per_fold_rows = []
    for fold in unique_folds:
        idx = folds == fold
        per_fold_rows.append({
            **metric_row(
                "C76", y[idx], probabilities["C76"][idx],
                probabilities["C76"][idx], zero_route[idx]
            ),
            "fold": int(fold),
        })
    for method in methods[1:]:
        route_tag = method.split("__", 1)[0]
        route = route_map[route_tag]
        summary_rows.append(metric_row(
            method, y, probabilities[method], probabilities["C76"], route
        ))
        for fold in unique_folds:
            idx = folds == fold
            per_fold_rows.append({
                **metric_row(
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
            raw_name = method_name(route_tag, model, RAW_TAG)
            z_name = method_name(route_tag, model, Z_TAG)
            raw_z_name = method_name(route_tag, model, RAW_Z_TAG)
            comparison_pairs.extend([
                (raw_name, z_name),
                (raw_name, raw_z_name),
                (z_name, raw_z_name),
            ])
    for structure_tag in structures:
        comparison_pairs.extend([
            (
                method_name(route_tags[0], "Joint", structure_tag),
                method_name(route_tags[0], residual_tag, structure_tag),
            ),
            (
                method_name(route_tags[1], "Joint", structure_tag),
                method_name(route_tags[1], residual_tag, structure_tag),
            ),
            (
                method_name(route_tags[0], residual_tag, structure_tag),
                method_name(route_tags[1], residual_tag, structure_tag),
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
        left_only, right_only, p_value = exact_mcnemar(left_correct, right_correct)
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

    reference = check_previous_reference(
        args.previous_reference,
        manifest,
        probabilities,
        {"ambiguity": ambiguity_routes, "acn": acn_routes},
        budget_tag,
        args.eta,
    )
    write_csv(args.output / "source_summary.csv", summary_rows)
    write_csv(args.output / "source_per_fold.csv", per_fold_rows)
    write_csv(args.output / "source_predictions.csv", prediction_rows)
    write_csv(args.output / "pairwise_tests.csv", pairwise_rows)
    write_csv(args.output / "fit_audit.csv", fit_rows)
    write_csv(args.output / "fold_audit.csv", fold_rows)
    write_csv(args.output / "pairz_construction_audit.csv", pairz_audit_rows)
    np.savez_compressed(
        args.output / "pairz_features.npz",
        sha256=np.asarray([norm_sha(row["sha256"]) for row in cohort_rows]),
        pair_raw=pair_raw,
        pair_z=pair_z,
    )
    (args.output / "reference_check.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "experiment": "pairz-structure-ablation-2017-v1",
        "source_only": True,
        "source_apks": int(n),
        "future_rows_loaded": False,
        "test_label_used_by_structure": False,
        "test_label_used_by_router": False,
        "coarse": "C76 allratio76_ratio52",
        "structures": {tag: int(matrix.shape[1]) for tag, matrix in structures.items()},
        "pairz": {
            "role_count": 31,
            "pair_count": 465,
            "null_model": "binary Host-Role bipartite double-edge swap",
            "preserved": ["each Host degree", "each Role degree"],
            "null_samples_per_apk": int(args.null_samples),
            "successful_swaps_target_per_sample": (
                str(args.swaps_per_edge) + " x incidence_edge_count"
            ),
            "zero_variance_policy": "PairZ=0",
            "post_null_transform": "none; fold-local standardization only",
        },
        "routes": route_tags,
        "budget": float(args.budget),
        "knn_k": int(args.knn_k),
        "inner_folds": int(args.inner_folds),
        "residual_eta": float(args.eta),
        "residual_l2": float(args.residual_l2),
        "primary_question": (
            "Under identical folds, routes and learners, does degree-corrected PairZ "
            "replace or incrementally improve PairRaw?"
        ),
        "interpretation_rule": (
            "Prefer a corrected structure only if global BA/net improve without fold-local "
            "collapse; Ambiguity and ACN are both retained as interaction tests."
        ),
        "previous_reference": reference,
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Done:", args.output)


if __name__ == "__main__":
    main()
