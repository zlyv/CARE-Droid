#!/usr/bin/env python3
"""Source-only evidence-aware routing ablation for the frozen C76+A20+S7 line.

The predictive models are held fixed:

* layer 1: C76 (24 non-role dimensions + 52 role-ratio dimensions);
* layer 2: JointLogistic on C76 + S7, where
  S7 = Stats120 + PairRaw465 + InterZ961 clipped to [-5, 5];
* routing budget: a fixed fraction (20% by default).

Only the routing score changes.  The experiment compares:

* A20:       ambiguity only (the frozen reference);
* A20-D:     ambiguity anchored by Global24/Role52 disagreement;
* A20-R:     ambiguity anchored by label-free S7 evidence readiness;
* A20-DR:    ambiguity anchored by both disagreement and readiness.

All predictive routing quantities are cross-fitted inside each outer training
fold.  Readiness uses only structural availability counts and never uses a
label.  Future rows and labels are not loaded.  The optional previous-result
check reproduces the frozen C76/A20/S7 probabilities before accepting results.

Copy this file into ``scripts/`` in the tasa-droid repository and run it from
the repository root.  It expects the existing helper scripts used by the
previous experiments to remain in ``scripts/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_acn_routing_ablation_2017 as acn
import run_allratio70_vs_nodrop76_experiment as audit


THRESHOLD = 0.5
VARIANT = "allratio76_ratio52"
S7_TAG = "Stats120_PairRaw465_InterZClip5"
READINESS_COLUMNS = (
    "active_role_count",
    "behavior_host_count",
    "host_role_incidence_count",
    "host_host_directed_edge_count",
    "pair_raw_nonzero_dims",
)


def norm_sha(value):
    return str(value).strip().lower()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clipped_logit(probability, epsilon=1e-6):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64), epsilon, 1.0 - epsilon
    )
    return np.log(probability / (1.0 - probability))


def view_disagreement(global_probability, role_probability):
    """Absolute log-odds disagreement between two calibrated-by-rank heads."""
    return np.abs(
        clipped_logit(global_probability) - clipped_logit(role_probability)
    )


def split_view_indices(names):
    role = np.asarray([
        name.startswith("role_node__") or name.startswith("role_edge__")
        for name in names
    ], dtype=bool)
    global_view = ~role
    if int(np.sum(global_view)) != 24 or int(np.sum(role)) != 52:
        raise RuntimeError(
            "C76 view split mismatch: expected Global24+Role52, got "
            + repr((int(np.sum(global_view)), int(np.sum(role))))
        )
    return np.flatnonzero(global_view), np.flatnonzero(role)


def load_s7_reference(path, cohort_rows, raw_structure):
    path = Path(path)
    if not path.exists():
        raise RuntimeError("S7 multiscale reference not found: " + str(path))
    with np.load(path) as reference:
        required = {"sha256", "pair_raw", "inter_z_clipped"}
        missing = sorted(required - set(reference.files))
        if missing:
            raise RuntimeError("S7 reference is missing arrays: " + repr(missing))
        expected_sha = np.asarray([norm_sha(row["sha256"]) for row in cohort_rows])
        actual_sha = reference["sha256"].astype(str)
        pair_raw = np.asarray(raw_structure[:, 120:], dtype=np.float64)
        reference_pair = np.asarray(reference["pair_raw"], dtype=np.float64)
        inter_clip = np.asarray(reference["inter_z_clipped"], dtype=np.float64)
        check = {
            "path": str(path),
            "sha_order_equal": bool(np.array_equal(expected_sha, actual_sha)),
            "pair_raw_max_abs_difference": float(
                np.max(np.abs(pair_raw - reference_pair))
            ),
            "inter_clip_finite": bool(np.isfinite(inter_clip).all()),
            "inter_shape": list(inter_clip.shape),
        }
    if (
        not check["sha_order_equal"]
        or check["pair_raw_max_abs_difference"] > 0.0
        or not check["inter_clip_finite"]
        or tuple(check["inter_shape"]) != (799, 961)
    ):
        raise RuntimeError("invalid S7 multiscale reference: " + repr(check))
    s7 = np.concatenate([raw_structure[:, :120], pair_raw, inter_clip], axis=1)
    if s7.shape != (799, 1546) or not np.isfinite(s7).all():
        raise RuntimeError("expected finite S7 shape (799,1546), got " + repr(s7.shape))
    return s7, check


def load_readiness_matrix(path, cohort_rows):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            "Host-role asset audit not found: " + str(path)
            + ". Run audit_host_role_null_model_assets.py first or pass --asset-audit."
        )
    rows = read_csv(path)
    by_sha = {norm_sha(row.get("sha256", "")): row for row in rows}
    expected = [norm_sha(row["sha256"]) for row in cohort_rows]
    missing_sha = [sha for sha in expected if sha not in by_sha]
    if missing_sha:
        raise RuntimeError("asset audit is missing source APKs: " + repr(missing_sha[:10]))
    missing_columns = [
        name for name in READINESS_COLUMNS
        if any(name not in by_sha[sha] for sha in expected)
    ]
    if missing_columns:
        raise RuntimeError("asset audit is missing readiness columns: " + repr(missing_columns))
    matrix = np.asarray([
        [float(by_sha[sha][name]) for name in READINESS_COLUMNS]
        for sha in expected
    ], dtype=np.float64)
    if matrix.shape != (799, len(READINESS_COLUMNS)) or not np.isfinite(matrix).all():
        raise RuntimeError("invalid readiness matrix: " + repr(matrix.shape))
    if np.any(matrix < 0.0):
        raise RuntimeError("readiness counts must be nonnegative")
    # log1p has no effect on within-column ranks, but makes exported raw values
    # and future extensions less dominated by very large count ranges.
    return np.log1p(matrix), {
        "path": str(path),
        "rows": int(len(rows)),
        "matched_source_rows": int(len(expected)),
        "columns": list(READINESS_COLUMNS),
    }


def readiness_against(reference_matrix, values_matrix):
    """Mean empirical percentile across predeclared availability counts."""
    reference_matrix = np.asarray(reference_matrix, dtype=np.float64)
    values_matrix = np.asarray(values_matrix, dtype=np.float64)
    if reference_matrix.ndim != 2 or values_matrix.ndim != 2:
        raise ValueError("readiness matrices must be two-dimensional")
    if reference_matrix.shape[1] != values_matrix.shape[1]:
        raise ValueError("readiness dimension mismatch")
    component = np.column_stack([
        acn.percentile_against(reference_matrix[:, index], values_matrix[:, index])
        for index in range(reference_matrix.shape[1])
    ])
    return np.mean(component, axis=1), component


def anchored_scores(ambiguity_percentile, disagreement_percentile, readiness_percentile):
    """Predeclared gate formulas that keep ambiguity as the primary anchor."""
    a = np.asarray(ambiguity_percentile, dtype=np.float64)
    d = np.asarray(disagreement_percentile, dtype=np.float64)
    r = np.asarray(readiness_percentile, dtype=np.float64)
    return {
        "A": a,
        "A_D": a * (0.5 + 0.5 * d),
        "A_R": a * (0.5 + 0.5 * r),
        "A_DR": a * ((1.0 + d + r) / 3.0),
    }


def metric_row(method, labels, probability, coarse, route):
    return acn.metric_row(method, labels, probability, coarse, route)


def route_diagnostic_row(method, labels, coarse, expert, route, score):
    row = acn.route_diagnostic_row(method, labels, coarse, route, score)
    labels = np.asarray(labels, dtype=np.int64)
    coarse_correct = (np.asarray(coarse) >= THRESHOLD) == labels
    expert_correct = (np.asarray(expert) >= THRESHOLD) == labels
    route = np.asarray(route, dtype=bool)
    fixable = ~coarse_correct & expert_correct
    breakable = coarse_correct & ~expert_correct
    row.update({
        "expert_fixable_total": int(np.sum(fixable)),
        "routed_expert_fixable": int(np.sum(route & fixable)),
        "expert_fixable_capture_rate": (
            float(np.sum(route & fixable) / np.sum(fixable)) if np.any(fixable) else 0.0
        ),
        "expert_breakable_total": int(np.sum(breakable)),
        "routed_expert_breakable": int(np.sum(route & breakable)),
        "expert_breakable_exposure_rate": (
            float(np.sum(route & breakable) / np.sum(breakable)) if np.any(breakable) else 0.0
        ),
    })
    return row


def check_previous_reference(path, manifest, coarse, expert, a_route):
    path = Path(path)
    if not path.exists():
        return {"available": False, "path": str(path)}
    rows = read_csv(path)
    by_sha = {norm_sha(row.get("sha256", "")): row for row in rows}
    s7_column = "A20__Joint__S7_Stats120_PairRaw465_InterZClip5__prob"
    required = {"C76__prob", s7_column, "ambiguity_routed"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("previous prediction file lacks frozen S7 columns")
    coarse_diff = []
    routed_s7_diff = []
    route_mismatch = 0
    for index, row in enumerate(manifest):
        sha = norm_sha(row["sha256"])
        if sha not in by_sha:
            raise RuntimeError("previous predictions missing SHA: " + sha)
        previous = by_sha[sha]
        coarse_diff.append(abs(float(previous["C76__prob"]) - float(coarse[index])))
        expected_routed = int(previous["ambiguity_routed"])
        route_mismatch += int(expected_routed != int(a_route[index]))
        expected_probability = float(coarse[index])
        if expected_routed:
            expected_probability = float(expert[index])
        routed_s7_diff.append(abs(float(previous[s7_column]) - expected_probability))
    result = {
        "available": True,
        "path": str(path),
        "rows": int(len(rows)),
        "max_abs_C76_probability_difference": float(max(coarse_diff)),
        "max_abs_A20_S7_probability_difference": float(max(routed_s7_diff)),
        "ambiguity_route_mismatches": int(route_mismatch),
    }
    if (
        result["max_abs_C76_probability_difference"] > 1e-10
        or result["max_abs_A20_S7_probability_difference"] > 1e-10
        or result["ambiguity_route_mismatches"]
    ):
        raise RuntimeError("frozen C76+A20+S7 reproduction failed: " + repr(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument(
        "--cache", type=Path,
        default=Path("artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz"),
    )
    parser.add_argument(
        "--p72-oof", type=Path,
        default=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),
    )
    parser.add_argument(
        "--multiscale-features", type=Path,
        default=Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"),
    )
    parser.add_argument(
        "--asset-audit", type=Path,
        default=Path("artifacts/host-role-null-model-asset-audit-v1-full/apk_asset_audit.csv"),
    )
    parser.add_argument(
        "--previous-predictions", type=Path,
        default=Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/source_predictions.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/evidence-aware-routing-ablation-2017-v1"),
    )
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260827)
    args = parser.parse_args()

    if not 0.0 < args.budget < 1.0:
        raise SystemExit("--budget must be in (0,1)")
    if args.inner_folds < 2:
        raise SystemExit("--inner-folds must be at least 2")

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
    print("Reconstructing source Stats120+PairRaw465...", flush=True)
    raw_structure, _ = audit.future.reconstruct_structure(
        cohort_rows, cache, int(audit.exp.ROLE_VOCABULARY_SIZE)
    )
    if raw_structure.shape != (799, 585):
        raise RuntimeError("expected source structure (799,585), got " + repr(raw_structure.shape))
    s7, s7_reference = load_s7_reference(
        args.multiscale_features, cohort_rows, raw_structure
    )
    readiness, readiness_reference = load_readiness_matrix(args.asset_audit, cohort_rows)

    labels = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    unique_folds = sorted(np.unique(folds))
    n = len(labels)
    budget_tag = str(int(round(100.0 * args.budget)))
    route_names = {
        "A": "A" + budget_tag,
        "A_D": "A" + budget_tag + "_D",
        "A_R": "A" + budget_tag + "_R",
        "A_DR": "A" + budget_tag + "_DR",
    }
    methods = ["C76", "JointS7_100"] + list(route_names.values())
    probabilities = {method: np.full(n, np.nan) for method in methods}
    routes = {name: np.zeros(n, dtype=bool) for name in route_names.values()}
    head_probability = {
        "Global24": np.full(n, np.nan),
        "Role52": np.full(n, np.nan),
    }
    outputs = {
        "ambiguity_raw": np.full(n, np.nan),
        "disagreement_raw": np.full(n, np.nan),
        "ambiguity_percentile": np.full(n, np.nan),
        "disagreement_percentile": np.full(n, np.nan),
        "readiness_percentile": np.full(n, np.nan),
        **{"score_" + key: np.full(n, np.nan) for key in route_names},
    }
    readiness_components = np.full((n, readiness.shape[1]), np.nan)
    fold_rows = []

    for fold in unique_folds:
        train_index = np.flatnonzero(folds != fold)
        validation_index = np.flatnonzero(folds == fold)
        train_rows = [cohort_rows[index] for index in train_index]
        validation_rows = [cohort_rows[index] for index in validation_index]
        train_labels = labels[train_index]
        x_train, x_validation, names = audit.compact_matrix_variant(
            VARIANT,
            train_rows,
            validation_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        global_index, role_index = split_view_indices(names)
        coarse, _, _ = audit.p72.fit_predict(
            x_train, train_labels, x_validation, seed=int(fold) - 1
        )
        global_probability, _, _ = audit.p72.fit_predict(
            x_train[:, global_index], train_labels, x_validation[:, global_index],
            seed=args.split_seed + 100 * int(fold) + 1,
        )
        role_probability, _, _ = audit.p72.fit_predict(
            x_train[:, role_index], train_labels, x_validation[:, role_index],
            seed=args.split_seed + 100 * int(fold) + 2,
        )
        expert = audit.fit_config(
            x_train,
            x_validation,
            s7[train_index],
            s7[validation_index],
            train_labels,
            seed=int(fold) - 1,
        )

        inner_ambiguity = np.full(len(train_index), np.nan)
        inner_disagreement = np.full(len(train_index), np.nan)
        splitter = StratifiedKFold(
            n_splits=int(args.inner_folds),
            shuffle=True,
            random_state=int(args.split_seed + 1000 * int(fold)),
        )
        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(
            splitter.split(np.zeros(len(train_index)), train_labels), start=1
        ):
            inner_train_rows = [train_rows[index] for index in inner_train_local]
            inner_validation_rows = [train_rows[index] for index in inner_validation_local]
            x_inner_train, x_inner_validation, inner_names = audit.compact_matrix_variant(
                VARIANT,
                inner_train_rows,
                inner_validation_rows,
                compact_by_sha,
                compact_schema,
                original_names,
            )
            inner_global_index, inner_role_index = split_view_indices(inner_names)
            full_part, _, _ = audit.p72.fit_predict(
                x_inner_train,
                train_labels[inner_train_local],
                x_inner_validation,
                # Keep this seed exactly aligned with the frozen ACN/A20
                # reference implementation so A20 is reproduced bit-for-bit.
                seed=args.split_seed + 10000 * int(fold) + inner_fold,
            )
            global_part, _, _ = audit.p72.fit_predict(
                x_inner_train[:, inner_global_index],
                train_labels[inner_train_local],
                x_inner_validation[:, inner_global_index],
                seed=args.split_seed + 10000 * int(fold) + 10 * inner_fold + 1,
            )
            role_part, _, _ = audit.p72.fit_predict(
                x_inner_train[:, inner_role_index],
                train_labels[inner_train_local],
                x_inner_validation[:, inner_role_index],
                seed=args.split_seed + 10000 * int(fold) + 10 * inner_fold + 2,
            )
            inner_ambiguity[inner_validation_local] = acn.ambiguity(full_part)
            inner_disagreement[inner_validation_local] = view_disagreement(
                global_part, role_part
            )

        if (
            np.any(~np.isfinite(inner_ambiguity))
            or np.any(~np.isfinite(inner_disagreement))
        ):
            raise RuntimeError("incomplete inner OOF gate quantities in fold " + str(fold))

        ambiguity_raw = acn.ambiguity(coarse)
        disagreement_raw = view_disagreement(global_probability, role_probability)
        ambiguity_percentile = acn.percentile_against(inner_ambiguity, ambiguity_raw)
        disagreement_percentile = acn.percentile_against(
            inner_disagreement, disagreement_raw
        )
        readiness_train, _ = readiness_against(
            readiness[train_index], readiness[train_index]
        )
        readiness_validation, readiness_component = readiness_against(
            readiness[train_index], readiness[validation_index]
        )
        readiness_percentile = acn.percentile_against(
            readiness_train, readiness_validation
        )
        gate_scores = anchored_scores(
            ambiguity_percentile, disagreement_percentile, readiness_percentile
        )

        probabilities["C76"][validation_index] = coarse
        probabilities["JointS7_100"][validation_index] = expert
        head_probability["Global24"][validation_index] = global_probability
        head_probability["Role52"][validation_index] = role_probability
        outputs["ambiguity_raw"][validation_index] = ambiguity_raw
        outputs["disagreement_raw"][validation_index] = disagreement_raw
        outputs["ambiguity_percentile"][validation_index] = ambiguity_percentile
        outputs["disagreement_percentile"][validation_index] = disagreement_percentile
        outputs["readiness_percentile"][validation_index] = readiness_percentile
        readiness_components[validation_index] = readiness_component

        fold_routes = {}
        for key, method in route_names.items():
            outputs["score_" + key][validation_index] = gate_scores[key]
            route = acn.top_fraction(gate_scores[key], args.budget)
            fold_routes[method] = route
            routes[method][validation_index] = route
            probabilities[method][validation_index] = acn.routed_probability(
                coarse, expert, route
            )

        fold_rows.append({
            "fold": int(fold),
            "outer_train_n": int(len(train_index)),
            "outer_validation_n": int(len(validation_index)),
            "inner_folds": int(args.inner_folds),
            "compact_dim": int(len(names)),
            "global_dim": int(len(global_index)),
            "role_dim": int(len(role_index)),
            "s7_dim": int(s7.shape[1]),
            "budget": float(args.budget),
            **{
                method + "__routed_n": int(np.sum(fold_routes[method]))
                for method in route_names.values()
            },
            "inner_disagreement_min": float(np.min(inner_disagreement)),
            "inner_disagreement_max": float(np.max(inner_disagreement)),
            "validation_readiness_mean": float(np.mean(readiness_percentile)),
        })
        print("fold", int(fold), "done", flush=True)

    vectors = list(probabilities.values()) + list(head_probability.values()) + list(outputs.values())
    if any(np.any(~np.isfinite(value)) for value in vectors):
        raise RuntimeError("incomplete OOF output")

    reference_check = check_previous_reference(
        args.previous_predictions,
        manifest,
        probabilities["C76"],
        probabilities["JointS7_100"],
        routes[route_names["A"]],
    )

    zero_route = np.zeros(n, dtype=bool)
    full_route = np.ones(n, dtype=bool)
    method_routes = {
        "C76": zero_route,
        "JointS7_100": full_route,
        **routes,
    }
    summary_rows = [
        metric_row(
            method,
            labels,
            probabilities[method],
            probabilities["C76"],
            method_routes[method],
        )
        for method in methods
    ]
    per_fold_rows = []
    for fold in unique_folds:
        selected = folds == fold
        for method in methods:
            per_fold_rows.append({
                **metric_row(
                    method,
                    labels[selected],
                    probabilities[method][selected],
                    probabilities["C76"][selected],
                    method_routes[method][selected],
                ),
                "fold": int(fold),
            })

    diagnostic_rows = []
    diagnostic_fold_rows = []
    for key, method in route_names.items():
        diagnostic_rows.append(route_diagnostic_row(
            method,
            labels,
            probabilities["C76"],
            probabilities["JointS7_100"],
            routes[method],
            outputs["score_" + key],
        ))
        for fold in unique_folds:
            selected = folds == fold
            diagnostic_fold_rows.append({
                **route_diagnostic_row(
                    method,
                    labels[selected],
                    probabilities["C76"][selected],
                    probabilities["JointS7_100"][selected],
                    routes[method][selected],
                    outputs["score_" + key][selected],
                ),
                "fold": int(fold),
            })

    pairwise_rows = []
    for left, right in combinations(methods, 2):
        left_correct = (probabilities[left] >= THRESHOLD) == labels
        right_correct = (probabilities[right] >= THRESHOLD) == labels
        left_only, right_only, p_value = acn.exact_mcnemar(left_correct, right_correct)
        pairwise_rows.append({
            "left": left,
            "right": right,
            "left_only_correct": int(left_only),
            "right_only_correct": int(right_only),
            "discordant": int(left_only + right_only),
            "mcnemar_exact_p": float(p_value),
        })

    overlap_rows = []
    for left, right in combinations(route_names.values(), 2):
        intersection = int(np.sum(routes[left] & routes[right]))
        union = int(np.sum(routes[left] | routes[right]))
        overlap_rows.append({
            "left": left,
            "right": right,
            "intersection": intersection,
            "union": union,
            "jaccard": float(intersection / union) if union else 1.0,
        })

    prediction_rows = []
    for index, row in enumerate(manifest):
        output = {
            "sha256": row["sha256"],
            "label": int(labels[index]),
            "fold": int(folds[index]),
            "C76__prob": float(probabilities["C76"][index]),
            "Global24__prob": float(head_probability["Global24"][index]),
            "Role52__prob": float(head_probability["Role52"][index]),
            "JointS7_100__prob": float(probabilities["JointS7_100"][index]),
            **{name: float(value[index]) for name, value in outputs.items()},
        }
        for component_index, component_name in enumerate(READINESS_COLUMNS):
            output["readiness_component__" + component_name] = float(
                readiness_components[index, component_index]
            )
        for method in route_names.values():
            output[method + "__prob"] = float(probabilities[method][index])
            output[method + "__routed"] = int(routes[method][index])
        prediction_rows.append(output)

    write_csv(args.output / "source_summary.csv", summary_rows)
    write_csv(args.output / "source_per_fold.csv", per_fold_rows)
    write_csv(args.output / "source_predictions.csv", prediction_rows)
    write_csv(args.output / "route_diagnostics.csv", diagnostic_rows)
    write_csv(args.output / "route_diagnostics_per_fold.csv", diagnostic_fold_rows)
    write_csv(args.output / "route_overlap.csv", overlap_rows)
    write_csv(args.output / "pairwise_tests.csv", pairwise_rows)
    write_csv(args.output / "fold_audit.csv", fold_rows)

    summary_by_method = {row["method"]: row for row in summary_rows}
    baseline = route_names["A"]
    candidates = [route_names[key] for key in ("A_D", "A_R", "A_DR")]
    baseline_fold_ba = {
        int(row["fold"]): float(row["BA"])
        for row in per_fold_rows if row["method"] == baseline
    }
    retention = {}
    for candidate in candidates:
        candidate_fold_ba = {
            int(row["fold"]): float(row["BA"])
            for row in per_fold_rows if row["method"] == candidate
        }
        noninferior = sum(
            candidate_fold_ba[int(fold)] + 1e-12 >= baseline_fold_ba[int(fold)]
            for fold in unique_folds
        )
        retention[candidate] = {
            "noninferior_outer_folds": int(noninferior),
            "BA_delta_vs_A": float(
                summary_by_method[candidate]["BA"] - summary_by_method[baseline]["BA"]
            ),
            "net_delta_vs_A": int(
                summary_by_method[candidate]["net"] - summary_by_method[baseline]["net"]
            ),
            "passes_predeclared_source_screen": bool(
                noninferior >= 4
                and summary_by_method[candidate]["BA"]
                > summary_by_method[baseline]["BA"] + 1e-12
                and summary_by_method[candidate]["net"]
                >= summary_by_method[baseline]["net"]
            ),
        }

    metadata = {
        "experiment": "evidence-aware-routing-ablation-2017-v1",
        "source_only": True,
        "future_rows_loaded": False,
        "source_apks": int(n),
        "outer_folds": [int(value) for value in unique_folds],
        "coarse": "C76 allratio76_ratio52",
        "coarse_views": {"Global24": 24, "Role52": 52},
        "expert": "JointLogistic(C76+S7)",
        "s7": S7_TAG,
        "s7_dim": int(s7.shape[1]),
        "budget": float(args.budget),
        "gate_definitions": {
            "A": "ambiguity percentile",
            "D": "percentile of abs(logit(p_Global24)-logit(p_Role52))",
            "R": "percentile of mean training-fold empirical percentiles across predeclared availability counts",
            "A_D": "A*(0.5+0.5*D)",
            "A_R": "A*(0.5+0.5*R)",
            "A_DR": "A*(1+D+R)/3",
        },
        "readiness_columns": list(READINESS_COLUMNS),
        "test_label_used_by_router": False,
        "inner_crossfit": {
            "folds": int(args.inner_folds),
            "seed": int(args.split_seed),
            "quantities": ["C76 ambiguity", "Global24/Role52 disagreement"],
        },
        "s7_reference": s7_reference,
        "readiness_reference": readiness_reference,
        "frozen_reference_check": reference_check,
        "predeclared_retention": {
            "criterion": "candidate BA > A baseline, net >= A baseline, and fold BA noninferior in at least 4/5 folds",
            "results": retention,
        },
        "interpretation_guardrail": (
            "Fixable/breakable labels are output diagnostics only. They are not used "
            "to fit, tune, or score the deployment-time router."
        ),
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
