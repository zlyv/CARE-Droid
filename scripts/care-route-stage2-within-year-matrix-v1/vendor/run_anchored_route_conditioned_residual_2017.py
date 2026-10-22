#!/usr/bin/env python3
"""2017 anchored and route-conditioned linear residual experiment.

This experiment keeps the fixed C76 representation and the existing
Stats120+PairRaw465 structure.  It compares the current JointLogistic expert
against a linear structural residual with the C76 logit fixed at coefficient 1.
Both Ambiguity20 and ACN20 are retained so that router/expert interaction is
measured rather than assuming that the current expert is final.

Training is leakage-safe:
  * each outer training fold receives cross-fitted C76 probabilities;
  * Ambiguity/ACN training routes are computed from those OOF probabilities;
  * ACN calibration uses only pooled OOF risks inside the outer training fold;
  * the outer validation fold is used only for final evaluation;
  * no future rows or labels are loaded.

Copy this file into ``scripts/`` beside
``run_acn_routing_ablation_2017.py`` and run from the repository root.
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


audit = acn.audit
THRESHOLD = 0.5
VARIANT = acn.VARIANT


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sigmoid(logit):
    logit = np.clip(np.asarray(logit, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logit))


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


def fit_offset_residual(
    structure,
    labels,
    base_probability,
    sample_weight,
    l2,
    max_iterations,
    tolerance,
):
    """Minimize weighted CE(y, sigmoid(logit(p0)+S beta)) + L2.

    There is no residual intercept.  beta=0 is exactly the frozen C76 model.
    """
    structure = np.asarray(structure, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    base_logit = logit(base_probability)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if structure.ndim != 2 or structure.shape[0] != len(labels):
        raise ValueError("invalid residual training matrix")
    if len(sample_weight) != len(labels):
        raise ValueError("invalid sample weights")
    if np.any(sample_weight < 0.0) or float(np.sum(sample_weight)) <= 0.0:
        raise ValueError("sample weights must be nonnegative with positive sum")
    normalizer = float(np.sum(sample_weight))

    def objective(beta):
        z = base_logit + structure @ beta
        probability = sigmoid(z)
        loss_rows = np.logaddexp(0.0, z) - labels * z
        loss = float(np.sum(sample_weight * loss_rows) / normalizer)
        loss += 0.5 * float(l2) * float(beta @ beta)
        gradient = (
            structure.T @ (sample_weight * (probability - labels)) / normalizer
            + float(l2) * beta
        )
        return loss, gradient

    # Small deterministic NumPy L-BFGS optimizer.  Keeping the implementation
    # here avoids adding a SciPy dependency to the established project env.
    beta = np.zeros(structure.shape[1], dtype=np.float64)
    loss, gradient = objective(beta)
    function_evaluations = 1
    memory = 10
    s_history = []
    y_history = []
    rho_history = []
    success = False
    status = 1
    message = "maximum iterations reached"
    iterations = 0
    for iteration in range(1, int(max_iterations) + 1):
        iterations = iteration
        gradient_inf = float(np.max(np.abs(gradient)))
        if gradient_inf <= float(tolerance):
            success = True
            status = 0
            message = "gradient tolerance reached"
            break

        q = gradient.copy()
        alphas = []
        for s_vec, y_vec, rho in zip(
            reversed(s_history), reversed(y_history), reversed(rho_history)
        ):
            alpha = rho * float(s_vec @ q)
            alphas.append(alpha)
            q = q - alpha * y_vec
        if s_history:
            sy = float(s_history[-1] @ y_history[-1])
            yy = float(y_history[-1] @ y_history[-1])
            scale = sy / yy if yy > 1e-20 else 1.0
        else:
            scale = 1.0
        r = scale * q
        for s_vec, y_vec, rho, alpha in zip(
            s_history, y_history, rho_history, reversed(alphas)
        ):
            beta_two_loop = rho * float(y_vec @ r)
            r = r + s_vec * (alpha - beta_two_loop)
        direction = -r
        directional_derivative = float(gradient @ direction)
        if (
            not np.isfinite(directional_derivative)
            or directional_derivative >= -1e-14
        ):
            direction = -gradient
            directional_derivative = -float(gradient @ gradient)

        step = 1.0
        accepted = False
        candidate = beta
        candidate_loss = loss
        candidate_gradient = gradient
        for _ in range(40):
            candidate = beta + step * direction
            candidate_loss, candidate_gradient = objective(candidate)
            function_evaluations += 1
            if (
                np.isfinite(candidate_loss)
                and candidate_loss
                <= loss + 1e-4 * step * directional_derivative
            ):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            status = 2
            message = "backtracking line search failed"
            break

        s_vec = candidate - beta
        y_vec = candidate_gradient - gradient
        curvature = float(s_vec @ y_vec)
        if curvature > 1e-12:
            if len(s_history) == memory:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            s_history.append(s_vec)
            y_history.append(y_vec)
            rho_history.append(1.0 / curvature)

        previous_loss = loss
        beta = candidate
        loss = float(candidate_loss)
        gradient = candidate_gradient
        if abs(previous_loss - loss) <= float(tolerance) * max(
            1.0, abs(previous_loss)
        ):
            success = True
            status = 0
            message = "objective-change tolerance reached"
            break

    final_loss, gradient = objective(beta)
    function_evaluations += 1
    diagnostic = {
        "optimizer_success": bool(success),
        "optimizer_status": int(status),
        "optimizer_message": str(message),
        "iterations": int(iterations),
        "function_evaluations": int(function_evaluations),
        "objective": float(final_loss),
        "gradient_inf_norm": float(np.max(np.abs(gradient))),
        "beta_l2_norm": float(np.linalg.norm(beta)),
        "nonzero_beta": int(np.sum(np.abs(beta) > 1e-10)),
    }
    return beta, diagnostic


def crossfit_base_and_risks(
    rows,
    labels,
    compact_by_sha,
    compact_schema,
    original_names,
    inner_folds,
    split_seed,
    model_seed,
    knn_k,
):
    """OOF base probabilities and raw A/C/N risks for one outer train fold."""
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    probability = np.full(n, np.nan)
    ambiguity_raw = np.full(n, np.nan)
    collision_raw = np.full(n, np.nan)
    novelty_raw = np.full(n, np.nan)
    splitter = StratifiedKFold(
        n_splits=int(inner_folds),
        shuffle=True,
        random_state=int(split_seed),
    )
    for inner_fold, (train_local, val_local) in enumerate(
        splitter.split(np.zeros(n), labels), start=1
    ):
        train_rows = [rows[i] for i in train_local]
        val_rows = [rows[i] for i in val_local]
        x_train, x_val, _ = audit.compact_matrix_variant(
            VARIANT,
            train_rows,
            val_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        part_probability, _, _ = audit.p72.fit_predict(
            x_train,
            labels[train_local],
            x_val,
            seed=int(model_seed + inner_fold),
        )
        part_collision, part_novelty, _, _ = acn.neighborhood_risks(
            x_train, labels[train_local], x_val, int(knn_k)
        )
        probability[val_local] = part_probability
        ambiguity_raw[val_local] = acn.ambiguity(part_probability)
        collision_raw[val_local] = part_collision
        novelty_raw[val_local] = part_novelty
    values = (probability, ambiguity_raw, collision_raw, novelty_raw)
    if any(np.any(~np.isfinite(value)) for value in values):
        raise RuntimeError("incomplete cross-fitted base probability or risk")
    return values


def calibrated_routes(
    ambiguity_reference,
    collision_reference,
    novelty_reference,
    ambiguity_values,
    collision_values,
    novelty_values,
    budget,
):
    a = acn.percentile_against(ambiguity_reference, ambiguity_values)
    c = acn.percentile_against(collision_reference, collision_values)
    n = acn.percentile_against(novelty_reference, novelty_values)
    combined = np.maximum.reduce([a, c, n])
    return {
        "ambiguity_percentile": a,
        "collision_percentile": c,
        "novelty_percentile": n,
        "acn_score": combined,
        "ambiguity_route": acn.top_fraction(a, budget),
        "acn_route": acn.top_fraction(combined, budget),
    }


def method_metric(method, y, probability, coarse, route):
    return acn.metric_row(method, y, probability, coarse, route)


def exact_mcnemar(left_correct, right_correct):
    return audit.exact_mcnemar_p(
        np.asarray(left_correct, dtype=bool),
        np.asarray(right_correct, dtype=bool),
    )


def reference_check(path, manifest, probability, routes):
    path = Path(path)
    if not path.exists():
        return {"available": False, "path": str(path)}
    rows = read_csv(path)
    by_sha = {str(row["sha256"]).strip().casefold(): row for row in rows}
    missing = [
        row["sha256"] for row in manifest
        if str(row["sha256"]).strip().casefold() not in by_sha
    ]
    if missing:
        raise RuntimeError("ACN reference is missing SHAs: " + repr(missing[:5]))
    coarse_diff = []
    joint_diff = []
    ambiguity_mismatch = 0
    acn_mismatch = 0
    for i, item in enumerate(manifest):
        row = by_sha[str(item["sha256"]).strip().casefold()]
        coarse_diff.append(abs(float(row["C76__prob"]) - probability["C76"][i]))
        joint_diff.append(
            abs(float(row["JointLogistic100__prob"]) - probability["JointLogistic100"][i])
        )
        ambiguity_mismatch += int(
            int(row["Ambiguity20__routed"]) != int(routes["ambiguity"][i])
        )
        acn_mismatch += int(int(row["ACN20__routed"]) != int(routes["acn"][i]))
    result = {
        "available": True,
        "path": str(path),
        "rows": int(len(rows)),
        "max_abs_C76_probability_difference": float(max(coarse_diff)),
        "max_abs_joint_probability_difference": float(max(joint_diff)),
        "ambiguity_route_mismatches": int(ambiguity_mismatch),
        "acn_route_mismatches": int(acn_mismatch),
    }
    if (
        result["max_abs_C76_probability_difference"] > 1e-10
        or result["max_abs_joint_probability_difference"] > 1e-10
        or ambiguity_mismatch
        or acn_mismatch
    ):
        raise RuntimeError("new experiment does not reproduce ACN v1 baseline: " + repr(result))
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
        "--acn-reference", type=Path,
        default=Path(
            "artifacts/acn-routing-ablation-2017-v1/source_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/anchored-route-conditioned-residual-2017-v1"),
    )
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260827)
    parser.add_argument("--residual-l2", type=float, default=0.03)
    parser.add_argument(
        "--eta-values", type=float, nargs="+", default=[0.0, 0.1, 0.25]
    )
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    if not 0.0 < args.budget < 1.0:
        raise SystemExit("--budget must be between zero and one")
    if args.knn_k < 2:
        raise SystemExit("--knn-k must be at least two")
    if args.inner_folds < 2:
        raise SystemExit("--inner-folds must be at least two")
    if args.residual_l2 <= 0.0:
        raise SystemExit("--residual-l2 must be positive")
    eta_values = []
    for value in args.eta_values:
        value = float(value)
        if not 0.0 <= value < 1.0:
            raise SystemExit("each --eta-values entry must be in [0,1)")
        if value not in eta_values:
            eta_values.append(value)

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
    structure, structure_names = audit.future.reconstruct_structure(
        cohort_rows, cache, int(audit.exp.ROLE_VOCABULARY_SIZE)
    )
    if structure.shape != (799, 585):
        raise RuntimeError(
            "Expected source structure shape (799,585), got "
            + repr(structure.shape)
        )

    y = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    unique_folds = sorted(np.unique(folds))
    n = len(y)
    budget_tag = str(int(round(100.0 * args.budget)))
    a_joint = "A" + budget_tag + "_JointLogistic"
    acn_joint = "ACN" + budget_tag + "_JointLogistic"
    a_global = "A" + budget_tag + "_AnchoredGlobal"
    acn_global = "ACN" + budget_tag + "_AnchoredGlobal"
    methods = [
        "C76",
        "JointLogistic100",
        a_joint,
        acn_joint,
        "AnchoredResidual100",
        a_global,
        acn_global,
    ]
    matched_method = {}
    for eta in eta_values:
        eta_tag = format(eta, "g")
        matched_method[("ambiguity", eta)] = (
            "A" + budget_tag + "_RouteResidual_eta" + eta_tag
        )
        matched_method[("acn", eta)] = (
            "ACN" + budget_tag + "_RouteResidual_eta" + eta_tag
        )
        methods.extend([
            matched_method[("ambiguity", eta)],
            matched_method[("acn", eta)],
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
    fit_audit_rows = []
    fold_audit_rows = []

    for fold in unique_folds:
        train_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        train_rows = [cohort_rows[i] for i in train_idx]
        val_rows = [cohort_rows[i] for i in val_idx]
        x_train, x_val, names = audit.compact_matrix_variant(
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
        joint = audit.fit_config(
            x_train,
            x_val,
            structure[train_idx],
            structure[val_idx],
            y[train_idx],
            seed=int(fold) - 1,
        )

        (
            train_oof_probability,
            train_ambiguity_raw,
            train_collision_raw,
            train_novelty_raw,
        ) = crossfit_base_and_risks(
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
        train_route_data = calibrated_routes(
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
        outer_ambiguity = acn.ambiguity(coarse)
        val_route_data = calibrated_routes(
            train_ambiguity_raw,
            train_collision_raw,
            train_novelty_raw,
            outer_ambiguity,
            outer_collision,
            outer_novelty,
            args.budget,
        )
        a_train_route = train_route_data["ambiguity_route"]
        acn_train_route = train_route_data["acn_route"]
        a_val_route = val_route_data["ambiguity_route"]
        acn_val_route = val_route_data["acn_route"]

        feature_names = [
            "structure__" + format(i, "03d") for i in range(structure.shape[1])
        ]
        s_train, s_val, structure_audit = audit.p72.standardize_added(
            structure[train_idx], structure[val_idx], feature_names
        )
        usable_structure = int(
            sum(not bool(row["constant_in_train"]) for row in structure_audit)
        )

        global_beta, global_diagnostic = fit_offset_residual(
            s_train,
            y[train_idx],
            train_oof_probability,
            np.ones(len(train_idx), dtype=np.float64),
            args.residual_l2,
            args.max_iterations,
            args.optimizer_tolerance,
        )
        global_delta = s_val @ global_beta
        global_probability = sigmoid(logit(coarse) + global_delta)

        probabilities["C76"][val_idx] = coarse
        probabilities["JointLogistic100"][val_idx] = joint
        probabilities[a_joint][val_idx] = acn.routed_probability(
            coarse, joint, a_val_route
        )
        probabilities[acn_joint][val_idx] = acn.routed_probability(
            coarse, joint, acn_val_route
        )
        probabilities["AnchoredResidual100"][val_idx] = global_probability
        probabilities[a_global][val_idx] = acn.routed_probability(
            coarse, global_probability, a_val_route
        )
        probabilities[acn_global][val_idx] = acn.routed_probability(
            coarse, global_probability, acn_val_route
        )
        fit_audit_rows.append({
            "fold": int(fold),
            "model": "AnchoredGlobal",
            "route_training": "all",
            "eta": 1.0,
            "l2": float(args.residual_l2),
            "train_weight_sum": float(len(train_idx)),
            "train_routed_n": int(len(train_idx)),
            "eval_delta_mean": float(np.mean(global_delta)),
            "eval_delta_std": float(np.std(global_delta)),
            "eval_delta_max_abs": float(np.max(np.abs(global_delta))),
            **global_diagnostic,
        })

        for route_name, train_route, val_route in (
            ("ambiguity", a_train_route, a_val_route),
            ("acn", acn_train_route, acn_val_route),
        ):
            for eta in eta_values:
                weights = np.where(train_route, 1.0, float(eta))
                beta, diagnostic = fit_offset_residual(
                    s_train,
                    y[train_idx],
                    train_oof_probability,
                    weights,
                    args.residual_l2,
                    args.max_iterations,
                    args.optimizer_tolerance,
                )
                delta = s_val @ beta
                enhanced = sigmoid(logit(coarse) + delta)
                method = matched_method[(route_name, eta)]
                probabilities[method][val_idx] = acn.routed_probability(
                    coarse, enhanced, val_route
                )
                fit_audit_rows.append({
                    "fold": int(fold),
                    "model": method,
                    "route_training": route_name,
                    "eta": float(eta),
                    "l2": float(args.residual_l2),
                    "train_weight_sum": float(np.sum(weights)),
                    "train_routed_n": int(np.sum(train_route)),
                    "eval_delta_mean": float(np.mean(delta)),
                    "eval_delta_std": float(np.std(delta)),
                    "eval_delta_max_abs": float(np.max(np.abs(delta))),
                    **diagnostic,
                })

        ambiguity_routes[val_idx] = a_val_route
        acn_routes[val_idx] = acn_val_route
        for key in risk_outputs:
            risk_outputs[key][val_idx] = val_route_data[key]
        fold_audit_rows.append({
            "fold": int(fold),
            "outer_train_n": int(len(train_idx)),
            "outer_val_n": int(len(val_idx)),
            "compact_dim": int(len(names)),
            "structure_dim": int(structure.shape[1]),
            "usable_structure_dim": int(usable_structure),
            "inner_folds": int(args.inner_folds),
            "knn_k": int(args.knn_k),
            "budget": float(args.budget),
            "ambiguity_train_route_n": int(np.sum(a_train_route)),
            "acn_train_route_n": int(np.sum(acn_train_route)),
            "ambiguity_val_route_n": int(np.sum(a_val_route)),
            "acn_val_route_n": int(np.sum(acn_val_route)),
        })
        print("fold", int(fold), "done")

    if any(np.any(~np.isfinite(value)) for value in probabilities.values()):
        raise RuntimeError("incomplete OOF probabilities")
    if any(np.any(~np.isfinite(value)) for value in risk_outputs.values()):
        raise RuntimeError("incomplete OOF risk output")

    zero_route = np.zeros(n, dtype=bool)
    full_route = np.ones(n, dtype=bool)

    def route_for_method(method):
        if method == "C76":
            return zero_route
        if method.endswith("100"):
            return full_route
        if method.startswith("ACN"):
            return acn_routes
        return ambiguity_routes

    summary_rows = []
    per_fold_rows = []
    for method in methods:
        route = route_for_method(method)
        summary_rows.append(
            method_metric(method, y, probabilities[method], probabilities["C76"], route)
        )
        for fold in unique_folds:
            idx = folds == fold
            per_fold_rows.append({
                **method_metric(
                    method,
                    y[idx],
                    probabilities[method][idx],
                    probabilities["C76"][idx],
                    route[idx],
                ),
                "fold": int(fold),
            })

    route_diagnostic_rows = [
        acn.route_diagnostic_row(
            "Ambiguity" + budget_tag,
            y,
            probabilities["C76"],
            ambiguity_routes,
            risk_outputs["ambiguity_percentile"],
        ),
        acn.route_diagnostic_row(
            "ACN" + budget_tag,
            y,
            probabilities["C76"],
            acn_routes,
            risk_outputs["acn_score"],
        ),
    ]

    comparison_pairs = [
        ("C76", a_joint),
        ("C76", acn_joint),
        (a_joint, acn_joint),
        (a_joint, a_global),
        (acn_joint, acn_global),
        (a_global, acn_global),
    ]
    for eta in eta_values:
        a_method = matched_method[("ambiguity", eta)]
        acn_method = matched_method[("acn", eta)]
        comparison_pairs.extend([
            (a_joint, a_method),
            (acn_joint, acn_method),
            (a_global, a_method),
            (acn_global, acn_method),
            (a_method, acn_method),
        ])
    seen_pairs = set()
    pairwise_rows = []
    for left, right in comparison_pairs:
        if left == right or (left, right) in seen_pairs:
            continue
        seen_pairs.add((left, right))
        left_correct = (probabilities[left] >= THRESHOLD) == y
        right_correct = (probabilities[right] >= THRESHOLD) == y
        left_only, right_only, p_value = exact_mcnemar(left_correct, right_correct)
        pairwise_rows.append({
            "left": left,
            "right": right,
            "left_only_correct": int(left_only),
            "right_only_correct": int(right_only),
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

    reference_result = reference_check(
        args.acn_reference,
        manifest,
        probabilities,
        {"ambiguity": ambiguity_routes, "acn": acn_routes},
    )

    write_csv(args.output / "source_summary.csv", summary_rows)
    write_csv(args.output / "source_per_fold.csv", per_fold_rows)
    write_csv(args.output / "source_predictions.csv", prediction_rows)
    write_csv(args.output / "fit_audit.csv", fit_audit_rows)
    write_csv(args.output / "fold_audit.csv", fold_audit_rows)
    write_csv(args.output / "route_diagnostics.csv", route_diagnostic_rows)
    write_csv(args.output / "pairwise_tests.csv", pairwise_rows)
    (args.output / "reference_check.json").write_text(
        json.dumps(reference_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "experiment": "anchored-route-conditioned-residual-2017-v1",
        "source_only": True,
        "source_apks": int(n),
        "outer_folds": [int(value) for value in unique_folds],
        "coarse": "C76 allratio76_ratio52 with frozen coefficient 1 in residual models",
        "structure": "Stats120+PairRaw465",
        "structure_dim": int(structure.shape[1]),
        "structure_name_count": int(len(structure_names)),
        "routes": ["Ambiguity" + budget_tag, "ACN" + budget_tag],
        "budget": float(args.budget),
        "knn_k": int(args.knn_k),
        "inner_folds": int(args.inner_folds),
        "residual_objective": (
            "weighted mean CE(y,sigmoid(logit(C76_OOF)+S beta)) "
            "+ 0.5*l2*||beta||^2; no residual intercept"
        ),
        "residual_l2": float(args.residual_l2),
        "l2_reason": "fixed to the existing C76 logistic L2 for this first falsification experiment",
        "eta_values_reported_independently": eta_values,
        "eta_one_equivalent": "AnchoredGlobal; all training samples receive weight 1",
        "eta_selection": "none; no best eta is selected inside this script",
        "optimizer": "deterministic NumPy L-BFGS with analytic gradient and Armijo backtracking",
        "test_label_used_by_router": False,
        "future_rows_loaded": False,
        "primary_questions": [
            "Does anchored structural correction beat the current JointLogistic under the same route?",
            "Does route-conditioned weighting reduce breaks or improve net correction?",
            "Can ACN become competitive when paired with a different second-layer learner?",
        ],
        "decision_rule": (
            "Do not eliminate ACN from future expert interaction tests based only on the current expert. "
            "A final combination should improve global BA/net correction and avoid fold-local collapse; "
            "eta variants are interpreted as robustness evidence rather than post-hoc tuning."
        ),
        "reference_reproduction": reference_result,
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Done:", args.output)


if __name__ == "__main__":
    main()
