#!/usr/bin/env python3
"""Fair Global24 + R-routing + Joint-S7 experiment for 2017--2021.

This experiment removes all 52 role-ratio dimensions from layer 1 and changes
all three dependent parts together:

1. Global24 produces the layer-1 probability;
2. Global24 ambiguity, optionally anchored by label-free readiness R, selects
   the fixed-budget layer-2 route;
3. Joint-S7 is retrained on Global24 + S7 rather than C76 + S7.

The frozen C76+A20+Joint-S7 line and its C76+A20-R counterpart are evaluated
under the same protocol as references.  The strict 2017 result uses the fixed
five outer folds.  The 2018--2021 results train on all 2017 rows and are
diagnostic only; future labels are used only after all probabilities and routes
have been fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_host_role_null_model_assets as host_asset
import run_acn_routing_ablation_2017 as acn
import run_allratio70_vs_nodrop76_experiment as audit
import run_evidence_aware_routing_ablation_2017 as evidence
import run_s7_interz_future_diagnostic as future_s7


VARIANT = "allratio76_ratio52"
THRESHOLD = 0.5
READINESS_COLUMNS = evidence.READINESS_COLUMNS


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


def readiness_from_cache(rows, cache, raw_structure, progress_every=200):
    """Reconstruct the exact five R counts without labels."""
    result = []
    n = len(rows)
    for index, row in enumerate(rows):
        sha = evidence.norm_sha(row["sha256"])
        if sha not in cache:
            raise RuntimeError("endpoint cache is missing: " + sha)
        asset = host_asset.reconstruct_host_roles(cache[sha]["payload"])
        host_roles = asset["host_roles"]
        incidence = host_asset.incidence_edges(host_roles)
        result.append([
            len({role for _, role in incidence}),
            len(host_roles),
            len(incidence),
            len(asset["host_host_edges"]),
            int(np.count_nonzero(raw_structure[index, 120:])),
        ])
        done = index + 1
        if progress_every and (done % progress_every == 0 or done == n):
            print("  readiness:", done, "/", n, flush=True)
    matrix = np.log1p(np.asarray(result, dtype=np.float64))
    if matrix.shape != (n, len(READINESS_COLUMNS)) or not np.isfinite(matrix).all():
        raise RuntimeError("invalid readiness matrix: " + repr(matrix.shape))
    return matrix


def readiness_calibration(reference, values):
    reference_r, _ = evidence.readiness_against(reference, reference)
    values_r, components = evidence.readiness_against(reference, values)
    percentile = acn.percentile_against(reference_r, values_r)
    return percentile, components


def route_scores(ambiguity_percentile, readiness_percentile):
    a = np.asarray(ambiguity_percentile, dtype=np.float64)
    r = np.asarray(readiness_percentile, dtype=np.float64)
    return {"A": a, "A_R": a * (0.5 + 0.5 * r)}


def fit_view(x_train, x_test, labels, indices, seed):
    probability, _, _ = audit.p72.fit_predict(
        x_train[:, indices], labels, x_test[:, indices], seed=int(seed)
    )
    return np.asarray(probability, dtype=np.float64)


def fit_joint_view(x_train, x_test, s_train, s_test, labels, indices, seed):
    return audit.fit_config(
        x_train[:, indices], x_test[:, indices], s_train, s_test, labels, seed
    )


def nested_ambiguity_reference(
    train_rows,
    train_labels,
    compact_by_sha,
    compact_schema,
    original_names,
    view,
    inner_folds,
    split_seed,
    model_seed,
):
    n = len(train_rows)
    result = np.full(n, np.nan)
    splitter = StratifiedKFold(
        n_splits=int(inner_folds), shuffle=True, random_state=int(split_seed)
    )
    for inner_fold, (fit_local, val_local) in enumerate(
        splitter.split(np.zeros(n), train_labels), start=1
    ):
        fit_rows = [train_rows[i] for i in fit_local]
        val_rows = [train_rows[i] for i in val_local]
        x_fit, x_val, names = audit.compact_matrix_variant(
            VARIANT,
            fit_rows,
            val_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        global_index, _ = evidence.split_view_indices(names)
        indices = np.arange(x_fit.shape[1]) if view == "C76" else global_index
        part = fit_view(
            x_fit,
            x_val,
            train_labels[fit_local],
            indices,
            model_seed + inner_fold,
        )
        result[val_local] = acn.ambiguity(part)
    if np.any(~np.isfinite(result)):
        raise RuntimeError("incomplete nested ambiguity reference")
    return result


def crossfit_ambiguity_reference(
    rows,
    labels,
    compact_by_sha,
    compact_schema,
    original_names,
    view,
    folds,
    split_seed,
    model_seed,
):
    """Source-only calibration used when routing the pooled future batch."""
    n = len(rows)
    probability = np.full(n, np.nan)
    splitter = StratifiedKFold(
        n_splits=int(folds), shuffle=True, random_state=int(split_seed)
    )
    for inner_fold, (fit_local, val_local) in enumerate(
        splitter.split(np.zeros(n), labels), start=1
    ):
        fit_rows = [rows[i] for i in fit_local]
        val_rows = [rows[i] for i in val_local]
        x_fit, x_val, names = audit.compact_matrix_variant(
            VARIANT,
            fit_rows,
            val_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        global_index, _ = evidence.split_view_indices(names)
        indices = np.arange(x_fit.shape[1]) if view == "C76" else global_index
        probability[val_local] = fit_view(
            x_fit,
            x_val,
            labels[fit_local],
            indices,
            model_seed + inner_fold,
        )
    if np.any(~np.isfinite(probability)):
        raise RuntimeError("incomplete future ambiguity calibration")
    return acn.ambiguity(probability)


def metric_row(method, labels, probability, coarse, route, year=None):
    row = acn.metric_row(method, labels, probability, coarse, route)
    if year is not None:
        row["year"] = int(year)
        row["diagnostic_only"] = True
    return row


def route_diagnostic(method, labels, coarse, expert, route, score, year=None):
    row = evidence.route_diagnostic_row(
        method, labels, coarse, expert, route, score
    )
    if year is not None:
        row["year"] = int(year)
        row["diagnostic_only"] = True
    return row


def evaluate_methods(labels, probability, coarse_by_family, routes, year=None):
    rows = []
    for method, values in probability.items():
        family = "Global24" if method.startswith("Global24") else "C76"
        route = routes.get(method, np.zeros(len(labels), dtype=bool))
        rows.append(metric_row(
            method, labels, values, coarse_by_family[family], route, year
        ))
    return rows


def mean_future_rows(annual_rows):
    result = []
    methods = list(dict.fromkeys(row["method"] for row in annual_rows))
    for method in methods:
        rows = [row for row in annual_rows if row["method"] == method]
        result.append({
            "method": method,
            "mean_BA": float(np.mean([row["BA"] for row in rows])),
            "min_BA": float(np.min([row["BA"] for row in rows])),
            "mean_AUC": float(np.mean([row["AUC"] for row in rows])),
            "mean_F1": float(np.mean([row["F1"] for row in rows])),
            "mean_fixed": float(np.mean([row["fixed"] for row in rows])),
            "mean_broken": float(np.mean([row["broken"] for row in rows])),
            "mean_net": float(np.mean([row["net"] for row in rows])),
            "diagnostic_only": True,
        })
    return result


def append_report_table(lines, title, rows, fields):
    lines.extend(["", title])
    header = " | ".join(["method"] + fields)
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        values = [str(row.get("method", row.get("route", "")))]
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append(" | ".join(values))


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
        "artifacts/global24-r-joints7-2017-2021-v1"
    ))
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--source-crossfit-folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260827)
    parser.add_argument("--null-samples", type=int, default=64)
    parser.add_argument("--swaps-per-edge", type=int, default=10)
    parser.add_argument("--null-seed", type=int, default=20260827)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    if not 0.0 < args.budget < 1.0:
        raise SystemExit("--budget must be in (0,1)")
    if args.inner_folds < 2 or args.source_crossfit_folds < 2:
        raise SystemExit("cross-fitting needs at least two folds")
    years = sorted({int(value.strip()) for value in args.years.split(",") if value.strip()})
    if not years:
        raise SystemExit("--years is empty")
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
        raise RuntimeError("requested/observed future years differ")

    source_y = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    source_fold = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    target_y = np.asarray([audit.future.label_of(row) for row in target_rows], dtype=np.int64)
    target_year = np.asarray([int(row["metadata_year"]) for row in target_rows], dtype=np.int64)
    cache = audit.exp.read_cache(args.cache)

    print("Reconstructing source/future Stats120+PairRaw465...", flush=True)
    source_s3, _ = audit.future.reconstruct_structure(
        source_rows, cache, int(audit.exp.ROLE_VOCABULARY_SIZE)
    )
    target_s3, _ = audit.future.reconstruct_structure(
        target_rows, cache, int(audit.exp.ROLE_VOCABULARY_SIZE)
    )
    if source_s3.shape != (799, 585) or target_s3.shape[1] != 585:
        raise RuntimeError("bad S3 shapes: " + repr((source_s3.shape, target_s3.shape)))
    source_s7, source_reference = future_s7.source_s7_from_reference(
        args.source_multiscale_features, source_rows, source_s3
    )
    target_inter, construction_rows, future_cache_audit = future_s7.load_or_build_future_inter(
        target_rows, target_s3, cache, args
    )
    target_s7 = np.concatenate([target_s3, target_inter], axis=1)

    print("Reconstructing label-free R readiness counts...", flush=True)
    source_readiness = readiness_from_cache(
        source_rows, cache, source_s3, args.progress_every
    )
    target_readiness = readiness_from_cache(
        target_rows, cache, target_s3, args.progress_every
    )

    method_names = [
        "Global24",
        "Global24__JointS7_100",
        "Global24__A20__JointS7",
        "Global24__A20_R__JointS7",
        "C76",
        "C76__JointS7_100",
        "C76__A20__JointS7",
        "C76__A20_R__JointS7",
    ]
    source_probability = {
        name: np.full(len(source_y), np.nan) for name in method_names
    }
    source_routes = {
        name: np.zeros(len(source_y), dtype=bool) for name in method_names
    }
    source_scores = {
        "Global24__A20__JointS7": np.full(len(source_y), np.nan),
        "Global24__A20_R__JointS7": np.full(len(source_y), np.nan),
        "C76__A20__JointS7": np.full(len(source_y), np.nan),
        "C76__A20_R__JointS7": np.full(len(source_y), np.nan),
    }
    source_r_percentile = np.full(len(source_y), np.nan)
    fold_audit = []

    print("Running strict 2017 five-fold comparison...", flush=True)
    for fold in sorted(np.unique(source_fold)):
        train_index = np.flatnonzero(source_fold != fold)
        val_index = np.flatnonzero(source_fold == fold)
        train_rows = [source_rows[i] for i in train_index]
        val_rows = [source_rows[i] for i in val_index]
        train_y = source_y[train_index]
        x_train, x_val, names = audit.compact_matrix_variant(
            VARIANT,
            train_rows,
            val_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        global_index, _ = evidence.split_view_indices(names)
        c76_index = np.arange(x_train.shape[1])
        view_indices = {"Global24": global_index, "C76": c76_index}

        expert_by_view = {}
        coarse_by_view = {}
        for view in ("Global24", "C76"):
            indices = view_indices[view]
            coarse = fit_view(x_train, x_val, train_y, indices, int(fold) - 1)
            expert = fit_joint_view(
                x_train,
                x_val,
                source_s7[train_index],
                source_s7[val_index],
                train_y,
                indices,
                int(fold) - 1,
            )
            coarse_by_view[view] = coarse
            expert_by_view[view] = expert
            source_probability[view][val_index] = coarse
            source_probability[view + "__JointS7_100"][val_index] = expert
            source_routes[view + "__JointS7_100"][val_index] = True

        r_percentile, _ = readiness_calibration(
            source_readiness[train_index], source_readiness[val_index]
        )
        source_r_percentile[val_index] = r_percentile
        fold_row = {
            "fold": int(fold),
            "train_n": int(len(train_index)),
            "validation_n": int(len(val_index)),
            "Global24_dim": int(len(global_index)),
            "C76_dim": int(len(c76_index)),
            "S7_dim": int(source_s7.shape[1]),
        }
        for view in ("Global24", "C76"):
            ambiguity_reference = nested_ambiguity_reference(
                train_rows,
                train_y,
                compact_by_sha,
                compact_schema,
                original_names,
                view,
                args.inner_folds,
                args.split_seed + 1000 * int(fold),
                args.split_seed + 10000 * int(fold) + (0 if view == "C76" else 50000),
            )
            ambiguity_value = acn.ambiguity(coarse_by_view[view])
            a_percentile = acn.percentile_against(
                ambiguity_reference, ambiguity_value
            )
            scores = route_scores(a_percentile, r_percentile)
            for key, suffix in (("A", "A20"), ("A_R", "A20_R")):
                method = view + "__" + suffix + "__JointS7"
                route = acn.top_fraction(scores[key], args.budget)
                source_routes[method][val_index] = route
                source_scores[method][val_index] = scores[key]
                source_probability[method][val_index] = acn.routed_probability(
                    coarse_by_view[view], expert_by_view[view], route
                )
                fold_row[method + "__routed_n"] = int(np.sum(route))
        fold_audit.append(fold_row)
        print("fold", int(fold), "done", flush=True)

    if any(np.any(~np.isfinite(value)) for value in source_probability.values()):
        raise RuntimeError("incomplete source probabilities")
    if any(np.any(~np.isfinite(value)) for value in source_scores.values()):
        raise RuntimeError("incomplete source route scores")

    source_coarse = {
        "Global24": source_probability["Global24"],
        "C76": source_probability["C76"],
    }
    source_summary = evaluate_methods(
        source_y, source_probability, source_coarse, source_routes
    )
    source_route_diagnostics = []
    for view in ("Global24", "C76"):
        for suffix in ("A20", "A20_R"):
            method = view + "__" + suffix + "__JointS7"
            source_route_diagnostics.append(route_diagnostic(
                method,
                source_y,
                source_probability[view],
                source_probability[view + "__JointS7_100"],
                source_routes[method],
                source_scores[method],
            ))

    x_source, x_target, names = audit.compact_matrix_variant(
        VARIANT,
        source_rows,
        target_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    )
    global_index, _ = evidence.split_view_indices(names)
    c76_index = np.arange(x_source.shape[1])
    future_probability = {}
    future_routes = {name: np.zeros(len(target_y), dtype=bool) for name in method_names}
    future_scores = {}
    future_a = {}
    future_r, future_r_components = readiness_calibration(
        source_readiness, target_readiness
    )

    print("Training fixed 2017 models and evaluating future diagnostics...", flush=True)
    for view, indices in (("Global24", global_index), ("C76", c76_index)):
        coarse = fit_view(x_source, x_target, source_y, indices, 0)
        expert = fit_joint_view(
            x_source, x_target, source_s7, target_s7, source_y, indices, 0
        )
        future_probability[view] = coarse
        future_probability[view + "__JointS7_100"] = expert
        future_routes[view + "__JointS7_100"][:] = True

        ambiguity_reference = crossfit_ambiguity_reference(
            source_rows,
            source_y,
            compact_by_sha,
            compact_schema,
            original_names,
            view,
            args.source_crossfit_folds,
            args.split_seed,
            args.split_seed + 10000 + (0 if view == "C76" else 50000),
        )
        a_percentile = acn.percentile_against(
            ambiguity_reference, acn.ambiguity(coarse)
        )
        future_a[view] = a_percentile
        scores = route_scores(a_percentile, future_r)
        for key, suffix in (("A", "A20"), ("A_R", "A20_R")):
            method = view + "__" + suffix + "__JointS7"
            route = acn.top_fraction(scores[key], args.budget)
            future_routes[method] = route
            future_scores[method] = scores[key]
            future_probability[method] = acn.routed_probability(
                coarse, expert, route
            )

    future_coarse = {
        "Global24": future_probability["Global24"],
        "C76": future_probability["C76"],
    }
    annual_rows = []
    annual_route_diagnostics = []
    for year in years:
        selected = target_year == year
        probability_part = {
            key: value[selected] for key, value in future_probability.items()
        }
        route_part = {
            key: value[selected] for key, value in future_routes.items()
        }
        coarse_part = {
            key: value[selected] for key, value in future_coarse.items()
        }
        annual_rows.extend(evaluate_methods(
            target_y[selected], probability_part, coarse_part, route_part, year
        ))
        for view in ("Global24", "C76"):
            for suffix in ("A20", "A20_R"):
                method = view + "__" + suffix + "__JointS7"
                annual_route_diagnostics.append(route_diagnostic(
                    method,
                    target_y[selected],
                    future_probability[view][selected],
                    future_probability[view + "__JointS7_100"][selected],
                    future_routes[method][selected],
                    future_scores[method][selected],
                    year,
                ))

    future_mean = mean_future_rows(annual_rows)
    future_pooled = evaluate_methods(
        target_y, future_probability, future_coarse, future_routes
    )
    route_methods = [
        view + "__" + suffix + "__JointS7"
        for view in ("Global24", "C76")
        for suffix in ("A20", "A20_R")
    ]
    future_route_mean = []
    all_year_route_coverage = []
    source_diagnostic_by_route = {
        row["route"]: row for row in source_route_diagnostics
    }
    for method in route_methods:
        rows = [
            row for row in annual_route_diagnostics if row["route"] == method
        ]
        future_errors = int(sum(row["coarse_errors"] for row in rows))
        future_captured = int(sum(row["routed_coarse_errors"] for row in rows))
        future_route_mean.append({
            "route": method,
            "macro_error_capture_rate": float(np.mean([
                row["error_capture_rate"] for row in rows
            ])),
            "micro_routed_coarse_errors": future_captured,
            "micro_coarse_errors": future_errors,
            "micro_error_capture_rate": (
                float(future_captured / future_errors) if future_errors else 0.0
            ),
            "mean_annual_routed_n": float(np.mean([row["routed_n"] for row in rows])),
            "diagnostic_only": True,
        })
        source_row = source_diagnostic_by_route[method]
        year_rows = [{
            "route": method,
            "year": 2017,
            "coarse_errors": int(source_row["coarse_errors"]),
            "routed_coarse_errors": int(source_row["routed_coarse_errors"]),
            "error_capture_rate": float(source_row["error_capture_rate"]),
            "strict_source_oof": True,
            "diagnostic_only": False,
        }] + [{
            "route": method,
            "year": int(row["year"]),
            "coarse_errors": int(row["coarse_errors"]),
            "routed_coarse_errors": int(row["routed_coarse_errors"]),
            "error_capture_rate": float(row["error_capture_rate"]),
            "strict_source_oof": False,
            "diagnostic_only": True,
        } for row in rows]
        all_year_route_coverage.extend(year_rows)
        all_errors = int(sum(row["coarse_errors"] for row in year_rows))
        all_captured = int(sum(row["routed_coarse_errors"] for row in year_rows))
        all_year_route_coverage.append({
            "route": method,
            "year": "2017-2021_macro_micro",
            "coarse_errors": all_errors,
            "routed_coarse_errors": all_captured,
            "error_capture_rate": float(np.mean([
                row["error_capture_rate"] for row in year_rows
            ])),
            "micro_error_capture_rate": (
                float(all_captured / all_errors) if all_errors else 0.0
            ),
            "strict_source_oof": "mixed",
            "diagnostic_only": True,
        })

    source_prediction_rows = []
    for i, row in enumerate(manifest):
        output = {
            "sha256": row["sha256"],
            "label": int(source_y[i]),
            "fold": int(source_fold[i]),
            "readiness_percentile": float(source_r_percentile[i]),
        }
        for method in method_names:
            output[method + "__prob"] = float(source_probability[method][i])
            output[method + "__routed"] = int(source_routes[method][i])
        for method, values in source_scores.items():
            output[method + "__route_score"] = float(values[i])
        source_prediction_rows.append(output)

    future_prediction_rows = []
    for i, row in enumerate(target_rows):
        output = {
            "sha256": row["sha256"],
            "year": int(target_year[i]),
            "label": int(target_y[i]),
            "Global24__ambiguity_percentile": float(future_a["Global24"][i]),
            "C76__ambiguity_percentile": float(future_a["C76"][i]),
            "readiness_percentile": float(future_r[i]),
            "diagnostic_only": True,
        }
        for j, name in enumerate(READINESS_COLUMNS):
            output["readiness_component__" + name] = float(future_r_components[i, j])
        for method in method_names:
            output[method + "__prob"] = float(future_probability[method][i])
            output[method + "__routed"] = int(future_routes[method][i])
        for method, values in future_scores.items():
            output[method + "__route_score"] = float(values[i])
        future_prediction_rows.append(output)

    write_csv(args.output / "source_summary.csv", source_summary)
    write_csv(args.output / "source_route_diagnostics.csv", source_route_diagnostics)
    write_csv(args.output / "source_predictions.csv", source_prediction_rows)
    write_csv(args.output / "fold_audit.csv", fold_audit)
    write_csv(args.output / "future_annual_summary_diagnostic.csv", annual_rows)
    write_csv(args.output / "future_mean_summary_diagnostic.csv", future_mean)
    write_csv(args.output / "future_pooled_summary_diagnostic.csv", future_pooled)
    write_csv(args.output / "future_route_diagnostics_diagnostic.csv", annual_route_diagnostics)
    write_csv(args.output / "future_route_mean_diagnostic.csv", future_route_mean)
    write_csv(args.output / "error_coverage_2017_2021.csv", all_year_route_coverage)
    write_csv(args.output / "future_predictions_diagnostic.csv", future_prediction_rows)
    write_csv(args.output / "future_inter_construction_audit.csv", construction_rows)

    metadata = {
        "experiment": "global24-r-joints7-2017-2021-v1",
        "source_protocol": "fixed 2017 five-fold OOF with nested four-fold gate calibration",
        "future_protocol": "train on all 2017, route pooled unlabeled 2018-2021 batch, evaluate by year",
        "future_diagnostic_only": True,
        "future_labels_used_for_training_routing_or_selection": False,
        "Global24": "C76 after removing all 52 role_node/role_edge ratio dimensions",
        "Global24_dim": 24,
        "C76_dim": 76,
        "S7": "Stats120+PairRaw465+InterZ961 clipped to [-5,5]",
        "S7_dim": 1546,
        "budget": float(args.budget),
        "R_columns": list(READINESS_COLUMNS),
        "R_definition": "mean source empirical percentile of five label-free availability counts, then percentile calibrated; A_R=A*(0.5+0.5*R)",
        "source_s7_reference": source_reference,
        "future_inter_cache": future_cache_audit,
        "selection_guardrail": "2018-2021 are diagnostic only and must not be used for further tuning",
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "Global24 + R routing + Joint-S7, 2017-2021",
        "",
        "Global24 removes all 52 role-ratio dimensions from C76.",
        "R does not use labels: it measures whether enough structural evidence is available for S7.",
        "A20_R uses A*(0.5+0.5*R) at the same fixed 20% total budget.",
        "2018-2021 are diagnostic only.",
    ]
    append_report_table(
        lines,
        "Strict 2017 source summary",
        source_summary,
        ["BA", "AUC", "F1", "fixed", "broken", "net", "routed_n"],
    )
    append_report_table(
        lines,
        "Strict 2017 route error coverage",
        source_route_diagnostics,
        ["coarse_errors", "routed_coarse_errors", "error_capture_rate", "routed_n"],
    )
    for year in years:
        append_report_table(
            lines,
            "Future diagnostic " + str(year),
            [row for row in annual_rows if row["year"] == year],
            ["BA", "AUC", "F1", "fixed", "broken", "net", "routed_n"],
        )
        append_report_table(
            lines,
            "Future diagnostic route error coverage " + str(year),
            [row for row in annual_route_diagnostics if row["year"] == year],
            ["coarse_errors", "routed_coarse_errors", "error_capture_rate", "routed_n"],
        )
    append_report_table(
        lines,
        "2018-2021 macro mean",
        future_mean,
        ["mean_BA", "min_BA", "mean_AUC", "mean_F1", "mean_fixed", "mean_broken", "mean_net"],
    )
    append_report_table(
        lines,
        "2018-2021 route error-coverage average",
        future_route_mean,
        ["macro_error_capture_rate", "micro_error_capture_rate", "micro_routed_coarse_errors", "micro_coarse_errors"],
    )
    append_report_table(
        lines,
        "2017-2021 route error-coverage macro/micro",
        [row for row in all_year_route_coverage if row["year"] == "2017-2021_macro_micro"],
        ["error_capture_rate", "micro_error_capture_rate", "routed_coarse_errors", "coarse_errors"],
    )
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
