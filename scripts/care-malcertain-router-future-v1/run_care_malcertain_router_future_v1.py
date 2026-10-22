#!/usr/bin/env python3
"""Frozen 2018--2021 audit of a MalCertain-inspired multi-view router.

Only 2017 labels train the five lightweight semantic views and the SVM error
router.  The already-frozen Global24 probability and deduplicated bounded local
expert probability are reused for 2018--2021.  Future labels are materialized
only after every score, route, and final probability has been fixed.

The primary comparison uses an exact 25 percent budget independently inside
each future year.  A secondary source-threshold transport audit applies the
75th-percentile score threshold fitted on 2017 without future calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
for path in (
    SCRIPT_DIR,
    SCRIPTS_ROOT / "care-malcertain-router-source-v1",
    SCRIPTS_ROOT / "care-multiview-conformal-router-source-v1",
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script",
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-future-script",
    # Local audit-bundle fallbacks; harmless on the experiment server.
    SCRIPTS_ROOT / "outputs" / "care-bounded-fixkeep-v3-script",
    SCRIPTS_ROOT / "outputs" / "care-joint-dualprototype-v2-script",
    SCRIPTS_ROOT / "outputs" / "care-local-dualpool-mil-v1-script",
    SCRIPTS_ROOT / "outputs" / "care-local-region-asset-audit-v1-script",
    SCRIPTS_ROOT / "outputs" / "global24-r-joints7-2017-2021-v1-script" / "scripts",
    SCRIPTS_ROOT / "work" / "compact76_trace" / "scripts",
    SCRIPTS_ROOT / "work" / "role_masking_code_bundle_uploaded"
    / "role_masking_code_bundle" / "repository" / "scripts",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import run_multiview_conformal_router_source_v1 as mv
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing established CARE multi-view dependencies: " + str(exc))

from future_router_helpers import (  # noqa: E402
    groupwise_top_fraction,
    routed_probability,
    source_quantile_threshold,
)
from uncertainty_router_core import (  # noqa: E402
    OFFICIAL7_NAMES,
    PAPER9_NAMES,
    error_risk_score,
    fit_error_router,
    router_features,
)


FROZEN_YEARS = (2018, 2019, 2020, 2021)
DEFAULT_EXPERT_COLUMN = (
    "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep__ensemble3__expert_prob"
)
VIEW_NAMES = mv.VIEW_NAMES


def norm_sha(value: object) -> str:
    return str(value or "").strip().casefold()


def read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_years(text: str) -> tuple[int, ...]:
    years = tuple(sorted({int(token.strip()) for token in text.split(",") if token.strip()}))
    if years != FROZEN_YEARS:
        raise SystemExit("This frozen audit requires --years 2018,2019,2020,2021")
    return years


def align_frozen_future(
    path: Path,
    target_rows: list[dict],
    target_year: np.ndarray,
    requested_expert_column: str,
) -> tuple[dict[str, np.ndarray], str]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError("future expert prediction file is empty: " + str(path))
    by_sha = {norm_sha(row.get("sha256")): row for row in rows}
    if len(by_sha) != len(rows):
        raise RuntimeError("duplicate SHA in future expert predictions")

    columns = set(rows[0])
    expert_column = requested_expert_column
    if expert_column not in columns:
        candidates = sorted(
            name for name in columns
            if "CARE_RGCondAttn_Dedup_FixKeep" in name
            and name.endswith("__expert_prob")
        )
        if len(candidates) != 1:
            raise RuntimeError(
                "future expert column missing; candidates=" + repr(candidates)
            )
        expert_column = candidates[0]

    required = {"sha256", "Global24__prob", "A25__score", "A25__routed", expert_column}
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError("future expert prediction columns missing: " + repr(missing))

    aligned = []
    for index, row in enumerate(target_rows):
        sha = norm_sha(row.get("sha256"))
        if sha not in by_sha:
            raise RuntimeError("future expert prediction is missing SHA: " + sha)
        record = by_sha[sha]
        if str(record.get("year", "")).strip():
            if int(record["year"]) != int(target_year[index]):
                raise RuntimeError("future expert year mismatch: " + sha)
        aligned.append(record)

    result = {
        "base": np.asarray([float(row["Global24__prob"]) for row in aligned]),
        "a25_score": np.asarray([float(row["A25__score"]) for row in aligned]),
        "a25_route": np.asarray(
            [int(float(row["A25__routed"])) for row in aligned], dtype=bool
        ),
        "expert": np.asarray([float(row[expert_column]) for row in aligned]),
    }
    for name, value in result.items():
        if value.shape != (len(target_rows),) or not np.isfinite(value).all():
            raise RuntimeError("invalid aligned future array: " + name)
    return result, expert_column


def metric_row(
    method: str,
    labels: np.ndarray,
    probability: np.ndarray,
    base: np.ndarray,
    route: np.ndarray,
    scope: str,
    year: int | None = None,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    coarse = np.asarray(base, dtype=np.float64)
    selected = np.asarray(route, dtype=bool)
    prediction = (p >= 0.5).astype(np.int64)
    base_prediction = (coarse >= 0.5).astype(np.int64)
    base_correct = base_prediction == y
    final_correct = prediction == y
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    row = {
        "scope": scope,
        "method": method,
        "n": int(len(y)),
        "routed_n": int(np.sum(selected)),
        "routed_ratio": float(np.mean(selected)),
        "fixed": int(np.sum(~base_correct & final_correct)),
        "broken": int(np.sum(base_correct & ~final_correct)),
        "net": int(np.sum(final_correct) - np.sum(base_correct)),
        "correction_precision": float(
            np.sum(~base_correct & final_correct)
            / max(1, np.sum(base_correct != final_correct))
        ),
        "ACC": float(accuracy_score(y, prediction)),
        "BA": float(balanced_accuracy_score(y, prediction)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "diagnostic_only": True,
    }
    if year is not None:
        row["year"] = int(year)
    return row


def routing_row(
    method: str,
    labels: np.ndarray,
    base: np.ndarray,
    route: np.ndarray,
    score: np.ndarray,
    scope: str,
    year: int | None = None,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    selected = np.asarray(route, dtype=bool)
    error = (np.asarray(base, dtype=np.float64) >= 0.5) != y
    accepted = ~selected
    captured = int(np.sum(selected & error))
    row = {
        "scope": scope,
        "method": method,
        "n": int(len(y)),
        "routed_n": int(np.sum(selected)),
        "routed_ratio": float(np.mean(selected)),
        "base_error_n": int(np.sum(error)),
        "routed_base_error_n": captured,
        "error_recall": float(captured / np.sum(error)) if np.any(error) else 0.0,
        "route_error_precision": float(captured / np.sum(selected)) if np.any(selected) else 0.0,
        "accepted_n": int(np.sum(accepted)),
        "accepted_error_n": int(np.sum(accepted & error)),
        "accepted_risk": float(np.mean(error[accepted])) if np.any(accepted) else 0.0,
        "score_mean": float(np.mean(score)),
        "score_max": float(np.max(score)),
        "diagnostic_only": True,
    }
    if year is not None:
        row["year"] = int(year)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--p72-oof", type=Path, default=Path(
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"
    ))
    parser.add_argument("--future-expert-predictions", type=Path, default=Path(
        "artifacts/care-rgcondattn-dedup-bounded-future-diagnostic-v1/"
        "future_predictions_diagnostic.csv"
    ))
    parser.add_argument("--future-expert-column", default=DEFAULT_EXPERT_COLUMN)
    parser.add_argument("--years", default="2018,2019,2020,2021")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-malcertain-router-future-v1"
    ))
    args = parser.parse_args()
    years = parse_years(args.years)
    if args.inner_folds < 2 or not 0.0 < args.budget < 1.0:
        raise SystemExit("invalid inner folds or budget")
    args.output.mkdir(parents=True, exist_ok=True)

    print("Frozen 2018-2021 MalCertain-inspired router audit", flush=True)
    print("Training all reliability components on 2017 only.", flush=True)
    (
        manifest,
        source_rows,
        clean_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    ) = mv.v4.audit.load_inputs(args)
    source_y = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
    if len(source_y) != 799:
        raise RuntimeError("expected the fixed 799-APK 2017 source cohort")

    requested = {str(year) for year in years}
    target_rows = [
        row for row in clean_rows
        if str(row.get("metadata_year", "")).strip() in requested
    ]
    target_year = np.asarray(
        [int(row["metadata_year"]) for row in target_rows], dtype=np.int64
    )
    if len(target_rows) != 1989 or tuple(sorted(set(target_year.tolist()))) != years:
        raise RuntimeError(
            "unexpected frozen future cohort: n=" + str(len(target_rows))
            + " years=" + repr(sorted(set(target_year.tolist())))
        )

    # Cross-fitted 2017 predictions are the only inputs used to learn error risk.
    source_oof, oof_names = mv.proper_oof_views(
        source_rows,
        source_y,
        compact_by_sha,
        compact_schema,
        original_names,
        args.inner_folds,
        args.seed,
    )
    source_matrix = mv.probability_matrix(source_oof)
    source_error = (
        (source_matrix[:, 0] >= 0.5).astype(np.int64) != source_y
    ).astype(np.int64)
    source_paper, paper_names = router_features(source_matrix, "paper9")
    source_code, code_names = router_features(source_matrix, "official7")
    paper_router = fit_error_router(source_paper, source_error, args.seed + 50000)
    code_router = fit_error_router(source_code, source_error, args.seed + 60000)
    source_paper_score = error_risk_score(paper_router, source_paper)
    source_code_score = error_risk_score(code_router, source_code)
    paper_threshold = source_quantile_threshold(source_paper_score, args.budget)
    code_threshold = source_quantile_threshold(source_code_score, args.budget)

    # Fit the five lightweight views once on all 2017 APKs, then predict future rows.
    x_source, x_target, names = mv.global_matrices(
        source_rows,
        target_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    )
    if names != oof_names:
        raise RuntimeError("Global24 feature names changed between source and future")
    groups = mv.group_indices(names)
    target_view_probability = mv.fit_raw_views(
        x_source,
        source_y,
        x_target,
        groups,
        args.seed + 70000,
    )
    target_matrix_aux = mv.probability_matrix(target_view_probability)
    frozen, expert_column = align_frozen_future(
        args.future_expert_predictions,
        target_rows,
        target_year,
        args.future_expert_column,
    )
    auxiliary_all_probability = target_matrix_aux[:, 0].copy()
    target_matrix = target_matrix_aux.copy()
    target_matrix[:, 0] = frozen["base"]

    target_paper, _ = router_features(target_matrix, "paper9")
    target_code, _ = router_features(target_matrix, "official7")
    paper_score = error_risk_score(paper_router, target_paper)
    code_score = error_risk_score(code_router, target_code)
    view_range = np.max(target_matrix, axis=1) - np.min(target_matrix, axis=1)

    # The reused artifact stores A25 as the source-reference percentile of the
    # raw Global24 ambiguity, not as the raw ambiguity itself.  Keep the frozen
    # calibrated score for routing and export the raw value separately so the
    # relationship can be audited without imposing an invalid equality check.
    raw_a25_ambiguity = 1.0 - 2.0 * np.abs(frozen["base"] - 0.5)
    a25_frozen_vs_raw_max_abs_diff = float(
        np.max(np.abs(raw_a25_ambiguity - frozen["a25_score"]))
    )

    # All routes and final probabilities are frozen before future labels are used.
    route_by_method = {
        "A25_Frozen": frozen["a25_route"],
        "A25_YearTop25": groupwise_top_fraction(
            frozen["a25_score"], target_year, args.budget
        ),
        "ViewRange_YearTop25": groupwise_top_fraction(
            view_range, target_year, args.budget
        ),
        "MVUQ9_SVM_YearTop25": groupwise_top_fraction(
            paper_score, target_year, args.budget
        ),
        "MVUQ7_SVM_YearTop25": groupwise_top_fraction(
            code_score, target_year, args.budget
        ),
        "MVUQ9_SVM_SourceQ75": paper_score >= paper_threshold,
        "MVUQ7_SVM_SourceQ75": code_score >= code_threshold,
    }
    score_by_method = {
        "A25_Frozen": frozen["a25_score"],
        "A25_YearTop25": frozen["a25_score"],
        "ViewRange_YearTop25": view_range,
        "MVUQ9_SVM_YearTop25": paper_score,
        "MVUQ7_SVM_YearTop25": code_score,
        "MVUQ9_SVM_SourceQ75": paper_score,
        "MVUQ7_SVM_SourceQ75": code_score,
    }
    final_by_method = {
        method: routed_probability(frozen["base"], frozen["expert"], route)
        for method, route in route_by_method.items()
    }

    # Future labels are first materialized here, after route/final freezing.
    target_y = np.asarray(
        [mv.v4.audit.future.label_of(row) for row in target_rows], dtype=np.int64
    )

    annual_metrics: list[dict] = []
    annual_routing: list[dict] = []
    pooled_metrics = [metric_row(
        "Global24", target_y, frozen["base"], frozen["base"],
        np.zeros(len(target_y), dtype=bool), "pooled_2018_2021"
    )]
    pooled_routing: list[dict] = []
    for year in years:
        selected = target_year == year
        annual_metrics.append(metric_row(
            "Global24", target_y[selected], frozen["base"][selected],
            frozen["base"][selected], np.zeros(int(np.sum(selected)), dtype=bool),
            "annual", year,
        ))
        for method in route_by_method:
            annual_metrics.append(metric_row(
                method,
                target_y[selected],
                final_by_method[method][selected],
                frozen["base"][selected],
                route_by_method[method][selected],
                "annual",
                year,
            ))
            annual_routing.append(routing_row(
                method,
                target_y[selected],
                frozen["base"][selected],
                route_by_method[method][selected],
                score_by_method[method][selected],
                "annual",
                year,
            ))
    for method in route_by_method:
        pooled_metrics.append(metric_row(
            method,
            target_y,
            final_by_method[method],
            frozen["base"],
            route_by_method[method],
            "pooled_2018_2021",
        ))
        pooled_routing.append(routing_row(
            method,
            target_y,
            frozen["base"],
            route_by_method[method],
            score_by_method[method],
            "pooled_2018_2021",
        ))

    # The feature matrices already contain the base probability followed by the
    # requested metric family; export them directly to avoid recomputation drift.
    prediction_rows: list[dict] = []
    for index, row in enumerate(target_rows):
        item = {
            "sha256": norm_sha(row["sha256"]),
            "year": int(target_year[index]),
            "label": int(target_y[index]),
            "Global24__prob": float(frozen["base"][index]),
            "FrozenExpert__prob": float(frozen["expert"][index]),
            "AuxiliaryAllView__prob": float(auxiliary_all_probability[index]),
            "A25__score": float(frozen["a25_score"][index]),
            "A25__raw_ambiguity": float(raw_a25_ambiguity[index]),
            "ViewRange__score": float(view_range[index]),
            "MVUQ9_SVM__score": float(paper_score[index]),
            "MVUQ7_SVM__score": float(code_score[index]),
        }
        for column, view in enumerate(VIEW_NAMES):
            item["view_" + view + "__prob"] = float(target_matrix[index, column])
        for column, name in enumerate(paper_names):
            item["paper9_" + name] = float(target_paper[index, column])
        for column, name in enumerate(code_names):
            item["official7_" + name] = float(target_code[index, column])
        for method in route_by_method:
            item[method + "__routed"] = int(route_by_method[method][index])
            item[method + "__prob"] = float(final_by_method[method][index])
        item["diagnostic_only"] = True
        prediction_rows.append(item)

    write_csv(args.output / "future_annual_metrics_diagnostic.csv", annual_metrics)
    write_csv(args.output / "future_annual_routing_diagnostic.csv", annual_routing)
    write_csv(args.output / "future_pooled_metrics_diagnostic.csv", pooled_metrics)
    write_csv(args.output / "future_pooled_routing_diagnostic.csv", pooled_routing)
    write_csv(args.output / "future_predictions_diagnostic.csv", prediction_rows)

    auxiliary_label_disagreement = float(np.mean(
        (auxiliary_all_probability >= 0.5) != (frozen["base"] >= 0.5)
    ))
    route_counts = {
        method: {
            str(year): int(np.sum(route[target_year == year]))
            for year in years
        }
        for method, route in route_by_method.items()
    }
    audit = {
        "experiment": "care-malcertain-router-future-v1",
        "diagnostic_only": True,
        "source_year": 2017,
        "source_n": int(len(source_y)),
        "future_years": list(years),
        "future_n": int(len(target_y)),
        "budget": float(args.budget),
        "seed": int(args.seed),
        "inner_folds": int(args.inner_folds),
        "view_order": list(VIEW_NAMES),
        "paper9_metrics": list(PAPER9_NAMES),
        "official7_metrics": list(OFFICIAL7_NAMES),
        "paper9_input_dimension": int(source_paper.shape[1]),
        "official7_input_dimension": int(source_code.shape[1]),
        "source_router_error_n": int(np.sum(source_error)),
        "source_router_error_ratio": float(np.mean(source_error)),
        "source_paper9_q75_threshold": float(paper_threshold),
        "source_official7_q75_threshold": float(code_threshold),
        "future_expert_predictions": str(args.future_expert_predictions),
        "future_expert_column": expert_column,
        "future_auxiliary_all_vs_frozen_base_label_disagreement": auxiliary_label_disagreement,
        "future_auxiliary_all_vs_frozen_base_max_abs_probability_difference": float(
            np.max(np.abs(auxiliary_all_probability - frozen["base"]))
        ),
        "a25_score_semantics": (
            "source-reference percentile calibration of raw Global24 ambiguity; "
            "reused unchanged from the frozen future artifact"
        ),
        "a25_raw_ambiguity_formula": "1 - 2*abs(p-0.5)",
        "a25_frozen_score_is_expected_to_equal_raw_ambiguity": False,
        "a25_frozen_score_vs_raw_max_abs_difference": (
            a25_frozen_vs_raw_max_abs_diff
        ),
        "a25_frozen_score_range": [
            float(np.min(frozen["a25_score"])),
            float(np.max(frozen["a25_score"])),
        ],
        "a25_raw_ambiguity_range": [
            float(np.min(raw_a25_ambiguity)),
            float(np.max(raw_a25_ambiguity)),
        ],
        "route_counts_by_year": route_counts,
        "future_labels_used_for_view_training_router_training_threshold_or_routing": False,
        "future_labels_first_used_after_scores_routes_and_final_probabilities_frozen": True,
        "primary_comparison": "A25_YearTop25 versus MVUQ9_SVM_YearTop25",
        "secondary_transport_audit": "MVUQ9_SVM_SourceQ75",
        "future_warning": (
            "2018-2021 were already inspected in this project; these are "
            "development diagnostics, not an untouched final test"
        ),
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    annual_lookup = {
        (int(row["year"]), str(row["method"])): row for row in annual_metrics
    }
    routing_lookup = {
        (int(row["year"]), str(row["method"])): row for row in annual_routing
    }
    lines = [
        "CARE-Droid MalCertain-inspired future router v1",
        "",
        "2017 only: five view models, SVM error router, source thresholds.",
        "2018-2021 labels are used only after scores/routes/probabilities freeze.",
        "WARNING: future years are diagnostic-only because they were inspected before.",
        "",
        "Primary equal-budget annual comparison:",
    ]
    for year in years:
        lines.append(str(year))
        for method in (
            "Global24",
            "A25_YearTop25",
            "MVUQ9_SVM_YearTop25",
            "MVUQ7_SVM_YearTop25",
            "ViewRange_YearTop25",
        ):
            metric = annual_lookup[(year, method)]
            if method == "Global24":
                lines.append(
                    f"  {method}: BA={metric['BA']:.6f} AUC={metric['AUC']:.6f} "
                    f"F1={metric['F1']:.6f}"
                )
            else:
                routing = routing_lookup[(year, method)]
                lines.append(
                    f"  {method}: routed={metric['routed_n']}/{metric['n']} "
                    f"error_recall={routing['error_recall']:.6f} "
                    f"accepted_risk={routing['accepted_risk']:.6f} "
                    f"BA={metric['BA']:.6f} AUC={metric['AUC']:.6f} "
                    f"F1={metric['F1']:.6f} fixed={metric['fixed']} "
                    f"broken={metric['broken']} net={metric['net']}"
                )
    lines.extend([
        "",
        "Source-threshold transport route ratios:",
    ])
    for year in years:
        for method in ("MVUQ9_SVM_SourceQ75", "MVUQ7_SVM_SourceQ75"):
            metric = annual_lookup[(year, method)]
            lines.append(
                f"  {year} {method}: routed_ratio={metric['routed_ratio']:.6f} "
                f"BA={metric['BA']:.6f} F1={metric['F1']:.6f}"
            )
    (args.output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
