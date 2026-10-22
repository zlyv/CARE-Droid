#!/usr/bin/env python3
"""Strict source-only audit of a MalCertain-inspired reliability router.

The established Global24 base probability and bounded local expert probability
are frozen.  Only the decision about which APK enters Stage II is changed.

For every outer fold, five inexpensive Logistic heads are obtained from the
Global24 feature set: the complete view and four semantic sub-views.  Their
strictly cross-fitted probabilities on the outer-training rows form uncertainty
features used to train an SVM that predicts whether the complete-view decision
is likely to be wrong.  The highest-risk 25 percent of outer-validation APKs
are routed to the unchanged local expert.

This is an adaptation of MalCertain's correction-model idea, not an exact
reproduction: MalCertain obtains repeated predictions from uncertainty-aware
deep models, whereas this experiment obtains them from five static views.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
for path in (
    SCRIPT_DIR,
    SCRIPTS_ROOT,
    SCRIPTS_ROOT / "care-multiview-conformal-router-source-v1",
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script",
    SCRIPTS_ROOT / "care-rgcondattn-dedup-bounded-source-v1",
    # Local audit-bundle fallbacks.  On the experiment server these modules
    # normally already live together under scripts/, so the paths are harmless.
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

from uncertainty_router_core import (  # noqa: E402
    OFFICIAL7_NAMES,
    PAPER9_NAMES,
    error_risk_score,
    fit_error_router,
    router_features,
    top_fraction,
)


VIEW_NAMES = mv.VIEW_NAMES
FROZEN_EXPERT_NAME = "CARE_RGCondAttn_Dedup_FixKeep"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
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


def per_fold_metric_rows(
    folds: np.ndarray,
    labels: np.ndarray,
    base: np.ndarray,
    final_probability: dict[str, np.ndarray],
    route_all: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for fold in sorted(np.unique(folds)):
        selected = folds == fold
        for method, probability in final_probability.items():
            if method == "Global24":
                route = np.zeros(int(np.sum(selected)), dtype=bool)
            else:
                route_name = method.split("__", 1)[0]
                route = route_all[route_name][selected]
            metric = mv.v4.v2.metric_rows(
                labels[selected],
                {method: probability[selected]},
                base[selected],
                route,
            )[0]
            rows.append({"fold": int(fold), **metric})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/p1-features/features.csv")
    )
    parser.add_argument(
        "--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json")
    )
    parser.add_argument(
        "--p72-oof",
        type=Path,
        default=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),
    )
    parser.add_argument(
        "--expert-predictions",
        type=Path,
        default=Path(
            "artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv"
        ),
    )
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/care-malcertain-router-source-v1"),
    )
    args = parser.parse_args()
    if args.inner_folds < 2:
        raise SystemExit("--inner-folds must be >= 2")
    if not 0.0 < args.budget < 1.0:
        raise SystemExit("--budget must be in (0,1)")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = (
        mv.v4.audit.load_inputs(args)
    )
    labels = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    years = {str(row.get("metadata_year", "")).strip() for row in source_rows}
    if years != {"2017"}:
        raise RuntimeError("source-only guard failed: " + repr(years))
    frozen = mv.load_frozen_predictions(args.expert_predictions, manifest)

    route_names = (
        "A25",
        "ViewRange25",
        "MVUQ9_SVM25",
        "MVUQ7_SVM25",
    )
    route_all = {name: np.zeros(len(labels), dtype=bool) for name in route_names}
    score_all = {name: np.full(len(labels), np.nan) for name in route_names}
    view_probability_all = {
        view: np.full(len(labels), np.nan) for view in VIEW_NAMES
    }
    paper_feature_all: dict[str, np.ndarray] = {}
    code_feature_all: dict[str, np.ndarray] = {}
    fold_audit: list[dict] = []
    per_fold_routing: list[dict] = []

    print("Strict 2017 MalCertain-inspired reliability router audit", flush=True)
    for fold in sorted(np.unique(folds)):
        outer_train = np.flatnonzero(folds != fold)
        outer_validation = np.flatnonzero(folds == fold)
        train_rows = [source_rows[index] for index in outer_train]
        validation_rows = [source_rows[index] for index in outer_validation]
        train_y = labels[outer_train]
        print(
            f"  fold {int(fold)} train={len(outer_train)} "
            f"validation={len(outer_validation)}",
            flush=True,
        )

        # Every training row receives probabilities from models that did not
        # train on that row.  These are the only rows used to learn reliability.
        oof_probability, oof_names = mv.proper_oof_views(
            train_rows,
            train_y,
            compact_by_sha,
            compact_schema,
            original_names,
            args.inner_folds,
            args.seed + 10000 * int(fold),
        )
        x_train, x_validation, names = mv.global_matrices(
            train_rows,
            validation_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        if names != oof_names:
            raise RuntimeError("Global24 names changed between OOF and final fit")
        groups = mv.group_indices(names)
        validation_probability = mv.fit_raw_views(
            x_train,
            train_y,
            x_validation,
            groups,
            args.seed + 40000 * int(fold),
        )

        oof_matrix = mv.probability_matrix(oof_probability)
        validation_matrix_aux = mv.probability_matrix(validation_probability)
        auxiliary_disagreement = float(np.mean(
            (validation_matrix_aux[:, 0] >= 0.5)
            != (frozen["base"][outer_validation] >= 0.5)
        ))
        # Use the actual frozen operational base as the complete-view member at
        # validation time.  The four auxiliary views are learned in this fold.
        validation_matrix = validation_matrix_aux.copy()
        validation_matrix[:, 0] = frozen["base"][outer_validation]

        training_error = (
            (oof_matrix[:, 0] >= 0.5).astype(np.int64) != train_y
        ).astype(np.int64)
        train_paper, paper_names = router_features(oof_matrix, "paper9")
        validation_paper, _ = router_features(validation_matrix, "paper9")
        train_code, code_names = router_features(oof_matrix, "official7")
        validation_code, _ = router_features(validation_matrix, "official7")

        paper_router = fit_error_router(
            train_paper, training_error, args.seed + 50000 * int(fold)
        )
        code_router = fit_error_router(
            train_code, training_error, args.seed + 60000 * int(fold)
        )
        paper_score = error_risk_score(paper_router, validation_paper)
        code_score = error_risk_score(code_router, validation_code)
        view_range = (
            np.max(validation_matrix, axis=1) - np.min(validation_matrix, axis=1)
        )

        fold_routes = {
            "A25": frozen["a25_route"][outer_validation],
            "ViewRange25": top_fraction(view_range, args.budget),
            "MVUQ9_SVM25": top_fraction(paper_score, args.budget),
            "MVUQ7_SVM25": top_fraction(code_score, args.budget),
        }
        fold_scores = {
            "A25": frozen["a25_score"][outer_validation],
            "ViewRange25": view_range,
            "MVUQ9_SVM25": paper_score,
            "MVUQ7_SVM25": code_score,
        }
        for route_name in route_names:
            route_all[route_name][outer_validation] = fold_routes[route_name]
            score_all[route_name][outer_validation] = fold_scores[route_name]
            per_fold_routing.append(mv.routing_row(
                route_name,
                labels[outer_validation],
                frozen["base"][outer_validation],
                fold_routes[route_name],
                fold_scores[route_name],
                fold,
            ))
        for column, view in enumerate(VIEW_NAMES):
            view_probability_all[view][outer_validation] = validation_matrix[:, column]
        for column, name in enumerate(paper_names):
            paper_feature_all.setdefault(name, np.full(len(labels), np.nan))[
                outer_validation
            ] = validation_paper[:, column]
        for column, name in enumerate(code_names):
            code_feature_all.setdefault(name, np.full(len(labels), np.nan))[
                outer_validation
            ] = validation_code[:, column]

        count = np.bincount(training_error, minlength=2)
        fold_audit.append({
            "fold": int(fold),
            "outer_train_n": int(len(outer_train)),
            "outer_validation_n": int(len(outer_validation)),
            "router_train_correct_n": int(count[0]),
            "router_train_error_n": int(count[1]),
            "router_train_error_ratio": float(np.mean(training_error)),
            "paper9_input_dimension": int(train_paper.shape[1]),
            "official7_input_dimension": int(train_code.shape[1]),
            "auxiliary_all_vs_frozen_base_label_disagreement": auxiliary_disagreement,
        })

    for collection_name, collection in (
        ("scores", score_all),
        ("views", view_probability_all),
        ("paper features", paper_feature_all),
        ("official-code features", code_feature_all),
    ):
        if any(np.any(~np.isfinite(value)) for value in collection.values()):
            raise RuntimeError("incomplete " + collection_name)

    routing_summary = [
        mv.routing_row(
            name, labels, frozen["base"], route_all[name], score_all[name]
        )
        for name in route_names
    ]
    final_probability = {"Global24": frozen["base"]}
    for name in route_names:
        method = name + "__" + FROZEN_EXPERT_NAME
        final_probability[method] = mv.routed_probability(
            frozen["base"], frozen["expert"], route_all[name]
        )

    metric_summary = []
    for method, probability in final_probability.items():
        route = np.zeros(len(labels), dtype=bool)
        if method != "Global24":
            route = route_all[method.split("__", 1)[0]]
        metric_summary.extend(mv.v4.v2.metric_rows(
            labels, {method: probability}, frozen["base"], route
        ))
    per_fold_metric = per_fold_metric_rows(
        folds, labels, frozen["base"], final_probability, route_all
    )

    prediction_rows = []
    for index, row in enumerate(manifest):
        item = {
            "sha256": row["sha256"],
            "label": int(labels[index]),
            "fold": int(folds[index]),
            "Global24__prob": float(frozen["base"][index]),
            "FrozenDedupExpert__prob": float(frozen["expert"][index]),
        }
        for view in VIEW_NAMES:
            item["view_" + view + "__prob"] = float(view_probability_all[view][index])
        for name, value in paper_feature_all.items():
            item["paper9__" + name] = float(value[index])
        for name, value in code_feature_all.items():
            item["official7__" + name] = float(value[index])
        for name in route_names:
            item[name + "__score"] = float(score_all[name][index])
            item[name + "__routed"] = int(route_all[name][index])
            item[name + "__final_prob"] = float(
                final_probability[name + "__" + FROZEN_EXPERT_NAME][index]
            )
        prediction_rows.append(item)

    write_csv(args.output / "routing_summary.csv", routing_summary)
    write_csv(args.output / "source_metrics.csv", metric_summary)
    write_csv(args.output / "source_per_fold_metrics.csv", per_fold_metric)
    write_csv(args.output / "source_per_fold_routing.csv", per_fold_routing)
    write_csv(args.output / "fold_audit.csv", fold_audit)
    write_csv(args.output / "source_predictions.csv", prediction_rows)

    architecture = {
        "experiment": "CARE-Droid MalCertain-inspired multi-view reliability router v1",
        "evaluation": "strict fixed 2017 outer folds only",
        "future_data_loaded": False,
        "base": "frozen Global24 outer-fold OOF probability",
        "expert": "frozen deduplicated bounded RGCondAttn probability",
        "changed_component": "route only",
        "route_budget": float(args.budget),
        "view_order": list(VIEW_NAMES),
        "uncertainty_members": (
            "complete Global24 view plus Manifest, code/call, sensitive-behaviour, "
            "and graph-statistics heads"
        ),
        "router_target": "whether cross-fitted complete-view prediction is wrong",
        "router": "standardized RBF-SVM with balanced class weights",
        "paper9_metrics": list(PAPER9_NAMES),
        "official_public_code_metrics": list(OFFICIAL7_NAMES),
        "important_boundary": (
            "This borrows MalCertain's uncertainty-feature/correction-model idea. "
            "It is not an exact reproduction because the repeated predictions come "
            "from semantic feature views rather than five Bayesian/ensemble methods."
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output / "architecture.json").write_text(
        json.dumps(architecture, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    route_by_name = {row["method"]: row for row in routing_summary}
    metric_by_name = {row["method"]: row for row in metric_summary}
    lines = [
        "CARE-Droid MalCertain-inspired multi-view reliability router v1",
        "",
        "Only the routing decision changes; Global24 and the local expert are frozen.",
        "Strict source-only nested evaluation; no 2018-2021 row is loaded.",
        f"Stage-II budget: {100.0 * float(args.budget):.2f}%.",
        "",
        "Routing quality against frozen Global24 errors:",
    ]
    for name in route_names:
        row = route_by_name[name]
        lines.append(
            f"{name}: routed={int(row['routed_n'])}/{int(row['n'])} "
            f"({100.0 * float(row['routed_ratio']):.2f}%) "
            f"error_recall={float(row['error_recall']):.6f} "
            f"precision={float(row['route_error_precision']):.6f} "
            f"accepted_risk={float(row['accepted_risk']):.6f}"
        )
    lines.extend(["", "End-to-end metrics with the same frozen local expert:"])
    for method, row in metric_by_name.items():
        lines.append(
            f"{method}: BA={float(row['BA']):.6f} AUC={float(row['AUC']):.6f} "
            f"F1={float(row['F1']):.6f} fixed={int(row['fixed'])} "
            f"broken={int(row['broken'])} net={int(row['net'])}"
        )
    lines.extend([
        "",
        "Primary comparison:",
        "Compare MVUQ9_SVM25 with A25 at the same budget. It should capture more",
        "Global24 errors, lower accepted-set risk, and avoid reducing final BA.",
        "MVUQ7_SVM25 is a sensitivity check matching the public extractor's seven",
        "implemented metric types. ViewRange25 tests whether an SVM is needed at all.",
        "Do not tune this router using future-year labels.",
    ])
    (args.output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
