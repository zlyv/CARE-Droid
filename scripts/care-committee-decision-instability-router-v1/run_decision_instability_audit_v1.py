#!/usr/bin/env python3
"""Post-hoc equal-budget audit of committee decision-instability routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decision_instability_core import (
    MEMBER_COLUMNS,
    baseline_metrics,
    decision_instability,
    evaluate_route,
    groupwise_top_fraction,
    overlap_row,
    read_csv,
    write_csv,
)


SOURCE_METHODS = ("A25", "Committee_PE25", "Committee_MI25", "Committee_DI25")
FUTURE_METHODS = (
    "A25_YearTop25",
    "Committee_PE_YearTop25",
    "Committee_MI_YearTop25",
    "Committee_DI_YearTop25",
)
YEARS = (2018, 2019, 2020, 2021)


def required_columns(rows: list[dict], names: set[str], context: str) -> None:
    if not rows:
        raise RuntimeError(context + " input is empty")
    missing = sorted(names - set(rows[0]))
    if missing:
        raise RuntimeError(context + " columns missing: " + repr(missing))


def array(rows: list[dict], name: str, dtype=float) -> np.ndarray:
    return np.asarray([dtype(row[name]) for row in rows])


def member_matrix(rows: list[dict]) -> np.ndarray:
    return np.column_stack([array(rows, name) for name in MEMBER_COLUMNS])


def source_audit(args: argparse.Namespace) -> dict:
    rows = read_csv(args.source_scores)
    required_columns(rows, {
        "sha256", "label", "fold", "p_stage1", "p_frozen_bounded_expert",
        *MEMBER_COLUMNS, "p_bar", "U_PE", "U_MI", "U_A25",
        "route_A25", "route_PE", "route_MI",
    }, "source")
    if len(rows) != 799:
        raise RuntimeError("expected 799 source score rows")
    labels = array(rows, "label", int).astype(np.int64)
    folds = array(rows, "fold", int).astype(np.int64)
    base = array(rows, "p_stage1")
    expert = array(rows, "p_frozen_bounded_expert")
    members = member_matrix(rows)
    detail = decision_instability(base, members)
    if not np.allclose(detail["p_bar"], array(rows, "p_bar"), atol=1e-10):
        raise RuntimeError("stored committee mean changed")
    score = {
        "A25": array(rows, "U_A25"),
        "Committee_PE25": array(rows, "U_PE"),
        "Committee_MI25": array(rows, "U_MI"),
        "Committee_DI25": detail["uncertainty"],
    }
    route = {
        "A25": array(rows, "route_A25", int).astype(bool),
        "Committee_PE25": array(rows, "route_PE", int).astype(bool),
        "Committee_MI25": array(rows, "route_MI", int).astype(bool),
        "Committee_DI25": groupwise_top_fraction(
            detail["uncertainty"], folds, args.budget
        ),
    }
    # Recompute old equal-budget routes and fail if the persisted audit changed.
    for method in ("Committee_PE25", "Committee_MI25"):
        recomputed = groupwise_top_fraction(score[method], folds, args.budget)
        if not np.array_equal(recomputed, route[method]):
            raise RuntimeError(method + " stored route is inconsistent")
    metrics = [baseline_metrics(labels, base)] + [
        evaluate_route(method, labels, base, expert, route[method], score[method])
        for method in SOURCE_METHODS
    ]
    overlap = []
    for old_method in SOURCE_METHODS[:-1]:
        overlap.append(overlap_row(
            "Committee_DI25", route["Committee_DI25"],
            old_method, route[old_method], scope="source_2017"
        ))

    sample_rows = []
    final_di = np.where(route["Committee_DI25"], expert, base)
    for index, row in enumerate(rows):
        item = dict(row)
        item.update({
            "committee_support_for_stage1": float(detail["committee_support"][index]),
            "U_DI": float(detail["uncertainty"][index]),
            "support_lr": float(detail["support_by_member"][index, 0]),
            "support_rf": float(detail["support_by_member"][index, 1]),
            "support_xgb": float(detail["support_by_member"][index, 2]),
            "support_svm": float(detail["support_by_member"][index, 3]),
            "route_DI": int(route["Committee_DI25"][index]),
            "final_probability_DI": float(final_di[index]),
        })
        sample_rows.append(item)
    args.source_output.mkdir(parents=True, exist_ok=True)
    write_csv(args.source_output / "source_router_metrics.csv", metrics)
    write_csv(args.source_output / "source_route_overlap.csv", overlap)
    write_csv(args.source_output / "source_sample_scores.csv", sample_rows)
    return {"metrics": metrics, "overlap": overlap}


def future_audit(args: argparse.Namespace) -> dict:
    rows = read_csv(args.future_scores)
    required_columns(rows, {
        "sha256", "year", "label", "p_stage1", "p_frozen_bounded_expert",
        *MEMBER_COLUMNS, "p_bar", "U_PE", "U_MI", "U_A25",
        "route_A25", "route_PE", "route_MI",
    }, "future")
    if len(rows) != 1989:
        raise RuntimeError("expected 1989 future score rows")
    years = array(rows, "year", int).astype(np.int64)
    if tuple(sorted(set(years.tolist()))) != YEARS:
        raise RuntimeError("future year cohort changed")
    base = array(rows, "p_stage1")
    expert = array(rows, "p_frozen_bounded_expert")
    members = member_matrix(rows)
    detail = decision_instability(base, members)
    if not np.allclose(detail["p_bar"], array(rows, "p_bar"), atol=1e-10):
        raise RuntimeError("stored future committee mean changed")
    score = {
        "A25_YearTop25": array(rows, "U_A25"),
        "Committee_PE_YearTop25": array(rows, "U_PE"),
        "Committee_MI_YearTop25": array(rows, "U_MI"),
        "Committee_DI_YearTop25": detail["uncertainty"],
    }
    route = {
        "A25_YearTop25": array(rows, "route_A25", int).astype(bool),
        "Committee_PE_YearTop25": array(rows, "route_PE", int).astype(bool),
        "Committee_MI_YearTop25": array(rows, "route_MI", int).astype(bool),
        "Committee_DI_YearTop25": groupwise_top_fraction(
            detail["uncertainty"], years, args.budget
        ),
    }
    for method in ("Committee_PE_YearTop25", "Committee_MI_YearTop25"):
        recomputed = groupwise_top_fraction(score[method], years, args.budget)
        if not np.array_equal(recomputed, route[method]):
            raise RuntimeError(method + " stored future route is inconsistent")

    # These labels were already exposed by the completed frozen diagnostic,
    # but even this post-hoc audit materializes them only after U_DI and every
    # equal-budget route have frozen.  They are evaluation-only.
    labels = array(rows, "label", int).astype(np.int64)

    metrics = []
    overlap = []
    for year in YEARS:
        selected = years == year
        metrics.append(baseline_metrics(labels[selected], base[selected], year=year))
        for method in FUTURE_METHODS:
            metrics.append(evaluate_route(
                method,
                labels[selected],
                base[selected],
                expert[selected],
                route[method][selected],
                score[method][selected],
                year=year,
            ))
        for old_method in FUTURE_METHODS[:-1]:
            overlap.append(overlap_row(
                "Committee_DI_YearTop25", route["Committee_DI_YearTop25"][selected],
                old_method, route[old_method][selected], scope=str(year)
            ))

    sample_rows = []
    final_di = np.where(route["Committee_DI_YearTop25"], expert, base)
    for index, row in enumerate(rows):
        item = dict(row)
        item.update({
            "committee_support_for_stage1": float(detail["committee_support"][index]),
            "U_DI": float(detail["uncertainty"][index]),
            "support_lr": float(detail["support_by_member"][index, 0]),
            "support_rf": float(detail["support_by_member"][index, 1]),
            "support_xgb": float(detail["support_by_member"][index, 2]),
            "support_svm": float(detail["support_by_member"][index, 3]),
            "route_DI": int(route["Committee_DI_YearTop25"][index]),
            "final_probability_DI": float(final_di[index]),
            "diagnostic_only": True,
        })
        sample_rows.append(item)
    args.future_output.mkdir(parents=True, exist_ok=True)
    write_csv(args.future_output / "future_annual_metrics.csv", metrics)
    write_csv(args.future_output / "future_route_overlap.csv", overlap)
    write_csv(args.future_output / "future_sample_scores.csv", sample_rows)
    return {"metrics": metrics, "overlap": overlap}


def summary_text(source: dict, future: dict, budget: float) -> str:
    lines = [
        "CARE-Droid committee decision-instability router v1",
        "",
        "U_DI = 1 - mean committee support for the frozen Stage-I hard decision.",
        "Equivalent: committee mean probability assigned to the class opposite Stage I.",
        "No retraining; reuses strict OOF/source-only calibrated committee probabilities.",
        f"Stage-II budget: {100.0 * budget:.2f}%.",
        "",
        "Source 2017:",
    ]
    for row in source["metrics"]:
        if row["method"] == "Global24":
            lines.append(
                f"Global24: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f}"
            )
        else:
            lines.append(
                f"{row['method']}: routed={row['routed_n']}/{row['n']} "
                f"error_recall={row['error_recall']:.6f} "
                f"highconf_error_recall={row['high_confidence_error_recall']:.6f} "
                f"RER@25={row['repairable_error_recall_at_budget']:.6f} "
                f"HarmfulCapture={row['harmful_capture']:.6f} "
                f"UnrepairableCapture={row['unrepairable_capture']:.6f} "
                f"BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} "
                f"fixed={row['fixed']} broken={row['broken']} net={row['net']}"
            )
    lines.extend(["", "Future diagnostic:"])
    lookup = {
        (int(row["year"]), str(row["method"])): row
        for row in future["metrics"]
    }
    for year in YEARS:
        lines.append(str(year))
        for method in ("Global24", *FUTURE_METHODS):
            row = lookup[(year, method)]
            if method == "Global24":
                lines.append(
                    f"  Global24: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
                    f"F1={row['F1']:.6f}"
                )
            else:
                lines.append(
                    f"  {method}: routed={row['routed_n']}/{row['n']} "
                    f"error_recall={row['error_recall']:.6f} "
                    f"highconf_error_recall={row['high_confidence_error_recall']:.6f} "
                    f"RER@25={row['repairable_error_recall_at_budget']:.6f} "
                    f"HarmfulCapture={row['harmful_capture']:.6f} "
                    f"UnrepairableCapture={row['unrepairable_capture']:.6f} "
                    f"accepted_risk={row['accepted_risk']:.6f} "
                    f"BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} "
                    f"fixed={row['fixed']} broken={row['broken']} net={row['net']}"
                )
    lines.extend([
        "",
        "IMPORTANT:",
        "Future years are post-hoc diagnostic-only and were not used to define U_DI.",
        "Do not tune this score using the future results.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scores", type=Path, default=Path(
        "artifacts/care-committee-uncertainty-router-source-v1/source_sample_scores.csv"
    ))
    parser.add_argument("--future-scores", type=Path, default=Path(
        "artifacts/care-committee-uncertainty-router-future-v1/future_sample_scores.csv"
    ))
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--source-output", type=Path, default=Path(
        "artifacts/care-committee-decision-instability-router-source-v1"
    ))
    parser.add_argument("--future-output", type=Path, default=Path(
        "artifacts/care-committee-decision-instability-router-future-v1"
    ))
    args = parser.parse_args()
    if not 0.0 < args.budget < 1.0:
        raise SystemExit("budget must be in (0,1)")
    source = source_audit(args)
    future = future_audit(args)
    text = summary_text(source, future, args.budget)
    (args.source_output / "summary.txt").write_text(text, encoding="utf-8")
    (args.future_output / "summary.txt").write_text(text, encoding="utf-8")
    protocol = {
        "experiment": "CARE-Droid committee decision-instability router v1",
        "formula": "U_DI = 1 - mean_m support(member_m, frozen Stage-I hard decision)",
        "equivalent_formula": (
            "if Stage-I predicts malware: 1-p_bar; otherwise: p_bar"
        ),
        "member_probabilities_retrained": False,
        "source_probabilities": "strict 2017 OOF from committee router v1",
        "future_probabilities": "four members and calibrators trained on 2017 only",
        "budget": float(args.budget),
        "stage2": "same frozen CARE RGCondAttn Dedup BoundedCorrection",
        "future_diagnostic_only": True,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    for output in (args.source_output, args.future_output):
        (output / "protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
