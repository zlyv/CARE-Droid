#!/usr/bin/env python3
"""Evaluate the source-frozen DI budget/bound on diagnostic future years."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from selection_core import (
    MEMBER_COLUMNS,
    array,
    baseline,
    decision_instability,
    evaluate,
    find_column,
    groupwise_top_fraction,
    norm_sha,
    read_csv,
    write_csv,
)


YEARS = (2018, 2019, 2020, 2021)
EXPERT_PREFERRED = "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep__ensemble3__prob"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path, default=Path(
        "artifacts/care-di-budget-bound-selection-v1/source_selection/selected_source_config.json"
    ))
    parser.add_argument("--source-scores", type=Path, default=Path(
        "artifacts/care-committee-uncertainty-router-source-v1/source_sample_scores.csv"
    ))
    parser.add_argument("--future-scores", type=Path, default=Path(
        "artifacts/care-committee-uncertainty-router-future-v1/future_sample_scores.csv"
    ))
    parser.add_argument("--future-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-di-budget-bound-selection-v1/future_diagnostic"
    ))
    args = parser.parse_args()

    selected = json.loads(args.selected_config.read_text(encoding="utf-8"))
    chosen = selected["source_selected_within_ceiling"]
    budget = float(chosen["budget"])
    cap = float(chosen["max_delta"])
    thresholds = selected["source_thresholds"]

    rows = read_csv(args.future_scores)
    if len(rows) != 1989:
        raise RuntimeError("expected 1989 fixed future rows")
    required = {"sha256", "year", "label", "p_stage1", "U_A25", *MEMBER_COLUMNS}
    if not required.issubset(rows[0]):
        raise RuntimeError("future committee score columns are incomplete")
    years = array(rows, "year", int).astype(np.int64)
    if tuple(sorted(set(years.tolist()))) != YEARS:
        raise RuntimeError("future year cohort changed")
    labels = array(rows, "label", int).astype(np.int64)
    base = array(rows, "p_stage1")
    members = np.column_stack([array(rows, name) for name in MEMBER_COLUMNS])
    scores = {"A25": array(rows, "U_A25"), "DI": decision_instability(base, members)}

    pred_rows = read_csv(args.future_predictions)
    by_sha = {norm_sha(row["sha256"]): row for row in pred_rows}
    ordered = [by_sha[norm_sha(row["sha256"])] for row in rows]
    expert_column = find_column(ordered, EXPERT_PREFERRED, ("Dedup", "ensemble3", "prob"))
    expert = array(ordered, expert_column)
    pred_base_column = find_column(ordered, "Global24__prob", ("Global24", "prob"))
    pred_base = array(ordered, pred_base_column)
    if not np.allclose(base, pred_base, atol=2e-6, rtol=0.0):
        raise RuntimeError("future Global24 probabilities changed")

    routes = {
        "A25_YearTopB": groupwise_top_fraction(scores["A25"], years, budget),
        "DI_YearTopB": groupwise_top_fraction(scores["DI"], years, budget),
        "A25_SourceThreshold": scores["A25"] >= float(thresholds["A25"]),
        "DI_SourceThreshold": scores["DI"] >= float(thresholds["DI"]),
    }
    annual = []
    for year in YEARS:
        index = years == year
        item = baseline(labels[index], base[index])
        item.update({"year": year, "method": "Global24", "diagnostic_only": True})
        annual.append(item)
        for method, route in routes.items():
            item = evaluate(labels[index], base[index], expert[index], route[index])
            item.update({
                "year": year, "method": method, "budget_selected_on_2017": budget,
                "max_delta_selected_on_2017": cap, "diagnostic_only": True,
            })
            annual.append(item)

    pooled = []
    item = baseline(labels, base)
    item.update({"scope": "pooled_2018_2021", "method": "Global24", "diagnostic_only": True})
    pooled.append(item)
    for method, route in routes.items():
        item = evaluate(labels, base, expert, route)
        item.update({
            "scope": "pooled_2018_2021", "method": method,
            "budget_selected_on_2017": budget,
            "max_delta_selected_on_2017": cap, "diagnostic_only": True,
        })
        pooled.append(item)

    sample_rows = []
    for index, row in enumerate(rows):
        sample_rows.append({
            "sha256": row["sha256"], "year": int(years[index]),
            "label": int(labels[index]), "p_stage1": float(base[index]),
            "p_expert": float(expert[index]), "U_A25": float(scores["A25"][index]),
            "U_DI": float(scores["DI"][index]),
            **{"route_" + name: int(route[index]) for name, route in routes.items()},
            "p_final_DI_YearTopB": float(expert[index] if routes["DI_YearTopB"][index] else base[index]),
            "diagnostic_only": True,
        })
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "future_annual_metrics.csv", annual)
    write_csv(args.output / "future_pooled_metrics.csv", pooled)
    write_csv(args.output / "future_sample_predictions.csv", sample_rows)
    protocol = {
        "experiment": "CARE-Droid source-selected DI budget/bound future diagnostic v1",
        "budget": budget,
        "max_delta_logit": cap,
        "selection_source": str(args.selected_config),
        "future_labels_used_for_selection": False,
        "future_diagnostic_only": True,
        "route_modes": {
            "YearTopB": "fixed source-selected quota; ranking is label-free within each year/batch",
            "SourceThreshold": "fixed 2017 score threshold; future route ratio may drift",
        },
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lookup = {(row.get("year"), row["method"]): row for row in annual}
    lines = [
        "CARE-Droid source-frozen DI budget/bound future diagnostic v1",
        "",
        "Selected on 2017 only: budget={:.2%}, max_delta(logit)={}.".format(budget, cap),
        "2018-2021 are diagnostic-only and did not change the selected values.",
        "",
    ]
    for year in YEARS:
        lines.append(str(year))
        for method in ("Global24", "A25_YearTopB", "DI_YearTopB", "A25_SourceThreshold", "DI_SourceThreshold"):
            row = lookup[(year, method)]
            if method == "Global24":
                lines.append("  Global24: BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f}".format(**row))
            else:
                lines.append(
                    "  {method}: routed={routed_n}/{n} ({routed_ratio:.2%}) "
                    "BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} "
                    "fixed={fixed} broken={broken} net={net}".format(**row)
                )
    lines.extend([
        "",
        "Do not use these future results to revise budget, max_delta, or router definition.",
    ])
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
