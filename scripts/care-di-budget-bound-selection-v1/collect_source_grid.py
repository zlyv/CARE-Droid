#!/usr/bin/env python3
"""Collect exact 2017 OOF max-delta retraining runs and select source settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from selection_core import (
    MEMBER_COLUMNS,
    array,
    baseline,
    cap_tag,
    decision_instability,
    evaluate,
    find_column,
    groupwise_top_fraction,
    norm_sha,
    parse_float_list,
    read_csv,
    route_threshold,
    write_csv,
)


EXPERT_PREFERRED = "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep__expert_prob"


def load_committee(path: Path) -> dict:
    rows = read_csv(path)
    required = {"sha256", "label", "fold", "p_stage1", "U_A25", *MEMBER_COLUMNS}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("committee source scores are missing required columns")
    if len(rows) != 799:
        raise RuntimeError("expected 799 fixed 2017 rows")
    base = array(rows, "p_stage1")
    members = np.column_stack([array(rows, name) for name in MEMBER_COLUMNS])
    return {
        "rows": rows,
        "sha": [norm_sha(row["sha256"]) for row in rows],
        "labels": array(rows, "label", int).astype(np.int64),
        "folds": array(rows, "fold", int).astype(np.int64),
        "base": base,
        "scores": {
            "A25": array(rows, "U_A25"),
            "DI": decision_instability(base, members),
        },
    }


def load_cap_predictions(root: Path, cap: float, source: dict) -> np.ndarray:
    path = root / cap_tag(cap) / "source_predictions.csv"
    rows = read_csv(path)
    by_sha = {norm_sha(row["sha256"]): row for row in rows}
    if len(by_sha) != len(rows):
        raise RuntimeError("duplicate SHA in " + str(path))
    if set(by_sha) != set(source["sha"]):
        missing = sorted(set(source["sha"]) - set(by_sha))[:3]
        extra = sorted(set(by_sha) - set(source["sha"]))[:3]
        raise RuntimeError("cap predictions misaligned; missing=%r extra=%r" % (missing, extra))
    ordered = [by_sha[sha] for sha in source["sha"]]
    expert_column = find_column(ordered, EXPERT_PREFERRED, ("Dedup", "expert_prob"))
    expert = array(ordered, expert_column)
    base_column = find_column(ordered, "Global24__prob", ("Global24", "prob"))
    base = array(ordered, base_column)
    if not np.allclose(base, source["base"], atol=2e-6, rtol=0.0):
        raise RuntimeError("Global24 probabilities changed for max_delta=%s" % cap)
    labels = array(ordered, "label", int)
    folds = array(ordered, "fold", int)
    if not np.array_equal(labels, source["labels"]) or not np.array_equal(folds, source["folds"]):
        raise RuntimeError("labels/folds changed for max_delta=%s" % cap)
    return expert


def select_row(rows: list[dict], budget_ceiling: float | None = None) -> dict:
    eligible = [row for row in rows if row["router"] == "DI"]
    if budget_ceiling is not None:
        eligible = [row for row in eligible if float(row["budget"]) <= budget_ceiling + 1e-12]
    if not eligible:
        raise RuntimeError("no eligible DI configuration")
    return max(
        eligible,
        key=lambda row: (
            float(row["BA"]),
            int(row["net"]),
            -int(row["broken"]),
            -float(row["budget"]),
            -float(row["max_delta"]),
        ),
    )


def mark_pareto(rows: list[dict]) -> None:
    for row in rows:
        peers = [item for item in rows if item["router"] == row["router"]]
        dominated = any(
            float(other["budget"]) <= float(row["budget"]) + 1e-12
            and float(other["BA"]) >= float(row["BA"]) - 1e-12
            and (
                float(other["budget"]) < float(row["budget"]) - 1e-12
                or float(other["BA"]) > float(row["BA"]) + 1e-12
            )
            for other in peers
        )
        row["pareto_BA_budget"] = not dominated


def candidate_metrics(source: dict, experts: dict[float, np.ndarray], budgets: list[float]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    fold_rows: list[dict] = []
    for router, score in source["scores"].items():
        for budget in budgets:
            route = groupwise_top_fraction(score, source["folds"], budget)
            for cap, expert in experts.items():
                result = evaluate(source["labels"], source["base"], expert, route)
                result.update({"router": router, "budget": budget, "max_delta": cap})
                rows.append(result)
                for fold in sorted(np.unique(source["folds"])):
                    selected = source["folds"] == fold
                    item = evaluate(
                        source["labels"][selected], source["base"][selected],
                        expert[selected], route[selected],
                    )
                    item.update({
                        "router": router, "budget": budget,
                        "max_delta": cap, "fold": int(fold),
                    })
                    fold_rows.append(item)
    mark_pareto(rows)
    return rows, fold_rows


def nested_crossfit_selection(source: dict, experts: dict[float, np.ndarray], budgets: list[float], ceiling: float) -> tuple[list[dict], dict, np.ndarray, np.ndarray]:
    choices: list[dict] = []
    final = np.array(source["base"], copy=True)
    routed = np.zeros(len(final), dtype=bool)
    folds = sorted(np.unique(source["folds"]))
    for heldout in folds:
        development = source["folds"] != heldout
        candidates: list[dict] = []
        for budget in budgets:
            if budget > ceiling + 1e-12:
                continue
            route_all = groupwise_top_fraction(source["scores"]["DI"], source["folds"], budget)
            for cap, expert in experts.items():
                item = evaluate(
                    source["labels"][development], source["base"][development],
                    expert[development], route_all[development],
                )
                item.update({"router": "DI", "budget": budget, "max_delta": cap})
                candidates.append(item)
        chosen = select_row(candidates, ceiling)
        choices.append({
            "heldout_fold": int(heldout),
            "selected_budget": float(chosen["budget"]),
            "selected_max_delta": float(chosen["max_delta"]),
            "development_BA": float(chosen["BA"]),
            "development_net": int(chosen["net"]),
            "selection_uses_heldout_labels": False,
        })
        evaluation = source["folds"] == heldout
        local_route = groupwise_top_fraction(
            source["scores"]["DI"][evaluation],
            np.full(int(np.sum(evaluation)), heldout),
            float(chosen["budget"]),
        )
        index = np.flatnonzero(evaluation)
        routed[index] = local_route
        cap = float(chosen["max_delta"])
        final[index[local_route]] = experts[cap][index[local_route]]
    # evaluate() expects one expert array; construct an effective expert whose
    # selected positions equal the fold-specific expert. Unselected values are irrelevant.
    effective_expert = np.array(source["base"], copy=True)
    for choice in choices:
        fold = int(choice["heldout_fold"])
        cap = float(choice["selected_max_delta"])
        selected = (source["folds"] == fold) & routed
        effective_expert[selected] = experts[cap][selected]
    metrics = evaluate(source["labels"], source["base"], effective_expert, routed)
    metrics.update({
        "router": "DI",
        "selection": "leave-one-outer-fold-out",
        "budget_ceiling": ceiling,
        "unbiased_for_global_choice": True,
    })
    return choices, metrics, final, routed


def bootstrap_rows(labels: np.ndarray, methods: dict[str, np.ndarray], repeats: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    y = np.asarray(labels, dtype=np.int64)
    class_index = [np.flatnonzero(y == value) for value in (0, 1)]
    metric_values = {(name, metric): [] for name in methods for metric in ("BA", "F1")}
    point = {}
    for name, p in methods.items():
        pred = p >= 0.5
        point[(name, "BA")] = float(balanced_accuracy_score(y, pred))
        point[(name, "F1")] = float(f1_score(y, pred, zero_division=0))
    for _ in range(repeats):
        index = np.concatenate([
            rng.choice(values, size=len(values), replace=True) for values in class_index
        ])
        for name, p in methods.items():
            pred = p[index] >= 0.5
            metric_values[(name, "BA")].append(balanced_accuracy_score(y[index], pred))
            metric_values[(name, "F1")].append(f1_score(y[index], pred, zero_division=0))
    rows = []
    for (name, metric), values in metric_values.items():
        values = np.asarray(values, dtype=np.float64)
        rows.append({
            "method": name,
            "metric": metric,
            "estimate": point[(name, metric)],
            "bootstrap_repeats": repeats,
            "ci_lower_2p5": float(np.quantile(values, 0.025)),
            "ci_upper_97p5": float(np.quantile(values, 0.975)),
            "seed": seed,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scores", type=Path, default=Path(
        "artifacts/care-committee-uncertainty-router-source-v1/source_sample_scores.csv"
    ))
    parser.add_argument("--retrain-root", type=Path, default=Path(
        "artifacts/care-di-budget-bound-selection-v1/retrained_source"
    ))
    parser.add_argument("--caps", default="0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--budgets", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50")
    parser.add_argument("--budget-ceiling", type=float, default=0.25)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-di-budget-bound-selection-v1/source_selection"
    ))
    args = parser.parse_args()
    caps = parse_float_list(args.caps)
    budgets = parse_float_list(args.budgets)
    if any(value <= 0 for value in caps):
        raise SystemExit("all max-delta values must be positive")
    if any(not 0 < value < 1 for value in budgets):
        raise SystemExit("all budgets must be in (0,1)")
    if args.budget_ceiling not in budgets:
        raise SystemExit("budget ceiling must be included in budget grid")

    source = load_committee(args.source_scores)
    experts = {cap: load_cap_predictions(args.retrain_root, cap, source) for cap in caps}
    grid, per_fold = candidate_metrics(source, experts, budgets)
    fixed25 = select_row([
        row for row in grid if abs(float(row["budget"]) - 0.25) < 1e-12
    ])
    constrained = select_row(grid, args.budget_ceiling)
    unrestricted = select_row(grid, None)
    choices, nested_metrics, nested_final, nested_route = nested_crossfit_selection(
        source, experts, budgets, args.budget_ceiling
    )

    chosen_budget = float(constrained["budget"])
    chosen_cap = float(constrained["max_delta"])
    chosen_route = groupwise_top_fraction(
        source["scores"]["DI"], source["folds"], chosen_budget
    )
    chosen_final = np.where(chosen_route, experts[chosen_cap], source["base"])
    a25_route = groupwise_top_fraction(
        source["scores"]["A25"], source["folds"], chosen_budget
    )
    a25_final = np.where(a25_route, experts[chosen_cap], source["base"])

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "joint_grid_metrics.csv", grid)
    write_csv(args.output / "joint_grid_per_fold.csv", per_fold)
    write_csv(args.output / "crossfit_selection_choices.csv", choices)
    write_csv(args.output / "crossfit_selection_metrics.csv", [nested_metrics])
    write_csv(args.output / "bootstrap_intervals.csv", bootstrap_rows(
        source["labels"], {
            "Global24": source["base"],
            "DI_source_selected": chosen_final,
            "A25_same_budget_and_bound": a25_final,
            "DI_crossfit_selected": nested_final,
        }, args.bootstrap_repeats, args.bootstrap_seed,
    ))
    sample_rows = []
    for index, row in enumerate(source["rows"]):
        sample_rows.append({
            "sha256": row["sha256"],
            "label": int(source["labels"][index]),
            "fold": int(source["folds"][index]),
            "p_stage1": float(source["base"][index]),
            "U_A25": float(source["scores"]["A25"][index]),
            "U_DI": float(source["scores"]["DI"][index]),
            "route_DI_source_selected": int(chosen_route[index]),
            "route_A25_same_budget": int(a25_route[index]),
            "p_expert_source_selected_cap": float(experts[chosen_cap][index]),
            "p_final_DI_source_selected": float(chosen_final[index]),
            "route_DI_crossfit_selected": int(nested_route[index]),
            "p_final_DI_crossfit_selected": float(nested_final[index]),
        })
    write_csv(args.output / "source_selected_predictions.csv", sample_rows)

    selected = {
        "experiment": "CARE-Droid DI budget and bounded-correction source selection v1",
        "source_only": True,
        "source_year": 2017,
        "future_labels_used_for_selection": False,
        "candidate_budgets": budgets,
        "candidate_max_delta_logit": caps,
        "budget_ceiling": args.budget_ceiling,
        "selection_rule": (
            "among DI configurations within the declared budget ceiling, maximize "
            "source OOF BA, then net repairs, then fewer broken samples, then smaller "
            "budget, then smaller max_delta"
        ),
        "source_selected_within_ceiling": constrained,
        "fixed_25pct_best_bound": fixed25,
        "unrestricted_best": unrestricted,
        "crossfit_selection_metrics": nested_metrics,
        "source_thresholds": {
            "DI": route_threshold(source["scores"]["DI"], chosen_route),
            "A25": route_threshold(source["scores"]["A25"], a25_route),
        },
        "important_interpretation": {
            "max_delta_space": "logit",
            "exact_retraining": True,
            "same_source_oof_selection_bias": (
                "The pooled selected row is a development choice. Use the leave-one-"
                "outer-fold-out cross-fit row for a less selection-biased 2017 estimate."
            ),
            "future_status": "2018-2021 may be used only after this JSON is frozen",
        },
    }
    (args.output / "selected_source_config.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    base_row = baseline(source["labels"], source["base"])
    lines = [
        "CARE-Droid DI budget / bounded-correction source selection v1",
        "",
        "All candidate correction heads were retrained with strict 2017 OOF only.",
        "No 2018-2021 row is loaded by this program.",
        "max_delta is measured in logit space.",
        "",
        "Global24: BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f}".format(**base_row),
        "",
        "Best bound at fixed 25% DI budget:",
        "  max_delta={max_delta} BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} fixed={fixed} broken={broken} net={net}".format(**fixed25),
        "",
        "Source-selected DI configuration within budget ceiling:",
        "  budget={budget:.2%} max_delta={max_delta} BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} fixed={fixed} broken={broken} net={net}".format(**constrained),
        "",
        "Unrestricted source best (diagnostic, not necessarily deployable):",
        "  budget={budget:.2%} max_delta={max_delta} BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} fixed={fixed} broken={broken} net={net}".format(**unrestricted),
        "",
        "Leave-one-outer-fold-out selection estimate:",
        "  BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} fixed={fixed} broken={broken} net={net}".format(**nested_metrics),
        "",
        "Interpretation warning:",
        "  The pooled selected configuration is a development choice on 2017 OOF.",
        "  Report cross-fit selection metrics when an estimate after parameter selection is needed.",
        "  Freeze selected_source_config.json before any future-year diagnostic.",
    ]
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

