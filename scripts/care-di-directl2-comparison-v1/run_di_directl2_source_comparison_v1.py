#!/usr/bin/env python3
"""Strict source OOF 2x2 audit: A25/DI25 x bounded/direct Stage II."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


ROUTES = ("A25", "DI25")
HEADS = ("BoundedCorrection", "DirectL2")


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
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def find_direct_expert_column(rows: list[dict]) -> str:
    columns = set(rows[0])
    exact = "Global24__A25__CARE_RGCondAttn_DirectL2__expert_prob"
    if exact in columns:
        return exact
    candidates = sorted(
        column for column in columns
        if "RGCondAttn_DirectL2" in column and column.endswith("__expert_prob")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "cannot identify one Direct-L2 expert probability column: "
            + repr(candidates)
        )
    return candidates[0]


def metric_row(
    method: str,
    route_name: str,
    head_name: str,
    labels: np.ndarray,
    base_probability: np.ndarray,
    expert_probability: np.ndarray,
    route: np.ndarray,
    *,
    high_confidence_threshold: float = 0.8,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    base = np.asarray(base_probability, dtype=np.float64)
    expert = np.asarray(expert_probability, dtype=np.float64)
    selected = np.asarray(route, dtype=bool)
    base_correct = (base >= 0.5) == y
    expert_correct = (expert >= 0.5) == y
    base_error = ~base_correct
    repairable = base_error & expert_correct
    unrepairable = base_error & ~expert_correct
    harmful = base_correct & ~expert_correct
    confidence = np.maximum(base, 1.0 - base)
    high_error = base_error & (confidence >= float(high_confidence_threshold))
    final_probability = np.where(selected, expert, base)
    final_prediction = final_probability >= 0.5
    fixed = int(np.sum(selected & repairable))
    broken = int(np.sum(selected & harmful))
    routed_error = int(np.sum(selected & base_error))
    routed_high_error = int(np.sum(selected & high_error))
    accepted = ~selected
    return {
        "method": method,
        "router": route_name,
        "stage2_head": head_name,
        "n": int(len(y)),
        "routed_n": int(np.sum(selected)),
        "routed_ratio": float(np.mean(selected)),
        "base_error_n": int(np.sum(base_error)),
        "routed_base_error_n": routed_error,
        "error_recall": safe_ratio(routed_error, int(np.sum(base_error))),
        "high_confidence_error_n": int(np.sum(high_error)),
        "routed_high_confidence_error_n": routed_high_error,
        "high_confidence_error_recall": safe_ratio(
            routed_high_error, int(np.sum(high_error))
        ),
        "repairable_error_n_for_head": int(np.sum(repairable)),
        "routed_repairable_error_n": fixed,
        "repairable_error_recall_at_budget": safe_ratio(
            fixed, int(np.sum(repairable))
        ),
        "unrepairable_error_n_for_head": int(np.sum(unrepairable)),
        "routed_unrepairable_error_n": int(np.sum(selected & unrepairable)),
        "unrepairable_capture": safe_ratio(
            int(np.sum(selected & unrepairable)), int(np.sum(unrepairable))
        ),
        "harmful_sample_n_for_head": int(np.sum(harmful)),
        "routed_harmful_sample_n": broken,
        "harmful_capture": safe_ratio(broken, int(np.sum(harmful))),
        "accepted_risk": float(np.mean(base_error[accepted])) if np.any(accepted) else 0.0,
        "BA": float(balanced_accuracy_score(y, final_prediction)),
        "AUC": float(roc_auc_score(y, final_probability)),
        "F1": float(f1_score(y, final_prediction, zero_division=0)),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
    }


def global24_row(labels: np.ndarray, base: np.ndarray) -> dict:
    prediction = base >= 0.5
    return {
        "method": "Global24",
        "router": "None",
        "stage2_head": "None",
        "n": int(len(labels)),
        "routed_n": 0,
        "routed_ratio": 0.0,
        "BA": float(balanced_accuracy_score(labels, prediction)),
        "AUC": float(roc_auc_score(labels, base)),
        "F1": float(f1_score(labels, prediction, zero_division=0)),
        "fixed": 0,
        "broken": 0,
        "net": 0,
    }


def head_comparison_rows(
    labels: np.ndarray,
    base: np.ndarray,
    bounded: np.ndarray,
    direct: np.ndarray,
    routes: dict[str, np.ndarray],
) -> list[dict]:
    y = np.asarray(labels, dtype=np.int64)
    result = []
    for route_name, route in routes.items():
        selected = np.asarray(route, dtype=bool)
        bounded_final = np.where(selected, bounded, base)
        direct_final = np.where(selected, direct, base)
        bounded_correct = (bounded_final >= 0.5) == y
        direct_correct = (direct_final >= 0.5) == y
        result.append({
            "router": route_name,
            "routed_n": int(np.sum(selected)),
            "both_correct": int(np.sum(bounded_correct & direct_correct)),
            "bounded_only_correct": int(np.sum(bounded_correct & ~direct_correct)),
            "direct_only_correct": int(np.sum(~bounded_correct & direct_correct)),
            "both_wrong": int(np.sum(~bounded_correct & ~direct_correct)),
            "net_direct_minus_bounded": int(
                np.sum(~bounded_correct & direct_correct)
                - np.sum(bounded_correct & ~direct_correct)
            ),
            "routed_prediction_disagreement": float(np.mean(
                (bounded[selected] >= 0.5) != (direct[selected] >= 0.5)
            )),
            "routed_probability_pearson": float(np.corrcoef(
                bounded[selected], direct[selected]
            )[0, 1]),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--di-source-scores", type=Path, default=Path(
        "artifacts/care-committee-decision-instability-router-source-v1/"
        "source_sample_scores.csv"
    ))
    parser.add_argument("--direct-source-predictions", type=Path, default=Path(
        "artifacts/care-rgcondattn-direct-l2-source-v1/source_predictions.csv"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-di-directl2-comparison-source-v1"
    ))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    di_rows = read_csv(args.di_source_scores)
    direct_rows = read_csv(args.direct_source_predictions)
    if len(di_rows) != 799 or len(direct_rows) != 799:
        raise RuntimeError(
            f"expected aligned 799-row source artifacts, got {len(di_rows)} and "
            f"{len(direct_rows)}"
        )
    required_di = {
        "sha256", "label", "fold", "p_stage1", "p_frozen_bounded_expert",
        "route_A25", "route_DI", "U_DI",
    }
    missing = sorted(required_di - set(di_rows[0]))
    if missing:
        raise RuntimeError("DI source columns missing: " + repr(missing))
    direct_column = find_direct_expert_column(direct_rows)
    direct_by_sha = {norm_sha(row["sha256"]): row for row in direct_rows}
    if len(direct_by_sha) != len(direct_rows):
        raise RuntimeError("duplicate SHA in Direct-L2 source predictions")
    aligned_direct = []
    for row in di_rows:
        sha = norm_sha(row["sha256"])
        if sha not in direct_by_sha:
            raise RuntimeError("Direct-L2 source prediction missing SHA: " + sha)
        record = direct_by_sha[sha]
        if int(record["label"]) != int(row["label"]):
            raise RuntimeError("label mismatch for SHA: " + sha)
        if int(record["fold"]) != int(row["fold"]):
            raise RuntimeError("fold mismatch for SHA: " + sha)
        aligned_direct.append(record)

    labels = np.asarray([int(row["label"]) for row in di_rows], dtype=np.int64)
    base = np.asarray([float(row["p_stage1"]) for row in di_rows])
    bounded = np.asarray([
        float(row["p_frozen_bounded_expert"]) for row in di_rows
    ])
    direct = np.asarray([float(row[direct_column]) for row in aligned_direct])
    routes = {
        "A25": np.asarray([int(row["route_A25"]) for row in di_rows], dtype=bool),
        "DI25": np.asarray([int(row["route_DI"]) for row in di_rows], dtype=bool),
    }
    if any(int(np.sum(route)) != 200 for route in routes.values()):
        raise RuntimeError("source route budget changed")
    if not all(np.isfinite(value).all() for value in (base, bounded, direct)):
        raise RuntimeError("non-finite source probability")

    heads = {"BoundedCorrection": bounded, "DirectL2": direct}
    metrics = [global24_row(labels, base)]
    for route_name in ROUTES:
        for head_name in HEADS:
            metrics.append(metric_row(
                route_name + "__" + head_name,
                route_name,
                head_name,
                labels,
                base,
                heads[head_name],
                routes[route_name],
            ))
    comparisons = head_comparison_rows(labels, base, bounded, direct, routes)

    base_correct = (base >= 0.5) == labels
    bounded_correct = (bounded >= 0.5) == labels
    direct_correct = (direct >= 0.5) == labels
    sample_rows = []
    for index, row in enumerate(di_rows):
        item = {
            "sha256": norm_sha(row["sha256"]),
            "label": int(labels[index]),
            "fold": int(row["fold"]),
            "p_stage1": float(base[index]),
            "p_bounded_expert": float(bounded[index]),
            "p_direct_l2_expert": float(direct[index]),
            "U_DI": float(row["U_DI"]),
            "route_A25": int(routes["A25"][index]),
            "route_DI25": int(routes["DI25"][index]),
            "stage1_correct": int(base_correct[index]),
            "bounded_expert_correct": int(bounded_correct[index]),
            "direct_l2_expert_correct": int(direct_correct[index]),
            "bounded_repairable": int(~base_correct[index] & bounded_correct[index]),
            "direct_l2_repairable": int(~base_correct[index] & direct_correct[index]),
            "bounded_harmful": int(base_correct[index] & ~bounded_correct[index]),
            "direct_l2_harmful": int(base_correct[index] & ~direct_correct[index]),
        }
        for route_name in ROUTES:
            item[route_name + "__BoundedCorrection__final_prob"] = float(
                bounded[index] if routes[route_name][index] else base[index]
            )
            item[route_name + "__DirectL2__final_prob"] = float(
                direct[index] if routes[route_name][index] else base[index]
            )
        sample_rows.append(item)

    write_csv(args.output / "source_2x2_metrics.csv", metrics)
    write_csv(args.output / "source_head_pairwise.csv", comparisons)
    write_csv(args.output / "source_sample_scores.csv", sample_rows)
    protocol = {
        "experiment": "CARE-Droid A25/DI25 x Bounded/DirectL2 source comparison v1",
        "evaluation": "strict fixed 2017 OOF",
        "routers_retrained": False,
        "stage2_retrained_per_router": False,
        "bounded_source": str(args.di_source_scores),
        "direct_source": str(args.direct_source_predictions),
        "direct_expert_column": direct_column,
        "direct_stage2_input": "[role graph; critical region; attended regions; summaries]",
        "direct_stage2_excludes": "Global24 and Stage-I probability/logit",
        "primary_comparison": "DI25__DirectL2 versus DI25__BoundedCorrection",
        "future_data_loaded": False,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "CARE-Droid direct Stage-II classification source comparison v1",
        "",
        "Strict fixed 2017 OOF; no future row is loaded.",
        "Routers are frozen; the same Direct-L2 OOF expert is applied to A25 and DI25.",
        "Direct-L2 predicts an absolute malware probability and excludes Stage-I inputs.",
        "",
        "2x2 results:",
    ]
    for row in metrics:
        lines.append(
            f"{row['method']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
            f"F1={row['F1']:.6f} fixed={row['fixed']} "
            f"broken={row['broken']} net={row['net']}"
        )
        if row["method"] != "Global24":
            lines.append(
                f"  error_recall={row['error_recall']:.6f} "
                f"highconf_error_recall={row['high_confidence_error_recall']:.6f} "
                f"RER@25_for_head={row['repairable_error_recall_at_budget']:.6f} "
                f"HarmfulCapture_for_head={row['harmful_capture']:.6f}"
            )
    lines.extend(["", "Direct-L2 versus bounded pairwise:"])
    for row in comparisons:
        lines.append(
            f"{row['router']}: direct-only-correct={row['direct_only_correct']} "
            f"bounded-only-correct={row['bounded_only_correct']} "
            f"net_direct_minus_bounded={row['net_direct_minus_bounded']} "
            f"routed_disagreement={row['routed_prediction_disagreement']:.6f}"
        )
    lines.extend([
        "",
        "Primary decision: keep Direct-L2 only if it improves DI25 net/BA without",
        "an unacceptable increase in broken samples. Run a future diagnostic only",
        "after this source-side decision is frozen.",
    ])
    text = "\n".join(lines) + "\n"
    (args.output / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
