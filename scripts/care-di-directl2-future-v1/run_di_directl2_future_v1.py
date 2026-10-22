#!/usr/bin/env python3
"""Frozen future diagnostic for DI25 routed Direct-L2 Stage II.

This script trains the independent RGCondAttn Direct-L2 expert on all fixed
2017 source APKs, predicts 2018--2021, then compares frozen A25 and DI25
routes under two Stage-II heads:

  - BoundedCorrection from the existing DI audit artifact.
  - Direct-L2 absolute second-layer classifier from this diagnostic run.

Future labels are used only after routes and probabilities are frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
SEARCH_PATHS = (
    SCRIPT_DIR,
    SCRIPTS_ROOT,
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script",
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-future-script",
    SCRIPTS_ROOT / "care-rgcondattn-direct-l2-source-v1",
    SCRIPTS_ROOT / "care-committee-decision-instability-router-v1",
)
for path in SEARCH_PATHS:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    import run_care_rolegraph_conditioned_v4_future as future
    from run_care_rgcondattn_direct_l2_source_v1 import (
        DIAGNOSTIC_NAMES,
        DirectRGCondAttn,
        ordinary_bce,
        ordinary_weights,
    )
    from decision_instability_core import (
        MEMBER_COLUMNS,
        decision_instability,
        groupwise_top_fraction,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing established CARE/DI dependencies: " + str(exc))


METHOD = "Global24__A25__CARE_RGCondAttn_DirectL2"
DIRECT_NAME = METHOD + "__ensemble3"
ALIAS_METHOD = "__internal_direct_l2_cache_alias__"
YEARS = (2018, 2019, 2020, 2021)


def norm_sha(value: object) -> str:
    return str(value or "").strip().casefold()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
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


def array(rows: list[dict], name: str, dtype=float) -> np.ndarray:
    return np.asarray([dtype(row[name]) for row in rows])


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def output_path() -> Path:
    if "--output" in sys.argv:
        return Path(sys.argv[sys.argv.index("--output") + 1])
    return Path("artifacts/care-di-directl2-future-diagnostic-v1")


def metric_row(
    method: str,
    labels: np.ndarray,
    base_probability: np.ndarray,
    expert_probability: np.ndarray,
    route: np.ndarray,
    year: int,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    base = np.asarray(base_probability, dtype=np.float64)
    expert = np.asarray(expert_probability, dtype=np.float64)
    selected = np.asarray(route, dtype=bool)
    base_correct = (base >= 0.5) == y
    expert_correct = (expert >= 0.5) == y
    error = ~base_correct
    repairable = error & expert_correct
    harmful = base_correct & ~expert_correct
    confidence = np.maximum(base, 1.0 - base)
    high_error = error & (confidence >= 0.8)
    accepted = ~selected
    final_probability = np.where(selected, expert, base)
    final_prediction = final_probability >= 0.5
    fixed = int(np.sum(selected & repairable))
    broken = int(np.sum(selected & harmful))
    captured_error = int(np.sum(selected & error))
    return {
        "year": int(year),
        "method": method,
        "n": int(len(y)),
        "routed_n": int(np.sum(selected)),
        "routed_ratio": float(np.mean(selected)),
        "base_error_n": int(np.sum(error)),
        "routed_base_error_n": captured_error,
        "error_recall": safe_ratio(captured_error, int(np.sum(error))),
        "high_confidence_error_n": int(np.sum(high_error)),
        "routed_high_confidence_error_n": int(np.sum(selected & high_error)),
        "high_confidence_error_recall": safe_ratio(
            int(np.sum(selected & high_error)), int(np.sum(high_error))
        ),
        "repairable_error_n_for_head": int(np.sum(repairable)),
        "routed_repairable_error_n": fixed,
        "repairable_error_recall_at_budget": safe_ratio(
            fixed, int(np.sum(repairable))
        ),
        "harmful_sample_n_for_head": int(np.sum(harmful)),
        "routed_harmful_sample_n": broken,
        "harmful_capture": safe_ratio(broken, int(np.sum(harmful))),
        "accepted_risk": float(np.mean(error[accepted])) if np.any(accepted) else 0.0,
        "BA": float(balanced_accuracy_score(y, final_prediction)),
        "AUC": float(roc_auc_score(y, final_probability)),
        "F1": float(f1_score(y, final_prediction, zero_division=0)),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "diagnostic_only": True,
    }


def global_row(labels: np.ndarray, base: np.ndarray, year: int) -> dict:
    prediction = np.asarray(base) >= 0.5
    return {
        "year": int(year),
        "method": "Global24",
        "n": int(len(labels)),
        "routed_n": 0,
        "routed_ratio": 0.0,
        "BA": float(balanced_accuracy_score(labels, prediction)),
        "AUC": float(roc_auc_score(labels, base)),
        "F1": float(f1_score(labels, prediction, zero_division=0)),
        "fixed": 0,
        "broken": 0,
        "net": 0,
        "diagnostic_only": True,
    }


def pairwise_row(
    scope: str,
    labels: np.ndarray,
    base: np.ndarray,
    bounded: np.ndarray,
    direct: np.ndarray,
    route: np.ndarray,
    route_name: str,
    year: int | None = None,
) -> dict:
    selected = np.asarray(route, dtype=bool)
    y = np.asarray(labels, dtype=np.int64)
    bounded_final = np.where(selected, bounded, base)
    direct_final = np.where(selected, direct, base)
    bounded_correct = (bounded_final >= 0.5) == y
    direct_correct = (direct_final >= 0.5) == y
    row = {
        "scope": scope,
        "router": route_name,
        "n": int(len(y)),
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
        "diagnostic_only": True,
    }
    if year is not None:
        row["year"] = int(year)
    return row


def run_future_model() -> None:
    v4.RoleGraphConditionedEvidence = DirectRGCondAttn
    v4.fixkeep_weights = ordinary_weights
    v4.batch_loss = ordinary_bce
    v4.DIAGNOSTIC_NAMES = DIAGNOSTIC_NAMES
    v4.METHOD_FOR_VARIANT = {
        "RGConcat": ALIAS_METHOD,
        "RGCondAttn": METHOD,
    }
    original_fit = v4.fit_predict_variant
    cache: dict[tuple[int, ...], tuple] = {}

    def cached_fit(global_train, global_test, s7_train, s7_test,
                   raw_train_bags, raw_test_bags, base_train, base_test,
                   labels, variant, seeds, args, device):
        del variant
        key = tuple(int(seed) for seed in seeds)
        if key not in cache:
            cache[key] = original_fit(
                global_train, global_test, s7_train, s7_test,
                raw_train_bags, raw_test_bags, base_train, base_test,
                labels, "RGCondAttn", seeds, args, device,
            )
        return cache[key]

    v4.fit_predict_variant = cached_fit
    future.main()


def sanitize_prediction_rows(output: Path) -> list[dict]:
    rows = read_rows(output / "future_predictions_diagnostic.csv")
    cleaned = [
        {key: value for key, value in row.items() if ALIAS_METHOD not in key}
        for row in rows
    ]
    write_rows(output / "future_predictions_diagnostic.csv", cleaned)
    fit_path = output / "fit_audit.csv"
    if fit_path.exists():
        fit_rows = read_rows(fit_path)
        unique: dict[int, dict] = {}
        for row in fit_rows:
            seed = int(row["logical_seed"])
            item = dict(row)
            item["variant"] = "RGCondAttn-DirectL2"
            unique[seed] = item
        write_rows(fit_path, [unique[key] for key in sorted(unique)])
    return cleaned


def build_route_tables(output: Path, args: argparse.Namespace) -> None:
    direct_rows = sanitize_prediction_rows(output)
    di_rows = read_rows(args.di_future_scores)
    if len(direct_rows) != 1989 or len(di_rows) != 1989:
        raise RuntimeError(
            f"expected 1989 aligned future rows, got {len(direct_rows)} and "
            f"{len(di_rows)}"
        )
    required = {
        "sha256", "year", "label", "p_stage1", "p_frozen_bounded_expert",
        "route_A25", *MEMBER_COLUMNS,
    }
    missing = sorted(required - set(di_rows[0]))
    if missing:
        raise RuntimeError("DI future score columns missing: " + repr(missing))
    direct_by_sha = {norm_sha(row["sha256"]): row for row in direct_rows}
    if len(direct_by_sha) != len(direct_rows):
        raise RuntimeError("duplicate SHA in Direct-L2 future predictions")
    aligned_direct = []
    for row in di_rows:
        sha = norm_sha(row["sha256"])
        if sha not in direct_by_sha:
            raise RuntimeError("Direct-L2 future prediction missing SHA: " + sha)
        record = direct_by_sha[sha]
        if int(record["year"]) != int(row["year"]) or int(record["label"]) != int(row["label"]):
            raise RuntimeError("year/label mismatch for SHA: " + sha)
        aligned_direct.append(record)

    years = array(di_rows, "year", int).astype(np.int64)
    if tuple(sorted(set(years.tolist()))) != YEARS:
        raise RuntimeError("future year cohort changed")
    labels = array(di_rows, "label", int).astype(np.int64)
    base = array(di_rows, "p_stage1")
    bounded = array(di_rows, "p_frozen_bounded_expert")
    direct = array(aligned_direct, DIRECT_NAME + "__expert_prob")
    members = np.column_stack([array(di_rows, name) for name in MEMBER_COLUMNS])
    detail = decision_instability(base, members)
    route_a25 = array(di_rows, "route_A25", int).astype(bool)
    route_di = groupwise_top_fraction(detail["uncertainty"], years, args.budget)

    metrics = []
    pairwise = []
    sample_rows = []
    for year in YEARS:
        selected = years == year
        metrics.append(global_row(labels[selected], base[selected], year))
        for route_name, route in (
            ("A25_YearTop25", route_a25),
            ("DI25_YearTop25", route_di),
        ):
            route_y = route[selected]
            metrics.append(metric_row(
                route_name + "__BoundedCorrection",
                labels[selected],
                base[selected],
                bounded[selected],
                route_y,
                year,
            ))
            metrics.append(metric_row(
                route_name + "__DirectL2",
                labels[selected],
                base[selected],
                direct[selected],
                route_y,
                year,
            ))
            pairwise.append(pairwise_row(
                "annual",
                labels[selected],
                base[selected],
                bounded[selected],
                direct[selected],
                route_y,
                route_name,
                year,
            ))
    for route_name, route in (
        ("A25_YearTop25", route_a25),
        ("DI25_YearTop25", route_di),
    ):
        pairwise.append(pairwise_row(
            "pooled_2018_2021",
            labels,
            base,
            bounded,
            direct,
            route,
            route_name,
        ))

    for index, row in enumerate(di_rows):
        item = dict(row)
        item.update({
            "p_direct_l2_expert": float(direct[index]),
            "U_DI": float(detail["uncertainty"][index]),
            "route_DI": int(route_di[index]),
            "DI25_BoundedCorrection_final_prob": float(
                bounded[index] if route_di[index] else base[index]
            ),
            "DI25_DirectL2_final_prob": float(
                direct[index] if route_di[index] else base[index]
            ),
            "A25_BoundedCorrection_final_prob": float(
                bounded[index] if route_a25[index] else base[index]
            ),
            "A25_DirectL2_final_prob": float(
                direct[index] if route_a25[index] else base[index]
            ),
            "diagnostic_only": True,
        })
        sample_rows.append(item)

    write_rows(output / "future_di_directl2_annual_metrics.csv", metrics)
    write_rows(output / "future_di_directl2_pairwise.csv", pairwise)
    write_rows(output / "future_di_directl2_sample_scores.csv", sample_rows)

    audit = {
        "experiment": "care-di-directl2-future-diagnostic-v1",
        "diagnostic_only": True,
        "future_warning": "2018-2021 were inspected previously and are development diagnostics",
        "direct_l2_training": "all fixed 2017 source labels only",
        "direct_l2_excludes": "Global24 features and Stage-I probability/logit",
        "routers": ["A25_YearTop25", "DI25_YearTop25"],
        "budget": float(args.budget),
        "di_future_scores": str(args.di_future_scores),
        "direct_method": DIRECT_NAME,
    }
    (output / "protocol.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    by_year_method = {
        (int(row["year"]), str(row["method"])): row
        for row in metrics
    }
    lines = [
        "CARE-Droid DI25 + Direct-L2 future diagnostic v1",
        "",
        "Direct-L2 is trained on all fixed 2017 source APKs only.",
        "A25 and DI25 routes are frozen before future labels are evaluated.",
        "WARNING: 2018-2021 are diagnostic-only.",
        "",
        "Annual results:",
    ]
    for year in YEARS:
        lines.append(str(year))
        for method in (
            "Global24",
            "A25_YearTop25__BoundedCorrection",
            "A25_YearTop25__DirectL2",
            "DI25_YearTop25__BoundedCorrection",
            "DI25_YearTop25__DirectL2",
        ):
            row = by_year_method[(year, method)]
            lines.append(
                f"  {method}: BA={float(row['BA']):.6f} "
                f"AUC={float(row['AUC']):.6f} F1={float(row['F1']):.6f} "
                f"fixed={int(float(row['fixed']))} "
                f"broken={int(float(row['broken']))} "
                f"net={int(float(row['net']))}"
            )
    lines.extend(["", "Direct-L2 versus bounded pairwise:"])
    for row in pairwise:
        if row["scope"] == "pooled_2018_2021":
            lines.append(
                f"{row['router']}: direct-only-correct={row['direct_only_correct']} "
                f"bounded-only-correct={row['bounded_only_correct']} "
                f"net_direct_minus_bounded={row['net_direct_minus_bounded']} "
                f"routed_disagreement={float(row['routed_prediction_disagreement']):.6f}"
            )
    lines.extend([
        "",
        "Primary readout: DI25_YearTop25__DirectL2 vs DI25_YearTop25__BoundedCorrection.",
        "If Direct-L2 improves BA/net without a large broken increase across years, it is",
        "a credible future diagnostic follow-up to the source-side result.",
    ])
    text = "\n".join(lines) + "\n"
    (output / "REPORT.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--di-future-scores", type=Path, default=Path(
        "artifacts/care-committee-decision-instability-router-future-v1/"
        "future_sample_scores.csv"
    ))
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--postprocess-only", action="store_true")
    args, _ = parser.parse_known_args()
    if not 0.0 < args.budget < 1.0:
        raise SystemExit("budget must be in (0,1)")
    output = output_path()
    if not args.postprocess_only:
        run_future_model()
    build_route_tables(output, args)


if __name__ == "__main__":
    main()
