#!/usr/bin/env python3
"""Frozen 2018--2021 diagnostic for deduplicated bounded RGCondAttn.

The architecture and hyperparameters come from the completed strict 2017
source ablation.  Future labels are used only after all per-seed probabilities
and the equal-weight three-seed ensemble have been fixed.  Because 2018--2021
have already been inspected in this project, every exported result is marked
diagnostic-only and must not be described as an untouched final test.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
SEARCH_PATHS = (
    SCRIPT_DIR,
    SCRIPTS_ROOT,
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script",
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-future-script",
    SCRIPTS_ROOT / "care-rgcondattn-dedup-bounded-source-v1",
)
for path in SEARCH_PATHS:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    import run_care_rolegraph_conditioned_v4_future as future
    from run_care_rgcondattn_dedup_bounded_source_v1 import (
        DedupBoundedRGCondAttn,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing established CARE source/future dependencies: " + str(exc))


METHOD = "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep"
ORIGINAL_METHOD = "Global24__A25__CARE_RGCondAttn_FixKeep__ensemble3"
ALIAS_METHOD = "__internal_dedup_cache_alias__"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_path() -> Path:
    if "--output" in sys.argv:
        return Path(sys.argv[sys.argv.index("--output") + 1])
    return Path("artifacts/care-rgcondattn-dedup-bounded-future-diagnostic-v1")


def original_reference_path() -> Path:
    return Path(os.environ.get(
        "ORIGINAL_FUTURE_RESULT",
        "artifacts/care-rolegraph-conditioned-v4-future-diagnostic-v1/"
        "future_predictions_diagnostic.csv",
    ))


def sanitize_and_compare(output: Path) -> None:
    predictions_path = output / "future_predictions_diagnostic.csv"
    prediction_rows = read_rows(predictions_path)
    prediction_rows = [
        {key: value for key, value in row.items() if ALIAS_METHOD not in key}
        for row in prediction_rows
    ]

    original_path = original_reference_path()
    original_loaded = False
    original_column = ORIGINAL_METHOD + "__prob"
    reference_name = ORIGINAL_METHOD + "__reference"
    if original_path.exists():
        original_rows = read_rows(original_path)
        by_sha = {str(row["sha256"]).strip().lower(): row for row in original_rows}
        if len(by_sha) != len(original_rows):
            raise RuntimeError("duplicate SHA in original RGCondAttn reference")
        for row in prediction_rows:
            sha = str(row["sha256"]).strip().lower()
            if sha not in by_sha or original_column not in by_sha[sha]:
                raise RuntimeError("unaligned original RGCondAttn reference: " + sha)
            row[reference_name + "__prob"] = by_sha[sha][original_column]
        original_loaded = True
    write_rows(predictions_path, prediction_rows)

    # Remove the internal cache alias emitted by the reused frozen evaluator.
    annual_path = output / "future_annual_summary_diagnostic.csv"
    annual_rows = [
        row for row in read_rows(annual_path)
        if ALIAS_METHOD not in str(row.get("method", ""))
        and str(row.get("method", "")) != reference_name
    ]

    labels = np.asarray([int(row["label"]) for row in prediction_rows], dtype=np.int64)
    years = np.asarray([int(row["year"]) for row in prediction_rows], dtype=np.int64)
    base = np.asarray([float(row["Global24__prob"]) for row in prediction_rows])
    route = np.asarray([int(row["A25__routed"]) for row in prediction_rows], dtype=bool)
    dedup_name = METHOD + "__ensemble3"
    dedup = np.asarray([
        float(row[dedup_name + "__prob"]) for row in prediction_rows
    ])

    if original_loaded:
        original = np.asarray([
            float(row[reference_name + "__prob"]) for row in prediction_rows
        ])
        for year in future.FROZEN_YEARS:
            selected = years == year
            annual_rows.extend(v4.v2.metric_rows(
                labels[selected],
                {reference_name: original[selected]},
                base[selected],
                route[selected],
                year,
            ))
    write_rows(annual_path, annual_rows)

    # Rows loaded from CSV contain numeric strings, while newly computed rows
    # contain Python numbers.  The established mean_future_rows helper assumes
    # numeric values throughout, so normalize its input explicitly.
    numeric_fields = ("BA", "AUC", "F1", "fixed", "broken", "net")
    mean_input = []
    for row in annual_rows:
        normalized = dict(row)
        for field in numeric_fields:
            normalized[field] = float(normalized[field])
        mean_input.append(normalized)
    mean_rows = v4.g24.mean_future_rows(mean_input)
    write_rows(output / "future_mean_summary_diagnostic.csv", mean_rows)
    seed_rows = [
        row for row in mean_rows
        if "__seed" in str(row.get("method", ""))
    ]
    write_rows(output / "future_seed_summary_diagnostic.csv", seed_rows)

    pairwise_path = output / "future_pairwise_diagnostic.csv"
    pairwise_rows = [
        row for row in read_rows(pairwise_path)
        if ALIAS_METHOD not in str(row.get("baseline", ""))
        and ALIAS_METHOD not in str(row.get("candidate", ""))
        and not (
            str(row.get("baseline", "")) == reference_name
            and str(row.get("candidate", "")) == dedup_name
        )
    ]
    if original_loaded:
        for year in future.FROZEN_YEARS:
            selected = years == year
            pairwise_rows.append(future.direct_comparison_row(
                "annual",
                labels[selected],
                original[selected],
                dedup[selected],
                reference_name,
                dedup_name,
                year,
            ))
        pairwise_rows.append(future.direct_comparison_row(
            "pooled_2018_2021",
            labels,
            original,
            dedup,
            reference_name,
            dedup_name,
        ))
    write_rows(pairwise_path, pairwise_rows)

    fit_path = output / "fit_audit.csv"
    fit_rows = read_rows(fit_path)
    unique_fit: dict[int, dict] = {}
    for row in fit_rows:
        logical_seed = int(row["logical_seed"])
        cleaned = dict(row)
        cleaned["variant"] = "RGCondAttn-Dedup"
        unique_fit[logical_seed] = cleaned
    write_rows(fit_path, [unique_fit[key] for key in sorted(unique_fit)])

    audit_path = output / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update({
        "experiment": "care-rgcondattn-dedup-bounded-future-diagnostic-v1",
        "diagnostic_only": True,
        "variants_frozen_on_2017": ["RGCondAttn-Dedup"],
        "source_architecture_decision": (
            "single attended vector; no unused RGCondDual projections; "
            "bounded FixKeep correction retained"
        ),
        "future_warning": (
            "2018-2021 were inspected previously and are development diagnostics, "
            "not an untouched final test"
        ),
        "original_rgcondattn_reference": str(original_path),
        "original_rgcondattn_reference_loaded": original_loaded,
    })
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    by_method = {str(row["method"]): row for row in mean_rows}
    lines = [
        "CARE-Droid deduplicated bounded RGCondAttn future diagnostic v1",
        "",
        "Architecture and hyperparameters were fixed by the strict 2017 source run.",
        "Seeds: 0,1,2; ensemble: untuned arithmetic mean; threshold: fixed 0.5.",
        "Future labels were used only after probabilities were frozen.",
        "WARNING: 2018-2021 were already inspected and are diagnostic-only.",
        "",
        "2018-2021 macro means:",
    ]
    preferred = [
        "Global24",
        future.BASE_METHOD,
        future.CARE_V3_METHOD + "__reference",
        reference_name,
        dedup_name,
    ]
    for name in preferred:
        if name not in by_method:
            continue
        row = by_method[name]
        lines.append(
            f"{name}: BA={float(row['mean_BA']):.6f} "
            f"minBA={float(row['min_BA']):.6f} "
            f"AUC={float(row['mean_AUC']):.6f} "
            f"F1={float(row['mean_F1']):.6f} "
            f"fixed={float(row['mean_fixed']):.2f} "
            f"broken={float(row['mean_broken']):.2f} "
            f"net={float(row['mean_net']):.2f}"
        )
    lines.extend(["", "Annual deduplicated results:"])
    for row in annual_rows:
        if str(row.get("method")) == dedup_name:
            lines.append(
                f"{int(float(row['year']))}: BA={float(row['BA']):.6f} "
                f"AUC={float(row['AUC']):.6f} F1={float(row['F1']):.6f} "
                f"fixed={int(float(row['fixed']))} "
                f"broken={int(float(row['broken']))} net={int(float(row['net']))}"
            )
    if original_loaded:
        pooled = [
            row for row in pairwise_rows
            if row.get("scope") == "pooled_2018_2021"
            and row.get("baseline") == reference_name
            and row.get("candidate") == dedup_name
        ][0]
        lines.extend([
            "",
            "Paired deduplicated vs original RGCondAttn (pooled 2018-2021):",
            "dedup-only-correct=" + str(pooled["candidate_only_correct"]),
            "original-only-correct=" + str(pooled["baseline_only_correct"]),
            "net=" + str(pooled["net_candidate_minus_baseline"]),
        ])
    (output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    if "--postprocess-only" in sys.argv:
        sanitize_and_compare(output_path())
        return

    # The established future evaluator requires two named slots for its
    # historical RGConcat-vs-RGCondAttn comparison.  Both slots are directed to
    # one cached deduplicated fit, then the internal alias is removed from every
    # public output.  Thus each logical seed is trained exactly once.
    v4.RoleGraphConditionedEvidence = DedupBoundedRGCondAttn
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
    sanitize_and_compare(output_path())


if __name__ == "__main__":
    main()
