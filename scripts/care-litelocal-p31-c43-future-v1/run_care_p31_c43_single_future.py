#!/usr/bin/env python3
"""Frozen 2018--2021 diagnostic for CARE-Droid v4 role-graph conditioning.

The source-side decision is already frozen: compare RGConcat with RGCondAttn;
do not run RGCondDual.  Each model is trained on all 799 fixed 2017 APKs using
only source labels.  The 2017 OOF Global24 probability is the correction base
during training, while the frozen deployment Global24 probability is used for
future prediction.  The frozen A25 route is applied after expert prediction.

Future Stats120+PairRaw465 are reconstructed without labels and future Inter-Z
is loaded from (or built into) the established label-free cache.  Future labels
are materialized only after every per-seed probability and the equal-weight
ensemble have been fixed.  The 2018--2021 metrics are diagnostic only because
these years have already been inspected in this project.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing CARE v4 and its dependencies in scripts/: " + str(exc)
    )


BASE_METHOD = "Global24__A25__JointS7_ordinary"
CARE_V3_METHOD = "Global24__A25__CARE_DualProto_ES_FixKeep__ensemble3"
FROZEN_VARIANTS = ("RGConcat", "RGCondAttn")
FROZEN_SEEDS = (0, 1, 2)
FROZEN_YEARS = (2018, 2019, 2020, 2021)


def norm_sha(value) -> str:
    return str(value or "").strip().casefold()


def parse_unique_ints(text: str) -> list[int]:
    values: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            value = int(token)
            if value not in values:
                values.append(value)
    if not values:
        raise ValueError("seed list is empty")
    return values


def strict_reference_probability(
    path: Path,
    target_rows: list[dict],
    target_year: np.ndarray,
    column: str,
) -> np.ndarray:
    records = v4.read_csv(path)
    if not records:
        raise RuntimeError("reference prediction file is empty: " + str(path))
    required = {"sha256", column}
    missing = required - set(records[0])
    if missing:
        raise RuntimeError("reference prediction columns missing: " + repr(sorted(missing)))
    by_sha = {norm_sha(row["sha256"]): row for row in records}
    if len(by_sha) != len(records):
        raise RuntimeError("duplicate SHA in reference predictions: " + str(path))
    values = []
    for index, row in enumerate(target_rows):
        sha = norm_sha(row["sha256"])
        if sha not in by_sha:
            raise RuntimeError("reference prediction is missing SHA: " + sha)
        record = by_sha[sha]
        if "year" in record and str(record["year"]).strip():
            if int(record["year"]) != int(target_year[index]):
                raise RuntimeError("reference year mismatch for SHA: " + sha)
        values.append(float(record[column]))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(target_rows),) or not np.isfinite(result).all():
        raise RuntimeError("invalid aligned reference probability")
    return result


def direct_comparison_row(
    scope: str,
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    baseline_name: str,
    candidate_name: str,
    year: int | None = None,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    baseline_correct = (np.asarray(baseline) >= 0.5) == labels
    candidate_correct = (np.asarray(candidate) >= 0.5) == labels
    candidate_only = int(np.sum(~baseline_correct & candidate_correct))
    baseline_only = int(np.sum(baseline_correct & ~candidate_correct))
    row = {
        "scope": scope,
        "baseline": baseline_name,
        "candidate": candidate_name,
        "n": int(len(labels)),
        "both_correct": int(np.sum(baseline_correct & candidate_correct)),
        "candidate_only_correct": candidate_only,
        "baseline_only_correct": baseline_only,
        "both_wrong": int(np.sum(~baseline_correct & ~candidate_correct)),
        "net_candidate_minus_baseline": int(candidate_only - baseline_only),
        "diagnostic_only": True,
    }
    if year is not None:
        row["year"] = int(year)
    return row


def validate_args(args) -> tuple[list[int], list[int], list[str]]:
    years = sorted({int(value.strip()) for value in args.years.split(",") if value.strip()})
    seeds = parse_unique_ints(args.seeds)
    variants = v4.parse_variants(args.variants)
    if tuple(years) != FROZEN_YEARS:
        raise SystemExit("frozen diagnostic requires --years 2018,2019,2020,2021")
    if tuple(seeds) != FROZEN_SEEDS:
        raise SystemExit("frozen diagnostic requires --seeds 0,1,2 in this order")
    if tuple(variants) != FROZEN_VARIANTS:
        raise SystemExit("frozen diagnostic requires --variants RGConcat,RGCondAttn")
    if args.inner_folds < 2 or args.max_epochs < 1:
        raise SystemExit("invalid folds/epochs")
    if not 0.0 < args.early_stop_fraction < 0.5:
        raise SystemExit("--early-stop-fraction must be in (0,0.5)")
    if args.early_stop_patience < 1 or args.max_delta <= 0.0:
        raise SystemExit("patience and max delta must be positive")
    if args.graph_dim % args.graph_heads:
        raise SystemExit("--graph-dim must be divisible by --graph-heads")
    if min(args.keep_weight, args.fix_weight) <= 0.0:
        raise SystemExit("Fix/Keep weights must be positive")
    if not 0.0 < args.internal_background_coverage <= 1.0:
        raise SystemExit("--internal-background-coverage must be in (0,1]")
    return years, seeds, variants


def main() -> None:
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
    parser.add_argument("--previous-result", type=Path, default=Path(
        "artifacts/global24-a25-route-conditioned-joints7-v1-full"
    ))
    parser.add_argument("--care-v3-future-result", type=Path, default=Path(
        "artifacts/care-fixkeep-future-three-seed-v1/future_predictions_diagnostic.csv"
    ))
    parser.add_argument("--region-cache", type=Path, default=Path(
        "artifacts/care-local-region-cache-v1/regions_2017_2021.npz"
    ))
    parser.add_argument("--rebuild-region-cache", action="store_true")
    parser.add_argument("--years", default="2018,2019,2020,2021")
    parser.add_argument("--variants", default="RGConcat,RGCondAttn")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260901)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stop-fraction", type=float, default=0.18)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=24)
    parser.add_argument("--graph-dim", type=int, default=32)
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-delta", type=float, default=1.5)
    parser.add_argument("--keep-weight", type=float, default=2.0)
    parser.add_argument("--fix-weight", type=float, default=1.0)
    parser.add_argument("--keep-consistency", type=float, default=0.5)
    parser.add_argument("--shared-prototypes", type=int, default=32)
    parser.add_argument("--benign-prototypes", type=int, default=24)
    parser.add_argument("--prototype-regions-per-apk", type=int, default=16)
    parser.add_argument("--internal-background-coverage", type=float, default=0.70)
    parser.add_argument("--null-samples", type=int, default=64)
    parser.add_argument("--swaps-per-edge", type=int, default=10)
    parser.add_argument("--null-seed", type=int, default=20260827)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-rolegraph-conditioned-v4-future-diagnostic-v1"
    ))
    args = parser.parse_args()
    years, seeds, variants = validate_args(args)
    device = v4.resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    print("Frozen CARE v4 future diagnostic", flush=True)
    print("Device:", device, "seeds:", seeds, "variants:", variants, flush=True)

    (
        manifest,
        source_rows,
        clean_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    ) = v4.audit.load_inputs(args)
    source_y = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    if len(source_y) != 799:
        raise RuntimeError("expected 799 fixed 2017 APKs")

    requested = {str(year) for year in years}
    target_rows = [
        row for row in clean_rows
        if str(row.get("metadata_year", "")).strip() in requested
    ]
    target_year = np.asarray(
        [int(row["metadata_year"]) for row in target_rows], dtype=np.int64
    )
    if sorted(set(target_year.tolist())) != years or len(target_rows) != 1989:
        raise RuntimeError(
            "unexpected frozen future cohort: years="
            + repr(sorted(set(target_year.tolist())))
            + " n=" + str(len(target_rows))
        )

    source_reference = v4.care_v1.reference_arrays(
        args.previous_result / "source_predictions.csv", manifest, True
    )
    target_reference = v4.care_v1.reference_arrays(
        args.previous_result / "future_predictions_diagnostic.csv", target_rows, False
    )
    endpoint_cache = v4.audit.exp.read_cache(args.cache)

    print("Reconstructing source and future Stats120+PairRaw465...", flush=True)
    source_s3, _ = v4.audit.future.reconstruct_structure(
        source_rows, endpoint_cache, int(v4.audit.exp.ROLE_VOCABULARY_SIZE)
    )
    target_s3, _ = v4.audit.future.reconstruct_structure(
        target_rows, endpoint_cache, int(v4.audit.exp.ROLE_VOCABULARY_SIZE)
    )
    if source_s3.shape != (799, 585) or target_s3.shape != (1989, 585):
        raise RuntimeError("bad source/future S3 shapes: " + repr((source_s3.shape, target_s3.shape)))
    source_s7, source_s7_audit = v4.v3.future_s7.source_s7_from_reference(
        args.source_multiscale_features, source_rows, source_s3
    )
    target_inter, inter_rows, future_inter_audit = (
        v4.v3.future_s7.load_or_build_future_inter(
            target_rows, target_s3, endpoint_cache, args
        )
    )
    target_s7 = np.concatenate([target_s3, target_inter], axis=1)
    if source_s7.shape != (799, v4.S7_DIM) or target_s7.shape != (1989, v4.S7_DIM):
        raise RuntimeError("bad source/future S7 shapes: " + repr((source_s7.shape, target_s7.shape)))

    all_rows = list(source_rows) + list(target_rows)
    host_bags, _, bag_audit, region_cache_loaded = v4.care_v1.get_region_bags(
        all_rows, endpoint_cache, args
    )
    source_host = host_bags[:len(source_rows)]
    target_host = host_bags[len(source_rows):]

    x_source, x_target, compact_names = v4.audit.compact_matrix_variant(
        v4.VARIANT,
        source_rows,
        target_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    )
    global_index, _ = v4.evidence.split_view_indices(compact_names)
    global_source = np.asarray(x_source[:, global_index], dtype=np.float32)
    global_target = np.asarray(x_target[:, global_index], dtype=np.float32)
    if global_source.shape != (799, 24) or global_target.shape != (1989, 24):
        raise RuntimeError("unexpected Global24 matrix shapes")

    expert_by_variant_seed: dict[str, dict[int, np.ndarray]] = {
        variant: {} for variant in variants
    }
    final_by_variant_seed: dict[str, dict[int, np.ndarray]] = {
        variant: {} for variant in variants
    }
    diagnostic_by_variant_seed: dict[str, dict[int, np.ndarray]] = {
        variant: {} for variant in variants
    }
    fit_audit: list[dict] = []
    for variant in variants:
        for logical_seed in seeds:
            model_seed = 900000 + int(logical_seed)
            print(
                "Training all 799 source APKs:", variant,
                "logical seed", logical_seed,
                "model seed", model_seed,
                flush=True,
            )
            probability, diagnostic, audit_rows = v4.fit_predict_variant(
                global_source,
                global_target,
                source_s7,
                target_s7,
                source_host,
                target_host,
                source_reference["coarse"],
                target_reference["coarse"],
                source_y,
                variant,
                [model_seed],
                args,
                device,
            )
            expert = np.asarray(probability, dtype=np.float64)
            final = v4.acn.routed_probability(
                target_reference["coarse"], expert, target_reference["route"]
            )
            expert_by_variant_seed[variant][logical_seed] = expert
            final_by_variant_seed[variant][logical_seed] = np.asarray(final, dtype=np.float64)
            diagnostic_by_variant_seed[variant][logical_seed] = np.asarray(
                diagnostic, dtype=np.float64
            )
            for row in audit_rows:
                fit_audit.append({
                    "stage": "all_2017_to_future",
                    "variant": variant,
                    "logical_seed": int(logical_seed),
                    "model_seed": int(model_seed),
                    "train_n": int(len(source_y)),
                    "future_n": int(len(target_rows)),
                    **row,
                })
            print(variant, "seed", logical_seed, "done", flush=True)

    # All future probabilities and untuned equal-weight ensembles are frozen here.
    expert_ensemble = {
        variant: np.mean(
            np.asarray([expert_by_variant_seed[variant][seed] for seed in seeds]), axis=0
        )
        for variant in variants
    }
    final_ensemble = {
        variant: np.mean(
            np.asarray([final_by_variant_seed[variant][seed] for seed in seeds]), axis=0
        )
        for variant in variants
    }
    for variant in variants:
        if not np.isfinite(final_ensemble[variant]).all():
            raise RuntimeError("non-finite ensemble probability: " + variant)

    # Future labels and reference CARE probabilities are used only post hoc below.
    target_y = np.asarray(
        [v4.audit.future.label_of(row) for row in target_rows], dtype=np.int64
    )
    care_v3_probability = strict_reference_probability(
        args.care_v3_future_result,
        target_rows,
        target_year,
        CARE_V3_METHOD + "__prob",
    )

    probabilities: dict[str, np.ndarray] = {
        "Global24": np.asarray(target_reference["coarse"], dtype=np.float64),
        BASE_METHOD: np.asarray(target_reference["ordinary"], dtype=np.float64),
        CARE_V3_METHOD + "__reference": care_v3_probability,
    }
    for variant in variants:
        method = v4.METHOD_FOR_VARIANT[variant]
        for seed in seeds:
            probabilities[method + f"__seed{seed}"] = final_by_variant_seed[variant][seed]
        probabilities[method + "__ensemble3"] = final_ensemble[variant]

    annual_rows: list[dict] = []
    for year in years:
        selected = target_year == year
        annual_rows.extend(v4.v2.metric_rows(
            target_y[selected],
            {name: value[selected] for name, value in probabilities.items()},
            target_reference["coarse"][selected],
            target_reference["route"][selected],
            year,
        ))
    future_mean = v4.g24.mean_future_rows(annual_rows)

    pairwise_rows: list[dict] = []
    concat_name = v4.METHOD_FOR_VARIANT["RGConcat"] + "__ensemble3"
    attention_name = v4.METHOD_FOR_VARIANT["RGCondAttn"] + "__ensemble3"
    comparisons = [
        (concat_name, final_ensemble["RGConcat"], attention_name, final_ensemble["RGCondAttn"]),
        (BASE_METHOD, target_reference["ordinary"], attention_name, final_ensemble["RGCondAttn"]),
        (CARE_V3_METHOD + "__reference", care_v3_probability, attention_name, final_ensemble["RGCondAttn"]),
    ]
    for baseline_name, baseline, candidate_name, candidate in comparisons:
        for year in years:
            selected = target_year == year
            pairwise_rows.append(direct_comparison_row(
                "annual",
                target_y[selected],
                baseline[selected],
                candidate[selected],
                baseline_name,
                candidate_name,
                year,
            ))
        pairwise_rows.append(direct_comparison_row(
            "pooled_2018_2021",
            target_y,
            baseline,
            candidate,
            baseline_name,
            candidate_name,
        ))

    prediction_rows: list[dict] = []
    for index, row in enumerate(target_rows):
        output = {
            "sha256": norm_sha(row["sha256"]),
            "year": int(target_year[index]),
            "label": int(target_y[index]),
            "Global24__prob": float(target_reference["coarse"][index]),
            "A25__score": float(target_reference["score"][index]),
            "A25__routed": int(target_reference["route"][index]),
            BASE_METHOD + "__prob": float(target_reference["ordinary"][index]),
            CARE_V3_METHOD + "__reference__prob": float(care_v3_probability[index]),
        }
        for variant in variants:
            method = v4.METHOD_FOR_VARIANT[variant]
            for seed in seeds:
                output[method + f"__seed{seed}__expert_prob"] = float(
                    expert_by_variant_seed[variant][seed][index]
                )
                output[method + f"__seed{seed}__prob"] = float(
                    final_by_variant_seed[variant][seed][index]
                )
                for column, name in enumerate(v4.DIAGNOSTIC_NAMES):
                    output[method + f"__seed{seed}__" + name] = float(
                        diagnostic_by_variant_seed[variant][seed][index, column]
                    )
            output[method + "__ensemble3__expert_prob"] = float(
                expert_ensemble[variant][index]
            )
            output[method + "__ensemble3__prob"] = float(
                final_ensemble[variant][index]
            )
        output["diagnostic_only"] = True
        prediction_rows.append(output)

    seed_mean_rows = [
        row for row in future_mean
        if any(f"__seed{seed}" in str(row.get("method", "")) for seed in seeds)
    ]
    v4.g24.write_csv(args.output / "future_predictions_diagnostic.csv", prediction_rows)
    v4.g24.write_csv(args.output / "future_annual_summary_diagnostic.csv", annual_rows)
    v4.g24.write_csv(args.output / "future_mean_summary_diagnostic.csv", future_mean)
    v4.g24.write_csv(args.output / "future_seed_summary_diagnostic.csv", seed_mean_rows)
    v4.g24.write_csv(args.output / "future_pairwise_diagnostic.csv", pairwise_rows)
    v4.g24.write_csv(args.output / "fit_audit.csv", fit_audit)
    v4.g24.write_csv(args.output / "region_bag_audit.csv", bag_audit)
    v4.g24.write_csv(args.output / "future_inter_construction_audit.csv", inter_rows)

    metadata = {
        "experiment": "care-rolegraph-conditioned-v4-future-diagnostic-v1",
        "diagnostic_only": True,
        "source_train_n": int(len(source_y)),
        "future_n": int(len(target_y)),
        "future_years": years,
        "variants_frozen_on_2017": variants,
        "logical_seeds": seeds,
        "model_seeds": [900000 + seed for seed in seeds],
        "ensemble": "equal arithmetic mean of final probabilities; no future tuning",
        "future_labels_used_for_training_epoch_selection_weighting_routing_or_ensemble": False,
        "future_warning": "2018-2021 are already-inspected development diagnostics",
        "first_layer": "frozen role-free Global24",
        "router": "frozen A25",
        "candidate_base": "2017 OOF Global24 for training; deployment Global24 for future",
        "role_graph": "Stats120 nodes/globals + PairRaw465 + label-free InterZ961",
        "local_regions": "Host regions from frozen CARE cache",
        "region_cache_loaded": bool(region_cache_loaded),
        "source_s7_audit": source_s7_audit,
        "future_inter_audit": future_inter_audit,
        "care_v3_reference": str(args.care_v3_future_result),
        "device": str(device),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "CARE-Droid v4 frozen future diagnostic",
        "",
        "RGConcat and RGCondAttn were selected and frozen using 2017 only.",
        "Each seed is trained on all 799 source APKs and is exported separately.",
        "The ensemble is the untuned equal arithmetic mean of seeds 0,1,2.",
        "Future Inter-Z is label-free. Future labels are post-hoc metrics only.",
        "2018-2021 are already-inspected development diagnostics.",
    ]
    v4.g24.append_report_table(
        lines,
        "Future annual diagnostics",
        annual_rows,
        ["BA", "AUC", "F1", "fixed", "broken", "net", "routed_n"],
    )
    v4.g24.append_report_table(
        lines,
        "2018-2021 macro mean",
        future_mean,
        ["mean_BA", "min_BA", "mean_AUC", "mean_F1", "mean_fixed", "mean_broken", "mean_net"],
    )
    lines.extend(["", "Pooled direct comparisons"])
    for row in pairwise_rows:
        if row["scope"] == "pooled_2018_2021":
            lines.append(
                str(row["candidate"]) + " vs " + str(row["baseline"])
                + ": candidate-only=" + str(row["candidate_only_correct"])
                + " baseline-only=" + str(row["baseline_only_correct"])
                + " net=" + str(row["net_candidate_minus_baseline"])
            )
    (args.output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
