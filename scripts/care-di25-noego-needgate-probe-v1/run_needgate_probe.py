#!/usr/bin/env python3
"""Strict same-year OOF DI25 need-correction probes with true Host-only Stage-II features."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HISTORY = HERE.parent / "care-route-stage2-within-year-matrix-v1"
for path in (HERE, HISTORY, HISTORY / "vendor"):
    sys.path.insert(0, str(path))

import run_within_year_arm as within
from noego_representation import train_seed_and_encode
from preflight import validate as preflight_validate
from probe_common import (
    ARMS, FOLDS, SEEDS, YEARS, blocked, fit_probe, need_correct, norm,
    read_csv, stage1_logit, validate_partition, write_csv, write_json,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_parser():
    parser = within.future.parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest in {"representation", "head", "output"}:
            action.required = False
    parser.set_defaults(
        representation="anomaly", head="bounded",
        resolved=Path("artifacts/care-route-stage2-within-year-matrix-v1/preflight/resolved_inputs.json"),
        future_inter_cache=Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"),
        output=Path("artifacts/care-di25-noego-needgate-probe-v1"),
    )
    parser.add_argument("--router-inputs", type=Path, default=Path(
        "artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv"))
    return parser


def arm_matrices(global_train, global_validation, logit_train, logit_validation,
                 structural_train=None, structural_validation=None):
    result = {
        "S0_STAGE1_LOGIT": (logit_train[:, None], logit_validation[:, None]),
        "S1_GLOBAL24_LOGIT": (
            np.concatenate([global_train, logit_train[:, None]], axis=1),
            np.concatenate([global_validation, logit_validation[:, None]], axis=1),
        ),
    }
    if structural_train is not None:
        result.update({
            "S2_NOEGO_HOST_ANOMALY": (structural_train, structural_validation),
            "S3_NOEGO_HOST_ANOMALY_LOGIT": (
                np.concatenate([structural_train, logit_train[:, None]], axis=1),
                np.concatenate([structural_validation, logit_validation[:, None]], axis=1),
            ),
        })
    return result


def main() -> None:
    args = make_parser().parse_args()
    if args.representation != "anomaly" or args.head != "bounded":
        raise blocked("historical parser compatibility flags are frozen to anomaly/bounded")
    if float(args.max_delta) != 2.25:
        raise blocked("historical Host-only training requires frozen max_delta=2.25")
    if tuple(within.v4.parse_ints(args.seeds)) != SEEDS:
        raise blocked("frozen Stage-II seed ensemble must be 900000,900001,900002")

    # Reuse the preflight implementation; this writes only new-directory audits.
    preflight_args = type("PreflightArgs", (), {
        "router_inputs": args.router_inputs, "future_regions": args.future_regions,
        "resolved": args.resolved, "output": args.output,
        "source_cache": args.source_cache, "future_cache": args.future_cache,
        "future_inter_cache": args.future_inter_cache,
    })()
    cohort_audit = preflight_validate(preflight_args)
    router_rows = read_csv(args.router_inputs)
    data = within.load_data(args)
    router_ids = [norm(row["sha256"]) for row in router_rows]
    if list(data.ids) != router_ids:
        raise blocked("historical data loader changed router row order")
    data.route = np.asarray([int(row["route_DI"]) == 1 for row in router_rows], dtype=bool)
    if int(np.sum(data.route)) != sum(cohort_audit["DI_routed_counts"].values()):
        raise blocked("DI route count changed after historical input loading")

    v4 = within.v4
    future = within.future
    anomaly = within.anomaly_bounded
    v4.AUGMENTED_REGION_DIM = future.ORIGINAL_AUGMENTED_DIM
    v4.v3.fit_scaled_dualproto_bags = future.ORIGINAL_ANOMALY_FIT
    v4.RoleGraphConditionedEvidence = anomaly.DedupBoundedRGCondAttn
    device = v4.resolve_device(args.device)

    prediction_by_sha = {}
    fold_rows = []
    representation_audits = []
    assigned = set()
    for year in YEARS:
        for fold in FOLDS:
            train = np.flatnonzero((data.years == year) & (data.folds != fold))
            validation = np.flatnonzero((data.years == year) & (data.folds == fold))
            validate_partition([data.ids[i] for i in train], [data.ids[i] for i in validation])
            train_routed = data.route[train]
            validation_routed = data.route[validation]
            if not np.any(train_routed) or not np.any(validation_routed):
                raise blocked(f"empty routed split: year={year} fold={fold}")
            global_train, global_validation = within.fold_global24(data, train, validation)
            base_train = within.inner_oof_base(data, train, int(args.split_seed) + year * 10 + fold)
            base_validation = np.asarray(data.base[validation], dtype=np.float64)
            logit_train = stage1_logit(base_train)
            logit_validation = stage1_logit(base_validation)
            target_train = need_correct(data.labels[train], base_train)
            target_validation = need_correct(data.labels[validation], base_validation)
            if set(np.unique(target_train[train_routed]).tolist()) != {0, 1}:
                raise blocked(f"routed outer-train need target has one class: year={year} fold={fold}")

            per_arm_seed_probability = {arm: [] for arm in ARMS}
            simple = arm_matrices(global_train, global_validation, logit_train, logit_validation)
            for arm in ("S0_STAGE1_LOGIT", "S1_GLOBAL24_LOGIT"):
                x_train, x_validation = simple[arm]
                per_arm_seed_probability[arm].append(fit_probe(
                    x_train[train_routed], target_train[train_routed], x_validation[validation_routed]))

            for seed in SEEDS:
                structural_train, structural_validation, audit = train_seed_and_encode(
                    v4, anomaly, data, train, validation, global_train, global_validation,
                    base_train, base_validation, args, device, int(seed),
                )
                audit.update({"year": int(year), "fold": int(fold), "device": str(device), "ego_consumed": False})
                representation_audits.append(audit)
                matrices = arm_matrices(
                    global_train, global_validation, logit_train, logit_validation,
                    structural_train, structural_validation,
                )
                for arm in ("S2_NOEGO_HOST_ANOMALY", "S3_NOEGO_HOST_ANOMALY_LOGIT"):
                    x_train, x_validation = matrices[arm]
                    per_arm_seed_probability[arm].append(fit_probe(
                        x_train[train_routed], target_train[train_routed], x_validation[validation_routed]))

            routed_validation_indices = validation[validation_routed]
            for arm in ARMS:
                probability = np.mean(np.asarray(per_arm_seed_probability[arm]), axis=0)
                if len(probability) != len(routed_validation_indices):
                    raise blocked("probe output length differs from routed validation cohort")
                for global_index, value in zip(routed_validation_indices, probability):
                    sha = data.ids[int(global_index)]
                    key = (arm, sha)
                    if key in assigned:
                        raise blocked(f"duplicate OOF assignment: {key}")
                    assigned.add(key)
                    if sha not in prediction_by_sha:
                        stage1_pred = int(data.base[int(global_index)] >= 0.5)
                        prediction_by_sha[sha] = {
                            "sha256": sha, "year": int(year), "fold": int(fold),
                            "label": int(data.labels[int(global_index)]),
                            "p_stage1": float(data.base[int(global_index)]),
                            "stage1_logit": float(stage1_logit([data.base[int(global_index)]])[0]),
                            "stage1_pred": stage1_pred, "route_DI": 1,
                            "need_correct": int(stage1_pred != int(data.labels[int(global_index)])),
                        }
                    short = arm.split("_", 1)[0]
                    prediction_by_sha[sha][f"score_{short}"] = float(value)
                    prediction_by_sha[sha][f"pred_{short}"] = int(value >= 0.5)
            fold_rows.append({
                "year": int(year), "fold": int(fold), "train_n": int(len(train)),
                "validation_n": int(len(validation)), "train_routed_n": int(np.sum(train_routed)),
                "validation_routed_n": int(np.sum(validation_routed)),
                "train_need_positive_n": int(np.sum(target_train[train_routed])),
                "validation_need_positive_n": int(np.sum(target_validation[validation_routed])),
                "outer_train_validation_disjoint": True,
            })
            partial_rows = sorted(prediction_by_sha.values(), key=lambda row: (row["year"], row["fold"], row["sha256"]))
            write_csv(args.output / "oof_predictions.partial.csv", partial_rows)
            print(f"no-Ego need-gate probes {year} fold={fold}/5 done", flush=True)

    expected_assignments = int(np.sum(data.route)) * len(ARMS)
    if len(assigned) != expected_assignments or len(prediction_by_sha) != int(np.sum(data.route)):
        raise blocked(f"OOF assignment coverage failed: {len(assigned)} != {expected_assignments}")
    prediction_rows = sorted(prediction_by_sha.values(), key=lambda row: (row["year"], row["fold"], row["sha256"]))
    required_score_fields = {f"score_S{i}" for i in range(4)} | {f"pred_S{i}" for i in range(4)}
    if any(not required_score_fields.issubset(row) for row in prediction_rows):
        raise blocked("at least one routed APK lacks an arm-specific OOF prediction")
    write_csv(args.output / "oof_predictions.csv", prediction_rows)
    write_csv(args.output / "fold_audit.csv", fold_rows)
    write_csv(args.output / "representation_training_audit.csv", representation_audits)
    protocol = {
        "name": "CARE-Droid DI25 no-Ego need-gate probe v1",
        "status": "same-year development/diagnostic OOF; not temporal generalization or unseen-future evidence",
        "cohort": "route_DI==1 only", "target": "need_correct = ((p_stage1>=0.5) != label)",
        "split": "existing fixed year/fold; train same year fold!=f, validate same year fold=f",
        "probe": "StandardScaler + class-balanced L2 LogisticRegression(C=1, liblinear)",
        "probe_scaler_fit_scope": "routed outer-train only",
        "stage2_representation_training_scope": "all APKs in same-year outer-train only, matching historical Stage-II protocol",
        "stage2_neural_training_invoked": True,
        "validation_labels_visible_to_training_or_preprocessing": False,
        "arms": list(ARMS), "seeds": list(SEEDS), "bootstrap_iterations": 10000,
        "no_ego": True, "no_router_rebuild": True, "no_apk_reanalysis": True,
        "historical_files": {
            "within_year_runner": str(HISTORY / "run_within_year_arm.py"),
            "within_year_runner_sha256": sha256_file(HISTORY / "run_within_year_arm.py"),
        },
    }
    write_json(args.output / "protocol.json", protocol)
    print("NOEGO_NEEDGATE_OOF_DONE", flush=True)


if __name__ == "__main__":
    main()
