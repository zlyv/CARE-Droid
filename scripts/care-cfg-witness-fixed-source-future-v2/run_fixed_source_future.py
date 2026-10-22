#!/usr/bin/env python3
"""Evaluate 2017-only frozen CFG-Witness models on 2018--2021.

This script reads only cached historical features, frozen 2017-trained models,
and frozen fixed-2017 Stage-I/router predictions. It never reads APK bytes and
writes only outputs/care_cfg_witness_fixed_source_future_v2.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/boyuanjie/projects/tasa-droid")
HERE = ROOT / "scripts/care-cfg-witness-fixed-source-future-v2"
CFG = ROOT / "scripts/care-cfg-witness-fixed-ablation-v1"
BFA = ROOT / "scripts/care-background-functional-attribution-v1"
SCBE_SCRIPT = ROOT / "scripts/care-scbe-v1"
MYD_SCRIPT = ROOT / "scripts/care-multiyear-development-v1"
CFG_OUT = ROOT / "outputs/care_cfg_witness_fixed_ablation_v1"
SCBE_OUT = ROOT / "outputs/care_scbe_v1"
MYD_OUT = ROOT / "outputs/care_multiyear_development_v1"
OUT = ROOT / "outputs/care_cfg_witness_fixed_source_future_v2"
sys.path[:0] = [str(CFG), str(BFA), str(SCBE_SCRIPT), str(MYD_SCRIPT)]

from cfgwa_common import (  # noqa: E402
    ARMS,
    SEEDS,
    SIZE_COLS,
    WITNESS_COLS,
    conditional_shuffle,
    raw_matrix,
    witness_map,
)

YEARS = (2018, 2019, 2020, 2021)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=encode) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_csv(path: Path, values) -> None:
    values = list(values)
    fields = list(dict.fromkeys(key for row in values for key in row)) or ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    temp.replace(path)


def digest(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": h.hexdigest()}


def cfg_train_module():
    spec = importlib.util.spec_from_file_location("cfgwa_train_module", CFG / "train.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load CFG-Witness training module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def asset_paths() -> list[Path]:
    paths = [
        HERE / "run_fixed_source_future.py",
        CFG / "common.py",
        CFG / "cfgwa_common.py",
        CFG / "train.py",
        BFA / "bfa_core.py",
        SCBE_SCRIPT / "train.py",
        MYD_SCRIPT / "train_stage2.py",
        MYD_OUT / "data.npz",
        MYD_OUT / "feature_rows.pkl",
        MYD_OUT / "sample_manifest.csv",
        MYD_OUT / "predictions/C0R_2017/expert.npz",
    ]
    paths += [MYD_OUT / f"predictions/T{i}/router.npz" for i in range(1, 5)]
    for seed in SEEDS:
        paths.append(SCBE_OUT / f"models/T1/{seed}/M0_REPRODUCED.pkl")
        for arm in ARMS[1:]:
            paths.append(CFG_OUT / f"models/T1/{seed}/{arm}.pkl")
    return paths


def verify_manifest(items: list[dict]) -> dict:
    changed = []
    for item in items:
        current = digest(Path(item["path"]))
        if current != item:
            changed.append({"before": item, "after": current})
    return {
        "status": "PASS" if not changed else "FAIL",
        "checked": len(items),
        "changed": changed,
        "scope": "named frozen code/features/models/routes; no APK bytes",
    }


def feature_for_target(data, target_index, train_index, arm, seed, checkpoint, mapping):
    ids = data["ids"][target_index]
    raw_size = raw_matrix(ids, SIZE_COLS, mapping)
    if arm == "W1_WITNESS3":
        raw = raw_matrix(ids, WITNESS_COLS, mapping)
    elif arm == "W2_SIZE3_CONTROL":
        raw = raw_size
    elif arm == "W3_SIZE_CONDITIONAL_SHUFFLE":
        witness = raw_matrix(ids, WITNESS_COLS, mapping)
        train_size = raw_matrix(data["ids"][train_index], SIZE_COLS, mapping)
        raw = conditional_shuffle(
            witness,
            raw_size,
            train_size,
            np.random.RandomState(seed + 47002),
        )
    else:
        raise ValueError(arm)
    protocol = checkpoint["feature_protocol"]
    mean = np.asarray(protocol["mean"], np.float32)
    std = np.asarray(protocol["std"], np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
        raise RuntimeError("invalid frozen auxiliary scaler")
    result = ((raw - mean) / std).astype(np.float32)
    if result.shape != (len(target_index), 3) or not np.isfinite(result).all():
        raise RuntimeError("invalid target auxiliary features")
    return result


def evaluate(device: str) -> None:
    from bfa_core import load_data_contract
    from metrics import cascade
    from myd_common import runtime
    from train import global24
    from train_stage2 import raw_bags, transform_saved

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    manifest_path = OUT / "asset_manifest_before.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = [digest(path) for path in asset_paths()]
        atomic_json(manifest_path, manifest)
    initial_integrity = verify_manifest(manifest)
    if initial_integrity["status"] != "PASS":
        raise RuntimeError("frozen asset integrity failed before evaluation")

    data, _ = load_data_contract()
    years = np.asarray(data["years"], int)
    train_index = np.flatnonzero(years == 2017)
    if len(train_index) != 799:
        raise RuntimeError("2017 training set changed")
    mapping = witness_map()
    fixed_expert = npz(MYD_OUT / "predictions/C0R_2017/expert.npz")
    expert_position = {str(sid): i for i, sid in enumerate(fixed_expert["ids"])}
    cfg_train = cfg_train_module()
    v4, _, args = runtime()
    args.device = device
    args.keep_weight = args.fix_weight = 1.0
    args.keep_consistency = 0.0
    dev = v4.resolve_device(device)

    performance = []
    per_apk = []
    replay = []
    runtime_rows = []
    for year_index, year in enumerate(YEARS, 1):
        target_index = np.flatnonzero(years == year)
        router = npz(MYD_OUT / f"predictions/T{year_index}/router.npz")
        if not np.array_equal(router["val_index"], target_index):
            raise RuntimeError(f"fixed router index mismatch for {year}")
        ids = list(map(str, data["ids"][target_index]))
        if list(map(str, router["ids"])) != ids:
            raise RuntimeError(f"fixed router SHA mismatch for {year}")
        base = np.asarray(router["s0"], float)
        route = np.asarray(router["old_route"], bool)
        expected_route = int(round(0.25 * len(target_index)))
        if int(route.sum()) != expected_route:
            raise RuntimeError(f"fixed Top25 route changed for {year}")
        labels = np.asarray(data["y"][target_index], int)
        positions = np.asarray([expert_position[sid] for sid in ids], int)
        probabilities = {
            ARMS[0]: [
                np.asarray(fixed_expert["seed_probability"][seed_i, positions], float)
                for seed_i in range(len(SEEDS))
            ]
        }

        started = time.perf_counter()
        _, global_target = global24(v4, data, train_index, target_index)
        raw_target = raw_bags(data, target_index)
        preparation_seconds = time.perf_counter() - started
        for seed in SEEDS:
            m0 = load_pickle(SCBE_OUT / f"models/T1/{seed}/M0_REPRODUCED.pkl")
            bags = transform_saved(v4, raw_target, m0["background"], args)
            graph = v4.transform_role_graphs(data["s7"][target_index], m0["graph_scaler"])
            for arm in ARMS[1:]:
                checkpoint_path = CFG_OUT / f"models/T1/{seed}/{arm}.pkl"
                checkpoint = load_pickle(checkpoint_path)
                if checkpoint["split"] != "T1" or checkpoint["seed"] != seed or checkpoint["arm"] != arm:
                    raise RuntimeError("checkpoint identity mismatch: " + str(checkpoint_path))
                auxiliary = feature_for_target(
                    data, target_index, train_index, arm, seed, checkpoint, mapping
                )
                global_augmented = np.concatenate([global_target, auxiliary], axis=1)
                model = cfg_train.auxiliary_model(v4, args).to(dev)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                started = time.perf_counter()
                prediction, _ = v4.predict_model(
                    model,
                    global_augmented,
                    np.full(len(target_index), 0.5),
                    graph,
                    bags,
                    dev,
                )
                inference_seconds = time.perf_counter() - started
                probabilities.setdefault(arm, []).append(np.asarray(prediction, float))
                runtime_rows.append(
                    {
                        "year": year,
                        "arm": arm,
                        "seed": seed,
                        "preparation_seconds_shared_by_year": preparation_seconds,
                        "inference_seconds": inference_seconds,
                    }
                )
                if year == 2018:
                    reference = np.asarray(checkpoint["probability"], float)
                    difference = float(np.max(np.abs(reference - prediction)))
                    replay.append({"seed": seed, "arm": arm, "max_probability_difference": difference})
                    if difference > 1e-6:
                        raise RuntimeError(f"2018 T1 replay mismatch {seed}/{arm}: {difference}")
                del model
                if dev.type == "cuda":
                    v4.torch.cuda.empty_cache()

        for arm in ARMS:
            seed_probabilities = probabilities[arm]
            if len(seed_probabilities) != len(SEEDS):
                raise RuntimeError("missing seed probabilities")
            for seed, probability in zip(SEEDS, seed_probabilities):
                performance.append(
                    {
                        "protocol": "TRAIN_2017_ONLY_TEST_FUTURE",
                        "train_years": "2017",
                        "test_year": year,
                        "arm": arm,
                        "seed": seed,
                        "train_n": 799,
                        "test_n": len(target_index),
                        "routed": int(route.sum()),
                        **cascade(labels, base, probability, route),
                    }
                )
            ensemble = np.mean(seed_probabilities, axis=0)
            metric = cascade(labels, base, ensemble, route)
            performance.append(
                {
                    "protocol": "TRAIN_2017_ONLY_TEST_FUTURE",
                    "train_years": "2017",
                    "test_year": year,
                    "arm": arm,
                    "seed": "ENSEMBLE",
                    "train_n": 799,
                    "test_n": len(target_index),
                    "routed": int(route.sum()),
                    **metric,
                }
            )
            final = np.where(route, ensemble, base)
            for i, sid in enumerate(ids):
                per_apk.append(
                    {
                        "train_years": "2017",
                        "test_year": year,
                        "arm": arm,
                        "sha256": sid,
                        "label": int(labels[i]),
                        "routed": int(route[i]),
                        "stage1_probability": float(base[i]),
                        "expert_probability": float(ensemble[i]),
                        "final_probability": float(final[i]),
                        "stage1_correct": int(int(base[i] >= 0.5) == int(labels[i])),
                        "final_correct": int(int(final[i] >= 0.5) == int(labels[i])),
                    }
                )

    source_rows = []
    with (CFG_OUT / "performance.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "SOURCE_2017_OOF" and row["seed"] == "ENSEMBLE":
                source_rows.append(row)
    if len(source_rows) != len(ARMS):
        raise RuntimeError("source OOF reference incomplete")

    ensembles = [row for row in performance if row["seed"] == "ENSEMBLE"]
    by_key = {(int(row["test_year"]), row["arm"]): row for row in ensembles}
    comparisons = []
    for arm in ARMS[1:]:
        for reference in (ARMS[0], ARMS[2], ARMS[3]):
            if arm == reference:
                continue
            ba = [by_key[(year, arm)]["BA"] - by_key[(year, reference)]["BA"] for year in YEARS]
            net = [by_key[(year, arm)]["Net"] - by_key[(year, reference)]["Net"] for year in YEARS]
            comparisons.append(
                {
                    "candidate": arm,
                    "reference": reference,
                    "mean_BA_gain_2018_2021": float(np.mean(ba)),
                    "positive_BA_years": int(np.sum(np.asarray(ba) > 0)),
                    "total_Net_gain_2018_2021": int(np.sum(net)),
                }
            )

    final_integrity = verify_manifest(manifest)
    if final_integrity["status"] != "PASS":
        raise RuntimeError("frozen asset integrity failed after evaluation")
    atomic_csv(OUT / "performance.csv", performance)
    atomic_csv(OUT / "per_apk_predictions.csv", per_apk)
    atomic_csv(OUT / "comparisons.csv", comparisons)
    atomic_csv(OUT / "runtime_cost.csv", runtime_rows)
    atomic_json(
        OUT / "replay_validation.json",
        {
            "status": "PASS",
            "reference": "existing T1 predictions",
            "tolerance": 1e-6,
            "details": replay,
            "maximum_difference": max(row["max_probability_difference"] for row in replay),
        },
    )
    atomic_json(OUT / "baseline_integrity_check.json", final_integrity)
    atomic_json(
        OUT / "protocol.json",
        {
            "status": "FROZEN",
            "training_years": [2017],
            "evaluation_years": list(YEARS),
            "stage1": "S0R_2017 fixed model",
            "router": "2017 committee MeanOpposition old_route Top25 per target cohort",
            "expert": "2017-only T1 checkpoints reused without retraining",
            "threshold": 0.5,
            "new_APKs_used": False,
            "rolling_training_used": False,
            "future_labels_used_for_training_or_routing": False,
            "future_years_are_development": True,
        },
    )
    atomic_json(
        OUT / "final_decision.json",
        {
            "status": "FIXED_2017_TO_2018_2021_EVALUATION_COMPLETE",
            "formal_model_changed": False,
            "model_training_performed": False,
            "rolling_training_used": False,
            "future_years_are_development": True,
            "interpretation": "Corrected fixed-source temporal transfer evaluation; descriptive, not independent validation.",
        },
    )

    lines = [
        "# CFG Witness固定2017源域到2018--2021评估",
        "",
        "所有Stage-I、MeanOpposition路由、背景、RoleGraph模型与第二层参数均只由2017训练。",
        "2018--2021仅用于冻结推理和评价，没有滚动加入训练。",
        "",
        "| 测试年 | 方案 | BA | AUC | ACC | FN | FP | Fixed | Broken | Net | FN-vs-TN AUC | TP-vs-FP AUC |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for year in YEARS:
        for arm in ARMS:
            row = by_key[(year, arm)]
            lines.append(
                f"| {year} | {arm} | {row['BA']:.4f} | {row['AUC']:.4f} | "
                f"{row['Accuracy']:.4f} | {row['FN']} | {row['FP']} | {row['Fixed']} | "
                f"{row['Broken']} | {row['Net']} | {row['FN_vs_TN_AUC']:.4f} | "
                f"{row['TP_vs_FP_AUC']:.4f} |"
            )
    lines += ["", "2017年内结果沿用原五折OOF，仅作为源域参考；本次没有重新训练。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("CFGW_FIXED_SOURCE_FUTURE_DONE", flush=True)


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    device = os.environ.get("CFGW_FS_DEVICE", "cuda")
    evaluate(device)
