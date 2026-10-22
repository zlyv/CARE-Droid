#!/usr/bin/env python3
"""Freeze the user-selected M1/W1 candidate by reference and hash only.

This job never trains a model, reads an APK, or changes an existing asset.  It
creates a new reviewable contract that pins the exact 2017-only deployment
ensemble and the extraction implementation required for future APKs.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/home/boyuanjie/projects/tasa-droid")
HERE = ROOT / "scripts/care-manifest-rf-m1-freeze-v1"
OUT = ROOT / "outputs/care_manifest_rf_m1_freeze_v1"
SEEDS = (900000, 900001, 900002)
EXPECTED_EPOCHS = {900000: 20, 900001: 12, 900002: 8}
WITNESS_COLUMNS = (
    "bidirectional_pairs",
    "witness_pairs_interz_zero",
    "witness_unique_fraction_vs_interz",
)


def digest(path: Path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            hasher.update(block)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": hasher.hexdigest()}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=encode) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def model_path(seed: int):
    return ROOT / "outputs/care_cfg_witness_fixed_ablation_v1/models/T1" / str(seed) / "W1_WITNESS3.pkl"


def support_path(seed: int):
    return ROOT / "outputs/care_scbe_v1/models/T1" / str(seed) / "M0_REPRODUCED.pkl"


def required_assets():
    paths = [
        HERE / "freeze.py",
        HERE / "run.sh",
        ROOT / "scripts/care-cfg-witness-fixed-ablation-v1/common.py",
        ROOT / "scripts/care-cfg-witness-fixed-ablation-v1/train.py",
        ROOT / "scripts/care-background-functional-attribution-v1/bfa_core.py",
        ROOT / "scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
        ROOT / "scripts/care-p31-classifier-family-stage2-v1/model_adapter.py",
        ROOT / "scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py",
        ROOT / "scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_models.py",
        ROOT / "scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py",
        ROOT / "scripts/care-di25-graphglobal-path-source-v1/graphglobal_path_models.py",
        ROOT / "scripts/care-di25-compact43-relationquery-source-v1/hostrel_features.py",
        ROOT / "scripts/care-di25-compact43-relationquery-diagnostics-v1/build_future_cache.py",
        ROOT / "scripts/care-di25-host-gatedgraph-source-v1/run_endpoint_centered_tensor_cache_ag.py",
        ROOT / "scripts/care-droid-formal-diagnostics-v1/prepare_future_s7.py",
        ROOT / "scripts/care-host-cfg-rolegraph-witness-all-years-v2/pipeline.py",
        ROOT / "scripts/care-host-cfg-rolegraph-witness-all-years-v2/scan_one.py",
        ROOT / "scripts/care-host-cfg-rolegraph-witness-all-years-v2/witness_core.py",
        ROOT / "scripts/care-local-control-constraint-feasibility-v2/control_core.py",
        ROOT / "scripts/care-local-control-constraint-feasibility-v2/dex_adapter.py",
        ROOT / "scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml",
        ROOT / "artifacts/care_cfg_witness_all_years_v2/final_decision.json",
        ROOT / "artifacts/care_cfg_witness_all_years_v2/baseline_integrity_check.json",
        ROOT / "artifacts/care_cfg_witness_all_years_v2/cfg_witness_apk.csv",
        ROOT / "outputs/care_cfg_witness_fixed_source_future_v2/replay_validation.json",
        ROOT / "outputs/care_cfg_witness_fixed_source_future_v2/baseline_integrity_check.json",
        ROOT / "outputs/care_manifest_cfg_witness_fixed_source_v2/protocol.json",
        ROOT / "outputs/care_manifest_cfg_witness_fixed_source_v2/final_decision.json",
        ROOT / "outputs/care_manifest_cfg_witness_fixed_source_v2/baseline_integrity_check.json",
    ]
    paths.extend(model_path(seed) for seed in SEEDS)
    paths.extend(support_path(seed) for seed in SEEDS)
    return paths


def validate_upstream():
    checks = {
        "witness_asset_integrity": ROOT / "artifacts/care_cfg_witness_all_years_v2/baseline_integrity_check.json",
        "future_replay": ROOT / "outputs/care_cfg_witness_fixed_source_future_v2/replay_validation.json",
        "future_asset_integrity": ROOT / "outputs/care_cfg_witness_fixed_source_future_v2/baseline_integrity_check.json",
        "manifest_cfg_integrity": ROOT / "outputs/care_manifest_cfg_witness_fixed_source_v2/baseline_integrity_check.json",
    }
    result = {}
    for name, path in checks.items():
        value = read_json(path)
        if value.get("status") != "PASS":
            raise RuntimeError(f"upstream check is not PASS: {name}: {path}")
        result[name] = {"path": str(path.resolve()), "status": "PASS"}
    decision = read_json(ROOT / "artifacts/care_cfg_witness_all_years_v2/final_decision.json")
    if decision.get("status") != "SUPPORTED_FOR_FIXED_ABLATION":
        raise RuntimeError("repaired all-years Witness artifact does not support the fixed ablation")
    return result


def inspect_models():
    models = []
    common_mean = None
    common_std = None
    common_train_hash = None
    for seed in SEEDS:
        with model_path(seed).open("rb") as handle:
            model = pickle.load(handle)
        with support_path(seed).open("rb") as handle:
            support = pickle.load(handle)
        protocol = model.get("feature_protocol", {})
        columns = tuple(protocol.get("columns", ()))
        conditions = [
            model.get("arm") == "W1_WITNESS3",
            model.get("split") == "T1",
            int(model.get("seed")) == seed,
            int(model.get("selected_epochs")) == EXPECTED_EPOCHS[seed],
            int(model.get("global_width")) == 27,
            int(model.get("cost", {}).get("parameters")) == 35508,
            columns == WITNESS_COLUMNS,
            model.get("Stage1_changed") is False,
            model.get("Router_changed") is False,
            model.get("Host_RoleGraph_backgrounds_changed") is False,
            support.get("arm") == "M0_REPRODUCED",
            support.get("split") == "T1",
            int(support.get("seed")) == seed,
            support.get("train_years") == [2017],
            "background" in support,
            "graph_scaler" in support,
        ]
        if not all(conditions):
            raise RuntimeError(f"M1/support contract mismatch for seed {seed}")
        mean = np.asarray(protocol["mean"], dtype=float)
        std = np.asarray(protocol["std"], dtype=float)
        if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
            raise RuntimeError(f"invalid Witness scaler for seed {seed}")
        if common_mean is None:
            common_mean, common_std = mean, std
            common_train_hash = support.get("train_sha_hash")
        elif not (np.array_equal(mean, common_mean) and np.array_equal(std, common_std)):
            raise RuntimeError("Witness scaler differs across T1 ensemble members")
        if support.get("train_sha_hash") != common_train_hash:
            raise RuntimeError("2017 training membership hash differs across T1 support checkpoints")
        models.append({
            "seed": seed,
            "selected_epochs": EXPECTED_EPOCHS[seed],
            "parameters": 35508,
            "model_checkpoint": digest(model_path(seed)),
            "frozen_background_and_graph_scaler": digest(support_path(seed)),
            "state_dict_keys": len(model["state_dict"]),
        })
    return models, common_mean, common_std, common_train_hash


def verify_manifest(before):
    changed = []
    for item in before:
        try:
            current = digest(Path(item["path"]))
        except OSError as exc:
            changed.append({"path": item["path"], "error": repr(exc)})
            continue
        if current != item:
            changed.append({"before": item, "after": current})
    return {
        "status": "PASS" if not changed else "FAIL",
        "checked": len(before),
        "changed": changed,
        "scope": "M1 checkpoints, frozen background/scaler checkpoints, extraction code, mapping, and named validation assets; no APK bytes",
    }


def main():
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if OUT.exists():
        raise RuntimeError("output exists; refusing to overwrite")
    missing = [str(path) for path in required_assets() if not path.is_file()]
    if missing:
        raise RuntimeError("required assets missing: " + repr(missing))

    upstream = validate_upstream()
    models, witness_mean, witness_std, train_hash = inspect_models()
    OUT.mkdir(parents=True)
    before = [digest(path) for path in required_assets()]
    atomic_json(OUT / "asset_manifest_before.json", before)

    atomic_json(OUT / "model_contract.json", {
        "status": "FROZEN_BY_REFERENCE",
        "candidate_name": "M1",
        "implementation_arm": "W1_WITNESS3",
        "stage2_family": "Direct-C43 CFG-Witness",
        "training_years": [2017],
        "training_membership_sha256": train_hash,
        "deployment_ensemble": "arithmetic mean of three sigmoid probabilities",
        "seeds": list(SEEDS),
        "members": models,
        "global_input": "Global24 concatenated with standardized Witness3",
        "global_width": 27,
        "host_input_width": 142,
        "s7_width": 1546,
        "direct_logit": True,
        "stage1_logit_input_is_zeroed": True,
        "bounded_correction": False,
        "classification_threshold": 0.5,
        "witness_columns": list(WITNESS_COLUMNS),
        "witness_transform": ["log1p(max(x,0))", "log1p(max(x,0))", "identity"],
        "witness_mean": witness_mean,
        "witness_std": witness_std,
        "stage1_frozen_by_this_package": False,
        "router_frozen_by_this_package": False,
        "future_labels_allowed_for_model_or_router_selection": False,
    })

    atomic_json(OUT / "extraction_contract.json", {
        "status": "FROZEN",
        "unit": "one APK identified by verified SHA256",
        "read_only_apk": True,
        "required_outputs_per_apk": {
            "global24": 24,
            "s7": {"total": 1546, "role_stats": 93, "graph_global": 27, "pair_raw": 465, "inter_z_directed": 961},
            "host_regions": {"stored_width": 169, "model_consumed_width": 142, "components": ["Compact43", "HAIG64_placeholder_or_reserved_capacity", "Background4", "Relation31"]},
            "witness3": list(WITNESS_COLUMNS),
        },
        "witness_scan": {
            "host_membership": "frozen Host method IDs and sensitive endpoint Role IDs reconstructed from the endpoint cache",
            "dex_parser": "Androguard APK/DEX; static parsing only",
            "cfg": "instruction-level CFG including exception edges",
            "alignment": "invoke instructions must match a frozen Host:endpoint pair; frozen Role IDs are authoritative",
            "pair_relations": ["A_TO_B", "B_TO_A", "BIDIRECTIONAL", "NEITHER", "UNKNOWN"],
            "loop_detection": "deterministic iterative Kosaraju SCC",
            "pair_cap_per_method": 200000,
            "interz_zero_tolerance": 1e-12,
        },
        "quality_gates": [
            "SHA256 is unique and matches lineage",
            "Manifest135 exists for Stage-I",
            "Compact/global/endpoint/S7/Host/Relation features are finite and schema-aligned",
            "all frozen Host IDs and endpoint Role mappings use the pinned semantic mapping",
            "Witness scan status is OK and pair-cap/failure fields are retained",
            "all three ensemble members load with their pinned background and graph scaler",
        ],
        "missing_or_failed_stage2_asset_policy": "do not run M1 for that APK; retain the Stage-I Manifest probability and record the reason",
        "labels_used_during_extraction": False,
        "gpu_required_for_extraction": False,
    })

    atomic_json(OUT / "validation_boundary.json", {
        "status": "DECLARED",
        "selection_reason": "user-selected candidate to hold fixed for evaluation after dataset expansion",
        "existing_2018_2021_role": "development evaluation only",
        "independent_expanded_dataset_result_available": False,
        "permitted_next_use": "apply the exact frozen ensemble and extraction contract to a separately audited expanded dataset",
        "prohibited_uses": [
            "retrain M1 on future or expanded-test labels",
            "change Witness definitions after viewing expanded-test outcomes",
            "change the 0.5 expert threshold after viewing future outcomes",
            "select Manifest router budget from 2018-2021 results",
            "silently insert new APKs into the 2017 Prototype Bank or frozen background assets",
        ],
        "routing_boundary": "M1 is frozen here; the Manifest135 router remains an external, separately versioned contract",
        "known_evidence_boundary": "the fixed ablation did not establish W1 superiority over equal-capacity controls; this freeze is a prospective candidate decision",
    })

    after = verify_manifest(before)
    atomic_json(OUT / "baseline_integrity_check.json", after)
    if after["status"] != "PASS":
        raise RuntimeError("protected assets changed during freeze")
    atomic_json(OUT / "upstream_validation.json", {"status": "PASS", "checks": upstream})
    atomic_json(OUT / "final_decision.json", {
        "status": "M1_CANDIDATE_FROZEN_FOR_EXPANDED_DATASET",
        "candidate": "M1/W1_WITNESS3",
        "checkpoint_changed": False,
        "model_training_performed": False,
        "apk_opened": False,
        "formal_source_modified": False,
        "prototype_bank_modified": False,
        "router_changed": False,
        "integrity": after["status"],
        "independent_validation_pending": True,
    })

    report = """# Manifest135 → M1 冻结说明

最终状态：**M1_CANDIDATE_FROZEN_FOR_EXPANDED_DATASET**。

- M1 对应 `W1_WITNESS3`：Direct-C43 的 Global24 后追加 Witness3，形成 27 维全局输入。
- 部署模型固定为 2017-only T1 的三个种子 checkpoint（900000、900001、900002），输出概率取算术平均。
- 每个种子同时绑定原 M0 checkpoint 中的训练期背景原型和 Graph scaler；没有重训、复制或改写 checkpoint。
- Stage-I 仍是 Manifest135；本冻结包没有改动 Manifest 模型或路由。路由预算必须另行冻结。
- 2018—2021 仍是开发评估，不能据此回调 M1、阈值、Witness 定义或路由。
- 扩充数据集必须先完成 SHA/标签/重复/年份审计，再按 `extraction_contract.json` 提取；任一第二层资产失败时保留第一层概率并记录原因。
- 当前固定消融没有证明 W1 稳定优于等容量对照，因此这里固定的是待扩充数据验证的候选方案，不是新的独立有效性结论。
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("M1_FREEZE_COMPLETE integrity=PASS checkpoints=3 training=false apk_opened=false", flush=True)


if __name__ == "__main__":
    main()
