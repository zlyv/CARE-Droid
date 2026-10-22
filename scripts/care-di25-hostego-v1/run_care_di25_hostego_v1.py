#!/usr/bin/env python3
"""Run one predeclared frozen-DI25 CARE Host/Ego source arm."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
for path in (
    HERE, PROJECT / "scripts", PROJECT / "scripts/care-rolegraph-conditioned-v4-script",
    PROJECT / "scripts/care-rgcondattn-dedup-bounded-source-v1",
    PROJECT / "scripts/care-malben-contrast-v2", PROJECT / "scripts/care-malben-host-v2",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_contract import MAX_DELTA, build_di_reference, write_csv, write_static_audits
from host_ego_encoder import load_source_packed_bags, make_fit_adapter
from host_ego_models import anomaly_classes, malben_classes

import run_care_rolegraph_conditioned_dualstream_v4 as v4
import run_care_rgcondattn_dedup_bounded_source_v1 as anomaly
import run_care_malben_role_source_v2 as malben
from shared_role_prototypes import evidence_columns, fit_role_prototypes, ROLE_COUNT
from source_label_contract import install_source_label_contract

ARMS = {
    "M0": ("host", "anomaly", "DI25__Host__Anomaly__Bounded"),
    "M1": ("hostego", "anomaly", "DI25__HostEgo__Anomaly__Bounded"),
    "M2": ("host", "malben", "DI25__Host__MalBen__Bounded"),
    "M3": ("hostego", "malben", "DI25__HostEgo__MalBen__Bounded"),
}
FINAL_FROZEN_METHOD = "CARE-Droid-DI25-HostEgo-Anomaly-Bounded"

CAPTURES: list[dict] = []
PACKED_METADATA = []
PACKED_AUDIT = []
FEATURE_LOAD_SECONDS = 0.0
DATA_LOAD_SECONDS = 0.0
FEATURE_PREP_SECONDS = 0.0
TRAINING_SECONDS = 0.0
INFERENCE_SECONDS = 0.0
INFERENCE_APK_CALLS = 0
ORIGINAL_HOST_LOADER = v4.get_source_host_bags


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def install_phase_timers():
    """Instrument existing functions without changing their inputs or outputs."""
    original_load_inputs = v4.audit.load_inputs
    original_compact = v4.audit.compact_matrix_variant
    original_train_epochs = v4.train_epochs

    def timed_load_inputs(*args, **kwargs):
        global DATA_LOAD_SECONDS
        started = time.perf_counter(); result = original_load_inputs(*args, **kwargs)
        DATA_LOAD_SECONDS += time.perf_counter() - started
        return result

    def timed_compact(*args, **kwargs):
        global FEATURE_PREP_SECONDS
        started = time.perf_counter(); result = original_compact(*args, **kwargs)
        FEATURE_PREP_SECONDS += time.perf_counter() - started
        return result

    def timed_train_epochs(*args, **kwargs):
        global TRAINING_SECONDS
        _sync_if_cuda(); started = time.perf_counter()
        result = original_train_epochs(*args, **kwargs)
        _sync_if_cuda(); TRAINING_SECONDS += time.perf_counter() - started
        return result

    v4.audit.load_inputs = timed_load_inputs
    v4.audit.compact_matrix_variant = timed_compact
    v4.train_epochs = timed_train_epochs


def capture_predict(model, global_values, base_probability, graph_values, bags, device):
    global INFERENCE_SECONDS, INFERENCE_APK_CALLS
    load_started = time.perf_counter()
    global_tensor = torch.as_tensor(global_values, dtype=torch.float32, device=device)
    base_logit = torch.as_tensor(v4.probability_to_logit(base_probability), dtype=torch.float32, device=device)
    graph = v4.graph_tensors(graph_values, device)
    bag_tensors = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in bags]
    probability, diagnostic, ego = [], [], []
    model.eval()
    inference_started = time.perf_counter()
    with torch.no_grad():
        for index, bag in enumerate(bag_tensors):
            one_graph = {key: value[index] for key, value in graph.items()}
            logit, diag = model(global_tensor[index], base_logit[index], one_graph, bag)
            probability.append(float(torch.sigmoid(logit).cpu()))
            diagnostic.append(diag.detach().cpu().numpy())
            if getattr(model, "last_ego_capture", None) is not None:
                ego.append({key: np.asarray(value).copy() for key, value in model.last_ego_capture.items()})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    INFERENCE_SECONDS += time.perf_counter() - inference_started
    INFERENCE_APK_CALLS += len(bags)
    CAPTURES.append({"ego": ego})
    return np.asarray(probability), np.asarray(diagnostic)


def packed_loader(rows, endpoint_cache, args):
    global PACKED_METADATA, PACKED_AUDIT, FEATURE_LOAD_SECONDS
    started = time.perf_counter()
    root = Path(os.environ.get("CARE_SOURCE_LOCAL_ASSETS", "artifacts/care-source-local-assets-v2"))
    bags, audits, loaded, metadata = load_source_packed_bags(rows, args, root)
    PACKED_METADATA, PACKED_AUDIT = metadata, audits
    FEATURE_LOAD_SECONDS += time.perf_counter() - started
    return bags, audits, loaded


def timed_host_loader(rows, endpoint_cache, args):
    global FEATURE_LOAD_SECONDS
    started = time.perf_counter()
    result = ORIGINAL_HOST_LOADER(rows, endpoint_cache, args)
    FEATURE_LOAD_SECONDS += time.perf_counter() - started
    return result


def configure_arm(arm: str):
    representation, evidence, method = ARMS[arm]
    host_anomaly, ego_anomaly = anomaly_classes(anomaly, v4)
    host_malben, ego_malben = malben_classes(malben)
    v4.DIAGNOSTIC_NAMES = tuple(v4.DIAGNOSTIC_NAMES) + ("direct_stage2_probability",)
    if evidence == "anomaly":
        v4.MODEL_VARIANTS = ("RGCondAttn",)
        v4.METHOD_FOR_VARIANT = {"RGCondAttn": method}
        v4.RoleGraphConditionedEvidence = host_anomaly if representation == "host" else ego_anomaly
        original_fit = v4.v3.fit_scaled_dualproto_bags
    else:
        malben.fit_role_prototypes = fit_role_prototypes
        malben.evidence_columns = evidence_columns
        malben.ROLE_COUNT = ROLE_COUNT
        v4.MODEL_VARIANTS = ("MalBenContrast",)
        v4.METHOD_FOR_VARIANT = {"MalBenContrast": method}
        v4.DIAGNOSTIC_NAMES = tuple(malben.DIAGNOSTIC_NAMES) + ("direct_stage2_probability",)
        v4.AUGMENTED_REGION_DIM = malben.FULL_REGION_DIM
        v4.RoleGraphConditionedEvidence = host_malben if representation == "host" else ego_malben
        original_fit = malben.fit_scaled_role_bags
    if representation == "hostego":
        v4.get_source_host_bags = packed_loader
        v4.v3.fit_scaled_dualproto_bags = make_fit_adapter(original_fit)
    else:
        v4.get_source_host_bags = timed_host_loader
        v4.v3.fit_scaled_dualproto_bags = original_fit
    # This wrapper changes no prediction. It records inference separately and
    # appends the parameter-free Direct probability already emitted by each
    # arm's forward method.
    v4.predict_model = capture_predict
    return representation, evidence, method


def rewrite_argv(args, method_dir: Path, reference_dir: Path, representation: str, evidence: str):
    variant = "RGCondAttn" if evidence == "anomaly" else "MalBenContrast"
    sys.argv = [sys.argv[0],
        "--previous-result", str(reference_dir),
        "--cache", str(Path(os.environ.get("CARE_SOURCE_LOCAL_ASSETS", "artifacts/care-source-local-assets-v2")) / "endpoint_centered_tensor_cache_2017.jsonl.gz"),
        "--region-cache", str(Path(os.environ.get("CARE_SOURCE_LOCAL_ASSETS", "artifacts/care-source-local-assets-v2")) / "regions_2017.npz"),
        "--variants", variant, "--seeds", args.seeds, "--max-delta", str(MAX_DELTA),
        "--device", args.device, "--output", str(method_dir),
    ]


def emit_ego_diagnostics(output: Path, arm: str, method_dir: Path, seeds: list[int]):
    if not PACKED_METADATA:
        return
    with (method_dir / "source_predictions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fold_values = sorted({int(row["fold"]) for row in rows})
    expected = len(fold_values) * len(seeds)
    if len(CAPTURES) != expected:
        raise RuntimeError(f"Host/Ego capture count {len(CAPTURES)} != {expected}")
    host_rows, apk_rows, fallback_rows = [], [], []
    capture_group = 0
    for fold in fold_values:
        indices = [i for i, row in enumerate(rows) if int(row["fold"]) == fold]
        group = CAPTURES[capture_group:capture_group + len(seeds)]
        capture_group += len(seeds)
        for local_i, global_i in enumerate(indices):
            meta = PACKED_METADATA[global_i]
            values = [record["ego"][local_i] for record in group]
            keys = ("gamma", "used", "incoming", "outgoing", "host_norm", "ego_norm", "local_norm")
            avg = {key: np.mean(np.stack([np.asarray(v[key], dtype=float) for v in values]), axis=0) for key in keys}
            for h in range(len(meta.host_ids)):
                used = int(avg["used"][h] >= 0.5)
                diff = max(float(np.asarray(v["max_fallback_diff"])) for v in values) if not used else float("nan")
                host_rows.append({
                    "arm": arm, "sha256": meta.sha256, "fold": fold, "host_index_or_id": meta.host_ids[h],
                    "center_role": meta.primary_roles[h], "ego_used": used,
                    "neighbor_count": int(round(avg["incoming"][h] + avg["outgoing"][h])),
                    "incoming_neighbor_count_if_available": int(round(avg["incoming"][h])),
                    "outgoing_neighbor_count_if_available": int(round(avg["outgoing"][h])),
                    "host_feature_norm": float(avg["host_norm"][h]), "ego_feature_norm": float(avg["ego_norm"][h]),
                    "gamma": float(avg["gamma"][h]), "local_feature_norm": float(avg["local_norm"][h]),
                })
                fallback_rows.append({"arm": arm, "sha256": meta.sha256, "host_id": meta.host_ids[h],
                                      "has_nontrivial_ego": used, "max_abs_hostonly_vs_hostego_feature_diff": diff})
            used_mask = avg["used"] >= .5
            apk_rows.append({
                "arm": arm, "sha256": meta.sha256, "fold": fold,
                "routed": int(rows[global_i]["A25__routed"]), "host_count": meta.real_host_count,
                "nontrivial_ego_count": int(np.sum(used_mask)),
                "ego_usage_ratio": float(np.mean(used_mask)) if len(used_mask) else 0.0,
                "mean_gamma": float(np.mean(avg["gamma"][used_mask])) if np.any(used_mask) else 0.0,
                "max_gamma": float(np.max(avg["gamma"][used_mask])) if np.any(used_mask) else 0.0,
                "mean_neighbor_count": float(np.mean(avg["incoming"] + avg["outgoing"])),
            })
    write_csv(output / f"audits/ego_host_diagnostic_{arm}.csv", host_rows)
    write_csv(output / f"audits/ego_apk_diagnostic_{arm}.csv", apk_rows)
    write_csv(output / f"audits/ego_fallback_audit_{arm}.csv", fallback_rows)
    maximum = max((float(r["max_abs_hostonly_vs_hostego_feature_diff"]) for r in fallback_rows if str(r["max_abs_hostonly_vs_hostego_feature_diff"]) != "nan"), default=0.0)
    if maximum > 1e-6:
        raise RuntimeError(f"no-Ego fallback differs by {maximum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--output", type=Path, default=Path("artifacts/care-di25-hostego-v1"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if MAX_DELTA != 2.25:
        raise SystemExit("frozen max_delta contract failed")
    args.output.mkdir(parents=True, exist_ok=True)
    install_phase_timers()
    write_static_audits(args.output)
    install_source_label_contract(v4)
    contract = build_di_reference(PROJECT, args.output)
    representation, evidence, method = configure_arm(args.arm)
    method_dir = args.output / "source" / f"{args.arm}_{representation}_{evidence}_bounded"
    reference_dir = args.output / "preflight/di25_reference"
    rewrite_argv(args, method_dir, reference_dir, representation, evidence)
    started = time.perf_counter()
    v4.main()
    elapsed = time.perf_counter() - started
    emit_ego_diagnostics(args.output, args.arm, method_dir, [int(v) for v in args.seeds.split(",")])
    architecture = json.loads((method_dir / "architecture.json").read_text(encoding="utf-8"))
    architecture.update({
        "arm": args.arm, "router": "DI25", "local_representation": representation,
        "evidence": evidence, "max_delta_logit": MAX_DELTA, "correction_space": "logit",
        "stage1_probability": "Global24__prob", "future_data_loaded": False,
        "direct_ablation": "same raw correction-head output; no separately trained model",
        "wall_seconds_including_training": elapsed,
        "data_load_seconds": DATA_LOAD_SECONDS,
        "feature_prepare_seconds": FEATURE_PREP_SECONDS + FEATURE_LOAD_SECONDS,
        "feature_cache_loading_seconds": FEATURE_LOAD_SECONDS,
        "training_seconds": TRAINING_SECONDS,
        "model_inference_seconds": INFERENCE_SECONDS,
        "model_inference_apk_calls": INFERENCE_APK_CALLS,
        "model_inference_seconds_per_model_apk": INFERENCE_SECONDS / INFERENCE_APK_CALLS if INFERENCE_APK_CALLS else None,
        "FINAL_FROZEN_METHOD": FINAL_FROZEN_METHOD,
        "is_final_frozen_method_arm": args.arm == "M1",
    })
    (method_dir / "architecture.json").write_text(json.dumps(architecture, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"HOSTEGO_ARM_DONE {args.arm} method={method} seconds={elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
