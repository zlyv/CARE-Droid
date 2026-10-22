#!/usr/bin/env python3
"""Benchmark all-APK versus route-first RGCondAttn inference.

The model is trained exactly once on the fixed 2017 source cohort.  The same
frozen model is then evaluated in two modes on the same 2018--2021 cohort:

* all_expert: run the second-stage expert for every APK, then apply A25;
* route_first: run the expert only for APKs selected by frozen A25.

The script asserts that the two final probability vectors are numerically
identical.  It measures cached model-side inference; raw APK extraction is a
separate benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch

import run_care_rolegraph_conditioned_dualstream_v4 as v4


def current_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class RssMonitor:
    def __init__(self, interval: float = 0.01):
        self.interval = float(interval)
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.baseline = current_rss_bytes()
        self.peak = self.baseline

        def sample():
            while not self._stop.wait(self.interval):
                self.peak = max(self.peak, current_rss_bytes())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak = max(self.peak, current_rss_bytes())

    @property
    def peak_mib(self) -> float:
        return self.peak / (1024.0 * 1024.0)

    @property
    def delta_mib(self) -> float:
        return max(0, self.peak - self.baseline) / (1024.0 * 1024.0)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def subset_graph(values: dict[str, np.ndarray], index: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value[index]) for key, value in values.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure_prediction(
    mode: str,
    repeat: int,
    model,
    global_values,
    base_probability,
    graph_values,
    bags,
    device: torch.device,
):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    with RssMonitor() as memory:
        started = time.perf_counter()
        probability, diagnostic = v4.predict_model(
            model,
            global_values,
            base_probability,
            graph_values,
            bags,
            device,
        )
        sync(device)
        seconds = time.perf_counter() - started
    gpu_peak = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda" else 0.0
    )
    n = len(bags)
    row = {
        "mode": mode,
        "repeat": int(repeat),
        "n": int(n),
        "seconds": float(seconds),
        "milliseconds_per_apk": float(seconds * 1000.0 / n),
        "apk_per_second": float(n / seconds),
        "peak_rss_mib": float(memory.peak_mib),
        "incremental_peak_rss_mib": float(memory.delta_mib),
        "peak_torch_allocated_mib": float(gpu_peak),
    }
    return probability, diagnostic, row


def median_for(rows: list[dict], mode: str, column: str) -> float:
    return float(statistics.median(float(row[column]) for row in rows if row["mode"] == mode))


def validate_args(args) -> None:
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if not 0.0 < args.early_stop_fraction < 0.5:
        raise SystemExit("--early-stop-fraction must be in (0,0.5)")
    if not 0.0 < args.internal_background_coverage <= 1.0:
        raise SystemExit("--internal-background-coverage must be in (0,1]")


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
    parser.add_argument("--region-cache", type=Path, default=Path(
        "artifacts/care-local-region-cache-v1/regions_2017_2021.npz"
    ))
    parser.add_argument("--rebuild-region-cache", action="store_true")
    parser.add_argument("--years", default="2018,2019,2020,2021")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
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
        "artifacts/care-rgcondattn-route-first-benchmark-v1"
    ))
    args = parser.parse_args()
    validate_args(args)
    device = v4.resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    years = sorted({int(value.strip()) for value in args.years.split(",") if value.strip()})
    print("CARE RGCondAttn route-first benchmark", flush=True)
    print("Device:", device, "seed:", args.seed, "repeats:", args.repeats, flush=True)

    load_started = time.perf_counter()
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

    target_rows = [
        row for row in clean_rows
        if int(str(row.get("metadata_year", "0")).strip() or 0) in years
    ]
    if len(target_rows) != 1989:
        raise RuntimeError(f"expected 1989 fixed future APKs, got {len(target_rows)}")

    source_reference = v4.care_v1.reference_arrays(
        args.previous_result / "source_predictions.csv", manifest, True
    )
    target_reference = v4.care_v1.reference_arrays(
        args.previous_result / "future_predictions_diagnostic.csv", target_rows, False
    )
    route = np.asarray(target_reference["route"], dtype=bool)
    routed_index = np.flatnonzero(route)
    if len(routed_index) != 497:
        raise RuntimeError(f"expected frozen A25 route n=497, got {len(routed_index)}")

    endpoint_cache = v4.audit.exp.read_cache(args.cache)
    print("Reconstructing frozen S7 arrays...", flush=True)
    source_s3, _ = v4.audit.future.reconstruct_structure(
        source_rows, endpoint_cache, int(v4.audit.exp.ROLE_VOCABULARY_SIZE)
    )
    target_s3, _ = v4.audit.future.reconstruct_structure(
        target_rows, endpoint_cache, int(v4.audit.exp.ROLE_VOCABULARY_SIZE)
    )
    source_s7, _ = v4.v3.future_s7.source_s7_from_reference(
        args.source_multiscale_features, source_rows, source_s3
    )
    target_inter, _, _ = v4.v3.future_s7.load_or_build_future_inter(
        target_rows, target_s3, endpoint_cache, args
    )
    target_s7 = np.concatenate([target_s3, target_inter], axis=1)

    all_rows = list(source_rows) + list(target_rows)
    host_bags, _, bag_audit, cache_loaded = v4.care_v1.get_region_bags(
        all_rows, endpoint_cache, args
    )
    if not cache_loaded:
        raise RuntimeError("benchmark requires the frozen combined local-region cache")
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
    cached_input_load_seconds = time.perf_counter() - load_started

    model_seed = 900000 + int(args.seed)
    print("Selecting epochs on 2017 only...", flush=True)
    selection_started = time.perf_counter()
    selected_epochs, selection_loss = v4.select_epochs(
        global_source,
        source_s7,
        source_host,
        source_reference["coarse"],
        source_y,
        "RGCondAttn",
        args,
        device,
        model_seed,
    )
    selection_seconds = time.perf_counter() - selection_started

    print("Preparing dual-prototype bags...", flush=True)
    prep_started = time.perf_counter()
    train_bags, test_bags, prototype_audit = v4.v3.fit_scaled_dualproto_bags(
        source_host, target_host, source_y, args, model_seed + 1707
    )
    graph_scaler = v4.fit_role_graph_scaler(source_s7)
    train_graph = v4.transform_role_graphs(source_s7, graph_scaler)
    test_graph = v4.transform_role_graphs(target_s7, graph_scaler)
    target_preparation_seconds = time.perf_counter() - prep_started

    global_tensor = torch.as_tensor(global_source, dtype=torch.float32, device=device)
    train_graph_t = v4.graph_tensors(train_graph, device)
    train_bag_t = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in train_bags]
    weights, correct = v4.fixkeep_weights(source_y, source_reference["coarse"], args)
    v4.seed_everything(model_seed)
    model = v4.RoleGraphConditionedEvidence(
        global_source.shape[1], "RGCondAttn", args
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    print("Training final frozen model for", selected_epochs, "epochs...", flush=True)
    sync(device)
    training_started = time.perf_counter()
    final_loss = v4.train_epochs(
        model,
        optimizer,
        global_tensor,
        source_reference["coarse"],
        train_graph_t,
        train_bag_t,
        source_y,
        weights,
        correct,
        selected_epochs,
        args,
        model_seed + 3301,
    )
    sync(device)
    final_training_seconds = time.perf_counter() - training_started
    parameter_n = int(sum(value.numel() for value in model.parameters()))

    routed_global = global_target[routed_index]
    routed_base = np.asarray(target_reference["coarse"])[routed_index]
    routed_graph = subset_graph(test_graph, routed_index)
    routed_bags = [test_bags[int(index)] for index in routed_index]

    # Warm both paths before recording repeated timings.
    warm_index = routed_index[:min(3, len(routed_index))]
    v4.predict_model(
        model,
        global_target[warm_index],
        np.asarray(target_reference["coarse"])[warm_index],
        subset_graph(test_graph, warm_index),
        [test_bags[int(index)] for index in warm_index],
        device,
    )
    sync(device)

    timing_rows: list[dict] = []
    full_reference = route_reference = None
    for repeat in range(1, int(args.repeats) + 1):
        modes = ("all_expert", "route_first") if repeat % 2 else ("route_first", "all_expert")
        for mode in modes:
            if mode == "all_expert":
                probability, _, row = measure_prediction(
                    mode, repeat, model, global_target,
                    target_reference["coarse"], test_graph, test_bags, device
                )
                if full_reference is None:
                    full_reference = probability
            else:
                probability, _, row = measure_prediction(
                    mode, repeat, model, routed_global,
                    routed_base, routed_graph, routed_bags, device
                )
                if route_reference is None:
                    route_reference = probability
            timing_rows.append(row)
            print(
                mode, "repeat", repeat,
                f"seconds={row['seconds']:.6f}",
                f"ms/APK={row['milliseconds_per_apk']:.4f}",
                flush=True,
            )

    assert full_reference is not None and route_reference is not None
    routed_expert_max_abs_difference = float(
        np.max(np.abs(full_reference[routed_index] - route_reference))
    )
    sparse_expert = np.asarray(target_reference["coarse"], dtype=np.float64).copy()
    sparse_expert[routed_index] = route_reference
    full_final = v4.acn.routed_probability(
        target_reference["coarse"], full_reference, target_reference["route"]
    )
    route_final = v4.acn.routed_probability(
        target_reference["coarse"], sparse_expert, target_reference["route"]
    )
    final_max_abs_difference = float(np.max(np.abs(full_final - route_final)))
    if routed_expert_max_abs_difference > 1e-6 or final_max_abs_difference > 1e-6:
        raise RuntimeError(
            "route-first equivalence failed: expert="
            f"{routed_expert_max_abs_difference} final={final_max_abs_difference}"
        )

    full_seconds = median_for(timing_rows, "all_expert", "seconds")
    route_seconds = median_for(timing_rows, "route_first", "seconds")
    summary = {
        "benchmark_scope": "cached model-side inference; excludes raw APK extraction",
        "device": str(device),
        "seed": int(args.seed),
        "source_n": int(len(source_y)),
        "future_n": int(len(target_rows)),
        "routed_n": int(len(routed_index)),
        "routed_ratio": float(len(routed_index) / len(target_rows)),
        "all_local_regions": int(sum(len(value) for value in test_bags)),
        "routed_local_regions": int(sum(len(test_bags[int(index)]) for index in routed_index)),
        "parameter_n": parameter_n,
        "selected_epochs": int(selected_epochs),
        "selection_loss": float(selection_loss),
        "final_training_loss": float(final_loss),
        "cached_input_load_seconds": float(cached_input_load_seconds),
        "epoch_selection_seconds": float(selection_seconds),
        "target_preparation_all_seconds": float(target_preparation_seconds),
        "final_training_seconds": float(final_training_seconds),
        "all_expert_median_seconds": full_seconds,
        "route_first_median_seconds": route_seconds,
        "measured_inference_time_reduction": float(1.0 - route_seconds / full_seconds),
        "measured_inference_speedup": float(full_seconds / route_seconds),
        "routed_expert_max_abs_difference": routed_expert_max_abs_difference,
        "final_probability_max_abs_difference": final_max_abs_difference,
        "probability_equivalent": True,
        "prototype_audit": prototype_audit,
    }
    write_csv(args.output / "inference_repeats.csv", timing_rows)
    (args.output / "cost_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "CARE-Droid RGCondAttn route-first benchmark",
        "",
        "Scope: cached model-side inference; raw APK extraction excluded",
        f"Device: {device}",
        f"Seed: {args.seed}",
        f"Source/future/routed: {len(source_y)}/{len(target_rows)}/{len(routed_index)}",
        f"Route ratio: {100.0 * len(routed_index) / len(target_rows):.2f}%",
        f"Parameters: {parameter_n}",
        f"Selected epochs: {selected_epochs}",
        f"Cached input load: {cached_input_load_seconds:.3f} s",
        f"Epoch selection: {selection_seconds:.3f} s",
        f"All-target preparation: {target_preparation_seconds:.3f} s",
        f"Final training: {final_training_seconds:.3f} s",
        "",
        f"All-expert median inference: {full_seconds:.6f} s",
        f"Route-first median inference: {route_seconds:.6f} s",
        f"Measured inference reduction: {100.0 * (1.0 - route_seconds / full_seconds):.2f}%",
        f"Measured speedup: {full_seconds / route_seconds:.3f}x",
        f"Routed expert max abs difference: {routed_expert_max_abs_difference:.3e}",
        f"Final probability max abs difference: {final_max_abs_difference:.3e}",
        "Probability-equivalent: True",
        "",
        "Limitation: this run reuses frozen caches. It does not establish raw-APK",
        "feature-extraction time, end-to-end energy, or comparison-method cost.",
    ]
    (args.output / "REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print("Done:", args.output)


if __name__ == "__main__":
    main()
