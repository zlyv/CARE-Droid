#!/usr/bin/env python3
"""Dependency-light structural checks for the frozen future matrix package."""

from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    required = {
        "prepare_frozen_inputs.py": ["future_cohort_label_free.csv", "build_label_safe_features", "FUTURE_MATRIX_PREFLIGHT_PASS"],
        "build_frozen_router_inputs.py": ["FROZEN_A25_DI_ROUTER_REBUILD_PASS", "future_labels_loaded_by_router_builder", "0.8196741854636592"],
        "run_future_arm.py": ["future_labels_loaded_by_model_arm\": False", "future label seal failed", "--representation", "--head", "2.25"],
        "summarize_matrix.py": ["ARMS =", "AnomalyBaseline", "MalBenContrast", "DirectStageII", "BoundedCorrection", "future_metrics.csv"],
        "run_on_server.sh": ["anomaly direct", "anomaly bounded", "malben direct", "malben bounded"],
    }
    for name, markers in required.items():
        text = (root / name).read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                raise AssertionError(f"{name} missing marker {marker}")
    vendor = root / "vendor"
    for name in ("run_care_rolegraph_conditioned_dualstream_v4.py",
                 "run_care_bounded_fixkeep_ablation.py",
                 "run_allratio70_vs_nodrop76_experiment.py"):
        if not (vendor / name).is_file():
            raise AssertionError("missing vendored dependency " + name)
    print("FUTURE_MATRIX_SELFTEST_PASS")


if __name__ == "__main__": main()
