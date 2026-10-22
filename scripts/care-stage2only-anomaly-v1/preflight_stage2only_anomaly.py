#!/usr/bin/env python3
"""Self-contained hard preflight for the Stage-II-only anomaly control."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np


def csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def exact_oof(path: Path, probability_column: str) -> None:
    rows = csv_rows(path)
    ids = [str(row.get("sha256", "")).strip().lower() for row in rows]
    if len(rows) != 799 or len(set(ids)) != 799 or any(not value for value in ids):
        raise SystemExit(f"invalid 799-row OOF file: {path}")
    labels = [int(row["label"]) for row in rows]
    folds = [int(row["fold"]) for row in rows]
    probabilities = [float(row[probability_column]) for row in rows]
    if set(labels) != {0, 1} or set(folds) != {1, 2, 3, 4, 5}:
        raise SystemExit(f"invalid labels/folds in: {path}")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
        raise SystemExit(f"invalid probabilities in: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path("artifacts/care-source-local-assets-v2"))
    parser.add_argument("--malben", type=Path, default=Path("artifacts/care-stage2only-malben-v1/source/oof_predictions.csv"))
    parser.add_argument("--check-output", type=Path)
    args = parser.parse_args()

    required = [
        Path("scripts/care-stage2only-anomaly-v1/vendor/run_care_rolegraph_conditioned_dualstream_v4.py"),
        Path("scripts/care-stage2only-anomaly-v1/vendor/run_care_bounded_fixkeep_ablation.py"),
        Path("scripts/care-stage2only-anomaly-v1/vendor/run_care_joint_dualprototype_ablation.py"),
        Path("scripts/care-stage2only-anomaly-v1/vendor/run_care_local_dualpool_mil.py"),
        Path("scripts/care-stage2only-anomaly-v1/vendor/run_care_local_region_asset_audit.py"),
        Path("scripts/care-stage2only-anomaly-v1/vendor/run_global24_r_joints7_2017_2021.py"),
        Path("scripts/care-stage2only-anomaly-v1/vendor/run_allratio70_vs_nodrop76_experiment.py"),
        args.asset_root / "source_asset_gate.json",
        args.asset_root / "source_lineage_2017.csv",
        args.asset_root / "endpoint_centered_tensor_cache_2017.jsonl.gz",
        args.asset_root / "regions_2017.npz",
        args.asset_root / "global_role_graph_2017.npz",
        Path("artifacts/p1-features/features.csv"),
        Path("artifacts/p1-features/feature_schema.json"),
        Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),
        Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"),
        args.malben,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing frozen input(s): " + repr(missing))

    gate = json.loads((args.asset_root / "source_asset_gate.json").read_text(encoding="utf-8"))
    if gate.get("passed") is not True or int(gate.get("expected_apks", -1)) != 799:
        raise SystemExit("source local-asset hard gate is not a passed 799-APK gate")
    checks = gate.get("checks", {})
    if not checks or not all(value is True for value in checks.values()):
        raise SystemExit("one or more source local-asset hard checks are false")

    lineage = csv_rows(args.asset_root / "source_lineage_2017.csv")
    lineage_ids = [str(row.get("sha256", "")).strip().lower() for row in lineage]
    if len(lineage_ids) != 799 or len(set(lineage_ids)) != 799:
        raise SystemExit("source lineage is not exact unique 799")
    with np.load(args.asset_root / "regions_2017.npz", allow_pickle=False) as cache:
        cache_ids = cache["sha256"].astype(str).tolist()
        real_host_count = np.asarray(cache["real_host_count"], dtype=np.int64)
    if cache_ids != lineage_ids or len(real_host_count) != 799:
        raise SystemExit("regions_2017 cache does not match source lineage order")
    with gzip.open(args.asset_root / "endpoint_centered_tensor_cache_2017.jsonl.gz", "rt", encoding="utf-8") as handle:
        endpoint_ids = [json.loads(line)["sha256"].strip().lower() for line in handle if line.strip()]
    if endpoint_ids != lineage_ids or len(set(endpoint_ids)) != 799:
        raise SystemExit("endpoint cache does not match source lineage order")

    exact_oof(args.malben, "StageIIOnly_MalBen__prob")
    if args.check_output is not None:
        if not args.check_output.is_file():
            raise SystemExit("Stage-II-only anomaly output is not complete yet")
        exact_oof(args.check_output, "StageIIOnly_Anomaly__prob")
        print("STAGE2ONLY_ANOMALY_OUTPUT_799_PASS")
    else:
        print("STAGE2ONLY_ANOMALY_PREFLIGHT_PASS")
        print("STAGE2ONLY_ANOMALY_VENDOR_CLOSURE_PRESENT")


if __name__ == "__main__":
    main()
