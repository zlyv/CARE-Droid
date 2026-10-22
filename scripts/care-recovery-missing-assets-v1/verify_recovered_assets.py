#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/boyuanjie/projects/tasa-droid")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    expected = {
        "features": ROOT / "artifacts/p1-features/features.csv",
        "p72": ROOT / "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv",
        "multiscale": ROOT / "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz",
        "di": ROOT / "artifacts/care-route-stage2-future-matrix-v1/preflight/source_sample_scores.csv",
    }
    missing = [str(path) for path in expected.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing recovered assets: " + repr(missing))
    feature_rows = rows(expected["features"])
    p72_rows = rows(expected["p72"])
    di_rows = rows(expected["di"])
    ref = rows(ROOT / "scripts/care-source-recovery-stage2only-v1/assets/source_reference_799.csv")
    ref_ids = {row["sha256"].strip().lower() for row in ref}
    checks = {
        "features_rows_799": len(feature_rows) == 799,
        "features_unique_799": len({r["sha256"].strip().lower() for r in feature_rows}) == 799,
        "features_exact_cohort": {r["sha256"].strip().lower() for r in feature_rows} == ref_ids,
        "features_all_ok": all(r.get("status") == "OK" for r in feature_rows),
        "p72_rows_799": len(p72_rows) == 799,
        "p72_exact_cohort": {r["sha256"].strip().lower() for r in p72_rows} == ref_ids,
        "di_rows_799": len(di_rows) == 799,
        "di_exact_cohort": {r["sha256"].strip().lower() for r in di_rows} == ref_ids,
        "di_routed_200": sum(int(float(r["route_DI"])) for r in di_rows) == 200,
    }
    with np.load(expected["multiscale"], allow_pickle=False) as data:
        checks["multiscale_nonempty"] = bool(data.files)
        multiscale_schema = {key: list(data[key].shape) for key in data.files}
    passed = all(checks.values())
    audit = {
        "passed": passed,
        "checks": checks,
        "multiscale_schema": multiscale_schema,
        "note": "2018-2021 compact features were not used as source assets.",
    }
    out = ROOT / "artifacts/recovery-audit/missing_assets_recovery_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("RECOVERY_GATE_FAILED: " + json.dumps(audit))
    print("MISSING_ASSETS_RECOVERY_GATE_PASS")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
