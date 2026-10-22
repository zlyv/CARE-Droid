#!/usr/bin/env python3
"""Exhaustive gate for frozen DI25 Host/Ego future diagnostics."""
from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path("artifacts/care-di25-hostego-future-v1")
RESOLVED = Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json")
FUTURE_INPUT = Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv")
M0_FUTURE = Path("artifacts/care-route-stage2-future-matrix-v1/arms/anomaly_bounded/future_expert_predictions.csv")
M0_PROTOCOL = M0_FUTURE.parent / "protocol.json"
M0_SOURCE = Path("artifacts/care-di25-hostego-v1/source/M0_host_anomaly_bounded/source_predictions.csv")
M1_SOURCE = Path("artifacts/care-di25-hostego-v1/source/M1_hostego_anomaly_bounded/source_predictions.csv")
EXPECTED_YEARS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
EXPECTED_ROUTE = {2018: 198, 2019: 100, 2020: 99, 2021: 100}
EXPECTED_SOURCE = {
    "M0": (0.9111058897243107, 0.9410213032581454, 0.9086229086229086),
    "M1": (0.9148715538847118, 0.9433521303258148, 0.9132653061224489),
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normal(value: object) -> str:
    return str(value).strip().lower()


def unique(rows: list[dict], name: str, expected: int) -> dict[str, dict]:
    result = {}
    for row in rows:
        sha = normal(row.get("sha256", row.get("sample_id", "")))
        if not sha or sha in result:
            raise RuntimeError(f"{name}: missing/duplicate SHA {sha!r}")
        result[sha] = row
    if len(rows) != expected or len(result) != expected:
        raise RuntimeError(f"{name}: rows={len(rows)}, unique={len(result)}, expected={expected}")
    return result


def lock_source(path: Path, arm: str) -> None:
    rows = read_csv(path)
    unique(rows, arm, 799)
    columns = [key for key in rows[0] if key.startswith("DI25__") and key.endswith("__prob")]
    token = "HostEgo" if arm == "M1" else "Host"
    columns = [key for key in columns if token in key and (arm != "M0" or "HostEgo" not in key)]
    if len(columns) != 1:
        raise RuntimeError(f"{arm}: cannot identify bounded probability: {columns}")
    y = np.asarray([int(float(row["label"])) for row in rows])
    p = np.asarray([float(row[columns[0]]) for row in rows])
    actual = (balanced_accuracy_score(y, p >= .5), roc_auc_score(y, p), f1_score(y, p >= .5))
    if not np.allclose(actual, EXPECTED_SOURCE[arm], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{arm} source-result lock changed: {actual}")


def main() -> None:
    errors = []
    package = Path(__file__).resolve().parent
    required = [
        RESOLVED, FUTURE_INPUT, M0_FUTURE, M0_PROTOCOL, M0_SOURCE, M1_SOURCE,
        Path("artifacts/care-di25-hostego-v1/audits/route_integrity_M01_summary.json"),
        Path("artifacts/care-di25-hostego-v1/audits/host_row_count_reconciliation.txt"),
        Path("artifacts/care-source-local-assets-v2/source_asset_gate.txt"),
        Path("artifacts/care-source-local-assets-v2/host_regions_2017.npz"),
        Path("artifacts/care-source-local-assets-v2/ego_1hop_regions_2017.npz"),
        Path("artifacts/care-source-local-assets-v2/regions_2017.npz"),
        Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
        Path("artifacts/care-future-local-assets-v2/future_asset_gate.txt"),
        Path("artifacts/care-future-local-assets-v2/host_regions_2018_2021.npz"),
        Path("artifacts/care-future-local-assets-v2/ego_1hop_regions_2018_2021.npz"),
        Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz"),
        Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"),
        Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"),
        Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),
        Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"),
        package / "host_ego_encoder.py", package / "host_ego_models.py",
        package / "future_matrix/run_future_arm.py",
        package / "future_matrix/vendor/run_care_rolegraph_conditioned_dualstream_v4.py",
    ]
    errors.extend("missing file: " + str(path) for path in required if not path.is_file())
    for module in ("numpy", "sklearn", "torch", "yaml"):
        if importlib.util.find_spec(module) is None:
            errors.append("missing Python module: " + module)
    if errors:
        raise SystemExit("BLOCKED_BY_CONTRACT: exhaustive future preflight:\n- " + "\n- ".join(errors))

    for gate in (
        Path("artifacts/care-source-local-assets-v2/source_asset_gate.txt"),
        Path("artifacts/care-future-local-assets-v2/future_asset_gate.txt"),
    ):
        if "passed: true" not in gate.read_text(encoding="utf-8").lower():
            raise SystemExit("BLOCKED_BY_CONTRACT: asset gate is not PASS: " + str(gate))
    if "HOST_ROW_COUNT_RECONCILIATION_PASS" not in Path(
        "artifacts/care-di25-hostego-v1/audits/host_row_count_reconciliation.txt"
    ).read_text(encoding="utf-8"):
        raise SystemExit("BLOCKED_BY_CONTRACT: Host-row reconciliation did not pass")
    route_source = json.loads(Path(
        "artifacts/care-di25-hostego-v1/audits/route_integrity_M01_summary.json"
    ).read_text(encoding="utf-8"))
    for key in ("M0_vs_DI25_difference_count", "M1_vs_DI25_difference_count", "M0_vs_M1_difference_count"):
        if int(route_source[key]) != 0:
            raise SystemExit("BLOCKED_BY_CONTRACT: source route identity changed")
    lock_source(M0_SOURCE, "M0"); lock_source(M1_SOURCE, "M1")

    resolved = json.loads(RESOLVED.read_text(encoding="utf-8"))
    for key in ("features", "schema", "source_scores", "future_router_inputs_label_free",
                "future_label_file_posthoc_only"):
        if key not in resolved or not Path(resolved[key]).is_file():
            errors.append(f"resolved input unavailable: {key}={resolved.get(key)!r}")
    if errors:
        raise SystemExit("BLOCKED_BY_CONTRACT: resolved future inputs:\n- " + "\n- ".join(errors))

    future = read_csv(FUTURE_INPUT)
    by_sha = unique(future, "future DI input", 1989)
    years = Counter(int(row["year"]) for row in future)
    routes = Counter()
    for row in future:
        for key, value in row.items():
            if "label" in key.casefold() and str(value).strip():
                raise RuntimeError(f"future label visible in router/model input: {key}")
        route = int(float(row["route_DI"]))
        if route not in (0, 1):
            raise RuntimeError("future route_DI is not binary")
        routes[int(row["year"])] += route
    if dict(years) != EXPECTED_YEARS or dict(routes) != EXPECTED_ROUTE:
        raise RuntimeError(f"future cohort/route changed: years={dict(years)} routes={dict(routes)}")

    m0 = read_csv(M0_FUTURE)
    m0_index = unique(m0, "frozen M0 future", 1989)
    if set(m0_index) != set(by_sha):
        raise RuntimeError("frozen M0 future cohort differs from DI25 cohort")
    protocol = json.loads(M0_PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("representation") != "anomaly" or protocol.get("head") != "bounded":
        raise RuntimeError("frozen M0 future protocol is not Host/Anomaly/Bounded")
    if not protocol.get("source_only_training") or protocol.get("future_labels_loaded_by_model_arm"):
        raise RuntimeError("frozen M0 future protocol violates source-only contract")

    for path, n in (
        (Path("artifacts/care-source-local-assets-v2/host_regions_2017.npz"), 799),
        (Path("artifacts/care-source-local-assets-v2/ego_1hop_regions_2017.npz"), 799),
        (Path("artifacts/care-future-local-assets-v2/host_regions_2018_2021.npz"), 1989),
        (Path("artifacts/care-future-local-assets-v2/ego_1hop_regions_2018_2021.npz"), 1989),
    ):
        with np.load(path, allow_pickle=False) as saved:
            ids = [normal(value) for value in saved["sha256"].astype(str)]
            if len(ids) != n or len(set(ids)) != n:
                raise RuntimeError(f"asset lineage failed: {path}")

    ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": True, "source_n": 799, "future_n": 1989,
        "future_year_counts": EXPECTED_YEARS, "future_DI25_routes": EXPECTED_ROUTE,
        "M0_future_reused_not_retrained": True,
        "M1_source_definition_locked": True,
        "future_labels_visible_to_model": False,
    }
    (ROOT / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("DI25_HOSTEGO_FUTURE_PREFLIGHT_PASS")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
