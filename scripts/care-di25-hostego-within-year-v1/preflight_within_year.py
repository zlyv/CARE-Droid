#!/usr/bin/env python3
"""Fail-closed gate for the frozen DI25 Host/Ego same-year experiment."""
from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


ROOT = Path("artifacts/care-di25-hostego-within-year-v1")
MATRIX_ROOT = Path("artifacts/care-route-stage2-within-year-matrix-v1")
ROUTER = MATRIX_ROOT / "shared/within_year_router_inputs.csv"
ROUTER_PROTOCOL = MATRIX_ROOT / "shared/router_protocol.json"
RESOLVED = MATRIX_ROOT / "preflight/resolved_inputs.json"
M0 = MATRIX_ROOT / "arms/anomaly_bounded/within_year_expert_predictions.csv"
M0_PROTOCOL = MATRIX_ROOT / "arms/anomaly_bounded/protocol.json"
M1_SOURCE = Path("artifacts/care-di25-hostego-v1/source/M1_hostego_anomaly_bounded/source_predictions.csv")
M1_SOURCE_EXPECTED = (0.9148715538847118, 0.9433521303258148, 0.9132653061224489)
EXPECTED_YEARS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
EXPECTED_ROUTES = {2018: 198, 2019: 100, 2020: 99, 2021: 100}


def normal(value: object) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def unique(rows: list[dict], name: str, expected: int = 1989) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        sha = normal(row.get("sha256", row.get("sample_id", "")))
        if not sha or sha in result:
            raise RuntimeError(f"{name}: missing/duplicate SHA {sha!r}")
        result[sha] = row
    if len(rows) != expected or len(result) != expected:
        raise RuntimeError(f"{name}: rows={len(rows)} unique={len(result)} expected={expected}")
    return result


def source_lock() -> None:
    rows = read_csv(M1_SOURCE)
    unique(rows, "source M1", 799)
    columns = [name for name in rows[0] if name.startswith("DI25__") and name.endswith("__prob") and "HostEgo" in name]
    if len(columns) != 1:
        raise RuntimeError("cannot identify locked source M1 probability: " + repr(columns))
    y = np.asarray([int(float(row["label"])) for row in rows], dtype=int)
    p = np.asarray([float(row[columns[0]]) for row in rows], dtype=float)
    actual = (
        float(balanced_accuracy_score(y, p >= 0.5)),
        float(roc_auc_score(y, p)),
        float(f1_score(y, p >= 0.5)),
    )
    if not np.allclose(actual, M1_SOURCE_EXPECTED, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"source-selected M1 definition changed: {actual}")


def main() -> None:
    package = Path(__file__).resolve().parent
    required = [
        ROUTER, ROUTER_PROTOCOL, RESOLVED, M0, M0_PROTOCOL, M1_SOURCE,
        Path("artifacts/care-future-local-assets-v2/future_asset_gate.txt"),
        Path("artifacts/care-future-local-assets-v2/host_regions_2018_2021.npz"),
        Path("artifacts/care-future-local-assets-v2/ego_1hop_regions_2018_2021.npz"),
        Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz"),
        Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"),
        Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
        Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"),
        package / "host_ego_encoder.py", package / "host_ego_models.py",
        package / "within_matrix/run_within_year_arm.py",
        package / "within_matrix/vendor/run_care_rolegraph_conditioned_dualstream_v4.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    for module in ("numpy", "sklearn", "torch", "yaml"):
        if importlib.util.find_spec(module) is None:
            missing.append("Python module: " + module)
    if missing:
        raise SystemExit("BLOCKED_BY_CONTRACT: missing prerequisites:\n- " + "\n- ".join(missing))

    gate = Path("artifacts/care-future-local-assets-v2/future_asset_gate.txt").read_text(encoding="utf-8").lower()
    if "passed: true" not in gate:
        raise SystemExit("BLOCKED_BY_CONTRACT: future Host/Ego asset gate is not PASS")
    source_lock()

    router = read_csv(ROUTER)
    router_index = unique(router, "same-year DI25 router")
    years = Counter(int(row["year"]) for row in router)
    routes = Counter()
    fold_counts: dict[int, Counter] = {}
    for row in router:
        year = int(row["year"])
        fold = int(row["fold"])
        if fold not in range(1, 6):
            raise RuntimeError(f"invalid same-year fold {year}/{fold}")
        fold_counts.setdefault(year, Counter())[fold] += 1
        value = int(float(row["route_DI"]))
        if value not in (0, 1):
            raise RuntimeError("route_DI is not binary")
        routes[year] += value
    if dict(years) != EXPECTED_YEARS or dict(routes) != EXPECTED_ROUTES:
        raise RuntimeError(f"same-year cohort/routes changed: years={dict(years)}, routes={dict(routes)}")
    if any(set(counts) != set(range(1, 6)) for counts in fold_counts.values()):
        raise RuntimeError("same-year fold coverage is incomplete")

    router_protocol = json.loads(ROUTER_PROTOCOL.read_text(encoding="utf-8"))
    if router_protocol.get("protocol") != "strict independent within-year five-fold OOF":
        raise RuntimeError("router is not strict independent within-year five-fold OOF")
    if router_protocol.get("feature_contract_version") != "source-frozen-compact76-foldsafe-v1":
        raise RuntimeError("same-year router feature contract changed")

    m0_rows = read_csv(M0)
    m0_index = unique(m0_rows, "same-year M0")
    if set(m0_index) != set(router_index):
        raise RuntimeError("same-year M0/router cohorts differ")
    for sha, row in m0_index.items():
        reference = router_index[sha]
        if int(row["year"]) != int(reference["year"]) or int(row["fold"]) != int(reference["fold"]):
            raise RuntimeError("same-year M0 fold/year alignment changed: " + sha)
        if row.get("feature_contract_version") != "source-frozen-compact76-foldsafe-v1":
            raise RuntimeError("same-year M0 predates the fold-safe Compact76 contract")
    protocol = json.loads(M0_PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("representation") != "anomaly" or protocol.get("head") != "bounded":
        raise RuntimeError("existing same-year M0 is not Host/Anomaly/Bounded")

    resolved = json.loads(RESOLVED.read_text(encoding="utf-8"))
    for key in ("features", "schema", "source_manifest"):
        if key not in resolved or not Path(resolved[key]).is_file():
            raise RuntimeError(f"resolved same-year input unavailable: {key}={resolved.get(key)!r}")

    for path in (
        Path("artifacts/care-future-local-assets-v2/host_regions_2018_2021.npz"),
        Path("artifacts/care-future-local-assets-v2/ego_1hop_regions_2018_2021.npz"),
    ):
        with np.load(path, allow_pickle=False) as saved:
            ids = [normal(value) for value in saved["sha256"].astype(str)]
            if len(ids) != 1989 or len(set(ids)) != 1989 or set(ids) != set(router_index):
                raise RuntimeError("future Host/Ego asset lineage differs: " + str(path))

    ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": True,
        "protocol": "independent per-year five-fold OOF",
        "years": EXPECTED_YEARS,
        "DI25_routes": EXPECTED_ROUTES,
        "M0_reused_not_retrained": True,
        "M1_source_definition_locked": True,
        "max_delta_logit": 2.25,
        "temporal_generalization_result": False,
    }
    (ROOT / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("DI25_HOSTEGO_WITHIN_YEAR_PREFLIGHT_PASS")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
