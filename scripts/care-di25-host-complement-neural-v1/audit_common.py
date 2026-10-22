#!/usr/bin/env python3
"""Shared, side-effect-free contracts for Host Complement Neural v1."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/care-di25-host-complement-neural-v1")
def _pick(primary: str, local: str) -> Path:
    return Path(primary) if Path(primary).exists() else Path(local)
M1_ROOT = _pick("scripts/care-di25-hostego-within-year-v1", "care-di25-hostego-within-year-v1")
MATRIX_ROOT = M1_ROOT / "within_matrix"
LOCKED_M0 = Path("artifacts/care-di25-hostego-within-year-v1/m1_hostego_anomaly_bounded/within_year_m1_predictions.csv")
ROUTER = Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
HOST = Path("artifacts/care-future-local-assets-v2/host_regions_2018_2021.npz")
EGO = Path("artifacts/care-future-local-assets-v2/ego_1hop_regions_2018_2021.npz")
ENDPOINT = Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz")
ASSET_MANIFEST = Path("artifacts/care-future-local-assets-v2/local_asset_manifest.json")
EXPECTED_M0_SHA256 = "5a660ff60b252682a1b975a0dbd993437d5dfa7c6ab6044ae94ab5e76528339b"
EXPECTED_ROUTES = {2018: 198, 2019: 100, 2020: 99, 2021: 100}
ARM_NAMES = {
    "A": "M_A_COMPOSITION",
    "B": "M_B_POSITION",
    "AB": "M_AB_COMPOSITION_POSITION",
}


def normal(value: object) -> str:
    return str(value or "").strip().lower()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_m1_modules():
    for path in (M1_ROOT, MATRIX_ROOT, MATRIX_ROOT / "vendor"):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
    import run_m1_within_year as m1  # type: ignore
    return m1


def router_contract(rows: list[dict]) -> dict:
    ids = [normal(r["sha256"]) for r in rows]
    years = np.asarray([int(r["year"]) for r in rows])
    routes = np.asarray([int(r["route_DI"]) for r in rows])
    checks = {
        "rows_exact_1989": len(rows) == 1989,
        "ids_unique_1989": len(set(ids)) == 1989,
        "years_exact": sorted(set(years.tolist())) == [2018, 2019, 2020, 2021],
        "route_binary": set(routes.tolist()) <= {0, 1},
        "total_routed_exact_497": int(routes.sum()) == 497,
    }
    counts = {int(y): int(routes[years == y].sum()) for y in sorted(set(years.tolist()))}
    checks["annual_routes_exact"] = counts == EXPECTED_ROUTES
    return {"passed": all(checks.values()), "checks": checks, "annual_routes": counts}


def protected_paths() -> list[Path]:
    paths = [LOCKED_M0, ROUTER, HOST, EGO, ENDPOINT, ASSET_MANIFEST]
    paths += [
        M1_ROOT / "run_m1_within_year.py",
        M1_ROOT / "host_ego_encoder.py",
        M1_ROOT / "host_ego_models.py",
        MATRIX_ROOT / "run_within_year_arm.py",
        MATRIX_ROOT / "run_future_arm.py",
        MATRIX_ROOT / "run_care_rgcondattn_dedup_bounded_source_v1.py",
        (MATRIX_ROOT / "run_care_rolegraph_conditioned_dualstream_v4.py")
        if (MATRIX_ROOT / "run_care_rolegraph_conditioned_dualstream_v4.py").is_file()
        else MATRIX_ROOT / "vendor/run_care_rolegraph_conditioned_dualstream_v4.py",
    ]
    return paths


def hash_manifest(paths: list[Path]) -> dict[str, str]:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing protected inputs:\n- " + "\n- ".join(missing))
    return {str(p): sha256_file(p) for p in paths}


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    pred = p >= 0.5
    yb = y.astype(bool)
    tn = int(np.sum(~yb & ~pred)); fp = int(np.sum(~yb & pred))
    fn = int(np.sum(yb & ~pred)); tp = int(np.sum(yb & pred))
    return {
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "Wrong": fp + fn,
    }


def exact_mcnemar(old_correct: np.ndarray, new_correct: np.ndarray) -> dict:
    from scipy.stats import binomtest
    n01 = int(np.sum(~old_correct & new_correct))
    n10 = int(np.sum(old_correct & ~new_correct))
    return {"old_wrong_new_correct": n01, "old_correct_new_wrong": n10,
            "p_value": float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}


def stable_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
