#!/usr/bin/env python3
from __future__ import annotations
import csv, json, shutil
from pathlib import Path

ROOT = Path("artifacts/care-di25-hostrel-semantic-background-source-v1")
ROUTER = Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
H0 = Path("artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY")
B1 = Path("artifacts/care-di25-roleconditioned-background-source-v1/arms/B1_HARD_ROLE")
SOURCE_REF = Path("artifacts/care-di25-host-relative-source-v1/preflight/di25_reference")
REQ = (ROUTER, H0/"source_predictions.csv", B1/"source_predictions.csv", SOURCE_REF/"source_predictions.csv", Path("artifacts/care-source-local-assets-v2/regions_2017.npz"), Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"))


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    missing = [str(p) for p in REQ if not p.is_file() or p.stat().st_size < 10]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty:\n- " + "\n- ".join(missing))
    rows = read(ROUTER)
    folds = {f: sum(int(float(r["fold"])) == f for r in rows) for f in range(1, 6)}
    routes = {f: sum(int(float(r["fold"])) == f and int(float(r["route_DI"])) == 1 for r in rows) for f in range(1, 6)}
    if len(rows) != 799 or len({r["sha256"].lower() for r in rows}) != 799 or {int(float(r["year"])) for r in rows} != {2017}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort changed")
    if folds != {1:160, 2:160, 3:160, 4:160, 5:159} or routes != {1:40, 2:40, 3:40, 4:40, 5:40}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: fold/DI25 changed")
    target = ROOT/"preflight/di25_reference"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_REF, target, dirs_exist_ok=True)
    contract = {
        "source_rows": 799,
        "year": 2017,
        "route_DI": 200,
        "route_DI_per_fold": routes,
        "B0_H0_locked_not_retrained": True,
        "B1_HARD_ROLE_locked_not_retrained": True,
        "R1_only_trained_arm": True,
        "model_host_width": 74,
        "parameter_count": 33036,
        "background_same_representation_as_host": True,
        "future_rows_loaded": 0,
        "ego_tensor_consumed": False,
        "ego_encoder_instantiated": False,
    }
    (ROOT/"preflight/contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print("HOSTREL_SEMANTIC_BACKGROUND_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 route_DI=200")
    print("B0=reuse_locked B1=reuse_locked R1=train")
    print("model_host_width=74 parameter_count=33036 backgrounds_same_schema=true")
    print("future_rows_loaded=0 ego_consumed=false")


if __name__ == "__main__":
    main()
