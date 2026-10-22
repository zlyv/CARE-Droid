#!/usr/bin/env python3
"""Run or validate the predeclared five-replicate M0/M1 seed audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from freeze_final_method import ROOT, ROUTER, SEED_SETS, blocked, contract, read_csv, sha256_file


OUT = ROOT / "seed_robustness"
RUNNER = Path("scripts/care-di25-hostego-v1/run_care_di25_hostego_v1.py")
LOCKED = {
    "M0": ROOT / "source/M0_host_anomaly_bounded/source_predictions.csv",
    "M1": ROOT / "source/M1_hostego_anomaly_bounded/source_predictions.csv",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def exact_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if not n:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def resolve_column(rows: list[dict], token: str) -> str:
    columns = [name for name in rows[0] if name.startswith("DI25__") and name.endswith("__prob")]
    columns = [name for name in columns if token in name and (token != "Host" or "HostEgo" not in name)]
    if len(columns) != 1:
        blocked(f"cannot resolve {token} bounded output: {columns}")
    return columns[0]


def load_arm(path: Path, arm: str) -> tuple[list[dict], str]:
    rows = read_csv(path)
    if len(rows) != 799 or len({r["sha256"].strip().lower() for r in rows}) != 799:
        blocked(f"{arm} replicate is not exact 799")
    return rows, resolve_column(rows, "Host" if arm == "M0" else "HostEgo")


def metric_row(arm: str, rows: list[dict], column: str) -> dict:
    y = np.asarray([int(float(r["label"])) for r in rows], int)
    base = np.asarray([float(r["Global24__prob"]) for r in rows], float)
    p = np.asarray([float(r[column]) for r in rows], float)
    route = np.asarray([int(float(r["A25__routed"])) == 1 for r in rows], bool)
    pred, base_pred = p >= .5, base >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fixed = int(np.sum(route & (base_pred != y) & (pred == y)))
    broken = int(np.sum(route & (base_pred == y) & (pred != y)))
    return {
        "arm": arm, "method": arm, "parameter_count": 33388 if arm == "M0" else 34613,
        "n": len(y), "routed": int(route.sum()), "routing_rate": float(route.mean()),
        "BA": float(balanced_accuracy_score(y, pred)), "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, pred)), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "fixed": fixed, "broken": broken, "net": fixed - broken,
        "correction_precision": fixed / (fixed + broken) if fixed + broken else float("nan"),
    }


def paired_rows(m0_rows: list[dict], m0_col: str, m1_rows: list[dict], m1_col: str) -> tuple[list[dict], dict]:
    one = {r["sha256"].strip().lower(): r for r in m0_rows}
    two = {r["sha256"].strip().lower(): r for r in m1_rows}
    if set(one) != set(two):
        blocked("M0/M1 replicate cohorts differ")
    output, counts = [], {"same_correct": 0, "same_wrong": 0, "M0_wrong_M1_correct": 0, "M0_correct_M1_wrong": 0}
    for sha in [r["sha256"].strip().lower() for r in m0_rows]:
        a, b = one[sha], two[sha]
        y = int(float(a["label"]))
        c0 = (float(a[m0_col]) >= .5) == bool(y)
        c1 = (float(b[m1_col]) >= .5) == bool(y)
        status = "same_correct" if c0 and c1 else "same_wrong" if not c0 and not c1 else "M0_wrong_M1_correct" if not c0 else "M0_correct_M1_wrong"
        counts[status] += 1
        output.append({"sha256": sha, "label": y, "fold": a["fold"], "M0_prob": a[m0_col], "M1_prob": b[m1_col], "M0_correct": int(c0), "M1_correct": int(c1), "status": status})
    n01, n10 = counts["M0_wrong_M1_correct"], counts["M0_correct_M1_wrong"]
    return output, {**counts, "n01": n01, "n10": n10, "exact_mcnemar_two_sided_p": exact_p(n01, n10)}


def locate_generated(work: Path, arm: str) -> Path:
    folder = "M0_host_anomaly_bounded" if arm == "M0" else "M1_hostego_anomaly_bounded"
    return work / "source" / folder / "source_predictions.csv"


def architecture(work: Path, arm: str) -> dict:
    folder = "M0_host_anomaly_bounded" if arm == "M0" else "M1_hostego_anomaly_bounded"
    path = work / "source" / folder / "architecture.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def timing_row(arm: str, arch: dict, historical: bool) -> dict:
    total = arch.get("wall_seconds_including_training", "unavailable")
    load = arch.get("feature_cache_loading_seconds", "unavailable")
    infer = arch.get("model_inference_seconds", "unavailable")
    feature = arch.get("feature_prepare_seconds", "unavailable")
    training = arch.get("training_seconds", "unavailable")
    residual = "unavailable"
    try:
        residual = max(0.0, float(total) - float(load) - float(infer))
    except (TypeError, ValueError):
        pass
    return {
        "arm": arm,
        "total_seconds": total,
        "data_load_seconds": load,
        "feature_prepare_seconds": feature if not historical else arch.get("feature_prepare_seconds", "unavailable_historical_run"),
        "training_seconds": training if not historical else arch.get("training_seconds", "unavailable_historical_run"),
        "feature_prepare_plus_training_and_overhead_seconds": residual,
        "inference_seconds": infer,
        "seconds_per_model_apk": arch.get("model_inference_seconds_per_model_apk", "unavailable"),
        "seconds_per_routed_apk": arch.get("model_inference_seconds_per_model_apk", "unavailable"),
        "timing_origin": "historical_locked_run" if historical else "new_replicate_instrumentation",
        "timing_note": "R1-R4 use phase instrumentation with CUDA synchronization around training/inference. R0 keeps historical unavailable fields rather than manufacturing a split.",
    }


def output_valid(repdir: Path) -> bool:
    marker = repdir / "COMPLETE.json"
    if not marker.is_file():
        return False
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
        return all(Path(name).is_file() and sha256_file(Path(name)) == digest for name, digest in state["outputs"].items())
    except Exception:
        return False


def run_replicate(name: str, device: str, force: bool) -> None:
    seeds = SEED_SETS[name]
    repdir = OUT / name
    if output_valid(repdir) and not force:
        print(f"[resume] {name} is complete and checksum-valid", flush=True)
        return
    repdir.mkdir(parents=True, exist_ok=True)
    protected_before = {str(path): sha256_file(path) for path in LOCKED.values()}
    work = repdir / "work"
    if name == "R0":
        source_paths = LOCKED
    else:
        source_paths = {}
        for arm in ("M0", "M1"):
            command = [sys.executable, str(RUNNER), "--arm", arm, "--output", str(work), "--seeds", ",".join(map(str, seeds)), "--device", device]
            started = time.perf_counter()
            subprocess.run(command, check=True)
            print(f"{name} {arm} subprocess_seconds={time.perf_counter()-started:.3f}", flush=True)
            source_paths[arm] = locate_generated(work, arm)
    canonical = {}
    metrics, timing = [], []
    for arm in ("M0", "M1"):
        target = repdir / f"{arm}_predictions.csv"
        shutil.copyfile(source_paths[arm], target)
        rows, column = load_arm(target, arm)
        canonical[arm] = (rows, column)
        metrics.append({"replicate": name, "seed_triplet": ",".join(map(str, seeds)), **metric_row(arm, rows, column)})
        timing.append(timing_row(arm, architecture(ROOT if name == "R0" else work, arm), name == "R0"))
    frozen_route = {r["sha256"].strip().lower(): (int(float(r["fold"])), int(float(r["route_DI"]))) for r in read_csv(ROUTER)}
    for arm, (rows, _) in canonical.items():
        differences = [r["sha256"] for r in rows if frozen_route.get(r["sha256"].strip().lower()) != (int(float(r["fold"])), int(float(r["A25__routed"])))]
        if differences:
            blocked(f"{name} {arm} differs from frozen DI25/folds for {len(differences)} APKs; first={differences[:3]}")
    paired, mc = paired_rows(*canonical["M0"], *canonical["M1"])
    write_csv(repdir / "metrics.csv", metrics)
    write_csv(repdir / "paired_changes.csv", paired)
    write_csv(repdir / "mcnemar.csv", [{"replicate": name, **mc}])
    write_csv(repdir / "timing.csv", timing)
    protected_after = {str(path): sha256_file(path) for path in LOCKED.values()}
    if protected_before != protected_after:
        blocked(f"locked source prediction modified during {name}")
    run_contract = {
        "replicate": name, "seeds": seeds, "arms": ["M0", "M1"], "router": "DI25", "budget": .25,
        "max_delta_logit": 2.25, "source_only_2017": True, "future_rows_loaded": False,
        "parameter_counts": {"M0": 33388, "M1": 34613}, "protected_source_hashes": protected_after,
        "timing_caveat": timing[0]["timing_note"],
    }
    (repdir / "run_contract.json").write_text(json.dumps(run_contract, indent=2) + "\n", encoding="utf-8")
    outputs = [repdir / name for name in ("M0_predictions.csv", "M1_predictions.csv", "metrics.csv", "paired_changes.csv", "mcnemar.csv", "timing.csv", "run_contract.json")]
    marker = {"replicate": name, "outputs": {str(path): sha256_file(path) for path in outputs}}
    (repdir / "COMPLETE.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    by_arm = {row["arm"]: row for row in metrics}
    elapsed = sum(float(row["total_seconds"]) for row in timing if str(row["total_seconds"]) not in {"unavailable", "unavailable_historical_run"})
    print("[SEED ROBUSTNESS]", flush=True)
    print(f"replicate={name}", flush=True); print(f"seed_triplet={','.join(map(str, seeds))}", flush=True)
    print(f"M0 BA={by_arm['M0']['BA']:.9f}", flush=True); print(f"M1 BA={by_arm['M1']['BA']:.9f}", flush=True)
    print(f"delta_BA={by_arm['M1']['BA']-by_arm['M0']['BA']:+.9f}", flush=True)
    print(f"M0 net={by_arm['M0']['net']}", flush=True); print(f"M1 net={by_arm['M1']['net']}", flush=True)
    print(f"delta_net={by_arm['M1']['net']-by_arm['M0']['net']:+d}", flush=True); print(f"elapsed={elapsed:.3f}", flush=True)
    print(f"SEED_REPLICATE_PASS n01={mc['n01']} n10={mc['n10']} p={mc['exact_mcnemar_two_sided_p']:.6f}", flush=True)


def dry_run(replicate: str) -> None:
    frozen = contract()
    if not RUNNER.is_file():
        blocked(f"seed runner missing: {RUNNER}")
    value = {
        "passed": True, "mode": "dry-run", "replicate": replicate, "seeds": SEED_SETS[replicate],
        "would_execute_arms": ["M0", "M1"], "would_train": replicate != "R0",
        "output": str(OUT / replicate), "runner": str(RUNNER), "final_method": frozen["method_name"],
        "future_rows_loaded": False, "locked_source_hashes": {str(p): sha256_file(p) for p in LOCKED.values()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dry_run_contract.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print("SEED_ROBUSTNESS_DRY_RUN_PASS")
    print(json.dumps(value, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "run"), default="dry-run")
    parser.add_argument("--replicate", choices=tuple(SEED_SETS) + ("all",), default="R1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    contract()
    if args.mode == "dry-run":
        dry_run("R1" if args.replicate == "all" else args.replicate)
        return
    names = list(SEED_SETS) if args.replicate == "all" else [args.replicate]
    for name in names:
        run_replicate(name, args.device, args.force)


if __name__ == "__main__":
    main()
