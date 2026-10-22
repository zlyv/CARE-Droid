#!/usr/bin/env python3
"""Read-only transfer audit of frozen BFA expert predictions onto Manifest-RF routes.

Reads only existing prediction CSVs. Writes only to a new --output directory.
Does not read APKs, train models, or alter historical assets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


ARMS = (
    "B0_ORIGINAL",
    "B1_EXTERNAL_NEUTRAL",
    "B2_SHARED_NEUTRAL",
    "B3_INTERNAL_NEUTRAL",
    "B4_EXTERNAL_ROLE_SHUFFLE",
)
STRATEGIES = ("ANCHOR_UNC25_REF", "OTHER3_OP25_REF")
EXPERT_PARITY_TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def metrics(rows: list[dict]) -> dict:
    y = np.array([r["label"] for r in rows], dtype=int)
    p = np.array([r["probability"] for r in rows], dtype=float)
    s = np.array([r["stage1_probability"] for r in rows], dtype=float)
    routed = np.array([r["routed"] for r in rows], dtype=int)
    pred = p >= 0.5
    base = s >= 0.5
    errors = base != y
    fixed = int(np.sum(errors & (pred == y)))
    broken = int(np.sum((~errors) & (pred != y)))
    return {
        "n": len(rows),
        "routed": int(routed.sum()),
        "stage1_errors": int(errors.sum()),
        "routed_stage1_errors": int(np.sum(errors & (routed == 1))),
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "FN": int(np.sum((y == 1) & (~pred))),
        "FP": int(np.sum((y == 0) & pred)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-pred", type=Path, required=True)
    ap.add_argument("--bfa-pred", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.manifest_pred, args.bfa_pred):
        if not p.is_file():
            raise RuntimeError(f"Missing frozen prediction table: {p}")
    if args.output.exists():
        raise RuntimeError(f"Output already exists; refusing overwrite: {args.output}")
    before = {str(p): sha256(p) for p in (args.manifest_pred, args.bfa_pred)}

    bfa = {}
    for r in read_csv(args.bfa_pred):
        key = (r["year"], r["sha256"], r["arm"])
        if key in bfa:
            raise RuntimeError(f"Duplicate BFA key {key}")
        bfa[key] = r
    prediction_rows = []
    parity_max = 0.0
    seen = set()
    selected = [
        r for r in read_csv(args.manifest_pred)
        if r["anchor"] == "RF" and r["strategy"] in STRATEGIES
    ]
    if len(selected) != 3 * 2788 * len(STRATEGIES):
        raise RuntimeError(f"Unexpected Manifest-RF row count: {len(selected)}")
    for r in selected:
        key = (r["seed"], r["year"], r["sha256"], r["strategy"])
        if key in seen:
            raise RuntimeError(f"Duplicate Manifest key {key}")
        seen.add(key)
        split = f"S{r['fold']}" if r["year"] == "2017" else f"T{int(r['year']) - 2017}"
        y = int(r["label"])
        base_key = (r["year"], r["sha256"], "B0_ORIGINAL")
        if base_key not in bfa:
            raise RuntimeError(f"Missing BFA baseline: {base_key}")
        original = bfa[base_key]
        if int(original["label"]) != y or original["split"] != split:
            raise RuntimeError(f"Label or fold mismatch: {base_key}")
        diff = abs(float(original["expert_probability"]) - float(r["expert_probability"]))
        parity_max = max(parity_max, diff)
        if diff > EXPERT_PARITY_TOLERANCE:
            raise RuntimeError(f"Frozen expert parity mismatch {base_key}: {diff}")
        stage1 = float(r["stage1_probability"])
        routed = int(r["routed"])
        if routed not in (0, 1):
            raise RuntimeError("Nonbinary route")
        for arm in ARMS:
            k = (r["year"], r["sha256"], arm)
            if k not in bfa:
                raise RuntimeError(f"Missing BFA arm: {k}")
            br = bfa[k]
            if int(br["label"]) != y or br["split"] != split:
                raise RuntimeError(f"Arm label or fold mismatch: {k}")
            expert = float(br["expert_probability"])
            prediction_rows.append({
                "seed": r["seed"],
                "year": r["year"],
                "fold": r["fold"],
                "sha256": r["sha256"],
                "strategy": r["strategy"],
                "arm": arm,
                "label": y,
                "routed": routed,
                "stage1_probability": stage1,
                "expert_probability": expert,
                "probability": expert if routed else stage1,
            })

    after = {str(p): sha256(p) for p in (args.manifest_pred, args.bfa_pred)}
    if before != after:
        raise RuntimeError("Reused input changed during audit")
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "per_apk_predictions.csv", prediction_rows)
    grouped = defaultdict(list)
    for r in prediction_rows:
        grouped[(r["seed"], r["year"], r["strategy"], r["arm"])].append(r)
    yearly = []
    for (seed, year, strategy, arm), rr in sorted(grouped.items()):
        yearly.append({"seed": seed, "year": year, "strategy": strategy, "arm": arm, **metrics(rr)})
    write_csv(args.output / "yearly_metrics.csv", yearly)

    future = defaultdict(list)
    for r in prediction_rows:
        if r["year"] != "2017":
            future[(r["seed"], r["strategy"], r["arm"])].append(r)
    future_rows = []
    for (seed, strategy, arm), rr in sorted(future.items()):
        future_rows.append({"seed": seed, "strategy": strategy, "arm": arm, **metrics(rr)})
    write_csv(args.output / "future_pooled_metrics.csv", future_rows)

    comparison = []
    for strategy in STRATEGIES:
        for arm in ARMS[1:]:
            for scope in ("2017", "future"):
                subset = [
                    x for x in (yearly if scope == "2017" else future_rows)
                    if x["strategy"] == strategy and (scope != "2017" or x["year"] == "2017")
                ]
                base = {x["seed"]: x for x in subset if x["arm"] == "B0_ORIGINAL"}
                variant = {x["seed"]: x for x in subset if x["arm"] == arm}
                if set(base) != set(variant) or len(base) != 3:
                    raise RuntimeError(f"Missing seed comparison: {strategy}/{arm}/{scope}")
                comparison.append({
                    "strategy": strategy,
                    "arm": arm,
                    "scope": scope,
                    "mean_delta_BA_pp": np.mean([
                        100 * (variant[s]["BA"] - base[s]["BA"]) for s in base
                    ]),
                    "mean_delta_net": np.mean([
                        variant[s]["net"] - base[s]["net"] for s in base
                    ]),
                    "mean_delta_fixed": np.mean([
                        variant[s]["fixed"] - base[s]["fixed"] for s in base
                    ]),
                    "mean_delta_broken": np.mean([
                        variant[s]["broken"] - base[s]["broken"] for s in base
                    ]),
                    "positive_BA_seeds": sum(variant[s]["BA"] > base[s]["BA"] for s in base),
                    "positive_net_seeds": sum(variant[s]["net"] > base[s]["net"] for s in base),
                })
    write_csv(args.output / "paired_comparison.csv", comparison)
    integrity = {
        "status": "PASS",
        "before_sha256": before,
        "after_sha256": after,
        "expert_parity_max_difference": parity_max,
        "expert_parity_tolerance": EXPERT_PARITY_TOLERANCE,
        "manifest_selected_rows": len(selected),
        "scope": "Two reused prediction CSVs only; APKs/checkpoints/source files not opened",
    }
    (args.output / "asset_integrity_check.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    decision = {
        "status": "FROZEN_BACKGROUND_TRANSFER_AUDIT_COMPLETE",
        "new_training": False,
        "new_APK_scans": 0,
        "future_years_are_development": True,
        "formal_model_changed": False,
        "next_action": "Compare B1/B2/B3 against B0 and B4 before any new background training.",
    }
    (args.output / "final_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **integrity, **decision}))


if __name__ == "__main__":
    main()
