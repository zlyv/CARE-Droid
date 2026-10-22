#!/usr/bin/env python3
"""Strict-2017 integration: LiteLocal FUSED63 Stage-I -> DI25 -> existing CARE M0 Host expert -> bounded correction.

Design contract
---------------
1. Stage-I is the modified LiteLocal FUSED63 committee (P31 + COOC16 + TRANS16).
2. DI25 is recomputed from the selected Stage-I definition using MeanOpposition,
   with exactly 25% routed independently in each fixed outer fold.
3. Stage-II is NOT reimplemented.  The script reuses the established
   care-di25-host-input-replay-v1 FULL arm (Host-only RGCondAttn Dedup +
   FixKeep + bounded correction).
4. Outer-validation Stage-I probabilities are strict five-fold OOF.  For each
   outer fold, Stage-II training receives a NEW inner-OOF Stage-I probability
   generated only from the other three source folds.  Thus no held-out outer
   fold leaks into the Stage-II training base probabilities.
5. The existing 24-D Global24 vector is retained only as Stage-II context for
   the routed expert.  The Stage-I classifier/router is LiteLocal FUSED63.
6. No future row or future label is loaded.

Compatibility note
------------------
The historical CARE reference reader expects columns named A25__score and
A25__routed.  This script writes the NEW LiteLocal DI25 score/route into those
legacy compatibility columns; semantically they are DI25-LiteLocal, not the
old A25/Global24 router.  Explicit DI25_LiteLocal__* columns are also written.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

PROJECT = Path("/home/boyuanjie/projects/tasa-droid")
LITE_FEATURES = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
ROUTER = PROJECT / "artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv"
LITE_UPSTREAM = PROJECT / "scripts/care-litelocal-semantic-source-v1/run_source_oof.py"
ATTRIBUTION_OOF = PROJECT / "artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv"
HOST_ARM = PROJECT / "scripts/care-di25-host-input-replay-v1/run_arm.py"
HOST_DIR = HOST_ARM.parent
DEFAULT_OUT = PROJECT / "artifacts/care-litelocal-di25-host-m0-source-v1"
MEMBERS = ("LR", "RF", "XGB", "SVM")
DEFINITIONS = ("LR_ONLY", "MEAN_COMMITTEE", "MEDIAN_COMMITTEE")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def norm(x):
    return str(x or "").strip().casefold()


def stage1_probability(member_probabilities: np.ndarray, definition: str) -> np.ndarray:
    p = np.asarray(member_probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 4:
        raise ValueError("expected N x 4 member probabilities")
    if definition == "LR_ONLY":
        return p[:, 0]
    if definition == "MEAN_COMMITTEE":
        return p.mean(axis=1)
    if definition == "MEDIAN_COMMITTEE":
        return np.median(p, axis=1)
    raise KeyError(definition)


def mean_opposition(member_probabilities: np.ndarray, stage1_p: np.ndarray) -> np.ndarray:
    p = np.asarray(member_probabilities, dtype=float)
    decision = np.asarray(stage1_p, dtype=float) >= 0.5
    return np.where(decision[:, None], 1.0 - p, p).mean(axis=1)


def foldwise_top_budget(score, folds, ids, fraction=0.25):
    score = np.asarray(score, dtype=float)
    folds = np.asarray(folds, dtype=int)
    ids = np.asarray(ids).astype(str)
    route = np.zeros(len(score), dtype=bool)
    for fold in sorted(set(folds.tolist())):
        idx = np.flatnonzero(folds == fold)
        count = int(round(len(idx) * fraction))
        order = np.lexsort((ids[idx], -score[idx]))
        route[idx[order[:count]]] = True
    return route


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p >= 0.5
    return {
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "errors": int(np.sum(pred != y)),
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_source_arrays():
    router = read_csv(ROUTER)
    rmap = {norm(r["sha256"]): r for r in router}
    if len(router) != 799 or len(rmap) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: router cohort is not fixed 799")
    with np.load(LITE_FEATURES, allow_pickle=False) as z:
        ids = np.asarray([norm(x) for x in z["sha256"]])
        prevalence = z["role_prevalence"].astype(float)
        cooc = z["within_method_cooccurrence"].astype(float)
        transitions = z["directed_adjacent_transitions"].astype(float)
    if len(ids) != 799 or len(set(ids.tolist())) != 799 or set(ids.tolist()) != set(rmap):
        raise RuntimeError("BLOCKED_BY_CONTRACT: LiteLocal/source-router SHA mismatch")
    y = np.asarray([int(float(rmap[s]["label"])) for s in ids], dtype=int)
    folds = np.asarray([int(float(rmap[s]["fold"])) for s in ids], dtype=int)
    if [int(np.sum(folds == f)) for f in range(1, 6)] != [160, 160, 160, 160, 159]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: fixed fold sizes changed")
    return ids, y, folds, prevalence, cooc, transitions, rmap


def load_outer_member_oof_from_attribution(ids):
    if not ATTRIBUTION_OOF.is_file():
        return None
    rows = read_csv(ATTRIBUTION_OOF)
    by_sha = {norm(r["sha256"]): r for r in rows}
    cols = [f"p_FUSED63__{m}" for m in MEMBERS]
    if len(by_sha) != 799 or not all(c in rows[0] for c in cols):
        return None
    try:
        out = np.asarray([[float(by_sha[s][c]) for c in cols] for s in ids], dtype=float)
    except Exception:
        return None
    if out.shape != (799, 4) or not np.isfinite(out).all():
        return None
    return out


def build_stage1_cache(cache_path: Path, jobs: int, rebuild: bool = False):
    ids, y, folds, prevalence, cooc, transitions, _ = load_source_arrays()
    if cache_path.is_file() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as z:
            cached_ids = np.asarray([norm(x) for x in z["sha256"]])
            cached_folds = z["folds"].astype(int)
            cached_y = z["labels"].astype(int)
            outer = z["outer_member_oof"].astype(float)
            nested = {f: z[f"nested_outer_{f}"] .astype(float) for f in range(1, 6)}
        if np.array_equal(cached_ids, ids) and np.array_equal(cached_folds, folds) and np.array_equal(cached_y, y):
            for f in range(1, 6):
                tr = folds != f
                if not np.isfinite(nested[f][tr]).all():
                    raise RuntimeError(f"BLOCKED_BY_CONTRACT: cached nested outer {f} incomplete")
            if not np.isfinite(outer).all():
                raise RuntimeError("BLOCKED_BY_CONTRACT: cached outer OOF incomplete")
            print(f"Loaded strict LiteLocal cache: {cache_path}", flush=True)
            return ids, y, folds, outer, nested

    upstream = load_module("litelocal_stage1_upstream_for_integration", LITE_UPSTREAM)
    outer = load_outer_member_oof_from_attribution(ids)
    if outer is not None:
        print("Reusing audited outer FUSED63 member OOF predictions", flush=True)
    else:
        outer = np.full((len(ids), 4), np.nan, dtype=float)
        print("Attribution OOF not found; rebuilding outer FUSED63 OOF", flush=True)
        for fold in range(1, 6):
            tr = np.flatnonzero(folds != fold)
            va = np.flatnonzero(folds == fold)
            seed = 20260925 + 100000 * fold
            pred, _, _, _ = upstream.committee_nested(
                "FUSED63", prevalence, cooc, transitions, y, tr, va, seed, jobs=jobs
            )
            outer[va] = pred
            print(f"Stage-I outer OOF fold {fold}/5 done", flush=True)

    nested: Dict[int, np.ndarray] = {}
    audit_rows = []
    for outer_fold in range(1, 6):
        arr = np.full((len(ids), 4), np.nan, dtype=float)
        for inner_fold in [f for f in range(1, 6) if f != outer_fold]:
            inner_train = np.flatnonzero((folds != outer_fold) & (folds != inner_fold))
            inner_val = np.flatnonzero(folds == inner_fold)
            # New strict nested seed contract for the integration experiment.
            # It is source-only and deterministic; it does not alter the
            # already-audited outer OOF Stage-I predictions.
            seed = 20260925 + 100000 * outer_fold + 1000 * inner_fold
            pred, _, _, _ = upstream.committee_nested(
                "FUSED63", prevalence, cooc, transitions, y,
                inner_train, inner_val, seed, jobs=jobs
            )
            arr[inner_val] = pred
            audit_rows.append({
                "outer_fold": outer_fold,
                "inner_validation_fold": inner_fold,
                "inner_train_n": int(len(inner_train)),
                "inner_validation_n": int(len(inner_val)),
                "seed": seed,
            })
            print(
                f"Stage-I strict nested outer={outer_fold} inner_val={inner_fold} done",
                flush=True,
            )
        nested[outer_fold] = arr
        trmask = folds != outer_fold
        if not np.isfinite(arr[trmask]).all():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: nested Stage-I incomplete for outer {outer_fold}")

    if not np.isfinite(outer).all():
        raise RuntimeError("BLOCKED_BY_CONTRACT: outer Stage-I OOF incomplete")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sha256=ids,
        labels=y,
        folds=folds,
        outer_member_oof=outer,
        **{f"nested_outer_{f}": nested[f] for f in range(1, 6)},
    )
    write_csv(cache_path.with_suffix(".audit.csv"), audit_rows)
    print(f"Saved strict LiteLocal cache: {cache_path}", flush=True)
    return ids, y, folds, outer, nested


def build_reference(out_dir: Path, ids, y, folds, outer_members, definition: str):
    p = stage1_probability(outer_members, definition)
    score = mean_opposition(outer_members, p)
    route = foldwise_top_budget(score, folds, ids, 0.25)
    rows = []
    for i, sid in enumerate(ids):
        rows.append({
            "sha256": sid,
            "label": int(y[i]),
            "fold": int(folds[i]),
            # Legacy names required by the established CARE reference reader.
            "Global24__prob": float(p[i]),
            "A25__score": float(score[i]),
            "A25__routed": int(route[i]),
            "Global24__A25__JointS7_ordinary__prob": float(p[i]),
            # Explicit semantic names for this experiment.
            "LiteLocal_FUSED63__prob": float(p[i]),
            "DI25_LiteLocal__score": float(score[i]),
            "DI25_LiteLocal__routed": int(route[i]),
        })
    ref_dir = out_dir / "preflight" / "di25_reference"
    write_csv(ref_dir / "source_predictions.csv", rows)
    err = (p >= 0.5) != y
    route_rows = []
    for fold in range(1, 6):
        idx = folds == fold
        route_rows.append({
            "fold": fold,
            "n": int(idx.sum()),
            "routed": int(np.sum(route & idx)),
            "stage1_errors": int(np.sum(err & idx)),
            "routed_stage1_errors": int(np.sum(err & route & idx)),
            "error_recall": float(np.sum(err & route & idx) / max(np.sum(err & idx), 1)),
            "routing_precision": float(np.sum(err & route & idx) / max(np.sum(route & idx), 1)),
        })
    write_csv(ref_dir / "route_audit.csv", route_rows)
    return ref_dir, p, score, route


def load_host_arm_module():
    if str(HOST_DIR) not in sys.path:
        sys.path.insert(0, str(HOST_DIR))
    return load_module("care_host_m0_existing_run_arm", HOST_ARM)


def patch_host_runner(host, ref_dir: Path, definition: str, ids, y, folds, outer_members, nested_members):
    if not hasattr(host, "v4"):
        raise RuntimeError("BLOCKED_BY_CONTRACT: existing run_arm.py exposes no v4 module")
    if not hasattr(host.v4, "fit_predict_variant"):
        raise RuntimeError("BLOCKED_BY_CONTRACT: v4.fit_predict_variant missing")
    if not hasattr(host, "build_di_reference"):
        names = [n for n in dir(host) if "reference" in n.casefold()]
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: run_arm.py no longer exposes build_di_reference; candidates="
            + repr(names)
        )

    # Force the existing run_arm to consume our LiteLocal DI25 reference.
    host.build_di_reference = lambda *args, **kwargs: ref_dir

    original_fit = host.v4.fit_predict_variant
    seen_folds = []

    def patched_fit(global_train, global_test, s7_train, s7_test,
                    raw_train_bags, raw_test_bags, base_train, base_test,
                    labels, variant, seeds, args, device):
        if not seeds:
            raise RuntimeError("BLOCKED_BY_CONTRACT: empty Stage-II seed list")
        outer_fold = int(int(seeds[0]) // 100000)
        if outer_fold not in range(1, 6):
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: cannot infer outer fold from seeds={seeds}")
        tr = folds != outer_fold
        va = folds == outer_fold
        strict_train = stage1_probability(nested_members[outer_fold][tr], definition)
        strict_test = stage1_probability(outer_members[va], definition)
        if len(strict_train) != len(base_train) or len(strict_test) != len(base_test):
            raise RuntimeError(
                "BLOCKED_BY_CONTRACT: Stage-I/Stage-II split length mismatch "
                f"outer={outer_fold} strict_train={len(strict_train)} base_train={len(base_train)} "
                f"strict_test={len(strict_test)} base_test={len(base_test)}"
            )
        if not np.array_equal(np.asarray(labels, dtype=int), y[tr]):
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: Stage-II label alignment mismatch outer={outer_fold}")
        seen_folds.append(outer_fold)
        return original_fit(
            global_train, global_test, s7_train, s7_test,
            raw_train_bags, raw_test_bags,
            strict_train, strict_test,
            labels, variant, seeds, args, device,
        )

    host.v4.fit_predict_variant = patched_fit
    return seen_folds


def detect_final_probability_column(columns):
    candidates = [
        c for c in columns
        if c.endswith("__Bounded__prob") and "expert" not in c.casefold()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: expected one bounded final probability column, got {candidates}")
    return candidates[0]


def postprocess(out_dir: Path, definition: str, ids, y, folds, base_p, score, route):
    pred_path = out_dir / "arms" / "FULL" / "source_predictions.csv"
    if not pred_path.is_file():
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing Stage-II predictions: {pred_path}")
    rows = read_csv(pred_path)
    by_sha = {norm(r["sha256"]): r for r in rows}
    if len(by_sha) != 799 or set(by_sha) != set(ids.tolist()):
        raise RuntimeError("BLOCKED_BY_CONTRACT: Stage-II prediction SHA mismatch")
    final_col = detect_final_probability_column(rows[0].keys())
    final_p = np.asarray([float(by_sha[s][final_col]) for s in ids], dtype=float)
    base_pred = base_p >= 0.5
    final_pred = final_p >= 0.5
    base_correct = base_pred == y
    final_correct = final_pred == y
    fixed = (~base_correct) & final_correct
    broken = base_correct & (~final_correct)

    base_m = metrics(y, base_p)
    final_m = metrics(y, final_p)
    summary = [{
        "stage1_definition": definition,
        "n": 799,
        "routed": int(route.sum()),
        "stage1_BA": base_m["BA"],
        "stage1_AUC": base_m["AUC"],
        "stage1_F1": base_m["F1"],
        "stage1_errors": base_m["errors"],
        "routed_stage1_errors": int(np.sum((~base_correct) & route)),
        "route_error_recall": float(np.sum((~base_correct) & route) / max(np.sum(~base_correct), 1)),
        "route_precision": float(np.sum((~base_correct) & route) / max(np.sum(route), 1)),
        "final_BA": final_m["BA"],
        "final_AUC": final_m["AUC"],
        "final_F1": final_m["F1"],
        "final_errors": final_m["errors"],
        "fixed": int(fixed.sum()),
        "broken": int(broken.sum()),
        "net": int(fixed.sum() - broken.sum()),
        "delta_BA": float(final_m["BA"] - base_m["BA"]),
        "final_probability_column": final_col,
    }]
    per_fold = []
    integrated = []
    for fold in range(1, 6):
        idx = folds == fold
        bm = metrics(y[idx], base_p[idx])
        fm = metrics(y[idx], final_p[idx])
        per_fold.append({
            "fold": fold,
            "n": int(idx.sum()),
            "routed": int(np.sum(route & idx)),
            "stage1_BA": bm["BA"],
            "stage1_AUC": bm["AUC"],
            "stage1_F1": bm["F1"],
            "stage1_errors": bm["errors"],
            "routed_stage1_errors": int(np.sum((~base_correct) & route & idx)),
            "final_BA": fm["BA"],
            "final_AUC": fm["AUC"],
            "final_F1": fm["F1"],
            "final_errors": fm["errors"],
            "fixed": int(np.sum(fixed & idx)),
            "broken": int(np.sum(broken & idx)),
            "net": int(np.sum(fixed & idx) - np.sum(broken & idx)),
        })
    for i, sid in enumerate(ids):
        integrated.append({
            "sha256": sid,
            "label": int(y[i]),
            "fold": int(folds[i]),
            "stage1_definition": definition,
            "LiteLocal_FUSED63__prob": float(base_p[i]),
            "DI25_LiteLocal__score": float(score[i]),
            "DI25_LiteLocal__routed": int(route[i]),
            "CARE_M0__prob": float(final_p[i]),
            "stage1_correct": int(base_correct[i]),
            "final_correct": int(final_correct[i]),
            "fixed": int(fixed[i]),
            "broken": int(broken[i]),
        })
    write_csv(out_dir / "integration_summary.csv", summary)
    write_csv(out_dir / "integration_per_fold.csv", per_fold)
    write_csv(out_dir / "integration_predictions.csv", integrated)

    s = summary[0]
    lines = [
        "CARE-Droid LiteLocal -> DI25 -> Host-M0 bounded correction (strict 2017 source)",
        "",
        f"Stage-I definition: {definition}",
        "Stage-I representation: LiteLocal FUSED63 = P31 + COOC16 + TRANS16.",
        "DI25: fold-wise Top25% MeanOpposition from the four Stage-I committee members.",
        "Stage-II: existing care-di25-host-input-replay-v1 FULL arm, unchanged.",
        "Stage-II correction base: strict nested LiteLocal inner-OOF probabilities.",
        "Stage-II Global24 is context used only by the routed expert; it is not the Stage-I classifier/router.",
        "No future data loaded.",
        "",
        f"Stage-I: BA={s['stage1_BA']:.6f} AUC={s['stage1_AUC']:.6f} F1={s['stage1_F1']:.6f} errors={s['stage1_errors']}",
        f"DI25: routed={s['routed']}/799 routed_errors={s['routed_stage1_errors']} error_recall={s['route_error_recall']:.6f} precision={s['route_precision']:.6f}",
        f"End-to-end: BA={s['final_BA']:.6f} AUC={s['final_AUC']:.6f} F1={s['final_F1']:.6f} errors={s['final_errors']}",
        f"Correction: fixed={s['fixed']} broken={s['broken']} net={s['net']} delta_BA={s['delta_BA']:+.6f}",
        "",
        "Interpretation boundary: this experiment evaluates source-side integration only. It does not justify future-model selection or a cost claim by itself.",
    ]
    (out_dir / "REPORT_LITELOCAL_INTEGRATION.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-definition", choices=DEFINITIONS, required=True)
    parser.add_argument("--seeds", default="0,1,2", help="Existing Stage-II logical seeds, e.g. 0 or 0,1,2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jobs", type=int, default=4, help="LiteLocal Stage-I committee jobs")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rebuild-stage1-cache", action="store_true")
    args = parser.parse_args()

    for path in (LITE_FEATURES, ROUTER, LITE_UPSTREAM, HOST_ARM):
        if not path.is_file():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing required input {path}")

    definition = args.stage1_definition
    root = args.output_root
    out_dir = root / definition
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = root / "stage1_cache" / "strict_fused63_members.npz"

    ids, y, folds, outer_members, nested_members = build_stage1_cache(
        cache_path, jobs=args.jobs, rebuild=args.rebuild_stage1_cache
    )
    ref_dir, base_p, score, route = build_reference(
        out_dir, ids, y, folds, outer_members, definition
    )
    if int(route.sum()) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: DI25 routed count {int(route.sum())} != 200")
    expected_by_fold = [int(np.sum(route & (folds == f))) for f in range(1, 6)]
    if expected_by_fold != [40, 40, 40, 40, 40]:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: DI25 per-fold budget changed: {expected_by_fold}")

    host = load_host_arm_module()
    seen_folds = patch_host_runner(
        host, ref_dir, definition, ids, y, folds, outer_members, nested_members
    )

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(HOST_ARM),
            "--arm", "FULL",
            "--seeds", args.seeds,
            "--device", args.device,
            "--output", str(out_dir),
        ]
        host.main()
    finally:
        sys.argv = old_argv

    if sorted(set(seen_folds)) != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: patched Stage-I was not consumed by every outer fold: {seen_folds}")

    postprocess(out_dir, definition, ids, y, folds, base_p, score, route)
    contract = {
        "experiment": "care-litelocal-di25-host-m0-source-v1",
        "source_only": True,
        "future_rows_loaded": 0,
        "stage1": "LiteLocal FUSED63",
        "stage1_definition": definition,
        "router": "DI25 fold-wise Top25% MeanOpposition",
        "routed_total": int(route.sum()),
        "stage2": "existing care-di25-host-input-replay-v1 FULL Host M0",
        "correction": "existing bounded FixKeep correction",
        "strict_nested_stage1_for_stage2_training": True,
        "legacy_reference_column_note": "A25__score/A25__routed are compatibility names containing LiteLocal DI25 values",
        "stage2_global24_note": "Global24 is retained only as routed Stage-II context, not as Stage-I classifier/router",
        "stage1_outer_seed_anchor": "20260925 + 100000 * outer_fold",
        "stage1_nested_seed_anchor": "20260925 + 100000 * outer_fold + 1000 * inner_validation_fold",
        "stage2_seeds": args.seeds,
    }
    (out_dir / "integration_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print("LITELOCAL_DI25_HOST_M0_SOURCE_DONE", flush=True)


if __name__ == "__main__":
    main()
