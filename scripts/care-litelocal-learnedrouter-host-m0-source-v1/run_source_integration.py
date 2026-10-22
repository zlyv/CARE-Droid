#!/usr/bin/env python3
"""Strict 2017 source-only integration:
LiteLocal FUSED63 Stage-I -> learned Stage-I error-risk router -> existing Host-M0 -> bounded correction.

Only the router is changed relative to the fixed DI25 integration.

Leakage contract
----------------
For outer fold k:
1. Stage-I outer-validation member probabilities are ordinary fixed outer OOF
   predictions: the Stage-I models were trained only on the other four folds.
2. Router training features/targets use strict inner-OOF Stage-I member
   probabilities inside those four folds.  The held-out outer fold is never
   used to fit the router or to choose its threshold.
3. The router threshold is learned from router OOF risks inside the outer
   training set by maximizing balanced accuracy for the target
   `Stage-I prediction is wrong`.  Ties prefer a lower route rate.
4. After threshold selection, the router is refit on the complete outer
   training set and applied once to the held-out outer fold.
5. Stage-II is the established care-di25-host-input-replay-v1 FULL Host-M0
   implementation, unchanged.  Its correction base uses strict nested
   LiteLocal probabilities exactly as in the fixed-DI25 integration.
6. No 2018--2021 row or label is loaded.

Compatibility note
------------------
The historical CARE reader requires columns named A25__score/A25__routed.
For this experiment those legacy slots contain the learned error-risk score
and learned route decision.  Explicit LearnedErrorRisk__* fields are written
as well.  This experiment is NOT a fixed 25% router.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path("/home/boyuanjie/projects/tasa-droid")
LITE_FEATURES = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
ROUTER = PROJECT / "artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv"
LITE_UPSTREAM = PROJECT / "scripts/care-litelocal-semantic-source-v1/run_source_oof.py"
ATTRIBUTION_OOF = PROJECT / "artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv"
HOST_ARM = PROJECT / "scripts/care-di25-host-input-replay-v1/run_arm.py"
HOST_DIR = HOST_ARM.parent
DEFAULT_OUT = PROJECT / "artifacts/care-litelocal-learnedrouter-host-m0-source-v1"
DEFAULT_CACHE = PROJECT / "artifacts/care-litelocal-di25-host-m0-source-v1/stage1_cache/strict_fused63_members.npz"
MEMBERS = ("LR", "RF", "XGB", "SVM")
DEFINITIONS = ("LR_ONLY", "MEAN_COMMITTEE", "MEDIAN_COMMITTEE")
ROUTER_FEATURE_NAMES = (
    "p_stage1",
    "p_LR",
    "p_RF",
    "p_XGB",
    "p_SVM",
    "abs_margin",
    "binary_entropy",
    "member_std",
    "member_range",
    "vote_disagreement",
    "mean_opposition",
    "max_member_deviation",
)


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
    """Fixed DI25 baseline only; NOT used by the learned router."""
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


def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0)


def router_features(member_probabilities: np.ndarray, stage1_p: np.ndarray) -> np.ndarray:
    """Label-free router descriptors available before Stage-II."""
    members = np.asarray(member_probabilities, dtype=float)
    p = np.asarray(stage1_p, dtype=float)
    if members.ndim != 2 or members.shape[1] != 4 or len(p) != len(members):
        raise ValueError("router feature shape mismatch")
    vote_rate = np.mean(members >= 0.5, axis=1)
    vote_disagreement = np.minimum(vote_rate, 1.0 - vote_rate) * 2.0
    opp = mean_opposition(members, p)
    X = np.column_stack([
        p,
        members[:, 0], members[:, 1], members[:, 2], members[:, 3],
        np.abs(p - 0.5),
        binary_entropy(p),
        np.std(members, axis=1),
        np.max(members, axis=1) - np.min(members, axis=1),
        vote_disagreement,
        opp,
        np.max(np.abs(members - p[:, None]), axis=1),
    ])
    if X.shape[1] != len(ROUTER_FEATURE_NAMES) or not np.isfinite(X).all():
        raise RuntimeError("BLOCKED_BY_CONTRACT: invalid learned-router descriptors")
    return X


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


def safe_auc(y, score):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


def error_route_stats(error_target, route):
    error_target = np.asarray(error_target, dtype=bool)
    route = np.asarray(route, dtype=bool)
    n = len(error_target)
    errors = int(error_target.sum())
    routed = int(route.sum())
    routed_errors = int(np.sum(error_target & route))
    unrouted = n - routed
    unrouted_errors = errors - routed_errors
    overall_rate = errors / max(n, 1)
    routed_rate = routed_errors / max(routed, 1)
    unrouted_rate = unrouted_errors / max(unrouted, 1)
    return {
        "n": n,
        "errors": errors,
        "routed": routed,
        "route_rate": routed / max(n, 1),
        "routed_errors": routed_errors,
        "error_recall": routed_errors / max(errors, 1),
        "routing_precision": routed_errors / max(routed, 1),
        "overall_error_rate": overall_rate,
        "routed_error_rate": routed_rate,
        "unrouted_error_rate": unrouted_rate,
        "enrichment_vs_overall": routed_rate / max(overall_rate, 1e-12),
        "risk_ratio_routed_vs_unrouted": routed_rate / max(unrouted_rate, 1e-12),
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
    return ids, y, folds, prevalence, cooc, transitions


def load_outer_member_oof_from_attribution(ids):
    if not ATTRIBUTION_OOF.is_file():
        return None
    rows = read_csv(ATTRIBUTION_OOF)
    by_sha = {norm(r["sha256"]): r for r in rows}
    cols = [f"p_FUSED63__{m}" for m in MEMBERS]
    if len(by_sha) != 799 or not rows or not all(c in rows[0] for c in cols):
        return None
    try:
        out = np.asarray([[float(by_sha[s][c]) for c in cols] for s in ids], dtype=float)
    except Exception:
        return None
    if out.shape != (799, 4) or not np.isfinite(out).all():
        return None
    return out


def build_stage1_cache(cache_path: Path, jobs: int, rebuild: bool = False):
    ids, y, folds, prevalence, cooc, transitions = load_source_arrays()
    if cache_path.is_file() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as z:
            cached_ids = np.asarray([norm(x) for x in z["sha256"]])
            cached_folds = z["folds"].astype(int)
            cached_y = z["labels"].astype(int)
            outer = z["outer_member_oof"].astype(float)
            nested = {f: z[f"nested_outer_{f}"].astype(float) for f in range(1, 6)}
        if np.array_equal(cached_ids, ids) and np.array_equal(cached_folds, folds) and np.array_equal(cached_y, y):
            for f in range(1, 6):
                tr = folds != f
                if not np.isfinite(nested[f][tr]).all():
                    raise RuntimeError(f"BLOCKED_BY_CONTRACT: cached nested outer {f} incomplete")
            if not np.isfinite(outer).all():
                raise RuntimeError("BLOCKED_BY_CONTRACT: cached outer OOF incomplete")
            print(f"Loaded strict LiteLocal cache: {cache_path}", flush=True)
            return ids, y, folds, outer, nested

    upstream = load_module("litelocal_stage1_upstream_for_learned_router", LITE_UPSTREAM)
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
            print(f"Stage-I strict nested outer={outer_fold} inner_val={inner_fold} done", flush=True)
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


def fit_router_model(X, error_target, seed):
    X = np.asarray(X, dtype=float)
    target = np.asarray(error_target, dtype=int)
    unique = np.unique(target)
    if len(unique) < 2:
        return {"kind": "constant", "risk": float(target.mean())}
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=4000,
            random_state=int(seed),
        )),
    ])
    pipe.fit(X, target)
    return {"kind": "logistic", "model": pipe}


def predict_router_risk(model, X):
    X = np.asarray(X, dtype=float)
    if model["kind"] == "constant":
        return np.full(len(X), float(model["risk"]), dtype=float)
    return model["model"].predict_proba(X)[:, 1].astype(float)


def choose_threshold(error_target, risk):
    """Learn threshold on training-only router OOF risks.

    Primary objective: maximum balanced accuracy for detecting Stage-I errors.
    Tie 1: lower route rate (lower expected Stage-II cost / fewer correct rows exposed).
    Tie 2: higher threshold (deterministic conservative tie break).
    """
    y = np.asarray(error_target, dtype=int)
    r = np.asarray(risk, dtype=float)
    if len(y) == 0 or not np.isfinite(r).all():
        raise ValueError("invalid threshold input")
    if len(np.unique(y)) < 2:
        # No learnable discrimination target. Route none if all correct; all if all errors.
        tau = float(np.nextafter(np.max(r), np.inf)) if int(y[0]) == 0 else float(np.min(r))
        pred = r >= tau
        return tau, {
            "balanced_accuracy": 1.0,
            "route_rate": float(pred.mean()),
            "error_recall": float(np.mean(pred[y == 1])) if np.any(y == 1) else 0.0,
            "routing_precision": float(np.mean(y[pred])) if np.any(pred) else 0.0,
        }

    unique = np.unique(r)
    candidates = np.concatenate([
        [np.nextafter(unique.max(), np.inf)],
        unique[::-1],
        [np.nextafter(unique.min(), -np.inf)],
    ])
    best = None
    for tau in candidates:
        pred = r >= tau
        ba = float(balanced_accuracy_score(y, pred))
        route_rate = float(pred.mean())
        recall = float(np.sum((y == 1) & pred) / max(np.sum(y == 1), 1))
        precision = float(np.sum((y == 1) & pred) / max(np.sum(pred), 1))
        key = (ba, -route_rate, float(tau))
        if best is None or key > best[0]:
            best = (key, float(tau), {
                "balanced_accuracy": ba,
                "route_rate": route_rate,
                "error_recall": recall,
                "routing_precision": precision,
            })
    return best[1], best[2]


def build_learned_routes(out_dir: Path, ids, y, folds, outer_members, nested_members, definition: str):
    """Strict outer-OOF learned error-risk routing."""
    base_p = stage1_probability(outer_members, definition)
    risk = np.full(len(ids), np.nan, dtype=float)
    threshold_by_row = np.full(len(ids), np.nan, dtype=float)
    route = np.zeros(len(ids), dtype=bool)
    threshold_rows = []

    for outer_fold in range(1, 6):
        tr = folds != outer_fold
        va = folds == outer_fold
        tr_idx = np.flatnonzero(tr)
        va_idx = np.flatnonzero(va)

        train_members = nested_members[outer_fold][tr]
        train_p = stage1_probability(train_members, definition)
        train_error = ((train_p >= 0.5) != y[tr]).astype(int)
        X_train = router_features(train_members, train_p)

        # Router OOF risks used ONLY for threshold selection inside outer training.
        router_oof = np.full(len(tr_idx), np.nan, dtype=float)
        train_groups = folds[tr]
        for inner_val_fold in sorted(set(train_groups.tolist())):
            fit_mask = train_groups != inner_val_fold
            val_mask = train_groups == inner_val_fold
            model = fit_router_model(
                X_train[fit_mask], train_error[fit_mask],
                seed=20260928 + 100000 * outer_fold + 1000 * inner_val_fold,
            )
            router_oof[val_mask] = predict_router_risk(model, X_train[val_mask])
        if not np.isfinite(router_oof).all():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: router OOF incomplete outer={outer_fold}")

        tau, tau_stats = choose_threshold(train_error, router_oof)

        final_router = fit_router_model(
            X_train, train_error, seed=20260928 + 100000 * outer_fold + 999,
        )
        val_members = outer_members[va]
        val_p = base_p[va]
        X_val = router_features(val_members, val_p)
        val_risk = predict_router_risk(final_router, X_val)
        val_route = val_risk >= tau

        risk[va] = val_risk
        threshold_by_row[va] = tau
        route[va] = val_route

        val_error = ((val_p >= 0.5) != y[va]).astype(int)
        val_stats = error_route_stats(val_error, val_route)
        threshold_rows.append({
            "outer_fold": outer_fold,
            "router_train_n": int(tr.sum()),
            "router_train_errors": int(train_error.sum()),
            "router_train_error_rate": float(train_error.mean()),
            "router_oof_auc_for_stage1_error": safe_auc(train_error, router_oof),
            "learned_threshold": float(tau),
            "threshold_objective": "max_router_OOF_balanced_accuracy_tie_lower_route_rate",
            "router_oof_BA_at_threshold": tau_stats["balanced_accuracy"],
            "router_oof_route_rate": tau_stats["route_rate"],
            "router_oof_error_recall": tau_stats["error_recall"],
            "router_oof_precision": tau_stats["routing_precision"],
            "outer_val_n": int(va.sum()),
            "outer_val_errors": int(val_error.sum()),
            "outer_val_router_auc_for_stage1_error": safe_auc(val_error, val_risk),
            "outer_val_routed": val_stats["routed"],
            "outer_val_route_rate": val_stats["route_rate"],
            "outer_val_routed_errors": val_stats["routed_errors"],
            "outer_val_error_recall": val_stats["error_recall"],
            "outer_val_precision": val_stats["routing_precision"],
            "outer_val_routed_error_rate": val_stats["routed_error_rate"],
            "outer_val_unrouted_error_rate": val_stats["unrouted_error_rate"],
        })
        print(
            f"Learned router outer={outer_fold}: tau={tau:.6f} "
            f"route={val_stats['routed']}/{int(va.sum())} "
            f"errors={val_stats['routed_errors']}/{val_stats['errors']} "
            f"recall={val_stats['error_recall']:.4f} precision={val_stats['routing_precision']:.4f}",
            flush=True,
        )

    if not np.isfinite(risk).all() or not np.isfinite(threshold_by_row).all():
        raise RuntimeError("BLOCKED_BY_CONTRACT: learned router outer OOF incomplete")

    # Fixed DI25 is retained only as a routing baseline on the exact same Stage-I OOF.
    fixed_score = mean_opposition(outer_members, base_p)
    fixed_route = foldwise_top_budget(fixed_score, folds, ids, 0.25)
    stage1_error = (base_p >= 0.5) != y
    learned_stats = error_route_stats(stage1_error, route)
    fixed_stats = error_route_stats(stage1_error, fixed_route)

    comparison = []
    for name, stats in (("LEARNED_ERROR_RISK", learned_stats), ("FIXED_DI25", fixed_stats)):
        comparison.append({"router": name, **stats})
    write_csv(out_dir / "router_thresholds_by_outer_fold.csv", threshold_rows)
    write_csv(out_dir / "router_comparison.csv", comparison)

    rows = []
    for i, sid in enumerate(ids):
        rows.append({
            "sha256": sid,
            "label": int(y[i]),
            "fold": int(folds[i]),
            # Historical compatibility columns consumed by CARE.
            "Global24__prob": float(base_p[i]),
            "A25__score": float(risk[i]),
            "A25__routed": int(route[i]),
            "Global24__A25__JointS7_ordinary__prob": float(base_p[i]),
            # Explicit current semantics.
            "LiteLocal_FUSED63__prob": float(base_p[i]),
            "LearnedErrorRisk__risk": float(risk[i]),
            "LearnedErrorRisk__threshold": float(threshold_by_row[i]),
            "LearnedErrorRisk__routed": int(route[i]),
            # Fixed-DI25 audit baseline only.
            "FixedDI25__score": float(fixed_score[i]),
            "FixedDI25__routed": int(fixed_route[i]),
        })
    ref_dir = out_dir / "preflight" / "di25_reference"
    write_csv(ref_dir / "source_predictions.csv", rows)

    per_fold = []
    for fold in range(1, 6):
        idx = folds == fold
        fold_stats = error_route_stats(stage1_error[idx], route[idx])
        per_fold.append({"fold": fold, **fold_stats})
    write_csv(ref_dir / "route_audit.csv", per_fold)
    return ref_dir, base_p, risk, threshold_by_row, route, fixed_score, fixed_route


def load_host_arm_module():
    if str(HOST_DIR) not in sys.path:
        sys.path.insert(0, str(HOST_DIR))
    return load_module("care_host_m0_existing_run_arm_learned_router", HOST_ARM)


def patch_host_runner(host, ref_dir: Path, definition: str, y, folds, outer_members, nested_members):
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
        # Critical contract check: Stage-II fitting still receives all outer-train rows.
        # Therefore the learned route changes application/gating only; it does not
        # silently alter the established Host-M0 fitting cohort.
        if len(labels) != int(tr.sum()) or not np.array_equal(np.asarray(labels, dtype=int), y[tr]):
            raise RuntimeError(
                f"BLOCKED_BY_CONTRACT: Stage-II fitting cohort changed or is route-prefiltered outer={outer_fold}"
            )
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


def postprocess(out_dir: Path, definition: str, ids, y, folds,
                base_p, risk, threshold_by_row, route, fixed_score, fixed_route):
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
    route_stats = error_route_stats(~base_correct, route)
    fixed_route_stats = error_route_stats(~base_correct, fixed_route)
    summary = [{
        "stage1_definition": definition,
        "n": 799,
        "router": "LEARNED_ERROR_RISK",
        "routed": int(route.sum()),
        "route_rate": float(route.mean()),
        "stage1_BA": base_m["BA"],
        "stage1_AUC": base_m["AUC"],
        "stage1_F1": base_m["F1"],
        "stage1_errors": base_m["errors"],
        "routed_stage1_errors": route_stats["routed_errors"],
        "route_error_recall": route_stats["error_recall"],
        "route_precision": route_stats["routing_precision"],
        "route_error_enrichment_vs_overall": route_stats["enrichment_vs_overall"],
        "route_error_risk_ratio_vs_unrouted": route_stats["risk_ratio_routed_vs_unrouted"],
        "fixed_DI25_routed": fixed_route_stats["routed"],
        "fixed_DI25_error_recall": fixed_route_stats["error_recall"],
        "fixed_DI25_precision": fixed_route_stats["routing_precision"],
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
        rs = error_route_stats((~base_correct)[idx], route[idx])
        per_fold.append({
            "fold": fold,
            "n": int(idx.sum()),
            "learned_threshold": float(np.unique(threshold_by_row[idx])[0]),
            "routed": rs["routed"],
            "route_rate": rs["route_rate"],
            "stage1_BA": bm["BA"],
            "stage1_AUC": bm["AUC"],
            "stage1_F1": bm["F1"],
            "stage1_errors": bm["errors"],
            "routed_stage1_errors": rs["routed_errors"],
            "route_error_recall": rs["error_recall"],
            "route_precision": rs["routing_precision"],
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
            "LearnedErrorRisk__risk": float(risk[i]),
            "LearnedErrorRisk__threshold": float(threshold_by_row[i]),
            "LearnedErrorRisk__routed": int(route[i]),
            "FixedDI25__score": float(fixed_score[i]),
            "FixedDI25__routed": int(fixed_route[i]),
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
        "CARE-Droid LiteLocal -> learned error-risk router -> Host-M0 bounded correction (strict 2017 source)",
        "",
        f"Stage-I definition: {definition}",
        "Stage-I representation: LiteLocal FUSED63 = P31 + COOC16 + TRANS16.",
        "Router: logistic Stage-I-error risk model using only Stage-I committee descriptors.",
        "Threshold: learned independently inside each outer-train split from router OOF risks; objective=max error-detection balanced accuracy; ties prefer lower route rate.",
        "Stage-II: existing care-di25-host-input-replay-v1 FULL arm, unchanged.",
        "Stage-II correction base: strict nested LiteLocal inner-OOF probabilities.",
        "No future data loaded.",
        "",
        f"Stage-I: BA={s['stage1_BA']:.6f} AUC={s['stage1_AUC']:.6f} F1={s['stage1_F1']:.6f} errors={s['stage1_errors']}",
        f"Learned router: routed={s['routed']}/799 ({100*s['route_rate']:.2f}%) routed_errors={s['routed_stage1_errors']} error_recall={s['route_error_recall']:.6f} precision={s['route_precision']:.6f}",
        f"Fixed DI25 routing baseline: routed={s['fixed_DI25_routed']}/799 error_recall={s['fixed_DI25_error_recall']:.6f} precision={s['fixed_DI25_precision']:.6f}",
        f"End-to-end: BA={s['final_BA']:.6f} AUC={s['final_AUC']:.6f} F1={s['final_F1']:.6f} errors={s['final_errors']}",
        f"Correction: fixed={s['fixed']} broken={s['broken']} net={s['net']} delta_BA={s['delta_BA']:+.6f}",
        "",
        "Interpretation boundary: source-only router experiment. It can compare adaptive routing against fixed DI25 on 2017 OOF, but it does not authorize future-year tuning or a runtime-cost claim.",
    ]
    (out_dir / "REPORT_LEARNED_ROUTER_INTEGRATION.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-definition", choices=DEFINITIONS, required=True)
    parser.add_argument("--seeds", default="0,1,2", help="Existing Stage-II logical seeds")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage1-cache", type=Path, default=DEFAULT_CACHE,
                        help="Reuse strict FUSED63 outer/nested member cache from fixed-DI25 experiment")
    parser.add_argument("--rebuild-stage1-cache", action="store_true")
    args = parser.parse_args()

    for path in (LITE_FEATURES, ROUTER, LITE_UPSTREAM, HOST_ARM):
        if not path.is_file():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing required input {path}")

    definition = args.stage1_definition
    out_dir = args.output_root / definition
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, y, folds, outer_members, nested_members = build_stage1_cache(
        args.stage1_cache, jobs=args.jobs, rebuild=args.rebuild_stage1_cache
    )
    ref_dir, base_p, risk, thresholds, route, fixed_score, fixed_route = build_learned_routes(
        out_dir, ids, y, folds, outer_members, nested_members, definition
    )

    # Sanity: the new router must genuinely be adaptive, not silently fixed at 25%.
    per_fold_counts = [int(np.sum(route & (folds == f))) for f in range(1, 6)]
    print(f"Learned router routed total={int(route.sum())}/799 per_fold={per_fold_counts}", flush=True)

    host = load_host_arm_module()
    seen_folds = patch_host_runner(
        host, ref_dir, definition, y, folds, outer_members, nested_members
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

    postprocess(out_dir, definition, ids, y, folds, base_p, risk, thresholds, route, fixed_score, fixed_route)
    contract = {
        "experiment": "care-litelocal-learnedrouter-host-m0-source-v1",
        "source_only": True,
        "future_rows_loaded": 0,
        "stage1": "LiteLocal FUSED63",
        "stage1_definition": definition,
        "router": "Logistic learned Stage-I error-risk router",
        "router_target": "1 iff strict Stage-I prediction is wrong",
        "router_features": list(ROUTER_FEATURE_NAMES),
        "threshold_selection": "router OOF within outer-train; maximize error-detection balanced accuracy; ties lower route rate then higher threshold",
        "router_outer_fold_routed_counts": per_fold_counts,
        "routed_total": int(route.sum()),
        "stage2": "existing care-di25-host-input-replay-v1 FULL Host M0 unchanged",
        "correction": "existing bounded FixKeep correction unchanged",
        "strict_nested_stage1_for_router_training": True,
        "strict_nested_stage1_for_stage2_training": True,
        "legacy_reference_column_note": "A25__score/A25__routed are compatibility slots containing learned error-risk values",
        "stage2_fitting_cohort_guard": "fit_predict_variant must receive all outer-train labels; abort if route-prefiltered",
        "stage1_cache": str(args.stage1_cache),
        "stage2_seeds": args.seeds,
    }
    (out_dir / "integration_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print("LITELOCAL_LEARNED_ROUTER_HOST_M0_SOURCE_DONE", flush=True)


if __name__ == "__main__":
    main()
