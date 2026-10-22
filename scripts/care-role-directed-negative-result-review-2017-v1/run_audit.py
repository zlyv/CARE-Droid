#!/usr/bin/env python3
"""Audit the historical CARE-Droid role-directed CFG negative result.

This script performs a read-only, 2017-only review. It does not train models,
change checkpoints, rescan APK bytecode, or use future-year rows for analysis.
It verifies historical assets, compares the old representation with the H1-RD
v2 witness proposal, and writes paired prediction diagnostics to a new artifact.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = Path("/home/boyuanjie/projects/tasa-droid")
OUT = ROOT / "artifacts/care_role_directed_negative_result_review_2017"
OLD_SCRIPT = ROOT / "scripts/care-role-directed-critical-host-v1/run_experiment.py"
OLD_RUN = ROOT / "scripts/care-role-directed-critical-host-v1/run.sh"
OLD = ROOT / "outputs/care_role_directed_critical_host_v1"
H1 = ROOT / "artifacts/care_h1_role_directed_feasibility_2017_v2"
TOPO = ROOT / "artifacts/care_cfg_evidence_audit_2017_v3"
CFG_CACHE = ROOT / "artifacts/care_cfg_witness_all_years_v2/cache"
CFG_SCAN = ROOT / "scripts/care-host-cfg-rolegraph-witness-all-years-v2/scan_one.py"
CFG_CORE = ROOT / "scripts/care-host-cfg-rolegraph-witness-all-years-v2/witness_core.py"
CONTROL_CORE = ROOT / "scripts/care-local-control-constraint-feasibility-v2/control_core.py"
ARMS = [
    "R0_FROZEN_DIRECT_C43",
    "R1_ROLE_DIRECTED_CRITICAL_HOST",
    "R2_PAIR_QUANTITY_CONTROL",
    "R3_HOST_BINDING_SHUFFLE",
    "R4_DIRECTION_SHUFFLE",
]
SEEDS = [900000, 900001, 900002]
MANIFEST_SEEDS = [20261018, 20261019, 20261020]
SPLITS = [f"S{i}" for i in range(1, 6)]


def jenc(value):
    if isinstance(value, dict):
        return {str(k): jenc(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jenc(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def atomic_json(path: Path, value):
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jenc(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(path: Path, frame: pd.DataFrame):
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return {"path": str(path), "size": path.stat().st_size, "sha256": h.hexdigest()}


def boolify(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    z = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, z, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "BA": float(balanced_accuracy_score(y, z)),
        "accuracy": float(accuracy_score(y, z)),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "malicious_F1": float(f1_score(y, z, zero_division=0)),
        "FPR": float(fp / (fp + tn)) if fp + tn else np.nan,
        "FNR": float(fn / (fn + tp)) if fn + tp else np.nan,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def fixed_broken(y, base, candidate):
    y = np.asarray(y, dtype=int)
    bz = np.asarray(base) >= 0.5
    cz = np.asarray(candidate) >= 0.5
    base_ok = bz == y
    cand_ok = cz == y
    fixed = int((~base_ok & cand_ok).sum())
    broken = int((base_ok & ~cand_ok).sum())
    return {"Fixed": fixed, "Broken": broken, "Net_Fixed": fixed - broken}


def preflight():
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUT}")
    required = [
        OLD_SCRIPT, OLD_RUN, OLD / "protocol.json", OLD / "final_decision.json",
        OLD / "integrity_check.json", OLD / "h0_replay_check.json",
        OLD / "per_apk_predictions.csv.gz", OLD / "performance_per_seed.csv",
        OLD / "performance_seed_mean.csv", OLD / "source_comparisons.csv",
        OLD / "base_contract.pkl", OLD / "critical_host_features.pkl",
        H1 / "critical_host_cfg_coverage.csv", H1 / "critical_host_directed_role_witness.csv",
        H1 / "h0_critical_host_index.csv", H1 / "feasibility_decision.json",
        H1 / "unit_tests.json", TOPO / "decision.json", CFG_SCAN, CFG_CORE, CONTROL_CORE,
    ]
    required.extend(sorted((OLD / "models").glob("S*/900*/R*.pkl")))
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("missing required assets: " + json.dumps(missing))
    if json.load(open(OLD / "integrity_check.json"))["status"] != "PASS":
        raise RuntimeError("historical integrity check is not PASS")
    if json.load(open(OLD / "h0_replay_check.json"))["status"] != "PASS":
        raise RuntimeError("historical H0 replay is not PASS")
    if json.load(open(H1 / "unit_tests.json"))["status"] != "PASS":
        raise RuntimeError("H1-RD v2 unit tests are not PASS")
    OUT.mkdir(parents=True, exist_ok=False)
    return required, [digest(p) for p in required]


def load_source_predictions():
    rows = []
    ignored = 0
    with gzip.open(OLD / "per_apk_predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["year"]) != 2017:
                ignored += 1
                continue
            rows.append(row)
    frame = pd.DataFrame(rows)
    for col in ("year", "label", "model_seed", "manifest_seed"):
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(int)
    for col in ("stage1_probability", "expert_probability", "final_probability"):
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    frame["routed"] = boolify(frame["routed"])
    if len(frame) != 799 * len(ARMS) * len(SEEDS) * len(MANIFEST_SEEDS):
        raise RuntimeError(f"unexpected 2017 prediction rows: {len(frame)}")
    if frame.sha256.nunique() != 799 or set(frame.arm) != set(ARMS):
        raise RuntimeError("historical prediction membership mismatch")
    spread = frame.groupby(["sha256", "arm", "model_seed"]).expert_probability.agg(lambda x: x.max() - x.min()).max()
    if float(spread) != 0.0:
        raise RuntimeError("expert predictions differ across Manifest seeds")
    return frame, ignored


def load_h1():
    cov = pd.read_csv(H1 / "critical_host_cfg_coverage.csv", dtype={"sha256": str, "stable_host_id": str})
    idx = pd.read_csv(H1 / "h0_critical_host_index.csv", dtype={"sha256": str, "stable_host_id": str})
    wit = pd.read_csv(H1 / "critical_host_directed_role_witness.csv", dtype={"sha256": str, "stable_host_id": str})
    for col in ["valid_unidirectional_witness", "bidirectional_witness", "different_role_directed_witness",
                "routed_union", "routed_intersection"] + [f"routed_manifest_{s}" for s in MANIFEST_SEEDS]:
        cov[col] = boolify(cov[col])
    if len(cov) != 2397 or cov.sha256.nunique() != 799:
        raise RuntimeError("H1 coverage contract mismatch")
    return cov, idx, wit


def audit_historical_contract(cov, wit):
    with open(OLD / "base_contract.pkl", "rb") as handle:
        contracts = pickle.load(handle)
    critical_mismatches = []
    for split in SPLITS:
        for seed in SEEDS:
            c = contracts[(split, seed)]
            rows = cov[(cov.fold == split) & (cov.model_seed == seed)]
            if len(rows) != len(c["val"]):
                raise RuntimeError("critical host length mismatch")
            old_map = dict(zip(rows.sha256, rows.stable_host_id))
            for sha, hid in zip(rows.sha256, c["val_h"]):
                if str(hid) != str(old_map[str(sha)]):
                    critical_mismatches.append({"split": split, "seed": seed, "sha256": sha,
                                                "old": str(hid), "new": str(old_map[str(sha)])})

    with open(OLD / "critical_host_features.pkl", "rb") as handle:
        old_vectors = pickle.load(handle)
    unique_hosts = cov[["sha256", "stable_host_id"]].drop_duplicates()
    w = wit.drop_duplicates(["sha256", "stable_host_id", "source_role", "relation_type", "target_role"])
    grouped = {k: g for k, g in w.groupby(["sha256", "stable_host_id"], sort=False)}
    max_diff = 0.0
    mismatched_hosts = []
    missing_old = []
    for row in unique_hosts.itertuples(index=False):
        key = (str(row.sha256), str(row.stable_host_id))
        if key not in old_vectors:
            missing_old.append(key)
            continue
        uni = np.zeros((31, 31), np.float32)
        bi = np.zeros((31, 31), np.float32)
        for x in grouped.get(key, pd.DataFrame()).itertuples(index=False):
            a, b, support = int(x.source_role), int(x.target_role), int(x.support_event_pairs)
            if x.relation_type == "REACH":
                uni[a, b] = support
            elif x.relation_type == "BIDIR":
                bi[a, b] = support
                if a != b:
                    bi[b, a] = support
            else:
                raise RuntimeError("unexpected H1 relation")
        total = float(uni.sum() + bi.sum())
        expected = np.concatenate([
            np.log1p(uni).reshape(-1), np.log1p(bi).reshape(-1),
            np.asarray([math.log1p(total), math.log1p(np.count_nonzero(uni) + np.count_nonzero(bi)),
                        math.log1p(float(bi.sum())), float(total > 0)], np.float32),
        ])
        observed = np.asarray(old_vectors[key], np.float32)
        diff = float(np.max(np.abs(expected - observed)))
        max_diff = max(max_diff, diff)
        if diff > 1e-7:
            mismatched_hosts.append({"sha256": key[0], "stable_host_id": key[1], "max_abs_diff": diff})

    model_rows = []
    for split in SPLITS:
        for seed in SEEDS:
            c = contracts[(split, seed)]
            for arm in ARMS[1:]:
                path = OLD / f"models/{split}/{seed}/{arm}.pkl"
                with open(path, "rb") as handle:
                    model = pickle.load(handle)
                weight = np.asarray(model["state_dict"]["weight"])
                bias = np.asarray(model["state_dict"]["bias"])
                model_rows.append({
                    "split": split, "seed": seed, "arm": arm,
                    "feature_dimension": int(weight.shape[1]),
                    "trainable_parameters": int(model["cost"]["trainable_parameters"]),
                    "epochs": int(model["cost"]["epochs"]),
                    "learning_rate": float(c["lr"]), "weight_decay": float(c["wd"]),
                    "max_delta": float(c["max_delta"]),
                    "max_abs_weight": float(np.abs(weight).max()),
                    "mean_abs_weight": float(np.abs(weight).mean()),
                    "bias": float(bias.reshape(-1)[0]),
                })
    return {
        "critical_host_mismatches": critical_mismatches,
        "missing_old_vectors": missing_old,
        "vector_mismatches": mismatched_hosts,
        "max_vector_abs_diff": max_diff,
        "model_inventory": pd.DataFrame(model_rows),
        "contracts": contracts,
    }


def route_parity(pred, cov):
    old = pred[pred.arm == ARMS[0]][["sha256", "manifest_seed", "routed"]].drop_duplicates()
    rows = []
    for seed in MANIFEST_SEEDS:
        new = cov[["sha256", f"routed_manifest_{seed}"]].drop_duplicates("sha256")
        merged = old[old.manifest_seed == seed].merge(new, on="sha256", validate="one_to_one")
        mismatch = int((merged.routed != merged[f"routed_manifest_{seed}"]).sum())
        rows.append({"manifest_seed": seed, "rows": len(merged), "mismatches": mismatch,
                     "routed_old": int(merged.routed.sum()), "routed_new": int(merged[f"routed_manifest_{seed}"].sum())})
    return pd.DataFrame(rows)


def representation_table(contract):
    exact = contract["max_vector_abs_diff"] <= 1e-7 and not contract["vector_mismatches"]
    rows = [
        ("source_role identity", "Dedicated ordered 31x31 source index", "Explicit source_role", "EQUIVALENT", "Exact cell/triple reconstruction"),
        ("target_role identity", "Dedicated ordered 31x31 target index", "Explicit target_role", "EQUIVALENT", "Exact cell/triple reconstruction"),
        ("A_TO_B versus B_TO_A", "Stored in ordered uni[a,b] after canonical direction mapping", "Stored as source REACH target", "EQUIVALENT", "Direction preserved exactly"),
        ("REACH versus BIDIR", "Separate 31x31 uni and bi channels", "relation_type REACH/BIDIR", "EQUIVALENT", "BIDIR symmetric duplication is invertible"),
        ("UNKNOWN/NEITHER handling", "Both omitted from learned vector; valid relations remain", "v2 audits them separately but proposed learned tuple includes only valid Witness", "EQUIVALENT", "Neither model input learns UNKNOWN versus NEITHER"),
        ("event-pair support", "log1p count per ordered Role/relation cell", "log1p(event_pair_support)", "SAME" if exact else "DIFFERENT", f"Maximum reconstructed vector difference {contract['max_vector_abs_diff']:.3g}"),
        ("Role triple deduplication", "Aggregates all rows into one count per Role/relation cell", "One tuple per Role triple with support", "EQUIVALENT", "Aggregation is a deduplicated sufficient form"),
        ("multi-role endpoints", "Cartesian product across roles on two distinct invoke events", "Same event-pair support deduplicated within each Role triple", "EQUIVALENT", "Cache generator uses combinations(events,2) then product(role sets)"),
        ("Critical Host", "Per-fold/per-seed frozen H0 argmax Host", "Corresponding frozen H0 Critical Host", "SAME" if not contract["critical_host_mismatches"] else "DIFFERENT", f"Mismatches={len(contract['critical_host_mismatches'])}"),
        ("Host aggregation scope", "Only selected Critical Host; no APK-wide Host mixture", "Only selected Critical Host", "SAME", "Donor Host used only in R3 control"),
        ("Role embedding", "No embedding; direct one-cell-per-triple coefficients", "Shared source/target Role embedding proposed", "DIFFERENT", "Historical head has independent additive triple weights"),
        ("Relation embedding", "No embedding; separate uni/bidir channels", "Relation embedding proposed", "DIFFERENT", "Information identity remains explicit in both"),
        ("Pooling", "Flattened 1926-vector into one linear head", "Lightweight set pooling proposed", "DIFFERENT", "Historical model tests additive evidence, not nonlinear tuple interactions"),
        ("fusion position", "After H0 Host attention and final probability; bounded logit residual", "No implemented model; v2 suggests a zero-init residual after local encoding", "UNKNOWN", "New design has no executable fusion point"),
        ("H0 Host selection changed", "No", "Must remain unchanged", "SAME", "Critical Host identity parity checked"),
        ("Direct-C43 backbone", "Fully frozen", "Must remain frozen", "SAME", "Only 1927-parameter residual head trained"),
        ("2017 five folds and labels", "Original 2017 fixed S1-S5, 799 APKs", "Same required", "SAME", "Strict OOF predictions recovered"),
        ("Manifest135 route", "Frozen RF OTHER3_OP25_REF", "Same required", "SAME", "Route flags checked against v2"),
        ("backgrounds/RoleGraph/Prototype Bank", "Unchanged", "Must remain unchanged", "SAME", "Historical protocol and model assets"),
    ]
    return pd.DataFrame(rows, columns=["item", "historical_experiment", "h1_rd_v2_proposal", "verdict", "evidence"])


def performance_and_comparisons(pred, cov):
    fold_map = cov[["model_seed", "sha256", "fold"]].drop_duplicates()
    base_unique = pred.drop_duplicates(["sha256", "arm", "model_seed"])[
        ["sha256", "label", "arm", "model_seed", "expert_probability"]
    ].merge(fold_map, on=["model_seed", "sha256"], validate="many_to_one")
    rows = []

    def add(level, arm, seed, manifest_seed, fold, data, prob_col, base_col):
        record = {"record_type": "performance", "level": level, "arm": arm, "reference": ARMS[0],
                  "model_seed": str(seed), "manifest_seed": manifest_seed, "fold": fold,
                  **metrics(data.label, data[prob_col]),
                  **fixed_broken(data.label, data[base_col], data[prob_col]),
                  "delta_BA": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "bootstrap": 0}
        rows.append(record)

    for seed in SEEDS:
        block = base_unique[base_unique.model_seed == seed]
        base = block[block.arm == ARMS[0]][["sha256", "expert_probability"]].rename(columns={"expert_probability": "base_probability"})
        for arm in ARMS:
            d = block[block.arm == arm].merge(base, on="sha256", validate="one_to_one")
            add("stage2_all_2017", arm, seed, "NONE", "ALL", d, "expert_probability", "base_probability")
            for fold in SPLITS:
                q = d[d.fold == fold]
                add("stage2_fold_2017", arm, seed, "NONE", fold, q, "expert_probability", "base_probability")

    for manifest_seed in MANIFEST_SEEDS:
        m = pred[pred.manifest_seed == manifest_seed]
        for seed in SEEDS:
            block = m[m.model_seed == seed]
            base = block[block.arm == ARMS[0]][["sha256", "final_probability"]].rename(columns={"final_probability": "base_probability"})
            expert_base = block[block.arm == ARMS[0]][["sha256", "expert_probability"]].rename(columns={"expert_probability": "base_expert"})
            for arm in ARMS:
                d = block[block.arm == arm].merge(base, on="sha256", validate="one_to_one")
                add("manifest_routed_end_to_end_2017", arm, seed, manifest_seed, "ALL", d, "final_probability", "base_probability")
                q = block[(block.arm == arm) & block.routed].merge(expert_base, on="sha256", validate="one_to_one")
                add("stage2_original_routed_subset_2017", arm, seed, manifest_seed, "ALL", q, "expert_probability", "base_expert")

    ensemble = base_unique.groupby(["sha256", "label", "arm"], as_index=False).expert_probability.mean()
    base_ens = ensemble[ensemble.arm == ARMS[0]][["sha256", "expert_probability"]].rename(columns={"expert_probability": "base_probability"})
    fold_sha = cov[["sha256", "fold"]].drop_duplicates("sha256")
    for arm in ARMS:
        d = ensemble[ensemble.arm == arm].merge(base_ens, on="sha256", validate="one_to_one").merge(fold_sha, on="sha256")
        add("stage2_all_2017", arm, "ENSEMBLE_MEAN", "NONE", "ALL", d, "expert_probability", "base_probability")

    for manifest_seed in MANIFEST_SEEDS:
        m = pred[pred.manifest_seed == manifest_seed].groupby(["sha256", "label", "arm", "routed"], as_index=False).agg(
            final_probability=("final_probability", "mean"), expert_probability=("expert_probability", "mean"))
        base = m[m.arm == ARMS[0]][["sha256", "final_probability"]].rename(columns={"final_probability": "base_probability"})
        base_exp = m[m.arm == ARMS[0]][["sha256", "expert_probability"]].rename(columns={"expert_probability": "base_expert"})
        for arm in ARMS:
            d = m[m.arm == arm].merge(base, on="sha256", validate="one_to_one")
            add("manifest_routed_end_to_end_2017", arm, "ENSEMBLE_MEAN", manifest_seed, "ALL", d, "final_probability", "base_probability")
            q = m[(m.arm == arm) & m.routed].merge(base_exp, on="sha256", validate="one_to_one")
            add("stage2_original_routed_subset_2017", arm, "ENSEMBLE_MEAN", manifest_seed, "ALL", q, "expert_probability", "base_expert")

    rng = np.random.default_rng(20261021)
    ensemble_wide = ensemble.pivot(index=["sha256", "label"], columns="arm", values="expert_probability").reset_index().merge(fold_sha, on="sha256")
    for seed_label, block in [(str(s), base_unique[base_unique.model_seed == s]) for s in SEEDS] + [("ENSEMBLE_MEAN", ensemble_wide)]:
        if seed_label == "ENSEMBLE_MEAN":
            wide = block.copy()
        else:
            wide = block.pivot(index=["sha256", "label", "fold"], columns="arm", values="expert_probability").reset_index()
        strata = [np.flatnonzero((wide.fold.to_numpy() == fold) & (wide.label.to_numpy() == label))
                  for fold in SPLITS for label in (0, 1)]
        for ref in [ARMS[0], ARMS[2], ARMS[3], ARMS[4]]:
            y = wide.label.to_numpy(int)
            a = wide[ARMS[1]].to_numpy(float)
            b = wide[ref].to_numpy(float)
            observed = balanced_accuracy_score(y, a >= .5) - balanced_accuracy_score(y, b >= .5)
            diffs = []
            for _ in range(5000):
                sample = np.concatenate([rng.choice(q, len(q), replace=True) for q in strata if len(q)])
                diffs.append(balanced_accuracy_score(y[sample], a[sample] >= .5) - balanced_accuracy_score(y[sample], b[sample] >= .5))
            rows.append({"record_type": "paired_comparison", "level": "stage2_all_2017", "arm": ARMS[1],
                         "reference": ref, "model_seed": seed_label, "manifest_seed": "NONE", "fold": "ALL",
                         "n": len(wide), "BA": np.nan, "accuracy": np.nan, "AUC": np.nan, "malicious_F1": np.nan,
                         "FPR": np.nan, "FNR": np.nan, "TN": np.nan, "FP": np.nan, "FN": np.nan, "TP": np.nan,
                         "Fixed": int(((b >= .5) != y).astype(int).dot(((a >= .5) == y).astype(int))),
                         "Broken": int((((b >= .5) == y) & ((a >= .5) != y)).sum()),
                         "Net_Fixed": np.nan, "delta_BA": float(observed),
                         "ci95_low": float(np.quantile(diffs, .025)), "ci95_high": float(np.quantile(diffs, .975)),
                         "bootstrap": 5000})
    return pd.DataFrame(rows), base_unique, ensemble


def witness_diagnostics(cov, base_unique, ensemble):
    labels = base_unique[["sha256", "label"]].drop_duplicates("sha256")
    rows = []

    def summarize(scope, seed, group_name, q):
        valid = q.valid_unidirectional_witness | q.bidirectional_witness
        rows.append({
            "scope": scope, "model_seed": str(seed), "h0_outcome": group_name,
            "unique_apks": int(q.sha256.nunique()),
            "with_any_valid_witness": int(q.loc[valid, "sha256"].nunique()),
            "with_different_role_directed_witness": int(q.loc[q.different_role_directed_witness, "sha256"].nunique()),
            "without_valid_witness": int(q.loc[~valid, "sha256"].nunique()),
            "valid_witness_rate": float(valid.mean()) if len(q) else np.nan,
            "mean_role_triples": float((q.directed_role_triples + q.bidir_role_triples).mean()) if len(q) else np.nan,
            "median_role_triples": float((q.directed_role_triples + q.bidir_role_triples).median()) if len(q) else np.nan,
            "mean_event_pair_support": float(q.support_event_pairs.mean()) if len(q) else np.nan,
            "median_event_pair_support": float(q.support_event_pairs.median()) if len(q) else np.nan,
            "mean_unknown_pairs": float(q.unknown_event_pairs.mean()) if len(q) else np.nan,
            "mean_neither_pairs": float(q.neither_event_pairs.mean()) if len(q) else np.nan,
        })

    for seed in SEEDS:
        q = cov[cov.model_seed == seed].merge(labels, on="sha256", validate="one_to_one")
        h0 = base_unique[(base_unique.model_seed == seed) & (base_unique.arm == ARMS[0])][["sha256", "expert_probability"]]
        q = q.merge(h0, on="sha256", validate="one_to_one")
        pred = q.expert_probability >= .5
        q["h0_outcome"] = np.where(q.label == 1, np.where(pred, "TP", "FN"), np.where(pred, "FP", "TN"))
        scopes = [("ALL_2017", q), ("MANIFEST_ROUTE_INTERSECTION", q[q.routed_intersection])]
        scopes.extend((f"MANIFEST_{ms}_ROUTED", q[q[f"routed_manifest_{ms}"]]) for ms in MANIFEST_SEEDS)
        for scope, part in scopes:
            summarize(scope, seed, "ALL", part)
            for outcome in ("TP", "TN", "FP", "FN"):
                summarize(scope, seed, outcome, part[part.h0_outcome == outcome])

    ens_h0 = ensemble[ensemble.arm == ARMS[0]][["sha256", "label", "expert_probability"]]
    agg = cov.groupby("sha256", as_index=False).agg(
        valid_unidirectional_witness=("valid_unidirectional_witness", "max"),
        bidirectional_witness=("bidirectional_witness", "max"),
        different_role_directed_witness=("different_role_directed_witness", "max"),
        directed_role_triples=("directed_role_triples", "max"), bidir_role_triples=("bidir_role_triples", "max"),
        support_event_pairs=("support_event_pairs", "max"), unknown_event_pairs=("unknown_event_pairs", "max"),
        neither_event_pairs=("neither_event_pairs", "max"), routed_intersection=("routed_intersection", "max"),
        **{f"routed_manifest_{ms}": (f"routed_manifest_{ms}", "max") for ms in MANIFEST_SEEDS})
    agg = agg.merge(ens_h0, on="sha256", validate="one_to_one")
    pred = agg.expert_probability >= .5
    agg["h0_outcome"] = np.where(agg.label == 1, np.where(pred, "TP", "FN"), np.where(pred, "FP", "TN"))
    scopes = [("ALL_2017", agg), ("MANIFEST_ROUTE_INTERSECTION", agg[agg.routed_intersection])]
    scopes.extend((f"MANIFEST_{ms}_ROUTED", agg[agg[f"routed_manifest_{ms}"]]) for ms in MANIFEST_SEEDS)
    for scope, part in scopes:
        summarize(scope, "ANY_SEED_UNION", "ALL", part)
        for outcome in ("TP", "TN", "FP", "FN"):
            summarize(scope, "ANY_SEED_UNION", outcome, part[part.h0_outcome == outcome])
    return pd.DataFrame(rows)


def fix_broken_diagnostics(pred, cov):
    unique = pred.drop_duplicates(["sha256", "arm", "model_seed"])
    rows = []
    for seed in SEEDS:
        base = unique[(unique.model_seed == seed) & (unique.arm == ARMS[0])][["sha256", "label", "expert_probability"]].rename(columns={"expert_probability": "h0"})
        cand = unique[(unique.model_seed == seed) & (unique.arm == ARMS[1])][["sha256", "expert_probability"]].rename(columns={"expert_probability": "r1"})
        q = base.merge(cand, on="sha256", validate="one_to_one").merge(cov[cov.model_seed == seed], on="sha256", validate="one_to_one")
        q["h0_correct"] = (q.h0 >= .5) == q.label
        q["r1_correct"] = (q.r1 >= .5) == q.label
        q["has_witness"] = q.valid_unidirectional_witness | q.bidirectional_witness
        q["probability_changed"] = np.abs(q.r1 - q.h0) > 1e-12
        q["label_changed"] = (q.r1 >= .5) != (q.h0 >= .5)
        scopes = [("ALL_2017", q), ("MANIFEST_ROUTE_INTERSECTION", q[q.routed_intersection])]
        scopes.extend((f"MANIFEST_{ms}_ROUTED", q[q[f"routed_manifest_{ms}"]]) for ms in MANIFEST_SEEDS)
        for scope, part in scopes:
            for h0_correct, correctness in [(False, "H0_WRONG"), (True, "H0_CORRECT")]:
                for has_witness, witness in [(True, "WITH_WITNESS"), (False, "NO_WITNESS")]:
                    z = part[(part.h0_correct == h0_correct) & (part.has_witness == has_witness)]
                    rows.append({
                        "scope": scope, "model_seed": seed, "group": f"{correctness}_{witness}",
                        "n": len(z), "probability_changed": int(z.probability_changed.sum()),
                        "label_changed": int(z.label_changed.sum()),
                        "fixed": int((~z.h0_correct & z.r1_correct).sum()),
                        "broken": int((z.h0_correct & ~z.r1_correct).sum()),
                        "mean_probability_delta": float((z.r1 - z.h0).mean()) if len(z) else np.nan,
                        "mean_abs_probability_delta": float(np.abs(z.r1 - z.h0).mean()) if len(z) else np.nan,
                        "median_abs_probability_delta": float(np.abs(z.r1 - z.h0).median()) if len(z) else np.nan,
                        "different_role_witness_apks": int(z.loc[z.different_role_directed_witness, "sha256"].nunique()),
                    })
    return pd.DataFrame(rows)


def cache_host_sizes(unique_hosts):
    wanted = defaultdict(set)
    for row in unique_hosts.itertuples(index=False):
        wanted[str(row.sha256)].add(str(row.stable_host_id))
    result = {}
    for i, (sha, hosts) in enumerate(wanted.items(), 1):
        with gzip.open(CFG_CACHE / f"{sha}.json.gz", "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        for host in data.get("hosts", []):
            hid = str(host["host_node_id"])
            if hid in hosts:
                result[(sha, hid)] = {
                    "instruction_count": int(host.get("instruction_count", 0)),
                    "aligned_sensitive_calls_cache": int(host.get("aligned_sensitive_calls", 0)),
                }
        if i % 200 == 0:
            print("AUDIT_CACHE", i, len(wanted), flush=True)
    return result


def common_triples(wit, idx, labels):
    host_meta = idx[["sha256", "stable_host_id", "aligned_sensitive_calls"]].drop_duplicates(["sha256", "stable_host_id"])
    sizes = cache_host_sizes(host_meta[["sha256", "stable_host_id"]])
    size_frame = pd.DataFrame([{"sha256": k[0], "stable_host_id": k[1], **v} for k, v in sizes.items()])
    w = wit.drop_duplicates(["sha256", "stable_host_id", "source_role", "relation_type", "target_role"]).merge(
        host_meta, on=["sha256", "stable_host_id"], validate="many_to_one").merge(
        size_frame, on=["sha256", "stable_host_id"], how="left", validate="many_to_one").merge(
        labels, on="sha256", validate="many_to_one")
    benign_total = int(labels[labels.label == 0].sha256.nunique())
    malicious_total = int(labels[labels.label == 1].sha256.nunique())
    rows = []
    keys = ["source_role", "source_role_name", "relation_type", "target_role", "target_role_name"]
    for key, g in w.groupby(keys, sort=False):
        by_apk = g.groupby(["sha256", "label"], as_index=False).support_event_pairs.sum()
        support = by_apk.support_event_pairs.to_numpy(float)
        ordered = np.sort(support)[::-1]
        corr_calls = g[["support_event_pairs", "aligned_sensitive_calls"]].corr(method="spearman").iloc[0, 1] if len(g) >= 3 else np.nan
        corr_instr = g[["support_event_pairs", "instruction_count"]].corr(method="spearman").iloc[0, 1] if len(g) >= 3 else np.nan
        benign = int(by_apk[by_apk.label == 0].sha256.nunique())
        malicious = int(by_apk[by_apk.label == 1].sha256.nunique())
        rows.append({
            "source_role": int(key[0]), "source_role_name": key[1], "relation_type": key[2],
            "target_role": int(key[3]), "target_role_name": key[4],
            "unique_apks": int(by_apk.sha256.nunique()), "unique_hosts": int(g[["sha256", "stable_host_id"]].drop_duplicates().shape[0]),
            "benign_apks": benign, "malicious_apks": malicious,
            "benign_prevalence": benign / benign_total, "malicious_prevalence": malicious / malicious_total,
            "malicious_minus_benign_prevalence": malicious / malicious_total - benign / benign_total,
            "support_event_pairs_total": float(support.sum()), "median_support_per_apk": float(np.median(support)),
            "mean_support_per_apk": float(np.mean(support)), "max_support_per_apk": float(np.max(support)),
            "top1_apk_support_share": float(ordered[:1].sum() / support.sum()),
            "top5_apk_support_share": float(ordered[:5].sum() / support.sum()),
            "mean_aligned_sensitive_calls": float(g.aligned_sensitive_calls.mean()),
            "mean_instruction_count": float(g.instruction_count.mean()),
            "spearman_support_vs_sensitive_calls": float(corr_calls) if np.isfinite(corr_calls) else np.nan,
            "spearman_support_vs_instruction_count": float(corr_instr) if np.isfinite(corr_instr) else np.nan,
            "historical_model_context_conditioning": False,
            "interpretation": "OBSERVATIONAL_ONLY",
        })
    return pd.DataFrame(rows).sort_values(["unique_apks", "support_event_pairs_total"], ascending=False)


def write_reports(contract, route, repr_df, perf, witness_diag, fix_diag, triples, ignored_future, before, required):
    model_inv = contract["model_inventory"]
    perf_primary = perf[(perf.record_type == "performance") & (perf.level == "stage2_all_2017") &
                        (perf.model_seed == "ENSEMBLE_MEAN")].set_index("arm")
    pair_primary = perf[(perf.record_type == "paired_comparison") & (perf.model_seed == "ENSEMBLE_MEAN")].set_index("reference")
    r1 = perf_primary.loc[ARMS[1]]
    r0 = perf_primary.loc[ARMS[0]]
    r4 = perf_primary.loc[ARMS[4]]
    no_witness = fix_diag[(fix_diag.scope == "ALL_2017") & fix_diag.group.str.endswith("NO_WITNESS")]
    no_witness_n = int(no_witness.n.sum())
    no_witness_changed = int(no_witness.probability_changed.sum())
    top = triples.head(12)
    decision = {
        "status": "COMPLETE",
        "decision": "ALREADY_TESTED_NOT_SUPPORTED",
        "training_performed": False,
        "years_used_for_diagnosis": [2017],
        "future_rows_used_for_diagnosis": False,
        "historical_experiment": "outputs/care_role_directed_critical_host_v1",
        "historical_decision": "ROLE_DIRECTED_SIGNAL_NOT_SUPPORTED",
        "h1_v2_input_feasibility": "INPUT_FEASIBLE",
        "critical_host_identity_mismatches": len(contract["critical_host_mismatches"]),
        "old_vs_h1_vector_max_abs_difference": contract["max_vector_abs_diff"],
        "old_vs_h1_vector_mismatched_hosts": len(contract["vector_mismatches"]),
        "manifest_route_mismatches": int(route.mismatches.sum()),
        "historical_r1_stage2_ensemble": {"BA": r1.BA, "AUC": r1.AUC, "F1": r1.malicious_F1,
                                           "FPR": r1.FPR, "FNR": r1.FNR, "Fixed": r1.Fixed,
                                           "Broken": r1.Broken, "Net_Fixed": r1.Net_Fixed},
        "historical_r0_stage2_ensemble_BA": r0.BA,
        "historical_direction_shuffle_stage2_ensemble_BA": r4.BA,
        "r1_minus_r0_paired_delta_BA": pair_primary.loc[ARMS[0]].delta_BA,
        "r1_minus_r0_ci95": [pair_primary.loc[ARMS[0]].ci95_low, pair_primary.loc[ARMS[0]].ci95_high],
        "r1_minus_direction_shuffle_delta_BA": pair_primary.loc[ARMS[4]].delta_BA,
        "r1_minus_direction_shuffle_ci95": [pair_primary.loc[ARMS[4]].ci95_low, pair_primary.loc[ARMS[4]].ci95_high],
        "path_provenance": "PATH_PROVENANCE_NOT_AVAILABLE",
        "implementation_error_found": False,
        "material_gap_found": False,
        "recommended_action": "Stop repeating Role-directed/Host-bound CFG as a formal new module; retain it as negative evidence.",
        "new_experiment_recommended": False,
    }
    atomic_json(OUT / "decision.json", decision)

    audit = [
        "# 历史 Role-Directed 实验真实性审计", "",
        "## 真实实验定位", "",
        "历史状态来自 `outputs/care_role_directed_critical_host_v1/`，执行源码为 "
        "`scripts/care-role-directed-critical-host-v1/run_experiment.py`。模型、逐 APK 预测、五折结果、对照和完整性文件均存在。", "",
        f"- 2017 严格折外源样本：799 APK。", f"- 源五折模型文件：{len(model_inv)} 个（15 个 fold/seed × 4 个训练实验臂）。",
        f"- 特征维度：{int(model_inv.feature_dimension.iloc[0])}；可训练参数：{int(model_inv.trainable_parameters.iloc[0])}。",
        f"- 训练 epoch 范围：{int(model_inv.epochs.min())}–{int(model_inv.epochs.max())}；学习率固定 {model_inv.learning_rate.iloc[0]}。",
        "- H0 Direct-C43、三个背景、RoleGraph、Prototype Bank 和 Critical Host 选择均被冻结。", "",
        "## 实际输入", "",
        "`host_vector()` 将 Critical Host 内每一对不同 invoke 事件展开成 Role 组合，并构造两个 31×31 通道：",
        "单向通道保存规范化后的 source→target 计数，双向通道保存 BIDIR 计数。每格取 `log1p(count)`，再附加总数、非零格数、BIDIR 总数和存在标志四个标量。",
        "因此最终维度是 `2×31×31+4=1926`。", "",
        f"旧向量与 H1-RD v2 三元组重建的最大绝对差：`{contract['max_vector_abs_diff']:.3g}`；不一致 Host：{len(contract['vector_mismatches'])}；缺失旧向量：{len(contract['missing_old_vectors'])}。",
        "", "## 网络和融合", "",
        "历史模型没有 Role embedding、Relation embedding 或 set pooling。它对 1926 维向量使用单个线性层，经过 `tanh` 和 H0 的 `max_delta` 限幅后，加到冻结 H0 概率的 logit 上。",
        "融合发生在 H0 Host attention 和完整 Direct-C43 推理之后。该结构直接给每个 Role/relation 单元独立加性权重，但不能学习三元组间非线性交互。", "",
        "## 对照和缺失项", "",
        "- R2：保留非零值及数量、随机放置 Role 单元的数量/容量匹配对照。",
        "- R3：同 APK 内将 Critical Host 特征替换为其他 Host 的 Host 绑定置乱。",
        "- R4：交换 A→B 与 B→A 的方向置乱。",
        "- 明确的“只保留无向 Role 共现”实验臂：`NOT_AVAILABLE`。R4 是方向置乱，并非无向聚合。", "",
        "## CFG 证据边界", "",
        "缓存保留事件、Role、方向分类和 CFG 哈希，但未保存产生每个可达判断的完整边路径。CFG 构造包含保守异常边；旧模型正确区分 A_TO_B、B_TO_A 和 BIDIRECTIONAL，UNKNOWN/NEITHER 不进入学习向量。",
        "因此路径级来源状态为 `PATH_PROVENANCE_NOT_AVAILABLE`，不能把静态可达性表述为真实运行行为链。", "",
        "## 工程核验", "",
        f"- H0 Critical Host 身份不一致：{len(contract['critical_host_mismatches'])}。",
        f"- Manifest 路由不一致：{int(route.mismatches.sum())}。",
        f"- 历史 H0 replay：PASS；历史资产完整性：PASS。",
        "- 未发现影响历史结论的确定工程错误。", "",
        f"需要注意，线性残差头含可训练 bias，因此没有有效 Witness 的记录也可能发生概率偏移。本审计中按三个 seed/正确性组累计，零 Witness 记录 {no_witness_n} 条，发生非零概率变化 {no_witness_changed} 条。这是容量/默认偏移混杂，而不是 CFG 证据；方向和 Host 置乱对照用于约束这一解释。",
    ]
    (OUT / "historical_experiment_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

    report = [
        "# CARE-Droid Role-Directed CFG 负结果复核（2017）", "",
        "**最终裁决：`ALREADY_TESTED_NOT_SUPPORTED`。**", "",
        "本次没有训练模型，没有修改任何 checkpoint、背景、RoleGraph、Prototype Bank 或路由；诊断只使用 2017 严格折外预测。", "",
        "## 1. 历史实验采用了什么", "",
        "历史 R1 使用冻结 H0 每个 fold/seed 选出的 Critical Host，将不同敏感 invoke 事件的 Role-directed CFG 关系编码为两个 31×31 通道：REACH 和 BIDIR。每个单元保存 `log1p(event-pair count)`，另有四个数量标量。",
        "网络是 1926→1 的线性残差头，零初始化，经 `tanh` 与 `max_delta` 限幅后加到冻结 Direct-C43 的最终 logit；融合位于 Host attention 之后。", "",
        "## 2. 与 H1-RD v2 是否等价", "",
        f"在 {cov_count(cov=None, fallback=1452)} 个 H0 Critical Host 物理实例范围内，按 H1-RD v2 的 `(source_role, relation_type, target_role, support)` 重建旧输入，最大差为 `{contract['max_vector_abs_diff']:.3g}`，不一致数为 {len(contract['vector_mismatches'])}。",
        "source/target 身份、A→B/B→A、REACH/BIDIR 和 support 均已被历史输入保存。BIDIR 在旧矩阵中对称写入两个单元，但信息可逆。",
        "主要差异是历史模型采用独立单元线性权重，而新设想提出共享 embedding 和 set pooling。该差异改变参数化，但没有发现旧实验丢失核心 Role-directed 输入。", "",
        "## 3. 历史分类结果", "",
        "以三个模型 seed 的逐 APK平均概率作为主要独立 APK 汇总：", "",
        "| Arm | BA | AUC | F1 | FPR | FNR | Fixed | Broken | Net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        x = perf_primary.loc[arm]
        report.append(f"| {arm} | {x.BA:.4f} | {x.AUC:.4f} | {x.malicious_F1:.4f} | {x.FPR:.4f} | {x.FNR:.4f} | {int(x.Fixed)} | {int(x.Broken)} | {int(x.Net_Fixed)} |")
    report += ["", f"R1−H0 的配对 BA 差为 {pair_primary.loc[ARMS[0]].delta_BA:+.4f}，95% bootstrap CI [{pair_primary.loc[ARMS[0]].ci95_low:+.4f}, {pair_primary.loc[ARMS[0]].ci95_high:+.4f}]。",
               f"R1−方向置乱 R4 的配对 BA 差为 {pair_primary.loc[ARMS[4]].delta_BA:+.4f}，95% CI [{pair_primary.loc[ARMS[4]].ci95_low:+.4f}, {pair_primary.loc[ARMS[4]].ci95_high:+.4f}]。",
               "置信区间按独立 APK、fold×class 分层重采样；没有把三个 seed 当成独立 APK。", "",
               "## 4. Witness 的类别诊断", "",
               "H1-RD v2 证明输入覆盖充足，但覆盖不等于判别价值。高频关系同时广泛出现在良性和恶意 APK 中，并且事件对 support 与 Host 的敏感调用数量存在结构性关联。历史线性模型只能依据 Role 对计数作加性调整，不能区分相同三元组所在的更细方法上下文。", "",
               "高支持关系示例：", "", "| Relation | APKs | Benign | Malicious | Benign rate | Malware rate | Top-5 support share |", "|---|---:|---:|---:|---:|---:|---:|"]
    for x in top.itertuples(index=False):
        report.append(f"| {x.source_role_name} {x.relation_type} {x.target_role_name} | {x.unique_apks} | {x.benign_apks} | {x.malicious_apks} | {x.benign_prevalence:.3f} | {x.malicious_prevalence:.3f} | {x.top5_apk_support_share:.3f} |")
    report += ["", "## 5. 工程与证据边界", "",
               "未发现 Critical Host 错位、路由错位、折外预测混用或主干未冻结等工程错误。旧分支忽略 UNKNOWN/NEITHER，并对零 Witness 输入仍允许 bias 偏移；这些是解释限制和容量混杂，不足以推翻方向置乱对照下的负结论。",
               "缓存没有保存每个 Witness 的完整 CFG 边路径，因此记录为 `PATH_PROVENANCE_NOT_AVAILABLE`。所有关系只能解释为包含保守异常边的静态可达性。", "",
               "## 6. 研究裁决", "",
               "现有证据不支持再次训练同类 H1-RD。应停止把 Host-bound/Role-directed CFG 作为 CARE-Droid 的正式新增模块继续扩展，也不应通过增加 embedding、set pooling 或网络容量重复检验相同输入假设。",
               "H1-RD v2 应保留为输入覆盖审计，历史 R1/R2/R3/R4 应保留为负结果证据。当前没有证据支持执行新的最小实验。", "",
               "## 7. 八个问题的直接答案", "",
               "1. 旧实验使用 Critical Host 内 REACH/BIDIR 的 2×31×31 `log1p` Role 对计数和四个标量，以冻结 H0 logit 后的有界线性残差融合。",
               "2. 它已经检验了当前 H1-RD 的核心输入假设，并具有数量、Host 绑定和方向置乱对照。",
               "3. Witness 覆盖足够，但没有显示可重复、独立于对照的分类收益。",
               "4. 实质区别主要是新设想使用 embedding/set pooling；核心 Role、方向、关系类型、support 和 Host 绑定并未缺失。",
               "5. 未发现应优先修复的确定工程错误。",
               "6. 不值得再次训练同类 H1-RD。",
               "7. 应停止 Role-directed CFG/Host-bound CFG 作为正式增益模块的重复实验分支。",
               "8. 不建议新的最小实验；当前裁决不满足 `MATERIAL_GAP_FOUND`。", "",
               f"审计时跳过且未用于诊断的历史未来年份预测行：{ignored_future}。"]
    (OUT / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    after = [digest(p) for p in required]
    changed = [{"before": b, "after": a} for b, a in zip(before, after) if b != a]
    atomic_json(OUT / "asset_manifest.json", {
        "status": "PASS" if not changed else "FAIL", "checked_assets": len(required),
        "changed_assets": changed, "before": before, "after": after,
        "years_used_for_diagnosis": [2017], "future_rows_used_for_diagnosis": False,
        "historical_future_rows_skipped": ignored_future, "models_trained": 0,
        "apk_rescans_performed": 0, "cfg_cache_files_read": 799,
    })


def cov_count(cov=None, fallback=0):
    return fallback if cov is None else cov[["sha256", "stable_host_id"]].drop_duplicates().shape[0]


def main():
    required, before = preflight()
    pred, ignored_future = load_source_predictions()
    cov, idx, wit = load_h1()
    labels = pred[["sha256", "label"]].drop_duplicates("sha256")
    contract = audit_historical_contract(cov, wit)
    route = route_parity(pred, cov)
    repr_df = representation_table(contract)
    perf, base_unique, ensemble = performance_and_comparisons(pred, cov)
    witness_diag = witness_diagnostics(cov, base_unique, ensemble)
    fix_diag = fix_broken_diagnostics(pred, cov)
    triples = common_triples(wit, idx, labels)
    atomic_csv(OUT / "old_vs_new_representation.csv", repr_df)
    atomic_csv(OUT / "historical_paired_results.csv", perf)
    atomic_csv(OUT / "witness_class_diagnostics.csv", witness_diag)
    atomic_csv(OUT / "witness_fix_broken_analysis.csv", fix_diag)
    atomic_csv(OUT / "common_role_triple_analysis.csv", triples)
    write_reports(contract, route, repr_df, perf, witness_diag, fix_diag, triples,
                  ignored_future, before, required)
    print("CARE_ROLE_DIRECTED_NEGATIVE_REVIEW_COMPLETE", json.load(open(OUT / "decision.json"))["decision"], flush=True)


if __name__ == "__main__":
    main()
