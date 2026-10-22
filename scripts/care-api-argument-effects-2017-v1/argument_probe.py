"""Fixed CPU readout probe conditional on the frozen W0/router, source 2017 only."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from argument_core import CHANNELS, EFFECTS, COARSE
from argument_io import (MODEL_SEEDS, MANIFEST_SEEDS, SPLITS, read_json, write_json,
                         write_csv, source_rows, fold_representations, signature, status)

ARMS = ("CODE", "COARSE", "VALUE", "BOUND", "PLACEBO") + tuple(
    kind + str(i) for kind in ("EFFECT_SHUFFLE_", "BINDING_SHUFFLE_") for i in range(5))
BASE_ARMS = ("W0",) + ARMS
FEATURE_NAMES = (
    ["log_events", "log_resolved", "log_classified", "log_command_hosts",
     "log_ambiguous_hosts", "log_unrecovered_targets", "has_resolved", "unknown_event_fraction"]
    + ["coarse:" + k for k in COARSE] + ["channel:" + k for k in CHANNELS]
    + ["effect:" + k for k in EFFECTS]
    + ["binding:" + a + ":" + b for a in CHANNELS for b in EFFECTS]
)
COVER_END = 8
COARSE_END = COVER_END + len(COARSE)
EFFECT_START = COARSE_END + len(CHANNELS)
EFFECT_END = EFFECT_START + len(EFFECTS)
WIDTH = len(FEATURE_NAMES)


def feature_vector(record):
    s = record["summary"]
    hosts = record.get("hosts", [])
    v = [math.log1p(s["events"]), math.log1p(s["resolved"]), math.log1p(s["classified"]),
         math.log1p(len(hosts)), math.log1p(sum(h["status"] == "AMBIGUOUS_HOST_ID" for h in hosts)),
         math.log1p(sum(h.get("unrecovered_targets", h["expected_targets"]) for h in hosts)),
         float(s["resolved"] > 0), (s["events"] - s["resolved"]) / max(s["events"], 1)]
    for key, names in (("coarse", COARSE), ("channels", CHANNELS), ("effects", EFFECTS),
                       ("binding", tuple(a + ":" + b for a in CHANNELS for b in EFFECTS))):
        v.extend(math.log1p(s[key].get(name, 0)) for name in names)
    return np.asarray(v, dtype=np.float64)


def row_shuffle(values, context, ids, seed):
    """APK-dependent derangement, separately within train and target cohorts.

    Preserves the cohort's exact donor-row multiset, with approximate count/
    resolution/channel matching. This is not a global token permutation.
    No labels or correctness groups enter strata or donor selection.
    """
    groups = defaultdict(list)
    for i, row in enumerate(context):
        if row[0] == 0:
            continue
        dominant = int(np.argmax(row[COARSE_END:EFFECT_START]))
        key = (min(3, int(row[0] / math.log(2))), int(row[6] > .5), dominant)
        groups[key].append(i)
    donors = np.arange(len(ids))
    for key, indices in groups.items():
        order = sorted(indices, key=lambda i: hashlib.sha256(f"{seed}:{ids[i]}:{key}".encode()).digest())
        if len(order) > 1:
            for receiver, donor in zip(order, order[1:] + order[:1]):
                donors[receiver] = donor
    shuffled = np.asarray(values)[donors].copy()
    active = context[:, 0] > 0
    changed = np.any(shuffled != values, axis=1)
    return shuffled, {"active_apks": int(active.sum()), "changed_apks": int((active & changed).sum()),
                      "changed_fraction": float((active & changed).sum() / max(active.sum(), 1)),
                      "donor_indices": donors.tolist()}


def arm_features(arm, train, target, train_ids, target_ids, seed):
    a, b = train.copy(), target.copy()
    meta = {}
    if arm == "CODE":
        a[:] = 0; b[:] = 0
    elif arm == "COARSE":
        a[:, COARSE_END:] = 0; b[:, COARSE_END:] = 0
    elif arm == "VALUE":
        a[:, EFFECT_END:] = 0; b[:, EFFECT_END:] = 0
    elif arm == "BOUND":
        pass
    elif arm.startswith("EFFECT_SHUFFLE_"):
        repeat = int(arm.rsplit("_", 1)[1])
        a[:, EFFECT_START:EFFECT_END], ma = row_shuffle(train[:, EFFECT_START:EFFECT_END], train, train_ids, seed + repeat)
        b[:, EFFECT_START:EFFECT_END], mb = row_shuffle(target[:, EFFECT_START:EFFECT_END], target, target_ids, seed + repeat + 10000)
        a[:, EFFECT_END:] = 0; b[:, EFFECT_END:] = 0
        meta.update(train_null=ma, target_null=mb)
    elif arm.startswith("BINDING_SHUFFLE_"):
        repeat = int(arm.rsplit("_", 1)[1])
        a[:, EFFECT_END:], ma = row_shuffle(train[:, EFFECT_END:], train, train_ids, seed + repeat)
        b[:, EFFECT_END:], mb = row_shuffle(target[:, EFFECT_END:], target, target_ids, seed + repeat + 10000)
        meta.update(train_null=ma, target_null=mb)
    elif arm == "PLACEBO":
        mean, std = train[:, EFFECT_END:].mean(0), train[:, EFFECT_END:].std(0)
        for values, ids in ((a, train_ids), (b, target_ids)):
            for i, sid in enumerate(ids):
                rng = np.random.RandomState(int.from_bytes(hashlib.sha256(f"{seed}:{sid}:placebo".encode()).digest()[:4], "big"))
                values[i, EFFECT_END:] = (mean + std * rng.normal(size=len(mean))) if values[i, 0] > 0 else 0
        meta["placebo_moments"] = "outer-train only; inactive rows remain zero"
    else:
        raise ValueError(arm)
    return a, b, meta


def fit_offset(train, labels, offset, target, target_offset, ridge=.1, maximum_iterations=500):
    mean, std = train.mean(0), train.std(0)
    std[std < 1e-8] = 1
    x = np.column_stack([np.ones(len(train)), (train - mean) / std])
    z = np.column_stack([np.ones(len(target)), (target - mean) / std])
    labels, offset = np.asarray(labels, float), np.asarray(offset, float)
    def objective(w):
        logits = offset + x @ w
        loss = np.mean(np.logaddexp(0, logits) - labels * logits) + .5 * ridge * np.dot(w, w)
        grad = x.T @ (expit(logits) - labels) / len(labels) + ridge * w
        return loss, grad
    result = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True,
                      options={"maxiter": maximum_iterations, "ftol": 1e-12, "gtol": 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("offset readout did not converge: " + str(result.message))
    prediction = expit(np.asarray(target_offset) + z @ result.x)
    return prediction, {"weights": result.x.tolist(), "mean": mean.tolist(), "std": std.tolist(),
                        "iterations": int(result.nit), "loss": float(result.fun), "ridge": ridge}


def metrics(y, probabilities, stage1=None, route=None):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y, int), np.asarray(probabilities, float)
    if len(y) == 0 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("invalid evaluation labels")
    pred = p >= .5
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    ba = .5 * (tn / max(tn + fp, 1) + tp / max(tp + fn, 1))
    result = {"n": len(y), "BA": ba, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
              "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}
    if stage1 is not None:
        before, after = (np.asarray(stage1) >= .5) == y, pred == y
        fixed, broken = int(np.sum(~before & after)), int(np.sum(before & ~after))
        result.update(Fixed=fixed, Broken=broken, Net_Fixed=fixed - broken,
                      routed=int(np.sum(route)), routed_RF_errors=int(np.sum(np.asarray(route) & ~before)))
    return result


def paired_test(y, candidate, reference, seed, repeats=2000):
    """Class-stratified paired bootstrap; APK-clustered sign-flip comparison."""
    y = np.asarray(y, int)
    d = (((candidate >= .5) == y).astype(float) - ((reference >= .5) == y).astype(float)).mean(0)
    classes = [np.flatnonzero(y == k) for k in (0, 1)]
    if any(len(v) == 0 for v in classes):
        raise ValueError("two classes required for paired BA comparison")
    observed = .5 * sum(float(d[v].mean()) for v in classes)
    rng, draws = np.random.RandomState(seed), []
    for _ in range(repeats):
        draws.append(.5 * sum(float(d[rng.choice(v, len(v), replace=True)].mean()) for v in classes))
    weights = np.where(y == 0, .5 / len(classes[0]), .5 / len(classes[1]))
    null = [float(np.sum(rng.choice([-1, 1], len(y)) * d * weights)) for _ in range(repeats)]
    p = float((1 + np.sum(np.asarray(null) >= observed - 1e-15)) / (repeats + 1)) if observed > 0 else 1.
    return {"BA_difference": observed, "ci95_low": float(np.quantile(draws, .025)),
            "ci95_high": float(np.quantile(draws, .975)), "p_one_sided": p,
            "resamples": repeats, "unit": "APK, Manifest-seed rows correlated; no lineage/refit uncertainty"}


def adjust_holm(rows):
    running = 0.
    for rank, i in enumerate(sorted(range(len(rows)), key=lambda i: rows[i]["p_one_sided"])):
        running = max(running, min(1., (len(rows) - rank) * rows[i]["p_one_sided"]))
        rows[i]["p_holm"] = running


def run_probe(reference, out, config):
    labels, routes = source_rows(reference)
    ids = sorted(labels); index = {sid: i for i, sid in enumerate(ids)}
    representations, folds = fold_representations(reference, ids)
    x = np.stack([feature_vector(read_json(out / f"cache/{sid}.json.gz")) for sid in ids])
    y = np.asarray([labels[sid] for sid in ids])
    probabilities = {arm: np.full((3, 799), np.nan) for arm in BASE_ARMS}
    audits = []
    for split_no, split in enumerate(SPLITS, 1):
        for si, seed in enumerate(MODEL_SEEDS):
            rep = representations[(split, seed)]
            tr_ids, te_ids = list(map(str, rep["train_ids"])), list(map(str, rep["target_ids"]))
            tr, te = np.asarray([index[s] for s in tr_ids]), np.asarray([index[s] for s in te_ids])
            probabilities["W0"][si, te] = 1 / (1 + np.exp(-rep["logit_target"]))
            for arm in ARMS:
                atr, ate, meta = arm_features(arm, x[tr], x[te], tr_ids, te_ids, 20261022 + 100 * split_no)
                train = np.column_stack([rep["code_train"], atr])
                target = np.column_stack([rep["code_target"], ate])
                fingerprint = signature({"contract": config["contract_sha256"], "arm": arm, "split": split,
                                         "seed": seed, "features": hashlib.sha256(train.tobytes() + target.tobytes()).hexdigest()})
                path = out / f"models/{split}/{seed}/{arm}.json"
                if path.exists():
                    fitted = read_json(path)
                    if fitted["fingerprint"] != fingerprint or fitted["target_ids"] != te_ids:
                        raise ValueError("stale readout model")
                    prediction = np.asarray(fitted["probability"])
                else:
                    prediction, state = fit_offset(train, y[tr], rep["logit_train"], target, rep["logit_target"])
                    fitted = {"fingerprint": fingerprint, "target_ids": te_ids,
                              "probability": prediction.tolist(), "state": state, "feature_audit": meta}
                    write_json(path, fitted)
                if prediction.shape != (len(te),) or not np.isfinite(prediction).all():
                    raise ValueError("invalid readout prediction")
                probabilities[arm][si, te] = prediction
                audits.append({"split": split, "seed": seed, "arm": arm,
                               "train_null_fraction": meta.get("train_null", {}).get("changed_fraction"),
                               "target_null_fraction": meta.get("target_null", {}).get("changed_fraction")})
            print("SOURCE_READOUT_DONE", split, seed, flush=True)
            status(out, "RUNNING", "source_readout", split=split, W0_seed=seed)
    if any(not np.isfinite(v).all() for v in probabilities.values()):
        raise ValueError("incomplete source OOF")
    rows, per_apk, finals, per_seed = [], [], {}, []
    for arm in BASE_ARMS:
        expert = probabilities[arm].mean(0)
        fs = []
        for ms in MANIFEST_SEEDS:
            base = np.asarray([float(routes[ms][sid]["stage1_probability"]) for sid in ids])
            route = np.asarray([str(routes[ms][sid]["routed"]).lower() in {"1", "true"} for sid in ids])
            final = np.where(route, expert, base)
            fs.append(final)
            rows.append({"arm": arm, "manifest_seed": ms, "scope": "end_to_end", **metrics(y, final, base, route)})
            for split in SPLITS:
                take = np.asarray([folds[sid] == split for sid in ids])
                rows.append({"arm": arm, "manifest_seed": ms, "scope": split, **metrics(y[take], final[take], base[take], route[take])})
            for si, seed in enumerate(MODEL_SEEDS):
                pred = np.where(route, probabilities[arm][si], base)
                per_seed.append({"arm": arm, "manifest_seed": ms, "W0_seed": seed, **metrics(y, pred, base, route)})
            for i, sid in enumerate(ids):
                per_apk.append({"sha256": sid, "label": int(y[i]), "fold": folds[sid], "arm": arm,
                                "manifest_seed": ms, "routed": int(route[i]), "stage1_probability": float(base[i]),
                                "expert_probability": float(expert[i]), "final_probability": float(final[i])})
        finals[arm] = np.asarray(fs)
    b0_ba = float(np.mean([r["BA"] for r in rows if r["arm"] == "W0" and r["scope"] == "end_to_end"]))
    if abs(b0_ba - .9194653299916458) > 1e-8:
        raise RuntimeError("frozen cascade baseline replay failed")
    contrasts = [("VALUE", ref) for ref in ("W0", "CODE", "COARSE") + tuple("EFFECT_SHUFFLE_" + str(i) for i in range(5))]
    contrasts += [("BOUND", ref) for ref in ("W0", "VALUE", "PLACEBO") + tuple("BINDING_SHUFFLE_" + str(i) for i in range(5))]
    tests = [{"candidate": a, "reference": b, **paired_test(y, finals[a], finals[b], 20261022 + i)}
             for i, (a, b) in enumerate(contrasts)]
    adjust_holm(tests)
    def supported(arm):
        relevant = [r for r in tests if r["candidate"] == arm]
        return bool(relevant) and all(r["BA_difference"] > 0 and r["ci95_low"] > 0 and r["p_holm"] < .05 for r in relevant)
    # Nulls must actually perturb the evidence in every train/target fold.
    value_mobile = all(r["train_null_fraction"] >= .10 and r["target_null_fraction"] >= .10
                       for r in audits if r["arm"].startswith("EFFECT_SHUFFLE_"))
    binding_mobile = all(r["train_null_fraction"] >= .10 and r["target_null_fraction"] >= .10
                         for r in audits if r["arm"].startswith("BINDING_SHUFFLE_"))
    def stable(arm, ref):
        fold_positive = sum(np.mean([r["BA"] for r in rows if r["scope"] == split and r["arm"] == arm])
                            > np.mean([r["BA"] for r in rows if r["scope"] == split and r["arm"] == ref]) for split in SPLITS)
        seed_positive = sum(np.mean([r["BA"] for r in per_seed if r["W0_seed"] == seed and r["arm"] == arm])
                            > np.mean([r["BA"] for r in per_seed if r["W0_seed"] == seed and r["arm"] == ref]) for seed in MODEL_SEEDS)
        return bool(fold_positive >= 4 and seed_positive >= 2)
    def net_better(arm, ref):
        get = lambda a: np.mean([r["Net_Fixed"] for r in rows if r["scope"] == "end_to_end" and r["arm"] == a])
        return bool(get(arm) > get(ref))
    value = value_mobile and supported("VALUE") and stable("VALUE", "CODE") and stable("VALUE", "COARSE") and net_better("VALUE", "W0")
    binding = value and binding_mobile and supported("BOUND") and stable("BOUND", "VALUE") and net_better("BOUND", "W0")
    decision = {"status": "BOUND_ARGUMENT_EFFECTS_SUPPORTED_SOURCE" if binding else
                "VALUE_EFFECTS_SUPPORTED_SOURCE" if value else
                "INCONCLUSIVE_NULL_MOBILITY" if not value_mobile else "NO_SUPPORTED_GAIN_SOURCE",
                "value_supported": value, "binding_supported": binding,
                "value_null_mobile": value_mobile, "binding_null_mobile": binding_mobile,
                "binding_status": "SUPPORTED" if binding else
                    "INCONCLUSIVE_NULL_MOBILITY" if not binding_mobile else "NOT_SUPPORTED",
                "source_years": [2017], "future_read": False,
                "claim": config["probe"]["claim"], "baseline_BA": b0_ba}
    write_csv(out / "reports/source_metrics.csv", rows)
    write_csv(out / "reports/W0_seed_metrics.csv", per_seed)
    write_csv(out / "reports/source_predictions.csv", per_apk)
    write_csv(out / "reports/comparisons.csv", tests)
    write_csv(out / "reports/null_mobility.csv", audits)
    write_json(out / "reports/final_decision.json", decision, replace=True)
    write_json(out / "reports/feature_schema.json", {"names": FEATURE_NAMES, "width": WIDTH}, replace=True)
    return decision
