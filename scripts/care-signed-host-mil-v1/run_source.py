#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (ARMS, BATCH_SIZE, EPOCHS, MAX_DELTA, NULL_REPETITIONS, OUTPUT,
                    REAL_SEEDS, SEED, binary_auc, correction_metrics, dump_json,
                    finite, load_source, logit, sigmoid, spearman, write_csv)
from models import fit_model, make_model, predict_model, standardize


def host_count_bin(bag):
    return min(8, int(math.log2(max(1, len(bag)))))


def count_features(bags):
    result = []
    for bag in bags:
        role = np.asarray(bag[:, :31], float)
        result.append([math.log1p(len(bag)), float(np.sum(np.any(role > 0, axis=0))),
                       float(np.mean(np.sum(role > 0, axis=1))), math.log1p(float(np.count_nonzero(bag)))])
    return np.asarray(result, float)


def checkpoint_path(tag, arm, fold, seed):
    safe = tag.replace("/", "_")
    return OUTPUT / "checkpoints" / f"{safe}.{arm}.fold{fold}.seed{seed}.pt"


def train_or_load(arm, train_bags, train_y, test_bags, fold, seed, tag, device):
    normalized_train, normalized_test, mean, scale = standardize(train_bags, test_bags)
    path = checkpoint_path(tag, arm, fold, seed); path.parent.mkdir(parents=True, exist_ok=True)
    model = make_model(arm).to(device)
    if path.is_file():
        saved = torch.load(path, map_location=device); model.load_state_dict(saved["state_dict"])
    else:
        model = fit_model(arm, normalized_train, train_y, seed, device, EPOCHS, BATCH_SIZE)
        temporary = path.with_suffix(".tmp")
        torch.save({"state_dict": model.state_dict(), "mean": mean, "scale": scale,
                    "arm": arm, "fold": fold, "seed": seed, "epochs": EPOCHS}, temporary)
        temporary.replace(path)
    return model, normalized_test, predict_model(model, normalized_test, device)


def fit_real_arm(arm, bags, y, fold, route, device):
    fields = {key: np.zeros(len(y), float) for key in ("logit", "malware", "benign")}
    seed_values = {key: [] for key in fields}; deletion_rows = []
    for current in range(1, 6):
        train = np.flatnonzero(fold != current); test = np.flatnonzero(fold == current)
        per_seed = {key: [] for key in fields}; per_seed_deletion = []
        for seed in REAL_SEEDS:
            model, normalized_test, prediction = train_or_load(
                arm, [bags[i] for i in train], y[train], [bags[i] for i in test], current, seed, "real", device)
            for key in fields: per_seed[key].append(prediction[key])
            if arm == "M3_SIGNED_CLAM":
                selected = [local for local, global_index in enumerate(test) if route[global_index]]
                selected_prediction = {"logit": prediction["logit"][selected],
                    "host_signed": [prediction["host_signed"][local] for local in selected]}
                per_seed_deletion.append(deletion_effects(model,
                    [normalized_test[local] for local in selected], selected_prediction, seed, device))
        for key in fields: fields[key][test] = np.mean(np.stack(per_seed[key]), axis=0)
        if arm == "M3_SIGNED_CLAM":
            routed_test = [int(global_index) for global_index in test if route[global_index]]
            for local, global_index in enumerate(routed_test):
                for k in (1, 3, 5):
                    available = [rows[local][k] for rows in per_seed_deletion if k in rows[local]]
                    if available:
                        deletion_rows.append({"index": int(global_index), "fold": current, "k": k,
                            "top_direction_drop": float(np.mean([x[0] for x in available])),
                            "random_direction_drop": float(np.mean([x[1] for x in available])),
                            "only_top_signed_logit": float(np.mean([x[2] for x in available]))})
        print(f"SIGNED_HOST_MIL_REAL arm={arm} fold={current}/5", flush=True)
    return fields, deletion_rows


@torch.no_grad()
def deletion_effects(model, bags, prediction, seed, device):
    rng = np.random.default_rng(seed); result = []
    for index, bag in enumerate(bags):
        original = float(prediction["logit"][index]); signed = np.asarray(prediction["host_signed"][index], float)
        sign = 1.0 if original >= 0 else -1.0; order = np.argsort(-sign * signed); local = {}
        for k in (1, 3, 5):
            if len(bag) <= k: continue
            remove = set(order[:k].tolist()); kept = np.asarray([i for i in range(len(bag)) if i not in remove], int)
            top_prediction = predict_model(model, [bag[kept]], device)["logit"][0]
            random_drops = []
            for _ in range(20):
                removed = set(rng.choice(len(bag), size=k, replace=False).tolist())
                random_kept = np.asarray([i for i in range(len(bag)) if i not in removed], int)
                random_logit = predict_model(model, [bag[random_kept]], device)["logit"][0]
                random_drops.append(sign * (original - random_logit))
            only = predict_model(model, [bag[order[:k]]], device)["logit"][0]
            local[k] = (sign * (original - top_prediction), float(np.mean(random_drops)), float(only))
        result.append(local)
    return result


def fit_count_oof(bags, y, fold):
    x = count_features(bags); result = np.zeros(len(y), float)
    for current in range(1, 6):
        train = fold != current; test = fold == current
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000,
            class_weight="balanced", random_state=SEED + current))
        model.fit(x[train], y[train]); probability = model.predict_proba(x[test])[:, 1]
        result[test] = logit(probability)
    return result, x


def shuffled_labels(y, base, bags, indices, rng):
    value = np.asarray(y[indices], int).copy(); side = base[indices] >= .5
    bins = np.asarray([host_count_bin(bags[i]) for i in indices])
    for current_side in (False, True):
        for current_bin in np.unique(bins):
            local = np.flatnonzero((side == current_side) & (bins == current_bin))
            if len(local) > 1: value[local] = value[rng.permutation(local)]
    return value


def bag_permutation(bags, fold, base, seed):
    rng = np.random.default_rng(seed); mapping = np.arange(len(bags)); side = base >= .5
    bins = np.asarray([host_count_bin(bag) for bag in bags])
    for current in range(1, 6):
        for current_side in (False, True):
            for current_bin in np.unique(bins):
                index = np.flatnonzero((fold == current) & (side == current_side) & (bins == current_bin))
                if len(index) > 1:
                    rotated = rng.permutation(index)
                    if np.all(rotated == index): rotated = np.roll(rotated, 1)
                    mapping[index] = rotated
    return [bags[i] for i in mapping], mapping


def fit_null_oof(kind, repeat, bags, y, fold, base, device):
    cache = OUTPUT / "checkpoints" / f"null_{kind}_{repeat:03d}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as archive: return np.asarray(archive["logit"], float)
    used_bags = bags; mapping = None
    if kind == "bag": used_bags, mapping = bag_permutation(bags, fold, base, SEED + 50000 + repeat)
    score = np.zeros(len(y), float); rng = np.random.default_rng(SEED + 70000 + repeat)
    for current in range(1, 6):
        train = np.flatnonzero(fold != current); test = np.flatnonzero(fold == current)
        train_y = y[train] if kind == "bag" else shuffled_labels(y, base, bags, train, rng)
        _, _, prediction = train_or_load("M3_SIGNED_CLAM", [used_bags[i] for i in train], train_y,
            [used_bags[i] for i in test], current, SEED + 90000 + repeat,
            f"null_{kind}_{repeat:03d}", device)
        score[test] = prediction["logit"]
    np.savez_compressed(cache, logit=score, mapping=np.asarray(mapping if mapping is not None else [], int))
    return score


def direction_auc(y, fold, base, route, score, arm):
    predicted_malware = route & (base >= .5); predicted_benign = route & (base < .5)
    fp_auc = finite(binary_auc((y[predicted_malware] == 0).astype(int), -score[predicted_malware]))
    fn_auc = finite(binary_auc((y[predicted_benign] == 1).astype(int), score[predicted_benign]))
    pooled = {"arm": arm, "scope": "pooled", "FP_vs_TP_AUC": fp_auc, "FN_vs_TN_AUC": fn_auc,
              "FP_vs_TP_bootstrap95": bootstrap_auc((y[predicted_malware] == 0).astype(int), -score[predicted_malware], SEED + 301),
              "FN_vs_TN_bootstrap95": bootstrap_auc((y[predicted_benign] == 1).astype(int), score[predicted_benign], SEED + 307),
              "predicted_malware_n": int(predicted_malware.sum()), "predicted_benign_n": int(predicted_benign.sum())}
    folds = []
    for current in range(1, 6):
        pm = predicted_malware & (fold == current); pb = predicted_benign & (fold == current)
        folds.append({"arm": arm, "fold": current, "routed": int(np.sum(route & (fold == current))),
            "FP_vs_TP_AUC": finite(binary_auc((y[pm] == 0).astype(int), -score[pm])),
            "FN_vs_TN_AUC": finite(binary_auc((y[pb] == 1).astype(int), score[pb]))})
    return pooled, folds


def bootstrap_auc(y, score, seed, repetitions=2000):
    y = np.asarray(y, int); score = np.asarray(score, float)
    if len(np.unique(y)) < 2: return None
    rng = np.random.default_rng(seed); values = []
    for _ in range(repetitions):
        index = rng.integers(0, len(y), size=len(y)); value = binary_auc(y[index], score[index])
        if np.isfinite(value): values.append(float(value))
    if not values: return None
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def main():
    rows, ids, y, fold, base, c43, route, bags = load_source()
    if int(route.sum()) != 200: raise RuntimeError("BLOCKED_BY_CONTRACT: SVM route is not 200")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SIGNED_HOST_MIL_DEVICE={device} bags={len(bags)} hosts={sum(map(len,bags))}", flush=True)
    c43_direction = np.clip(logit(c43) - logit(base), -MAX_DELTA, MAX_DELTA)
    real = {}; deletion_rows = []
    for arm in ARMS:
        real[arm], local_deletion = fit_real_arm(arm, bags, y, fold, route, device); deletion_rows.extend(local_deletion)
    count_score, count_x = fit_count_oof(bags, y, fold)

    probe_rows = []; fold_probe_rows = []
    score_by_arm = {"M0_FORMAL_C43": c43_direction, "N1_COUNT_ONLY": count_score}
    score_by_arm.update({arm: real[arm]["logit"] for arm in ARMS})
    for arm, score in score_by_arm.items():
        pooled, fold_rows = direction_auc(y, fold, base, route, score, arm)
        probe_rows.append(pooled); fold_probe_rows.extend(fold_rows)

    null_rows = []
    for repeat in range(NULL_REPETITIONS):
        for kind in ("label", "bag"):
            score = fit_null_oof(kind, repeat, bags, y, fold, base, device)
            pooled, _ = direction_auc(y, fold, base, route, score, "N2_LABEL_SHUFFLE" if kind == "label" else "N3_BAG_SHUFFLE")
            null_rows.append({"control": kind, "repeat": repeat, **{key: value for key, value in pooled.items() if key not in {"arm", "scope"}}})
        print(f"SIGNED_HOST_MIL_NULL {repeat+1}/{NULL_REPETITIONS}", flush=True)

    write_csv(OUTPUT / "direction_probe_results.csv", probe_rows)
    write_csv(OUTPUT / "direction_probe_folds.csv", fold_probe_rows)
    write_csv(OUTPUT / "null_control_results.csv", null_rows)
    write_csv(OUTPUT / "deletion_audit.csv", deletion_rows)

    m3 = next(row for row in probe_rows if row["arm"] == "M3_SIGNED_CLAM")
    m0 = next(row for row in probe_rows if row["arm"] == "M0_FORMAL_C43")
    count = next(row for row in probe_rows if row["arm"] == "N1_COUNT_ONLY")
    label_fp = np.asarray([row["FP_vs_TP_AUC"] for row in null_rows if row["control"] == "label" and row["FP_vs_TP_AUC"] is not None], float)
    bag_fp = np.asarray([row["FP_vs_TP_AUC"] for row in null_rows if row["control"] == "bag" and row["FP_vs_TP_AUC"] is not None], float)
    null95 = max(float(np.quantile(label_fp, .95)), float(np.quantile(bag_fp, .95)))
    m3_fold = [row for row in fold_probe_rows if row["arm"] == "M3_SIGNED_CLAM" and row["FP_vs_TP_AUC"] is not None]
    routed_deletion = [row for row in deletion_rows if route[int(row["index"])] and int(row["k"]) == 3]
    top_drop = float(np.mean([row["top_direction_drop"] for row in routed_deletion])) if routed_deletion else float("nan")
    random_drop = float(np.mean([row["random_direction_drop"] for row in routed_deletion])) if routed_deletion else float("nan")
    criteria = {"FP_AUC>=0.65": m3["FP_vs_TP_AUC"] is not None and m3["FP_vs_TP_AUC"] >= .65,
        "FP_AUC>count+0.03": m3["FP_vs_TP_AUC"] is not None and count["FP_vs_TP_AUC"] is not None and m3["FP_vs_TP_AUC"] > count["FP_vs_TP_AUC"] + .03,
        "FP_AUC>null95+0.03": m3["FP_vs_TP_AUC"] is not None and m3["FP_vs_TP_AUC"] > null95 + .03,
        "FN_AUC>=0.55": m3["FN_vs_TN_AUC"] is not None and m3["FN_vs_TN_AUC"] >= .55,
        "FN_not_below_C43_minus_0.03": m3["FN_vs_TN_AUC"] is not None and m0["FN_vs_TN_AUC"] is not None and m3["FN_vs_TN_AUC"] >= m0["FN_vs_TN_AUC"] - .03,
        "FP_majority_folds": sum(row["FP_vs_TP_AUC"] > .5 for row in m3_fold) >= 3,
        "top3_deletion>random": np.isfinite(top_drop) and top_drop > random_drop}
    gate = all(criteria.values())

    size_audit = {"host_count": spearman(real["M3_SIGNED_CLAM"]["logit"], [len(bag) for bag in bags]),
        "active_role_count": spearman(real["M3_SIGNED_CLAM"]["logit"], count_x[:, 1]),
        "nonzero_entry_count": spearman(real["M3_SIGNED_CLAM"]["logit"], count_x[:, 3])}
    dump_json(OUTPUT / "direction_gate.json", {"status": "PASS" if gate else "FAIL", "criteria": criteria,
        "M3": m3, "M0": m0, "count_only": count, "matched_null_FP_AUC_95pct": null95,
        "deletion_top3_mean": top_drop, "deletion_random3_mean": random_drop,
        "size_correlations": size_audit, "future_rows_loaded": 0})

    correction_probabilities = {"S0_STAGE1_SVM": base, "M0_FORMAL_C43": c43}
    if gate:
        for arm in ARMS:
            delta = np.where(route, np.clip(real[arm]["logit"], -MAX_DELTA, MAX_DELTA), 0.0)
            correction_probabilities[arm + "_BOUNDED"] = sigmoid(logit(base) + delta)
    corrections = [{"arm": arm, **correction_metrics(y, base, probability, route)}
                   for arm, probability in correction_probabilities.items()]
    correction_folds = []
    for arm, probability in correction_probabilities.items():
        for current in range(1, 6):
            selected = fold == current
            correction_folds.append({"arm": arm, "fold": current,
                **correction_metrics(y[selected], base[selected], probability[selected], route[selected])})
    write_csv(OUTPUT / "source_correction_results.csv", corrections)
    write_csv(OUTPUT / "source_correction_folds.csv", correction_folds)

    prediction_rows = []
    for index, sid in enumerate(ids):
        prediction_rows.append({"sha256": sid, "fold": int(fold[index]), "label": int(y[index]),
            "stage1_prob": float(base[index]), "routed": int(route[index]), "formal_c43_prob": float(c43[index]),
            "host_count": len(bags[index]), "count_only_logit": float(count_score[index]),
            **{arm + "__logit": float(real[arm]["logit"][index]) for arm in ARMS},
            **{arm + "__malware_evidence": float(real[arm]["malware"][index]) for arm in ARMS},
            **{arm + "__benign_evidence": float(real[arm]["benign"][index]) for arm in ARMS}})
    write_csv(OUTPUT / "source_predictions.csv", prediction_rows)

    baseline = corrections[1]; integrated = corrections[-1] if gate else None
    fold_lookup = {(row["arm"], row["fold"]): row for row in correction_folds}
    noninferior_folds = (sum(fold_lookup[("M3_SIGNED_CLAM_BOUNDED", current)]["net"] >=
                             fold_lookup[("M0_FORMAL_C43", current)]["net"] for current in range(1, 6))
                         if gate else 0)
    integration_pass = bool(gate and integrated["net"] > baseline["net"] and integrated["broken"] <= baseline["broken"] and
                            integrated["fixed_FP"] >= baseline["fixed_FP"] and noninferior_folds >= 3)
    status = ("SIGNED_HOST_MIL_SUPPORTED_FOR_FROZEN_FUTURE" if integration_pass else
              "SIGNED_HOST_MIL_DIRECTION_PRESENT_BUT_INTEGRATION_NOT_SUPPORTED" if gate else
              "SIGNED_HOST_MIL_DIRECTION_NOT_SUPPORTED")
    dump_json(OUTPUT / "final_decision.json", {"status": status, "stopped_at": "SOURCE_INTEGRATION" if gate else "DIRECTION_GATE",
        "direction_gate_passed": gate, "integration_passed": integration_pass, "criteria": criteria,
        "formal_c43": baseline, "integrated_M3": integrated, "M3_noninferior_folds": noninferior_folds,
        "future_rows_loaded": 0,
        "formal_integrity_pending": True})
    print("SIGNED_HOST_MIL_SOURCE_DONE", flush=True)
    print(f"M3_FP_VS_TP_AUC={m3['FP_vs_TP_AUC']} M3_FN_VS_TN_AUC={m3['FN_vs_TN_AUC']} NULL95={null95}", flush=True)
    for row in corrections: print(f"{row['arm']}: BA={row['BA']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']}", flush=True)
    print("SIGNED_HOST_MIL_DECISION=" + status, flush=True)


if __name__ == "__main__":
    main()
