#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import FORMAL, HOST, METHOD, OUTPUT, P31, SEED, auc, dump, metrics, norm, read_csv, write_csv
from modeling import crossfit_calibrated_context, fit_calibrated_rf, rf


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def load_all():
    rows = read_csv(FORMAL)
    ids = np.asarray([norm(row["sha256"]) for row in rows])
    y = np.asarray([int(row["label"]) for row in rows])
    fold = np.asarray([int(row["fold"]) for row in rows])
    base = np.asarray([float(row["Global24__prob"]) for row in rows])
    route = np.asarray([int(float(row["A25__routed"])) for row in rows], bool)
    delta = np.asarray([float(row[METHOD + "__bounded_delta"]) for row in rows])
    with np.load(P31, allow_pickle=False) as data:
        p_ids = [norm(value) for value in data["sha256"]]
        p_values = np.asarray(data["role_prevalence"], float)
    p_pos = {value: index for index, value in enumerate(p_ids)}
    p31 = p_values[[p_pos[value] for value in ids]]
    with np.load(HOST, allow_pickle=False) as data:
        h_ids = [norm(value) for value in data["sha256"]]
        values = np.asarray(data["host_values"], float)
        offsets = np.asarray(data["host_offsets"], int)
        real = np.asarray(data["real_host_count"], int) if "real_host_count" in data else np.diff(offsets)
    h_pos = {value: index for index, value in enumerate(h_ids[:799])}
    bags = []
    for value in ids:
        index = h_pos[value]
        count = int(real[index])
        bag = values[offsets[index]: offsets[index] + count, :105]
        bags.append(bag if count else np.zeros((1, 105)))
    return rows, ids, y, fold, base, route, delta, p31, bags


def unit_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 1e-12)


def make_groups(p31):
    unit = unit_rows(np.asarray(p31, float))
    similarity = unit @ unit.T
    distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(similarity, -1, 1)))
    uf = UnionFind(len(p31))
    for left in range(len(p31)):
        neighbors = np.flatnonzero((similarity[left, left + 1:] >= .9999) &
                                   (distance[left, left + 1:] <= .015)) + left + 1
        for right in neighbors:
            uf.union(left, int(right))
    roots = [uf.find(index) for index in range(len(p31))]
    mapping = {root: index for index, root in enumerate(sorted(set(roots)))}
    groups = np.asarray([mapping[root] for root in roots], int)
    return groups, unit, similarity, distance


def group_audit(ids, y, original_fold, groups, similarity, distance):
    rows = []
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[int(group)].append(index)
    for group, indices in members.items():
        block = np.asarray(indices, int)
        pair_sim = similarity[np.ix_(block, block)]
        pair_dist = distance[np.ix_(block, block)]
        rows.append({
            "group": group, "size": len(block), "benign": int(np.sum(y[block] == 0)),
            "malware": int(np.sum(y[block] == 1)), "label_purity": float(max(np.mean(y[block] == 0), np.mean(y[block] == 1))),
            "original_fold_count": len(set(original_fold[block].tolist())),
            "min_pair_cosine": float(np.min(pair_sim)), "max_pair_unit_l2": float(np.max(pair_dist)),
            "chain_expanded": int(np.min(pair_sim) < .9999 or np.max(pair_dist) > .015),
            "sha256_members": ";".join(ids[block].tolist()),
        })
    write_csv(OUTPUT / "p31_group_structure_audit.csv", sorted(rows, key=lambda row: row["size"], reverse=True))
    dump(OUTPUT / "p31_group_structure_summary.json", {
        "groups": len(rows), "multi_sample_groups": sum(row["size"] > 1 for row in rows),
        "samples_in_multi_sample_groups": sum(row["size"] for row in rows if row["size"] > 1),
        "largest_group": max(row["size"] for row in rows),
        "chain_expanded_groups": sum(row["chain_expanded"] for row in rows),
        "largest_group_record": sorted(rows, key=lambda row: row["size"], reverse=True)[0],
    })


def group_splits(p31, y, groups):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261007)
    splits = list(splitter.split(p31, y, groups))
    assignment = np.full(len(y), -1, int)
    for held, (_, test) in enumerate(splits, 1):
        assignment[test] = held
    if np.any(assignment < 0):
        raise RuntimeError("group assignment incomplete")
    return assignment


def weights_for_groups(groups):
    counts = Counter(np.asarray(groups, int).tolist())
    return np.asarray([1.0 / counts[int(group)] for group in groups], float)


def grouped_stage1(ids, p31, y, groups, assignment):
    raw = np.full(len(y), np.nan)
    calibrated = np.full(len(y), np.nan)
    weighted = np.full(len(y), np.nan)
    all_weight = weights_for_groups(groups)
    fold_rows = []
    context_by_outer = {}
    for held in range(1, 6):
        train = np.flatnonzero(assignment != held)
        test = np.flatnonzero(assignment == held)
        # Exact seed used by the preceding information-audit grouped RF, so
        # the raw arm is a reproduction rather than a new random realization.
        plain = rf(20261007 + 800 + held)
        plain.fit(p31[train], y[train])
        raw[test] = plain.predict_proba(p31[test])[:, 1]
        calibrated[test] = fit_calibrated_rf(p31[train], y[train], p31[test], SEED + 2000 + held,
                                             groups_train=groups[train])
        weighted[test] = fit_calibrated_rf(p31[train], y[train], p31[test], SEED + 3000 + held,
                                           groups_train=groups[train], sample_weight=all_weight[train])
        # Strict outer-train context. Inner group folds are fixed without seeing outer test.
        local = np.full(len(y), -1, int)
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 4000 + held)
        for inner_id, (_, inner_test_local) in enumerate(inner.split(p31[train], y[train], groups[train]), 1):
            local[train[inner_test_local]] = inner_id
        context = crossfit_calibrated_context(p31, y, train, local, SEED + 5000 + held, groups=groups)
        context_by_outer[held] = context
        print(f"GROUP_STAGE1_OUTER_DONE {held}/5", flush=True)
    if not all(np.isfinite(value).all() for value in (raw, calibrated, weighted)):
        raise RuntimeError("incomplete grouped Stage-I")
    predictions = {"GROUP_RF_RAW": raw, "GROUP_RF_PLATT": calibrated, "GROUP_RF_INVERSE_GROUP_WEIGHT_PLATT": weighted}
    aggregate = []
    output = []
    for name, probability in predictions.items():
        aggregate.append({"arm": name, **metrics(y, probability)})
        for held in range(1, 6):
            mask = assignment == held
            fold_rows.append({"arm": name, "fold": held, **metrics(y[mask], probability[mask]),
                              "prob_benign_median": float(np.median(probability[mask & (y == 0)])),
                              "prob_malware_median": float(np.median(probability[mask & (y == 1)]))})
    for index in range(len(y)):
        output.append({"sha256": ids[index], "group": int(groups[index]), "group_fold": int(assignment[index]),
                       "label": int(y[index]), **{name + "__prob": float(value[index]) for name, value in predictions.items()}})
    write_csv(OUTPUT / "group_stage1_results.csv", aggregate)
    write_csv(OUTPUT / "group_stage1_fold_results.csv", fold_rows)
    write_csv(OUTPUT / "group_stage1_predictions.csv", output)
    return predictions, context_by_outer


def fixed_nested_context(p31, y, fold):
    result = {}
    for outer in range(1, 6):
        train = np.flatnonzero(fold != outer)
        context = crossfit_calibrated_context(p31, y, train, fold, SEED + 10000 + outer)
        result[outer] = context
        print(f"FIXED_NESTED_STAGE1_CONTEXT_DONE {outer}/5", flush=True)
    return result


def compact_features(bags):
    mean, std, max_abs, count = [], [], [], []
    for bag in bags:
        compact = np.asarray(bag[:, :43], float)
        role = np.maximum(np.asarray(bag[:, 43:74], float), 0)
        relation = np.asarray(bag[:, 74:105], float)
        mean.append(compact.mean(0))
        std.append(compact.std(0))
        max_abs.append(np.max(np.abs(compact), axis=0))
        role_mass = role.sum(0)
        total = role_mass.sum()
        p = role_mass / max(total, 1e-12)
        positive = p[p > 0]
        entropy = -np.sum(positive * np.log(positive)) / np.log(31) if len(positive) else 0.0
        count.append([math.log1p(len(bag)), int(np.sum(role_mass > 1e-12)), entropy,
                      math.log1p(float(total)), float(np.mean(np.sum(np.abs(relation), axis=1) > 1e-12)),
                      float(np.mean(np.sum(np.abs(relation), axis=1)))])
    mean, std, max_abs, count = map(np.asarray, (mean, std, max_abs, count))
    return {"COMPACT_MEAN_STD_MAX": np.c_[mean, std, max_abs], "COMPACT_MEAN": mean,
            "COMPACT_STD": std, "COMPACT_MAX_ABS": max_abs, "COUNT_ONLY": count}, count


def probe_model(width, seed):
    steps = [("scale", StandardScaler())]
    if width > 31:
        steps.append(("pca", PCA(n_components=31, svd_solver="full")))
    steps.append(("model", LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced",
                                                max_iter=5000, random_state=int(seed))))
    return Pipeline(steps)


def stratified_feature_shuffle(values, count, indices, seed):
    output = values.copy()
    index = np.asarray(indices, int)
    host_count = count[index, 0]
    cuts = np.unique(np.quantile(host_count, [0, .25, .5, .75, 1.0]))
    if len(cuts) < 2:
        strata = np.zeros(len(index), int)
    else:
        strata = np.digitize(host_count, cuts[1:-1], right=True)
    rng = np.random.RandomState(seed)
    for stratum in sorted(set(strata.tolist())):
        block = index[strata == stratum]
        output[block] = values[rng.permutation(block)]
    return output


def run_compact_protocol(protocol, ids, y, outer_assignment, test_stage1, train_contexts,
                         features, count, route=None):
    names = list(features)
    predictions = {name: np.full(len(y), np.nan) for name in names}
    predictions["COMPACT_COUNT_RESIDUAL"] = np.full(len(y), np.nan)
    shuffle_predictions = {rep: np.full(len(y), np.nan) for rep in range(5)}
    train_audit = []
    for outer in range(1, 6):
        train = np.flatnonzero(outer_assignment != outer)
        test = np.flatnonzero(outer_assignment == outer)
        context = train_contexts[outer]
        for side in (0, 1):
            side_train = train[(context[train] >= .5) == side]
            side_test = test[(test_stage1[test] >= .5) == side]
            train_audit.append({"protocol": protocol, "outer_fold": outer, "stage1_side": side,
                                "train_n": len(side_train), "train_benign": int(np.sum(y[side_train] == 0)),
                                "train_malware": int(np.sum(y[side_train] == 1)), "test_n": len(side_test),
                                "test_benign": int(np.sum(y[side_test] == 0)), "test_malware": int(np.sum(y[side_test] == 1))})
            if not len(side_test):
                continue
            if len(np.unique(y[side_train])) < 2:
                raise RuntimeError(f"conditional train lacks class protocol={protocol} outer={outer} side={side}")
            for ordinal, (name, values) in enumerate(features.items()):
                model = probe_model(values.shape[1], SEED + 20000 + outer * 100 + side * 10 + ordinal)
                model.fit(values[side_train], y[side_train])
                predictions[name][side_test] = model.predict_proba(values[side_test])[:, 1]
            real = features["COMPACT_MEAN_STD_MAX"]
            regressor = LinearRegression().fit(count[side_train], real[side_train])
            residual_train = real[side_train] - regressor.predict(count[side_train])
            residual_test = real[side_test] - regressor.predict(count[side_test])
            model = probe_model(real.shape[1], SEED + 21000 + outer * 100 + side)
            model.fit(residual_train, y[side_train])
            predictions["COMPACT_COUNT_RESIDUAL"][side_test] = model.predict_proba(residual_test)[:, 1]
            for rep in range(5):
                shuffled = stratified_feature_shuffle(real, count, side_train, SEED + 22000 + outer * 1000 + side * 100 + rep)
                model = probe_model(real.shape[1], SEED + 23000 + outer * 1000 + side * 100 + rep)
                model.fit(shuffled[side_train], y[side_train])
                shuffle_predictions[rep][side_test] = model.predict_proba(real[side_test])[:, 1]
        print(f"COMPACT_STRICT_{protocol}_OUTER_DONE {outer}/5", flush=True)
    for name, probability in predictions.items():
        if not np.isfinite(probability).all():
            raise RuntimeError("incomplete compact prediction " + protocol + " " + name)
    for probability in shuffle_predictions.values():
        if not np.isfinite(probability).all():
            raise RuntimeError("incomplete compact shuffle " + protocol)
    write_csv(OUTPUT / f"{protocol.lower()}_conditional_train_audit.csv", train_audit)
    scopes = {"ALL": np.ones(len(y), bool)}
    if route is not None:
        scopes["DI25_ROUTED"] = np.asarray(route, bool)
    aggregate, folds = [], []
    for name, probability in predictions.items():
        for scope_name, scope in scopes.items():
            predicted_benign = scope & (test_stage1 < .5)
            predicted_malware = scope & (test_stage1 >= .5)
            record = {"protocol": protocol, "representation": name, "scope": scope_name,
                      "FN_vs_TN_AUC": auc((y[predicted_benign] == 1).astype(int), probability[predicted_benign]),
                      "FP_vs_TP_AUC": auc((y[predicted_malware] == 0).astype(int), -probability[predicted_malware]),
                      "predicted_benign_n": int(predicted_benign.sum()), "predicted_malware_n": int(predicted_malware.sum())}
            aggregate.append(record)
            for outer in range(1, 6):
                fscope = scope & (outer_assignment == outer)
                f0, f1 = fscope & (test_stage1 < .5), fscope & (test_stage1 >= .5)
                folds.append({"protocol": protocol, "representation": name, "scope": scope_name, "fold": outer,
                              "FN_vs_TN_AUC": auc((y[f0] == 1).astype(int), probability[f0]),
                              "FP_vs_TP_AUC": auc((y[f1] == 0).astype(int), -probability[f1]),
                              "predicted_benign_n": int(f0.sum()), "predicted_malware_n": int(f1.sum())})
    shuffle = []
    for rep, probability in shuffle_predictions.items():
        for scope_name, scope in scopes.items():
            p0, p1 = scope & (test_stage1 < .5), scope & (test_stage1 >= .5)
            shuffle.append({"protocol": protocol, "control": "HOST_COUNT_STRATIFIED_TRAIN_ROW_SHUFFLE",
                            "repetition": rep, "scope": scope_name,
                            "FN_vs_TN_AUC": auc((y[p0] == 1).astype(int), probability[p0]),
                            "FP_vs_TP_AUC": auc((y[p1] == 0).astype(int), -probability[p1])})
    prediction_rows = []
    for index in range(len(y)):
        prediction_rows.append({"protocol": protocol, "sha256": ids[index], "fold": int(outer_assignment[index]),
                                "label": int(y[index]), "stage1_probability": float(test_stage1[index]),
                                **{name + "__prob": float(prob[index]) for name, prob in predictions.items()}})
    return aggregate, folds, shuffle, prediction_rows, predictions, shuffle_predictions


def stratified_paired_bootstrap(target, score_a, score_b, seed, repetitions=2000):
    target = np.asarray(target, int)
    score_a = np.asarray(score_a, float)
    score_b = np.asarray(score_b, float)
    positive = np.flatnonzero(target == 1)
    negative = np.flatnonzero(target == 0)
    if not len(positive) or not len(negative):
        return {"repetitions": 0, "AUC_A": float("nan"), "AUC_B": float("nan"),
                "difference": float("nan"), "difference_ci_low": float("nan"),
                "difference_ci_high": float("nan")}
    rng = np.random.RandomState(seed)
    differences = []
    for _ in range(repetitions):
        index = np.r_[rng.choice(positive, len(positive), replace=True),
                      rng.choice(negative, len(negative), replace=True)]
        differences.append(auc(target[index], score_a[index]) - auc(target[index], score_b[index]))
    return {"repetitions": repetitions, "AUC_A": auc(target, score_a), "AUC_B": auc(target, score_b),
            "difference": auc(target, score_a) - auc(target, score_b),
            "difference_ci_low": float(np.quantile(differences, .025)),
            "difference_ci_high": float(np.quantile(differences, .975))}


def main():
    _, ids, y, fixed_fold, formal_stage1, route, delta, p31, bags = load_all()
    groups, _, similarity, distance = make_groups(p31)
    group_audit(ids, y, fixed_fold, groups, similarity, distance)
    assignment = group_splits(p31, y, groups)
    write_csv(OUTPUT / "p31_group_assignments.csv", [
        {"sha256": ids[index], "label": int(y[index]), "original_fold": int(fixed_fold[index]),
         "group": int(groups[index]), "group_size": int(np.sum(groups == groups[index])),
         "group_fold": int(assignment[index])} for index in range(len(y))
    ])
    group_predictions, group_contexts = grouped_stage1(ids, p31, y, groups, assignment)
    fixed_contexts = fixed_nested_context(p31, y, fixed_fold)
    features, count = compact_features(bags)
    fixed_results = run_compact_protocol("FIXED_FOLDS", ids, y, fixed_fold, formal_stage1,
                                         fixed_contexts, features, count, route=route)
    group_results = run_compact_protocol("GROUP_FOLDS", ids, y, assignment,
                                         group_predictions["GROUP_RF_PLATT"], group_contexts,
                                         features, count, route=None)
    aggregate = fixed_results[0] + group_results[0]
    folds = fixed_results[1] + group_results[1]
    shuffles = fixed_results[2] + group_results[2]
    predictions = fixed_results[3] + group_results[3]
    # Frozen current C43 delta is a valid comparison only under the original fixed outer folds.
    current_score = 1.0 / (1.0 + np.exp(-delta))
    for scope_name, scope in (("ALL", np.ones(len(y), bool)), ("DI25_ROUTED", route)):
        p0, p1 = scope & (formal_stage1 < .5), scope & (formal_stage1 >= .5)
        aggregate.append({"protocol": "FIXED_FOLDS", "representation": "CURRENT_C43_DELTA_FROZEN_REFERENCE",
                          "scope": scope_name, "FN_vs_TN_AUC": auc((y[p0] == 1).astype(int), current_score[p0]),
                          "FP_vs_TP_AUC": auc((y[p1] == 0).astype(int), -current_score[p1]),
                          "predicted_benign_n": int(p0.sum()), "predicted_malware_n": int(p1.sum())})
    write_csv(OUTPUT / "compact43_strict_results.csv", aggregate)
    write_csv(OUTPUT / "compact43_strict_fold_results.csv", folds)
    write_csv(OUTPUT / "compact43_matched_shuffle.csv", shuffles)
    write_csv(OUTPUT / "compact43_strict_predictions.csv", predictions)
    bootstrap_rows = []
    fixed_probe = fixed_results[4]
    group_probe = group_results[4]
    fixed_real = fixed_probe["COMPACT_MEAN_STD_MAX"]
    for side, mask, target, score_sign in (
        ("FN_vs_TN", route & (formal_stage1 < .5), None, 1.0),
        ("FP_vs_TP", route & (formal_stage1 >= .5), None, -1.0),
    ):
        target = (y[mask] == (1 if side == "FN_vs_TN" else 0)).astype(int)
        comparisons = {
            "CURRENT_C43_DELTA": current_score,
            "COUNT_ONLY": fixed_probe["COUNT_ONLY"],
            "COMPACT_COUNT_RESIDUAL": fixed_probe["COMPACT_COUNT_RESIDUAL"],
        }
        for ordinal, (comparison, control_score) in enumerate(comparisons.items()):
            record = stratified_paired_bootstrap(
                target, score_sign * fixed_real[mask], score_sign * control_score[mask],
                SEED + 30000 + ordinal + (100 if side == "FP_vs_TP" else 0),
            )
            bootstrap_rows.append({"protocol": "FIXED_FOLDS", "scope": "DI25_ROUTED", "comparison": side,
                                   "candidate": "COMPACT_MEAN_STD_MAX", "control": comparison, **record})
    group_real_score = group_probe["COMPACT_MEAN_STD_MAX"]
    group_stage1 = group_predictions["GROUP_RF_PLATT"]
    for side, mask, score_sign in (
        ("FN_vs_TN", group_stage1 < .5, 1.0),
        ("FP_vs_TP", group_stage1 >= .5, -1.0),
    ):
        target = (y[mask] == (1 if side == "FN_vs_TN" else 0)).astype(int)
        record = stratified_paired_bootstrap(
            target, score_sign * group_real_score[mask], score_sign * group_probe["COUNT_ONLY"][mask],
            SEED + 31000 + (100 if side == "FP_vs_TP" else 0),
        )
        bootstrap_rows.append({"protocol": "GROUP_FOLDS", "scope": "ALL", "comparison": side,
                               "candidate": "COMPACT_MEAN_STD_MAX", "control": "COUNT_ONLY", **record})
    write_csv(OUTPUT / "compact43_paired_bootstrap.csv", bootstrap_rows)
    dump(OUTPUT / "run_state.json", {"status": "COMPLETE", "future_rows_loaded": 0})
    print("CARE_GROUP_CALIBRATION_COMPACT43_VALIDATION_DONE", flush=True)


if __name__ == "__main__":
    main()

