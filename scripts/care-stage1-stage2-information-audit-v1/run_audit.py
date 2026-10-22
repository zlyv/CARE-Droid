#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import (
    FORMAL_METHOD, FORMAL_PREDICTIONS, HOST_CACHE, OUTPUT, P31_CACHE, ROLE_NAMES,
    ROUTE_FIELD, SEED, STAGE1_FIELD, binary_auc, dump, ece, metric, norm, read_csv,
    write_csv,
)


def probability_column(rows, suffix):
    field = FORMAL_METHOD + suffix
    if field not in rows[0]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing formal field " + field)
    return np.asarray([float(row[field]) for row in rows])


def aligned_p31(ids):
    with np.load(P31_CACHE, allow_pickle=False) as data:
        source_ids = [norm(value) for value in data["sha256"]]
        values = np.asarray(data["role_prevalence"], float)
    position = {value: index for index, value in enumerate(source_ids)}
    return values[[position[value] for value in ids]]


def aligned_host(ids):
    with np.load(HOST_CACHE, allow_pickle=False) as data:
        source_ids = [norm(value) for value in data["sha256"]]
        values = np.asarray(data["host_values"], float)
        offsets = np.asarray(data["host_offsets"], int)
        real = np.asarray(data["real_host_count"], int) if "real_host_count" in data else np.diff(offsets)
    position = {value: index for index, value in enumerate(source_ids[:799])}
    bags = []
    for value in ids:
        index = position[value]
        count = int(real[index])
        bag = values[offsets[index]: offsets[index] + count, :105]
        if count == 0:
            bag = np.zeros((1, 105), float)
        bags.append(bag)
    return bags


def row_entropy(values):
    values = np.maximum(np.asarray(values, float), 0)
    total = values.sum()
    if total <= 1e-12:
        return 0.0
    p = values / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / max(np.log(len(values)), 1e-12))


def aggregate_representations(p31, bags):
    compact = []
    role = []
    relation = []
    full = []
    count = []
    for bag in bags:
        c = bag[:, :43]
        r = np.maximum(bag[:, 43:74], 0)
        q = bag[:, 74:105]
        c_summary = np.concatenate((c.mean(0), c.std(0), np.max(np.abs(c), axis=0)))
        role_mass = r.sum(0)
        role_norm = role_mass / max(role_mass.sum(), 1e-12)
        q_summary = np.concatenate((q.mean(0), q.std(0), np.max(np.abs(q), axis=0)))
        compact.append(c_summary)
        role.append(role_norm)
        relation.append(q_summary)
        full.append(np.concatenate((c_summary, role_norm, q_summary)))
        count.append([
            math.log1p(len(bag)), int(np.sum(role_mass > 1e-12)), row_entropy(role_mass),
            math.log1p(float(role_mass.sum())), float(np.mean(np.sum(np.abs(q), axis=1) > 1e-12)),
            float(np.mean(np.sum(np.abs(q), axis=1))), float(np.mean(np.linalg.norm(c, axis=1))),
        ])
    compact = np.asarray(compact)
    role = np.asarray(role)
    relation = np.asarray(relation)
    full = np.asarray(full)
    count = np.asarray(count)
    return {
        "P31": np.asarray(p31, float),
        "COMPACT43_SUMMARY": compact,
        "ROLE31": role,
        "RELATION31_SUMMARY": relation,
        "FULL_STATIC_HOST": full,
        "P31_PLUS_FULL_STATIC_HOST": np.concatenate((p31, full), axis=1),
        "COUNT_ONLY": count,
    }


def oof_predict(model_factory, x, y, fold):
    output = np.full(len(y), np.nan)
    for held in range(1, 6):
        train = fold != held
        test = fold == held
        model = model_factory(held)
        model.fit(x[train], y[train])
        output[test] = model.predict_proba(x[test])[:, list(model.classes_).index(1)]
    if not np.isfinite(output).all():
        raise RuntimeError("incomplete OOF prediction")
    return np.clip(output, 1e-7, 1 - 1e-7)


def count_features(p31):
    positive = np.maximum(p31, 0)
    total = positive.sum(1)
    normalized = np.divide(positive, total[:, None], out=np.zeros_like(positive), where=total[:, None] > 0)
    entropy = -np.sum(np.where(normalized > 0, normalized * np.log(np.maximum(normalized, 1e-12)), 0), axis=1) / np.log(31)
    return np.c_[total, np.sum(positive > 1e-12, axis=1), np.linalg.norm(p31, axis=1), np.max(positive, axis=1), entropy]


def model_lr(seed):
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(
        C=1.0, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed))])


def stage1_ladder(p31, y, fold, formal_probability):
    predictions = {"B5_FORMAL_CALIBRATED_RF": formal_probability}
    predictions["B0_PREVALENCE"] = np.full(len(y), float(np.mean(y)))
    predictions["B1_COUNT_ONLY"] = oof_predict(lambda f: model_lr(SEED + f), count_features(p31), y, fold)
    predictions["B3_P31_LOGISTIC"] = oof_predict(lambda f: model_lr(SEED + 100 + f), p31, y, fold)
    predictions["B4_DEPTH2_TREE"] = oof_predict(lambda f: DecisionTreeClassifier(
        max_depth=2, min_samples_leaf=10, class_weight="balanced", random_state=SEED + 200 + f), p31, y, fold)
    predictions["B5_RF_UNCALIBRATED_REPLAY"] = oof_predict(lambda f: RandomForestClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced",
        n_jobs=4, random_state=SEED + 300 + f), p31, y, fold)
    single = np.full(len(y), np.nan)
    selection = []
    for held in range(1, 6):
        train = fold != held
        test = fold == held
        candidates = []
        for column in range(31):
            auc = binary_auc(y[train], p31[train, column])
            signed = max(auc, 1 - auc)
            candidates.append((signed, -column, column))
        _, _, chosen = max(candidates)
        model = model_lr(SEED + 400 + held)
        model.fit(p31[train, chosen:chosen + 1], y[train])
        single[test] = model.predict_proba(p31[test, chosen:chosen + 1])[:, 1]
        selection.append({"fold": held, "selected_role": chosen, "selected_role_name": ROLE_NAMES[chosen],
                          "train_abs_direction_AUC": max(candidates)[0]})
    predictions["B2_TRAIN_SELECTED_SINGLE_ROLE"] = single
    rows = []
    folds = []
    for name, probability in predictions.items():
        rows.append({"arm": name, **metric(y, probability)})
        for held in range(1, 6):
            mask = fold == held
            folds.append({"arm": name, "fold": held, **metric(y[mask], probability[mask])})
    write_csv(OUTPUT / "stage1_capacity_ladder.csv", rows)
    write_csv(OUTPUT / "stage1_capacity_ladder_folds.csv", folds)
    write_csv(OUTPUT / "stage1_single_role_selection.csv", selection)
    return predictions


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left = self.find(left)
        right = self.find(right)
        if left != right:
            self.parent[right] = left


def p31_similarity_groups(p31):
    scaled = np.asarray(p31, float)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    unit = np.divide(scaled, norms, out=np.zeros_like(scaled), where=norms > 1e-12)
    similarity = unit @ unit.T
    normalized_distance = np.linalg.norm(unit[:, None, :] - unit[None, :, :], axis=2)
    uf = UnionFind(len(p31))
    for left in range(len(p31)):
        close = np.flatnonzero((similarity[left, left + 1:] >= 0.9999) &
                               (normalized_distance[left, left + 1:] <= 0.015)) + left + 1
        for right in close:
            uf.union(left, int(right))
    roots = [uf.find(index) for index in range(len(p31))]
    mapping = {root: index for index, root in enumerate(sorted(set(roots)))}
    return np.asarray([mapping[root] for root in roots], int), similarity


def grouped_robustness(p31, y, groups):
    from sklearn.model_selection import StratifiedGroupKFold

    unique, counts = np.unique(groups, return_counts=True)
    audit = {
        "group_definition": "label-free connected components: cosine>=0.9999 and unit-L2<=0.015",
        "groups": int(len(unique)), "multi_sample_groups": int(np.sum(counts > 1)),
        "samples_in_multi_sample_groups": int(np.sum(counts[counts > 1])),
        "largest_group": int(counts.max()), "status": "NOT_EVALUABLE",
    }
    assignments = np.full(len(y), -1, int)
    if len(unique) < 10 or np.sum(counts > 1) < 2:
        dump(OUTPUT / "stage1_grouped_split_audit.json", audit)
        return
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(p31, y, groups))
    for held, (_, test) in enumerate(splits, 1):
        assignments[test] = held
    if np.any(assignments < 0):
        raise RuntimeError("grouped split incomplete")
    model_factories = {
        "G1_COUNT_ONLY": lambda f: model_lr(SEED + 500 + f),
        "G2_P31_LOGISTIC": lambda f: model_lr(SEED + 600 + f),
        "G3_DEPTH2_TREE": lambda f: DecisionTreeClassifier(max_depth=2, min_samples_leaf=10,
            class_weight="balanced", random_state=SEED + 700 + f),
        "G4_RF": lambda f: RandomForestClassifier(n_estimators=300, max_features="sqrt",
            min_samples_leaf=2, class_weight="balanced", n_jobs=4, random_state=SEED + 800 + f),
    }
    inputs = {"G1_COUNT_ONLY": count_features(p31), "G2_P31_LOGISTIC": p31,
              "G3_DEPTH2_TREE": p31, "G4_RF": p31}
    results = []
    fold_rows = []
    for name, factory in model_factories.items():
        probability = oof_predict(factory, inputs[name], y, assignments)
        results.append({"arm": name, **metric(y, probability)})
        for held in range(1, 6):
            mask = assignments == held
            fold_rows.append({"arm": name, "fold": held, **metric(y[mask], probability[mask])})
    audit["status"] = "EVALUATED"
    audit["fold_sizes"] = [int(np.sum(assignments == held)) for held in range(1, 6)]
    dump(OUTPUT / "stage1_grouped_split_audit.json", audit)
    write_csv(OUTPUT / "stage1_grouped_results.csv", results)
    write_csv(OUTPUT / "stage1_grouped_fold_results.csv", fold_rows)
    write_csv(OUTPUT / "stage1_similarity_groups.csv", [
        {"row": index, "group": int(groups[index]), "group_size": int(np.sum(groups == groups[index]))}
        for index in range(len(y))
    ])


def masking_audit(p31, y, fold):
    base = np.full(len(y), np.nan)
    masked = np.full((len(y), 31), np.nan)
    for held in range(1, 6):
        train = fold != held
        test = fold == held
        model = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=4, random_state=SEED + 900 + held)
        model.fit(p31[train], y[train])
        base[test] = model.predict_proba(p31[test])[:, 1]
        replacement = np.median(p31[train], axis=0)
        for column in range(31):
            changed = p31[test].copy()
            changed[:, column] = replacement[column]
            masked[test, column] = model.predict_proba(changed)[:, 1]
    base_metric = metric(y, base)
    rows = []
    for column in range(31):
        result = metric(y, masked[:, column])
        rows.append({"role": column, "role_name": ROLE_NAMES[column],
                     "base_BA": base_metric["BA"], "masked_BA": result["BA"],
                     "BA_drop": base_metric["BA"] - result["BA"],
                     "base_AUC": base_metric["AUC"], "masked_AUC": result["AUC"],
                     "AUC_drop": base_metric["AUC"] - result["AUC"],
                     "mean_abs_probability_change": float(np.mean(np.abs(masked[:, column] - base)))})
    write_csv(OUTPUT / "stage1_feature_masking.csv", sorted(rows, key=lambda row: row["BA_drop"], reverse=True))


def probe_pipeline(width, seed):
    components = min(31, width)
    steps = [("scale", StandardScaler())]
    if width > 31:
        steps.append(("project", PCA(n_components=components, svd_solver="full")))
    steps.append(("model", LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced",
                                                max_iter=5000, random_state=seed)))
    return Pipeline(steps)


def stage2_probe(representations, y, fold, stage1_probability, route, current_delta, ids):
    stage1_pred = stage1_probability >= 0.5
    predictions = {name: np.full(len(y), np.nan) for name in representations}
    projected = {name: {} for name in representations}
    train_sizes = []
    for held in range(1, 6):
        for side in (0, 1):
            train = np.flatnonzero((fold != held) & (stage1_pred == side))
            test = np.flatnonzero((fold == held) & (stage1_pred == side))
            train_sizes.append({"fold": held, "stage1_pred": side, "train_n": len(train),
                                "train_benign": int(np.sum(y[train] == 0)), "train_malware": int(np.sum(y[train] == 1)),
                                "test_n": len(test), "test_routed_n": int(np.sum(route[test]))})
            if len(np.unique(y[train])) < 2:
                raise RuntimeError(f"conditional train side lacks a class fold={held} side={side}")
            for name, values in representations.items():
                model = probe_pipeline(values.shape[1], SEED + held * 100 + side * 10 + len(name))
                model.fit(values[train], y[train])
                predictions[name][test] = model.predict_proba(values[test])[:, 1]
                # Store fold-safe coordinates for collision analysis.
                transformer = Pipeline(model.steps[:-1]) if len(model.steps) > 1 else None
                train_z = transformer.transform(values[train]) if transformer is not None else values[train]
                test_z = transformer.transform(values[test]) if transformer is not None else values[test]
                projected[name][(held, side)] = (train, test, train_z, test_z)
    write_csv(OUTPUT / "stage2_conditional_train_sizes.csv", train_sizes)
    for name, probability in predictions.items():
        if not np.isfinite(probability).all():
            raise RuntimeError("incomplete conditional OOF probability for " + name)

    scopes = {
        "ALL_STAGE1_SIDE": np.ones(len(y), bool),
        "DI25_ROUTED": route,
    }
    aggregate = []
    fold_rows = []
    current_probability = 1.0 / (1.0 + np.exp(-current_delta))
    all_scores = {"CURRENT_C43_DELTA": current_probability, **predictions}
    for name, probability in all_scores.items():
        for scope_name, scope in scopes.items():
            pred0 = scope & ~stage1_pred
            pred1 = scope & stage1_pred
            fn_auc = binary_auc((y[pred0] == 1).astype(int), probability[pred0])
            fp_auc = binary_auc((y[pred1] == 0).astype(int), -probability[pred1])
            error = scope & ((stage1_pred != y))
            correct_direction = np.where(y[error] == 1, probability[error] >= 0.5, probability[error] < 0.5)
            aggregate.append({"representation": name, "scope": scope_name,
                              "FN_vs_TN_AUC": fn_auc, "FP_vs_TP_AUC": fp_auc,
                              "predicted_benign_n": int(pred0.sum()), "predicted_malware_n": int(pred1.sum()),
                              "stage1_errors_n": int(error.sum()),
                              "error_direction_accuracy": float(np.mean(correct_direction)) if np.any(error) else float("nan")})
            for held in range(1, 6):
                fold_scope = scope & (fold == held)
                f0 = fold_scope & ~stage1_pred
                f1 = fold_scope & stage1_pred
                fold_rows.append({"representation": name, "scope": scope_name, "fold": held,
                                  "FN_vs_TN_AUC": binary_auc((y[f0] == 1).astype(int), probability[f0]),
                                  "FP_vs_TP_AUC": binary_auc((y[f1] == 0).astype(int), -probability[f1]),
                                  "predicted_benign_n": int(f0.sum()), "predicted_malware_n": int(f1.sum())})
    write_csv(OUTPUT / "stage2_information_ceiling.csv", aggregate)
    write_csv(OUTPUT / "stage2_information_ceiling_folds.csv", fold_rows)

    # Same-capacity shuffled-label control, one deterministic repetition per fold/side.
    null_rows = []
    for name, values in representations.items():
        null = np.full(len(y), np.nan)
        for held in range(1, 6):
            for side in (0, 1):
                train = np.flatnonzero((fold != held) & (stage1_pred == side))
                test = np.flatnonzero((fold == held) & (stage1_pred == side))
                shuffled = np.random.RandomState(SEED + 10000 + held * 100 + side).permutation(y[train])
                model = probe_pipeline(values.shape[1], SEED + 11000 + held * 100 + side)
                model.fit(values[train], shuffled)
                null[test] = model.predict_proba(values[test])[:, 1]
        for scope_name, scope in scopes.items():
            pred0 = scope & ~stage1_pred
            pred1 = scope & stage1_pred
            null_rows.append({"representation": name, "scope": scope_name,
                              "FN_vs_TN_AUC": binary_auc((y[pred0] == 1).astype(int), null[pred0]),
                              "FP_vs_TP_AUC": binary_auc((y[pred1] == 0).astype(int), -null[pred1]),
                              "control": "OUTER_TRAIN_LABEL_SHUFFLE"})
    write_csv(OUTPUT / "stage2_label_shuffle_control.csv", null_rows)

    # Fold-safe nearest-neighbour collision audit on routed validation APKs.
    collision_rows = []
    for name in representations:
        for held in range(1, 6):
            for side in (0, 1):
                train, test, train_z, test_z = projected[name][(held, side)]
                test_keep = route[test]
                selected_test = test[test_keep]
                selected_z = test_z[test_keep]
                if not len(selected_test):
                    continue
                train_norm = np.linalg.norm(train_z, axis=1, keepdims=True)
                test_norm = np.linalg.norm(selected_z, axis=1, keepdims=True)
                train_unit = np.divide(train_z, train_norm, out=np.zeros_like(train_z), where=train_norm > 1e-12)
                test_unit = np.divide(selected_z, test_norm, out=np.zeros_like(selected_z), where=test_norm > 1e-12)
                similarity = test_unit @ train_unit.T
                nearest = np.argmax(similarity, axis=1)
                for local, sample in enumerate(selected_test):
                    neighbor = train[int(nearest[local])]
                    collision_rows.append({
                        "representation": name, "sha256": ids[sample], "fold": held,
                        "label": int(y[sample]), "stage1_pred": int(stage1_pred[sample]),
                        "stage1_error": int(stage1_pred[sample] != y[sample]),
                        "neighbor_sha256": ids[neighbor], "neighbor_label": int(y[neighbor]),
                        "neighbor_label_disagrees": int(y[sample] != y[neighbor]),
                        "cosine_similarity": float(similarity[local, nearest[local]]),
                    })
    write_csv(OUTPUT / "stage2_nearest_neighbor_collisions.csv", collision_rows)
    collision_summary = []
    for name in representations:
        values = [row for row in collision_rows if row["representation"] == name]
        errors = [row for row in values if row["stage1_error"] == 1]
        collision_summary.append({
            "representation": name, "routed_rows": len(values), "routed_error_rows": len(errors),
            "nearest_label_disagreement": float(np.mean([row["neighbor_label_disagrees"] for row in values])),
            "error_nearest_label_disagreement": float(np.mean([row["neighbor_label_disagrees"] for row in errors])) if errors else float("nan"),
            "mean_nearest_similarity": float(np.mean([row["cosine_similarity"] for row in values])),
            "error_mean_nearest_similarity": float(np.mean([row["cosine_similarity"] for row in errors])) if errors else float("nan"),
        })
    write_csv(OUTPUT / "stage2_collision_summary.csv", collision_summary)
    return aggregate, fold_rows, null_rows, predictions


def main():
    rows = read_csv(FORMAL_PREDICTIONS)
    ids = [norm(row["sha256"]) for row in rows]
    y = np.asarray([int(float(row["label"])) for row in rows])
    fold = np.asarray([int(float(row["fold"])) for row in rows])
    stage1 = np.asarray([float(row[STAGE1_FIELD]) for row in rows])
    route = np.asarray([int(float(row[ROUTE_FIELD])) for row in rows], bool)
    final = probability_column(rows, "__prob")
    delta = probability_column(rows, "__bounded_delta")
    p31 = aligned_p31(ids)
    bags = aligned_host(ids)

    formal = metric(y, stage1)
    formal_final = metric(y, final)
    base_correct = (stage1 >= 0.5) == y
    final_correct = (final >= 0.5) == y
    baseline = {
        "status": "PASS" if abs(formal["BA"] - 0.927380952381) < 1e-9 else "FAIL",
        "stage1": formal, "formal_full": formal_final,
        "route_rows": int(route.sum()), "captured_errors": int(np.sum(route & ~base_correct)),
        "stage1_errors": int(np.sum(~base_correct)),
        "error_recall_at_25": float(np.sum(route & ~base_correct) / np.sum(~base_correct)),
        "route_precision": float(np.sum(route & ~base_correct) / route.sum()),
        "fixed": int(np.sum(~base_correct & final_correct)),
        "broken": int(np.sum(base_correct & ~final_correct)),
        "net": int(np.sum(~base_correct & final_correct) - np.sum(base_correct & ~final_correct)),
        "future_rows_loaded": 0,
    }
    if baseline["status"] != "PASS" or baseline["stage1_errors"] != 58 or baseline["captured_errors"] != 45:
        dump(OUTPUT / "baseline_reproduction.json", baseline)
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal baseline did not reproduce")
    dump(OUTPUT / "baseline_reproduction.json", baseline)
    print("INFORMATION_AUDIT_BASELINE_REPRODUCTION_PASS", flush=True)

    ladder = stage1_ladder(p31, y, fold, stage1)
    groups, _ = p31_similarity_groups(p31)
    grouped_robustness(p31, y, groups)
    masking_audit(p31, y, fold)
    calibration, bins = ece(y, stage1)
    dump(OUTPUT / "stage1_calibration.json", {"ECE_10": calibration, "Brier": formal["Brier"], "bins": bins})
    high = []
    for index in np.flatnonzero(((stage1 <= 0.1) | (stage1 >= 0.9)) & ~base_correct):
        high.append({"sha256": ids[index], "fold": int(fold[index]), "label": int(y[index]),
                     "stage1_probability": float(stage1[index]), "confidence": float(max(stage1[index], 1-stage1[index])),
                     "routed": int(route[index])})
    write_csv(OUTPUT / "stage1_high_confidence_errors.csv", high)
    print("STAGE1_ROBUSTNESS_AUDIT_DONE", flush=True)

    representations = aggregate_representations(p31, bags)
    aggregate, fold_results, nulls, predictions = stage2_probe(representations, y, fold, stage1, route, delta, ids)
    output_rows = []
    for index in range(len(y)):
        row = {"sha256": ids[index], "fold": int(fold[index]), "label": int(y[index]),
               "stage1_probability": float(stage1[index]), "stage1_pred": int(stage1[index] >= .5),
               "stage1_error": int(not base_correct[index]), "routed": int(route[index]),
               "current_c43_delta": float(delta[index])}
        for name, values in predictions.items():
            row[name + "__malware_probability"] = float(values[index])
        output_rows.append(row)
    write_csv(OUTPUT / "stage2_oof_probe_predictions.csv", output_rows)
    print("STAGE2_STATIC_INFORMATION_CEILING_DONE", flush=True)

    # Predeclared evidence synthesis. This does not select or integrate a model.
    real = {(row["representation"], row["scope"]): row for row in aggregate}
    null = {(row["representation"], row["scope"]): row for row in nulls}
    count = real[("COUNT_ONLY", "DI25_ROUTED")]
    candidates = []
    for name in representations:
        if name == "COUNT_ONLY":
            continue
        row = real[(name, "DI25_ROUTED")]
        control = null[(name, "DI25_ROUTED")]
        per_fold = [item for item in fold_results
                    if item["representation"] == name and item["scope"] == "DI25_ROUTED"]
        fp_valid = [item for item in per_fold if np.isfinite(float(item["FP_vs_TP_AUC"]))]
        fn_valid = [item for item in per_fold if np.isfinite(float(item["FN_vs_TN_AUC"]))]
        fp_supporting = sum(float(item["FP_vs_TP_AUC"]) > 0.5 for item in fp_valid)
        fn_supporting = sum(float(item["FN_vs_TN_AUC"]) > 0.5 for item in fn_valid)
        fp_consistent = len(fp_valid) >= 3 and fp_supporting > len(fp_valid) / 2
        fn_consistent = len(fn_valid) >= 3 and fn_supporting > len(fn_valid) / 2
        fp_signal = (row["FP_vs_TP_AUC"] >= 0.60 and
                     row["FP_vs_TP_AUC"] >= count["FP_vs_TP_AUC"] + 0.05 and
                     row["FP_vs_TP_AUC"] >= control["FP_vs_TP_AUC"] + 0.05 and
                     fp_consistent)
        fn_signal = (row["FN_vs_TN_AUC"] >= 0.55 and
                     row["FN_vs_TN_AUC"] >= count["FN_vs_TN_AUC"] - 0.02 and
                     row["FN_vs_TN_AUC"] >= control["FN_vs_TN_AUC"] + 0.05 and
                     fn_consistent)
        candidates.append({"representation": name, "FP_signal": bool(fp_signal), "FN_signal": bool(fn_signal),
                           "bidirectional_signal": bool(fp_signal and fn_signal),
                           "FP_vs_TP_AUC": row["FP_vs_TP_AUC"], "FN_vs_TN_AUC": row["FN_vs_TN_AUC"],
                           "count_FP_AUC": count["FP_vs_TP_AUC"], "count_FN_AUC": count["FN_vs_TN_AUC"],
                           "shuffle_FP_AUC": control["FP_vs_TP_AUC"], "shuffle_FN_AUC": control["FN_vs_TN_AUC"],
                           "FP_supporting_folds": fp_supporting, "FP_valid_folds": len(fp_valid),
                           "FN_supporting_folds": fn_supporting, "FN_valid_folds": len(fn_valid)})
    if any(row["bidirectional_signal"] for row in candidates):
        decision = "STATIC_BIDIRECTIONAL_INFORMATION_PRESENT"
    elif any(row["FP_signal"] or row["FN_signal"] for row in candidates):
        decision = "STATIC_INFORMATION_PARTIAL_ONE_SIDED"
    else:
        decision = "STATIC_INFORMATION_CEILING_REACHED"
    dump(OUTPUT / "information_ceiling_decision.json", {
        "decision": decision, "criteria_are_diagnostic_not_model_selection": True,
        "candidates": candidates, "next_action": (
            "Design one minimal representation-specific validation only; do not integrate correction yet."
            if decision != "STATIC_INFORMATION_CEILING_REACHED" else
            "Stop adding static Stage-II modules; retain C43 for localization/explanation or abstention only."
        ), "future_rows_loaded": 0,
    })
    dump(OUTPUT / "run_state.json", {"status": "COMPLETE", "decision": decision, "future_rows_loaded": 0})
    print("CARE_INFORMATION_AUDIT_FULL_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()

