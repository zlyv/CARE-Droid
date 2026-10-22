#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_top5_context_incremental_2017 as audit
import run_sensitive_behavior_hypergraph_2017_eval as hyper


YEARS = (2018, 2019, 2020, 2021)
SEEDS = (0, 1, 2, 3, 4)
THRESHOLD = 0.5
MIN_TRAIN_SUPPORT = 2

SOURCE_PREDICTIONS = (
    ROOT
    / "artifacts/bounded-additive-residual-3way-2017/predictions.csv"
)
SOURCE_GRAPH = (
    ROOT
    / "artifacts/top5-semantic-behavior-graph-v1-2017-trigger551/"
    "binding_cores_v3_1_semantic_junction_train_safe.jsonl"
)
ENDPOINT_CACHE = (
    ROOT
    / "artifacts/p25-endpoint-cache-full-cap256/"
    "endpoint_centered_tensor_cache.jsonl.gz"
)
E_FUTURE_PREDICTIONS = (
    ROOT
    / "artifacts/bounded-additive-residual-3way-2017-2021-formal/"
    "predictions.csv"
)
FUTURE_GRAPH = {
    year: (
        ROOT
        / f"artifacts/top5-semantic-behavior-graph-v1-{year}-triggered/"
        "binding_cores_v3_1_semantic_junction_train_safe.jsonl"
    )
    for year in YEARS
}
OUTPUT = (
    ROOT
    / "artifacts/bounded-additive-h1-veto-2017-2021-formal"
)


def metric_dict(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "BA": balanced_accuracy_score(labels, predictions),
        "ACC": float(np.mean(labels == predictions)),
        "AUC": roc_auc_score(labels, probabilities),
        "FP": int(np.sum((labels == 0) & (predictions == 1))),
        "FN": int(np.sum((labels == 1) & (predictions == 0))),
    }


def load_endpoint_cache():
    return {
        audit.norm_sha(key): value
        for key, value in hyper.read_cache(
            ENDPOINT_CACHE.resolve(strict=True)
        ).items()
    }


def attach_shared_vectors(features, endpoint_cache):
    features = features.copy()
    vectors = []

    for sha in features["sha256"]:
        if sha not in endpoint_cache:
            raise RuntimeError("endpoint cache缺少APK: " + sha)

        vector, _ = hyper.sensitive_behavior_hypergraph_vector(
            endpoint_cache[sha]["payload"]
        )
        vectors.append(
            np.asarray(
                vector[:hyper.RELATION_DIMENSION],
                dtype=np.float64,
            )
        )

    features["shared_vector"] = vectors
    return features


def train_h1(endpoint_cache):
    # 与2017 forward audit完全相同的路由和五种子概率聚合。
    routed = audit.load_apk_table(SOURCE_PREDICTIONS)
    graphs = audit.load_graphs(
        SOURCE_GRAPH,
        set(routed["sha256"]),
    )
    features = audit.build_raw_features(routed, graphs)
    features = attach_shared_vectors(features, endpoint_cache)

    if len(features) != 213:
        raise RuntimeError(
            f"2017 H1 routed APK数量变化: {len(features)} != 213"
        )
    if int(features["label"].sum()) != 28:
        raise RuntimeError(
            "2017 H1 routed malware数量变化: "
            f"{int(features['label'].sum())} != 28"
        )

    allowed = audit.allowed_sparse_features(
        features,
        "R0_count_anchor",
        MIN_TRAIN_SUPPORT,
    )
    dictionaries = [
        audit.make_feature_dict(
            row,
            "R0_count_anchor",
            allowed,
        )
        for row in features.itertuples(index=False)
    ]

    vectorizer = DictVectorizer(sparse=True)
    base_matrix = vectorizer.fit_transform(dictionaries)

    shared = np.stack(features["shared_vector"].to_list())
    mean = shared.mean(axis=0)
    scale = shared.std(axis=0)
    scale[scale < 1e-12] = 1.0
    shared = (shared - mean) / scale

    matrix = sparse.hstack(
        [base_matrix, sparse.csr_matrix(shared)],
        format="csr",
    )
    labels = features["label"].astype(int).to_numpy()

    if len(allowed) != 5:
        raise RuntimeError(
            f"2017 anchor feature数量变化: {len(allowed)} != 5"
        )
    if matrix.shape[1] != 134:
        raise RuntimeError(
            f"2017 H1维数变化: {matrix.shape[1]} != 134"
        )

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        solver="liblinear",
        random_state=0,
    )
    model.fit(matrix, labels)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "allowed": allowed,
        "mean": mean,
        "scale": scale,
        "training_apks": len(features),
        "training_malware": int(labels.sum()),
        "dimensions": int(matrix.shape[1]),
    }


def transform_h1(features, bundle):
    dictionaries = [
        audit.make_feature_dict(
            row,
            "R0_count_anchor",
            bundle["allowed"],
        )
        for row in features.itertuples(index=False)
    ]
    base_matrix = bundle["vectorizer"].transform(dictionaries)

    shared = np.stack(features["shared_vector"].to_list())
    shared = (
        shared - bundle["mean"]
    ) / bundle["scale"]

    return sparse.hstack(
        [base_matrix, sparse.csr_matrix(shared)],
        format="csr",
    )


def build_future_h1_map(future_rows, bundle, endpoint_cache):
    work = future_rows.copy()
    work["sha256"] = work["sha256"].map(audit.norm_sha)
    work["base_prediction"] = (
        work["base_probability"].astype(float) >= THRESHOLD
    ).astype(int)

    consistency = (
        work.groupby(["year", "sha256"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            base_prediction_count=("base_prediction", "nunique"),
            graph_enabled_count=("graph_enabled", "nunique"),
        )
    )
    if (
        (consistency["seed_count"] != 5).any()
        or (consistency["base_prediction_count"] != 1).any()
        or (consistency["graph_enabled_count"] != 1).any()
    ):
        raise RuntimeError("未来Compact路由字段跨seed不一致")

    apk = (
        work.groupby(["year", "sha256"], as_index=False)
        .agg(
            base_probability=("base_probability", "mean"),
            graph_enabled=("graph_enabled", "first"),
            base_prediction=("base_prediction", "first"),
        )
    )

    h1_map = {}
    coverage_rows = []

    for year in YEARS:
        routed = apk[
            (apk["year"] == year)
            & (apk["graph_enabled"].astype(int) == 1)
            & (apk["base_prediction"].astype(int) == 0)
        ].copy()

        if routed.empty:
            coverage_rows.append({
                "year": year,
                "routed_apks": 0,
                "h1_apks": 0,
            })
            continue

        raw = pd.DataFrame({
            "split_name": [str(year)] * len(routed),
            "sha256": routed["sha256"].tolist(),
            "label": [0] * len(routed),
            "seed_count": [5] * len(routed),
            "graph_enabled": [1] * len(routed),
            "graph_count_prediction": [1.0] * len(routed),
            "base_probability": routed["base_probability"].tolist(),
            "base_prediction": [0] * len(routed),
        })

        target_shas = set(raw["sha256"])
        graphs = audit.load_graphs(
            FUTURE_GRAPH[year],
            target_shas,
        )
        features = audit.build_raw_features(raw, graphs)

        missing = target_shas - set(features["sha256"])
        if missing:
            raise RuntimeError(
                f"{year} routed APK缺少图: "
                + repr(sorted(missing)[:20])
            )

        features = attach_shared_vectors(
            features,
            endpoint_cache,
        )
        matrix = transform_h1(features, bundle)
        probabilities = bundle["model"].predict_proba(matrix)[:, 1]

        for sha, probability in zip(
            features["sha256"],
            probabilities,
        ):
            h1_map[(year, sha)] = float(probability)

        coverage_rows.append({
            "year": year,
            "routed_apks": len(routed),
            "h1_apks": len(features),
        })

    return h1_map, pd.DataFrame(coverage_rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading endpoint cache...", flush=True)
    endpoint_cache = load_endpoint_cache()

    print("Training locked H1 on 2017...", flush=True)
    bundle = train_h1(endpoint_cache)
    print(
        "H1 training:",
        bundle["training_apks"],
        "APKs,",
        bundle["training_malware"],
        "malware,",
        bundle["dimensions"],
        "dimensions",
        flush=True,
    )

    # 此阶段明确不读取未来label列。
    prediction_columns = [
        "year",
        "seed",
        "sha256",
        "graph_enabled",
        "base_probability",
        "delta",
        "final_probability",
    ]
    future = pd.read_csv(
        E_FUTURE_PREDICTIONS,
        usecols=prediction_columns,
        dtype={"sha256": str},
    )
    future["sha256"] = future["sha256"].map(audit.norm_sha)

    observed_years = tuple(sorted(future["year"].unique()))
    if observed_years != YEARS:
        raise RuntimeError(
            f"未来年份不一致: {observed_years} != {YEARS}"
        )

    print("Building future H1 probabilities...", flush=True)
    h1_map, coverage = build_future_h1_map(
        future,
        bundle,
        endpoint_cache,
    )

    h1_probability = []
    soft_probability = []
    vetoed = []

    for row in future.itertuples(index=False):
        key = (int(row.year), row.sha256)
        probability = h1_map.get(key, np.nan)

        e_probability = float(row.final_probability)
        e_prediction = int(e_probability >= THRESHOLD)

        apply_veto = (
            not np.isnan(probability)
            and probability < THRESHOLD
            and e_prediction == 1
        )

        final_probability = (
            min(e_probability, probability)
            if apply_veto
            else e_probability
        )

        h1_probability.append(probability)
        soft_probability.append(final_probability)
        vetoed.append(int(apply_veto))

    predicted = future.copy()
    predicted["h1_probability"] = h1_probability
    predicted["h1_available"] = (
        predicted["h1_probability"].notna().astype(int)
    )
    predicted["e_prediction"] = (
        predicted["final_probability"] >= THRESHOLD
    ).astype(int)
    predicted["h1_vetoed"] = vetoed
    predicted["soft_probability"] = soft_probability
    predicted["soft_prediction"] = (
        predicted["soft_probability"] >= THRESHOLD
    ).astype(int)

    print("Future predictions complete; loading labels for metrics.", flush=True)

    label_rows = pd.read_csv(
        E_FUTURE_PREDICTIONS,
        usecols=["year", "seed", "sha256", "label"],
        dtype={"sha256": str},
    )
    label_rows["sha256"] = label_rows["sha256"].map(audit.norm_sha)

    predicted = predicted.merge(
        label_rows,
        on=["year", "seed", "sha256"],
        how="left",
        validate="one_to_one",
    )
    if predicted["label"].isna().any():
        raise RuntimeError("未来预测缺少标签")

    per_seed_rows = []

    for (year, seed), part in predicted.groupby(
        ["year", "seed"],
        sort=True,
    ):
        labels = part["label"].astype(int).to_numpy()
        compact_probability = (
            part["base_probability"].astype(float).to_numpy()
        )
        e_probability = (
            part["final_probability"].astype(float).to_numpy()
        )
        soft_probability = (
            part["soft_probability"].astype(float).to_numpy()
        )

        compact_metrics = metric_dict(labels, compact_probability)
        e_metrics = metric_dict(labels, e_probability)
        soft_metrics = metric_dict(labels, soft_probability)

        e_prediction = (e_probability >= THRESHOLD).astype(int)
        soft_prediction = (soft_probability >= THRESHOLD).astype(int)

        corrected_fp = int(np.sum(
            (labels == 0)
            & (e_prediction == 1)
            & (soft_prediction == 0)
        ))
        regressed_tp = int(np.sum(
            (labels == 1)
            & (e_prediction == 1)
            & (soft_prediction == 0)
        ))

        per_seed_rows.append({
            "year": int(year),
            "seed": int(seed),
            "n": len(part),
            "h1_available": int(part["h1_available"].sum()),
            "vetoed": int(part["h1_vetoed"].sum()),
            **{
                "compact_" + key: value
                for key, value in compact_metrics.items()
            },
            **{
                "e_" + key: value
                for key, value in e_metrics.items()
            },
            **{
                "h1_" + key: value
                for key, value in soft_metrics.items()
            },
            "delta_BA_vs_E": (
                soft_metrics["BA"] - e_metrics["BA"]
            ),
            "delta_AUC_vs_E": (
                soft_metrics["AUC"] - e_metrics["AUC"]
            ),
            "corrected_FP_vs_E": corrected_fp,
            "regressed_TP_vs_E": regressed_tp,
        })

    per_seed = pd.DataFrame(per_seed_rows)
    numeric_columns = [
        column
        for column in per_seed.columns
        if column not in {"year", "seed"}
    ]
    per_year = (
        per_seed.groupby("year", as_index=False)[numeric_columns]
        .mean()
    )

    overall = {
        "future_years": len(per_year),
        "compact_mean_BA": per_year["compact_BA"].mean(),
        "e_mean_BA": per_year["e_BA"].mean(),
        "h1_mean_BA": per_year["h1_BA"].mean(),
        "delta_mean_BA_vs_E": per_year["delta_BA_vs_E"].mean(),
        "e_min_BA": per_year["e_BA"].min(),
        "h1_min_BA": per_year["h1_BA"].min(),
        "e_mean_ACC": per_year["e_ACC"].mean(),
        "h1_mean_ACC": per_year["h1_ACC"].mean(),
        "e_mean_AUC": per_year["e_AUC"].mean(),
        "h1_mean_AUC": per_year["h1_AUC"].mean(),
        "delta_mean_AUC_vs_E": per_year["delta_AUC_vs_E"].mean(),
        "e_min_AUC": per_year["e_AUC"].min(),
        "h1_min_AUC": per_year["h1_AUC"].min(),
        "e_mean_FP": per_year["e_FP"].mean(),
        "h1_mean_FP": per_year["h1_FP"].mean(),
        "e_mean_FN": per_year["e_FN"].mean(),
        "h1_mean_FN": per_year["h1_FN"].mean(),
        "total_mean_corrected_FP_vs_E": (
            per_year["corrected_FP_vs_E"].sum()
        ),
        "total_mean_regressed_TP_vs_E": (
            per_year["regressed_TP_vs_E"].sum()
        ),
    }
    summary = pd.DataFrame([overall])

    predicted.to_csv(OUTPUT / "predictions.csv", index=False)
    coverage.to_csv(OUTPUT / "coverage.csv", index=False)
    per_seed.to_csv(OUTPUT / "per_seed.csv", index=False)
    per_year.to_csv(OUTPUT / "per_year.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)

    np.savez(
        OUTPUT / "h1_model.npz",
        coefficients=bundle["model"].coef_,
        intercept=bundle["model"].intercept_,
        shared_mean=bundle["mean"],
        shared_scale=bundle["scale"],
    )

    with (OUTPUT / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "source_year": 2017,
                "future_years": list(YEARS),
                "variant": "H1_shared_caller_benign_veto",
                "training_apks": bundle["training_apks"],
                "training_malware": bundle["training_malware"],
                "dimensions": bundle["dimensions"],
                "min_train_support": MIN_TRAIN_SUPPORT,
                "threshold": THRESHOLD,
                "fusion": (
                    "if E positive and H1 negative: "
                    "min(E_probability, H1_probability)"
                ),
                "future_labels_used_for_training": False,
                "future_labels_used_for_prediction": False,
                "future_labels_used_for_metrics_only": True,
            },
            handle,
            indent=2,
        )

    print("\nCOVERAGE")
    print(coverage.to_string(index=False))
    print("\nPER YEAR")
    print(per_year.to_string(index=False))
    print("\nOVERALL")
    print(summary.to_string(index=False))
    print("\nsaved =", OUTPUT)


if __name__ == "__main__":
    main()
