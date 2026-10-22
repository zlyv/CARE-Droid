import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import run_sensitive_behavior_hypergraph_2017_eval as hyper


E_VARIANT = "E_local_bounded_additive"
ENDPOINT_CACHE = Path(
    "artifacts/p25-endpoint-cache-full-cap256/"
    "endpoint_centered_tensor_cache.jsonl.gz"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "APK-level temporal audit: test whether TOP-5 semantic context adds "
            "information beyond Compact probability and graph quantity."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "artifacts/bounded-additive-residual-3way-2017/predictions.csv"
        ),
    )
    parser.add_argument(
        "--graphs",
        type=Path,
        default=Path(
            "artifacts/top5-semantic-behavior-graph-v1-2017-trigger551/"
            "binding_cores_v3_1_semantic_junction_train_safe.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/top5-context-incremental-audit-2017"),
    )
    parser.add_argument(
        "--min-train-support",
        type=int,
        default=2,
        help="Minimum number of training APKs containing a sparse semantic feature.",
    )
    return parser.parse_args()


def norm_sha(value):
    return str(value).strip().lower()


def safe_logit(probability):
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def load_apk_table(path):
    data = pd.read_csv(path, dtype={"sha256": str})
    required = {
        "variant",
        "split_name",
        "seed",
        "sha256",
        "label",
        "graph_enabled",
        "graph_count",
        "base_probability",
        "base_prediction",
    }
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError("predictions.csv缺少字段: " + repr(sorted(missing)))

    data = data[data["variant"] == E_VARIANT].copy()
    if data.empty:
        raise RuntimeError("找不到E变体: " + E_VARIANT)
    data["sha256"] = data["sha256"].map(norm_sha)

    apk = (
        data.groupby(["split_name", "sha256"], as_index=False)
        .agg(
            label=("label", "first"),
            seed_count=("seed", "nunique"),
            graph_enabled=("graph_enabled", "first"),
            graph_count_prediction=("graph_count", "first"),
            base_probability=("base_probability", "mean"),
            base_prediction=("base_prediction", "first"),
        )
    )
    inconsistent = (
        data.groupby(["split_name", "sha256"])
        .agg(
            label_n=("label", "nunique"),
            graph_enabled_n=("graph_enabled", "nunique"),
            base_prediction_n=("base_prediction", "nunique"),
        )
        .reset_index()
    )
    bad = inconsistent[
        (inconsistent["label_n"] != 1)
        | (inconsistent["graph_enabled_n"] != 1)
        | (inconsistent["base_prediction_n"] != 1)
    ]
    if not bad.empty:
        raise RuntimeError(
            "同一APK跨种子的固定字段不一致:\n" + bad.head(20).to_string(index=False)
        )

    routed = apk[
        (apk["graph_enabled"].astype(int) == 1)
        & (apk["graph_count_prediction"].astype(float) > 0)
        & (apk["base_prediction"].astype(int) == 0)
    ].copy()
    if routed.empty:
        raise RuntimeError("没有找到 graph_enabled 且 Compact 判为阴性的APK")
    return routed


def extract_graph_context(graph):
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    by_id = {str(node.get("id")): node for node in nodes}

    semantic_by_id = {}
    anchor_roles = set()
    anchor_carriers = set()
    for node in nodes:
        node_id = str(node.get("id"))
        node_type = str(node.get("type", ""))
        if node_type == "semantic":
            role = str(node.get("role", "")).strip()
            if role:
                semantic_by_id[node_id] = role
                if int(node.get("is_anchor_role", 0)):
                    anchor_roles.add(role)
        elif node_type == "method" and int(node.get("is_anchor_carrier", 0)):
            anchor_carriers.add(node_id)

    if len(anchor_roles) != 1:
        raise RuntimeError(
            "每张图应有且仅有一个TOP-5 anchor: " + repr(sorted(anchor_roles))
        )
    anchor = next(iter(anchor_roles))

    roles_by_method = defaultdict(set)
    structural_out = defaultdict(set)
    for edge in edges:
        edge_type = str(edge.get("type", ""))
        if edge_type == "structural":
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            if source in by_id and target in by_id:
                structural_out[source].add(target)
            continue
        if edge_type != "method_to_semantic":
            continue
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        source_node = by_id.get(source)
        if not source_node or str(source_node.get("type", "")) != "method":
            continue
        role = semantic_by_id.get(target)
        if role:
            roles_by_method[source].add(role)

    core_roles = set(semantic_by_id.values()) - {anchor}
    direct_roles = set()
    for method_id in anchor_carriers:
        direct_roles.update(roles_by_method.get(method_id, set()))
    direct_roles.discard(anchor)

    # Anchor-relative semantic relation on the normalized directed DAG.
    # This deliberately avoids raw method names and raw hop counts. It asks only
    # whether a semantic carrier is the anchor carrier itself, upstream,
    # downstream, or on a sibling branch under the binding root.
    def reachable_from(start):
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for target in structural_out.get(current, ()):
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return seen

    method_ids = {
        node_id
        for node_id, node in by_id.items()
        if str(node.get("type", "")) == "method"
    }
    reach_by_method = {
        method_id: reachable_from(method_id)
        for method_id in method_ids
    }
    relation_features = set()
    for method_id, roles in roles_by_method.items():
        for role in roles:
            if role == anchor:
                continue
            for anchor_method in anchor_carriers:
                if method_id == anchor_method:
                    relation = "same_method"
                elif method_id in reach_by_method.get(anchor_method, set()):
                    relation = "downstream"
                elif anchor_method in reach_by_method.get(method_id, set()):
                    relation = "upstream"
                else:
                    relation = "sibling_branch"
                relation_features.add((relation, role))

    return {
        "anchor": anchor,
        "core_roles": core_roles,
        "direct_roles": direct_roles,
        "relation_features": relation_features,
        "method_count": sum(
            str(node.get("type", "")) == "method" for node in nodes
        ),
        "semantic_role_count": len(set(semantic_by_id.values())),
        "edge_count": len(edges),
    }


def load_graphs(path, target_shas):
    result = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            sha = norm_sha(item.get("sha256", ""))
            if sha in target_shas:
                result[sha].append(extract_graph_context(item))
    return result


def increment(counter, key):
    counter[key] += 1


def build_raw_features(apk, graphs_by_sha):
    rows = []
    for item in apk.itertuples(index=False):
        contexts = graphs_by_sha.get(item.sha256, [])
        if not contexts:
            continue

        counts = Counter()
        for context in contexts:
            anchor = context["anchor"]
            increment(counts, "anchor::" + anchor)
            for role in context["core_roles"]:
                increment(counts, "core::" + role)
                increment(counts, "anchor_core::" + anchor + "::" + role)
            for role in context["direct_roles"]:
                increment(counts, "direct::" + role)
                increment(counts, "anchor_direct::" + anchor + "::" + role)
            for relation, role in context["relation_features"]:
                increment(
                    counts,
                    "anchor_relation::" + anchor + "::" + relation + "::" + role,
                )

        graph_count = len(contexts)
        rows.append(
            {
                "split_name": item.split_name,
                "sha256": item.sha256,
                "label": int(item.label),
                "base_probability": float(item.base_probability),
                "graph_count_prediction": float(item.graph_count_prediction),
                "graph_count_loaded": graph_count,
                "mean_method_count": float(
                    np.mean([x["method_count"] for x in contexts])
                ),
                "mean_semantic_role_count": float(
                    np.mean([x["semantic_role_count"] for x in contexts])
                ),
                "mean_edge_count": float(
                    np.mean([x["edge_count"] for x in contexts])
                ),
                "feature_counts": counts,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("路由APK没有加载到局部图")
    return result


def split_order(names):
    def key(name):
        text = str(name)
        marker = "rolling_2017_"
        if marker in text:
            suffix = text.split(marker, 1)[1].split("_", 1)[0]
            try:
                return int(suffix), text
            except ValueError:
                pass
        return 10**9, text

    return sorted(names, key=key)


def allowed_sparse_features(train, block, min_support):
    prefixes = {
        "R0_count_anchor": ("anchor::",),
        "H1_shared_caller": ("anchor::",),
        "H2_compressed_reachability": ("anchor::",),
        "H3_behavior_hypergraph": ("anchor::",),
        "R1_core_context": ("anchor::", "core::"),
        "R2_anchor_interaction": (
            "anchor::",
            "core::",
            "direct::",
            "anchor_core::",
            "anchor_direct::",
        ),
        "R3_directional_relation": (
            "anchor::",
            "core::",
            "direct::",
            "anchor_core::",
            "anchor_direct::",
            "anchor_relation::",
        ),
    }[block]
    support = Counter()
    for counts in train["feature_counts"]:
        for feature in counts:
            if feature.startswith(prefixes):
                support[feature] += 1
    return {
        feature for feature, count in support.items() if count >= min_support
    }


def make_feature_dict(row, block, allowed):
    graph_count = max(1, int(row.graph_count_loaded))
    features = {
        "numeric::compact_logit": safe_logit(row.base_probability),
        "numeric::graph_count_log1p": math.log1p(graph_count),
        "numeric::mean_method_count_log1p": math.log1p(row.mean_method_count),
        "numeric::mean_edge_count_log1p": math.log1p(row.mean_edge_count),
    }
    if block in {
        "R1_core_context",
        "R2_anchor_interaction",
        "R3_directional_relation",
    }:
        features["numeric::mean_semantic_role_count"] = row.mean_semantic_role_count

    for feature, count in row.feature_counts.items():
        if feature not in allowed:
            continue
        features["present::" + feature] = 1.0
        features["rate::" + feature] = float(count) / graph_count
    return features


def metric_value(y_true, probability):
    prediction = (np.asarray(probability) >= 0.5).astype(int)
    result = {
        "BA": balanced_accuracy_score(y_true, prediction),
        "ACC": float(np.mean(prediction == np.asarray(y_true))),
    }
    if len(set(y_true)) == 2:
        result["AUC"] = roc_auc_score(y_true, probability)
    else:
        result["AUC"] = np.nan
    return result, prediction


def hypergraph_slice(block):
    width = hyper.RELATION_DIMENSION
    if block == "H1_shared_caller":
        return slice(0, width)
    if block == "H2_compressed_reachability":
        return slice(width, 2 * width)
    if block == "H3_behavior_hypergraph":
        return slice(0, 2 * width)
    return None


def fit_predict(train, test, block, min_support):
    hyper_slice = hypergraph_slice(block)
    base_block = "R0_count_anchor" if hyper_slice is not None else block

    allowed = allowed_sparse_features(train, base_block, min_support)
    train_dicts = [
        make_feature_dict(row, base_block, allowed)
        for row in train.itertuples(index=False)
    ]
    test_dicts = [
        make_feature_dict(row, base_block, allowed)
        for row in test.itertuples(index=False)
    ]

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(train_dicts)
    x_test = vectorizer.transform(test_dicts)

    if hyper_slice is not None:
        hyper_train = np.stack(
            train["hypergraph_vector"].to_list()
        )[:, hyper_slice]
        hyper_test = np.stack(
            test["hypergraph_vector"].to_list()
        )[:, hyper_slice]

        # Exactly the same train-only standardization as the P46 experiment.
        hyper_train, hyper_test = hyper.standardize_pair(
            hyper_train,
            hyper_test,
        )
        x_train = sparse.hstack(
            [x_train, sparse.csr_matrix(hyper_train)],
            format="csr",
        )
        x_test = sparse.hstack(
            [x_test, sparse.csr_matrix(hyper_test)],
            format="csr",
        )

    y_train = train["label"].astype(int).to_numpy()
    if len(set(y_train)) != 2:
        raise RuntimeError(
            "历史训练APK只包含一个类别，无法拟合: " + block
        )

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        solver="liblinear",
        random_state=0,
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    return probability, x_train.shape[1], len(allowed)


def evaluate_forward(features, min_support):
    splits = split_order(features["split_name"].unique())
    if len(splits) < 3:
        raise RuntimeError("至少需要3个按时间排序的验证切分")

    blocks = [
        "R0_count_anchor",
        "R1_core_context",
        "R2_anchor_interaction",
        "R3_directional_relation",
        "H1_shared_caller",
        "H2_compressed_reachability",
        "H3_behavior_hypergraph",
    ]
    prediction_rows = []
    metric_rows = []

    # Split 1 seeds the historical context learner. Splits 2 and 3 are untouched
    # forward tests. No feature support or vocabulary is learned from the test split.
    for test_index in range(1, len(splits)):
        test_split = splits[test_index]
        train_splits = splits[:test_index]
        train = features[features["split_name"].isin(train_splits)].copy()
        test = features[features["split_name"] == test_split].copy()
        y_test = test["label"].astype(int).to_numpy()

        if len(set(y_test)) != 2:
            raise RuntimeError("测试切分只包含一个类别: " + str(test_split))

        split_probabilities = {}
        for block in blocks:
            probability, dimensions, sparse_kept = fit_predict(
                train, test, block, min_support
            )
            split_probabilities[block] = probability
            metrics, prediction = metric_value(y_test, probability)
            metric_rows.append(
                {
                    "test_split": test_split,
                    "train_splits": ";".join(train_splits),
                    "block": block,
                    "train_apks": len(train),
                    "test_apks": len(test),
                    "train_malware": int(train["label"].sum()),
                    "test_malware": int(test["label"].sum()),
                    "dimensions": dimensions,
                    "sparse_features_kept": sparse_kept,
                    **metrics,
                }
            )

            for row, prob, pred in zip(
                test.itertuples(index=False), probability, prediction
            ):
                prediction_rows.append(
                    {
                        "test_split": test_split,
                        "sha256": row.sha256,
                        "label": row.label,
                        "base_probability": row.base_probability,
                        "graph_count": row.graph_count_loaded,
                        "block": block,
                        "probability": prob,
                        "prediction": int(pred),
                    }
                )

        r0_pred = (split_probabilities["R0_count_anchor"] >= 0.5).astype(int)
        for block in blocks[1:]:
            candidate_pred = (split_probabilities[block] >= 0.5).astype(int)
            corrected = int(np.sum((r0_pred != y_test) & (candidate_pred == y_test)))
            regressed = int(np.sum((r0_pred == y_test) & (candidate_pred != y_test)))
            for row in metric_rows:
                if row["test_split"] == test_split and row["block"] == block:
                    row["corrected_vs_R0"] = corrected
                    row["regressed_vs_R0"] = regressed

    metrics = pd.DataFrame(metric_rows)
    metrics["delta_BA_vs_R0"] = metrics["BA"] - metrics.groupby(
        "test_split"
    )["BA"].transform("first")
    metrics["delta_AUC_vs_R0"] = metrics["AUC"] - metrics.groupby(
        "test_split"
    )["AUC"].transform("first")
    return metrics, pd.DataFrame(prediction_rows)


def build_summary(metrics):
    rows = []
    for block, part in metrics.groupby("block", sort=False):
        rows.append(
            {
                "block": block,
                "evaluated_splits": len(part),
                "mean_BA": part["BA"].mean(),
                "min_BA": part["BA"].min(),
                "mean_AUC": part["AUC"].mean(),
                "min_AUC": part["AUC"].min(),
                "mean_delta_BA_vs_R0": part["delta_BA_vs_R0"].mean(),
                "mean_delta_AUC_vs_R0": part["delta_AUC_vs_R0"].mean(),
                "total_corrected_vs_R0": part["corrected_vs_R0"].fillna(0).sum(),
                "total_regressed_vs_R0": part["regressed_vs_R0"].fillna(0).sum(),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    routed = load_apk_table(args.predictions)
    graphs_by_sha = load_graphs(args.graphs, set(routed["sha256"]))
    features = build_raw_features(routed, graphs_by_sha)

    endpoint_cache = {
        norm_sha(key): value
        for key, value in hyper.read_cache(
            ENDPOINT_CACHE.expanduser().resolve(strict=True)
        ).items()
    }
    target_shas = set(features["sha256"])
    missing_endpoint = sorted(target_shas - set(endpoint_cache))
    if missing_endpoint:
        raise RuntimeError(
            "endpoint cache缺少路由APK: " + repr(missing_endpoint[:20])
        )

    hyper_vectors = {}
    for sha in sorted(target_shas):
        vector, _ = hyper.sensitive_behavior_hypergraph_vector(
            endpoint_cache[sha]["payload"]
        )
        hyper_vectors[sha] = vector
    features["hypergraph_vector"] = [
        hyper_vectors[sha] for sha in features["sha256"]
    ]

    missing = routed[~routed["sha256"].isin(features["sha256"])].copy()
    if not missing.empty:
        missing.to_csv(args.output / "routed_apks_missing_graphs.csv", index=False)

    metrics, predictions = evaluate_forward(features, args.min_train_support)
    summary = build_summary(metrics)

    apk_export = features.drop(
        columns=["feature_counts", "hypergraph_vector"]
    ).copy()
    apk_export.to_csv(args.output / "routed_apks.csv", index=False)
    metrics.to_csv(args.output / "per_split_metrics.csv", index=False)
    predictions.to_csv(args.output / "forward_predictions.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)

    print("=" * 100)
    print("TOP-5 CONTEXT INCREMENTAL TEMPORAL AUDIT")
    print("=" * 100)
    print("predictions =", args.predictions)
    print("graphs      =", args.graphs)
    print("output      =", args.output)
    print("\nROUTED APK COVERAGE BY SPLIT AND LABEL")
    print(
        features.groupby(["split_name", "label"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    if not missing.empty:
        print("\nWARNING: routed APKs missing graphs =", len(missing))
    print("\nPER-SPLIT FORWARD RESULTS")
    print(metrics.to_string(index=False))
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(
        "\nInterpretation rule: only claim semantic value if R1/R2 improve over R0 "
        "on both forward splits (especially min BA/AUC), with corrections not "
        "offset by regressions. Split 1 is training history, not a test result."
    )
    print(
        "All graph features are aggregated per APK. Test labels, vocabulary and "
        "feature support are never used during training."
    )


if __name__ == "__main__":
    main()
