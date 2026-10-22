import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact
except Exception:
    fisher_exact = None


E_VARIANT = "E_local_bounded_additive"
USEFUL_MIN_SEEDS = 4
HARMFUL_MIN_SEEDS = 3
OLD_GATE_BOUND = 0.30115


def norm_sha(value):
    return str(value).strip().lower()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "artifacts/bounded-additive-residual-3way-2017/"
            "predictions.csv"
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
        default=Path(
            "artifacts/stable-top5-context-contrast-2017"
        ),
    )
    return parser.parse_args()


def select_stable_apks(predictions):
    required = {
        "variant", "split_name", "seed", "sha256", "label",
        "graph_enabled", "graph_count", "base_probability",
        "base_prediction", "final_probability", "final_prediction",
        "actual_correction",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise RuntimeError("predictions.csv缺少字段: " + repr(sorted(missing)))

    data = predictions[predictions["variant"] == E_VARIANT].copy()
    if data.empty:
        raise RuntimeError("找不到E变体: " + E_VARIANT)

    data["sha256"] = data["sha256"].map(norm_sha)
    data["seed"] = data["seed"].astype(int)

    apk = (
        data.groupby(["split_name", "sha256"], as_index=False)
        .agg(
            label=("label", "first"),
            seed_count=("seed", "nunique"),
            graph_enabled=("graph_enabled", "first"),
            graph_count_prediction=("graph_count", "first"),
            base_probability=("base_probability", "mean"),
            base_prediction=("base_prediction", "first"),
            final_positive_seeds=("final_prediction", "sum"),
            mean_final_probability=("final_probability", "mean"),
            mean_actual_correction=("actual_correction", "mean"),
        )
    )

    if not (apk["seed_count"] == 5).all():
        bad = apk[apk["seed_count"] != 5]
        raise RuntimeError(
            "部分APK不是5个种子:\n" + bad.head(20).to_string(index=False)
        )

    useful = (
        (apk["label"] == 1)
        & (apk["base_prediction"] == 0)
        & (apk["final_positive_seeds"] >= USEFUL_MIN_SEEDS)
    )
    harmful = (
        (apk["label"] == 0)
        & (apk["base_prediction"] == 0)
        & (apk["final_positive_seeds"] >= HARMFUL_MIN_SEEDS)
    )

    selected = apk[useful | harmful].copy()
    selected["group"] = np.where(
        useful.loc[selected.index],
        "stable_useful_FN_fix",
        "stable_harmful_TN_to_FP",
    )
    selected["high_confidence_compact_FN"] = (
        (selected["group"] == "stable_useful_FN_fix")
        & (selected["base_probability"] < OLD_GATE_BOUND)
    ).astype(int)
    selected["flip_rate"] = selected["final_positive_seeds"] / 5.0

    if len(selected) < 10 or selected["group"].nunique() != 2:
        raise RuntimeError(
            "稳定对照组过小或缺少一组:\n"
            + selected.groupby("group").size().to_string()
        )

    return selected.sort_values(["group", "split_name", "sha256"])


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
    for edge in edges:
        if str(edge.get("type", "")) != "method_to_semantic":
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

    method_count = sum(
        str(node.get("type", "")) == "method" for node in nodes
    )
    return {
        "anchor": anchor,
        "core_roles": core_roles,
        "direct_roles": direct_roles,
        "method_count": method_count,
        "semantic_role_count": len(set(semantic_by_id.values())),
        "edge_count": len(edges),
    }


def load_selected_graphs(graph_path, target_shas):
    graphs = defaultdict(list)
    with graph_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            sha = norm_sha(item.get("sha256", ""))
            if sha not in target_shas:
                continue
            graphs[sha].append(extract_graph_context(item))
    return graphs


def add_feature(counter, family, name):
    counter[(family, name)] += 1


def aggregate_contexts(selected, graphs_by_sha):
    apk_feature_rows = []
    apk_summary_rows = []
    feature_maps = {}

    for row in selected.itertuples(index=False):
        contexts = graphs_by_sha.get(row.sha256, [])
        if not contexts:
            raise RuntimeError("稳定样本没有局部图: " + row.sha256)

        counts = Counter()
        anchors = Counter()
        all_core_roles = set()
        all_direct_roles = set()

        for context in contexts:
            anchor = context["anchor"]
            anchors[anchor] += 1
            add_feature(counts, "anchor", anchor)

            for role in context["core_roles"]:
                all_core_roles.add(role)
                add_feature(counts, "core_role", role)
                add_feature(counts, "anchor_core", anchor + "::" + role)

            for role in context["direct_roles"]:
                all_direct_roles.add(role)
                add_feature(counts, "direct_role", role)
                add_feature(counts, "anchor_direct", anchor + "::" + role)

        key = (row.split_name, row.sha256)
        feature_maps[key] = counts
        total_graphs = len(contexts)

        for (family, feature), hits in sorted(counts.items()):
            apk_feature_rows.append({
                "split_name": row.split_name,
                "sha256": row.sha256,
                "group": row.group,
                "feature_family": family,
                "feature": feature,
                "graph_hits": hits,
                "graph_rate": hits / total_graphs,
                "present": 1,
            })

        apk_summary_rows.append({
            **row._asdict(),
            "graph_count_loaded": total_graphs,
            "mean_method_count": float(np.mean([x["method_count"] for x in contexts])),
            "mean_semantic_role_count": float(
                np.mean([x["semantic_role_count"] for x in contexts])
            ),
            "mean_edge_count": float(np.mean([x["edge_count"] for x in contexts])),
            "anchors": ";".join(
                f"{name}:{count}" for name, count in sorted(anchors.items())
            ),
            "core_roles": ";".join(sorted(all_core_roles)),
            "direct_roles": ";".join(sorted(all_direct_roles)),
        })

    return (
        pd.DataFrame(apk_summary_rows),
        pd.DataFrame(apk_feature_rows),
        feature_maps,
    )


def build_feature_contrast(selected, feature_maps):
    useful_name = "stable_useful_FN_fix"
    harmful_name = "stable_harmful_TN_to_FP"
    group_by_key = {
        (row.split_name, row.sha256): row.group
        for row in selected.itertuples(index=False)
    }
    keys_by_group = {
        useful_name: [key for key, group in group_by_key.items() if group == useful_name],
        harmful_name: [key for key, group in group_by_key.items() if group == harmful_name],
    }

    all_features = sorted({
        feature
        for counts in feature_maps.values()
        for feature in counts
    })
    rows = []

    for family, feature in all_features:
        useful_keys = keys_by_group[useful_name]
        harmful_keys = keys_by_group[harmful_name]
        useful_hits = sum(feature_maps[key][(family, feature)] > 0 for key in useful_keys)
        harmful_hits = sum(feature_maps[key][(family, feature)] > 0 for key in harmful_keys)
        useful_absent = len(useful_keys) - useful_hits
        harmful_absent = len(harmful_keys) - harmful_hits
        support = useful_hits + harmful_hits
        if support < 2:
            continue

        useful_prev = useful_hits / len(useful_keys)
        harmful_prev = harmful_hits / len(harmful_keys)
        odds_useful = (useful_hits + 0.5) / (useful_absent + 0.5)
        odds_harmful = (harmful_hits + 0.5) / (harmful_absent + 0.5)
        log2_odds = math.log2(odds_useful / odds_harmful)

        useful_rates = [
            feature_maps[key][(family, feature)]
            / max(1, sum(v for (fam, _), v in feature_maps[key].items() if fam == "anchor"))
            for key in useful_keys
        ]
        harmful_rates = [
            feature_maps[key][(family, feature)]
            / max(1, sum(v for (fam, _), v in feature_maps[key].items() if fam == "anchor"))
            for key in harmful_keys
        ]

        if fisher_exact is None:
            fisher_p = np.nan
        else:
            _, fisher_p = fisher_exact(
                [[useful_hits, useful_absent], [harmful_hits, harmful_absent]],
                alternative="two-sided",
            )

        rows.append({
            "feature_family": family,
            "feature": feature,
            "support_apks": support,
            "useful_present": useful_hits,
            "useful_total": len(useful_keys),
            "useful_prevalence": useful_prev,
            "harmful_present": harmful_hits,
            "harmful_total": len(harmful_keys),
            "harmful_prevalence": harmful_prev,
            "prevalence_difference": useful_prev - harmful_prev,
            "log2_odds_useful": log2_odds,
            "useful_mean_graph_rate": float(np.mean(useful_rates)),
            "harmful_mean_graph_rate": float(np.mean(harmful_rates)),
            "mean_graph_rate_difference": float(
                np.mean(useful_rates) - np.mean(harmful_rates)
            ),
            "fisher_p_descriptive": fisher_p,
            "direction": (
                "useful_context"
                if useful_prev > harmful_prev
                else "harmful_context"
                if useful_prev < harmful_prev
                else "neutral"
            ),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("没有可比较的语境特征")
    return result.sort_values(
        ["prevalence_difference", "support_apks"],
        ascending=[False, False],
    )


def print_top(title, data, direction, n=20):
    subset = data[data["direction"] == direction].copy()
    if direction == "harmful_context":
        subset = subset.sort_values(
            ["prevalence_difference", "support_apks"],
            ascending=[True, False],
        )
    else:
        subset = subset.sort_values(
            ["prevalence_difference", "support_apks"],
            ascending=[False, False],
        )
    columns = [
        "feature_family", "feature", "support_apks",
        "useful_prevalence", "harmful_prevalence",
        "prevalence_difference", "log2_odds_useful",
        "fisher_p_descriptive",
    ]
    print("\n" + title)
    print(subset[columns].head(n).to_string(index=False))


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions, dtype={"sha256": str})
    selected = select_stable_apks(predictions)
    target_shas = set(selected["sha256"])
    graphs_by_sha = load_selected_graphs(args.graphs, target_shas)

    missing_graphs = sorted(target_shas - set(graphs_by_sha))
    if missing_graphs:
        raise RuntimeError("这些稳定APK没有图: " + repr(missing_graphs[:20]))

    apk_summary, apk_features, feature_maps = aggregate_contexts(
        selected,
        graphs_by_sha,
    )
    contrast = build_feature_contrast(selected, feature_maps)

    group_summary = (
        apk_summary.groupby("group", as_index=False)
        .agg(
            apks=("sha256", "size"),
            high_confidence_compact_FN=("high_confidence_compact_FN", "sum"),
            mean_base_probability=("base_probability", "mean"),
            mean_flip_rate=("flip_rate", "mean"),
            mean_graph_count=("graph_count_loaded", "mean"),
            median_graph_count=("graph_count_loaded", "median"),
            mean_method_count=("mean_method_count", "mean"),
            mean_semantic_role_count=("mean_semantic_role_count", "mean"),
        )
    )

    apk_summary.to_csv(args.output / "stable_apks.csv", index=False)
    apk_features.to_csv(args.output / "apk_context_features_long.csv", index=False)
    contrast.to_csv(args.output / "context_feature_contrast.csv", index=False)
    contrast[contrast["feature_family"] == "anchor"].to_csv(
        args.output / "anchor_contrast.csv",
        index=False,
    )
    group_summary.to_csv(args.output / "group_summary.csv", index=False)

    print("=" * 100)
    print("STABLE TOP-5 CONTEXT CONTRAST")
    print("=" * 100)
    print("predictions =", args.predictions)
    print("graphs =", args.graphs)
    print("output =", args.output)
    print("\nGROUP SUMMARY")
    print(group_summary.to_string(index=False))
    print("\nGROUPS BY SPLIT")
    print(
        apk_summary.groupby(["split_name", "group"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print_top(
        "TOP USEFUL-CONTEXT FEATURES",
        contrast,
        "useful_context",
    )
    print_top(
        "TOP HARMFUL-CONTEXT FEATURES",
        contrast,
        "harmful_context",
    )

    print("\n注意：Fisher p值仅作小样本描述，未做多重比较校正。")
    print("所有语境特征均先在APK内聚合，再使用APK级组别；没有给子图继承APK标签。")


if __name__ == "__main__":
    main()
