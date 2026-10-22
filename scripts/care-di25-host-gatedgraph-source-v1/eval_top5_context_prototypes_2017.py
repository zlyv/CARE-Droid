import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


sys.path.insert(0, "scripts")

import run_behavior_core_v3_1_residual_allratio70_2017 as m


GRAPH_FILE = Path(
    "artifacts/"
    "top5-semantic-behavior-graph-v1-2017-trigger551/"
    "binding_cores_v3_1_semantic_junction_train_safe.jsonl"
)

PREDICTIONS = Path(
    "artifacts/"
    "behavior-core-v3-1-residual-allratio70-2017/"
    "predictions.csv"
)

OUTPUT = Path(
    "artifacts/"
    "top5-context-prototype-2017"
)

TOP5 = [
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
]

VARIANTS = [
    "local_core_roles",
    "anchor_method_roles",
    "combined",
]


def norm_sha(x):
    return str(x).strip().lower()


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na <= 1e-12 or nb <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b) / (na * nb)
    )


def extract_graph(x):
    nodes = x.get("nodes", [])
    edges = x.get("edges", [])

    if str(
        x.get("representation", "")
    ) != "semantic_junction_v3_1":
        raise RuntimeError(
            "发现非v3.1图"
        )

    by_id = {
        str(n["id"]): n
        for n in nodes
    }

    semantic_by_id = {}

    anchor_roles = []
    anchor_carriers = set()

    for node in nodes:
        nid = str(node["id"])
        kind = str(
            node.get("type", "")
        )

        if kind == "semantic":
            role = str(
                node.get("role", "")
            ).strip()

            if role:
                semantic_by_id[nid] = role

            if int(
                node.get(
                    "is_anchor_role",
                    0,
                )
            ):
                anchor_roles.append(role)

        elif kind == "method":
            if int(
                node.get(
                    "is_anchor_carrier",
                    0,
                )
            ):
                anchor_carriers.add(nid)

    anchor_roles = sorted(
        set(anchor_roles)
    )

    if len(anchor_roles) != 1:
        raise RuntimeError(
            "每张图应只有一个anchor role: "
            + repr(anchor_roles)
        )

    anchor = anchor_roles[0]

    if anchor not in TOP5:
        raise RuntimeError(
            "未知TOP-5 anchor: "
            + anchor
        )

    roles_by_method = defaultdict(set)

    for edge in edges:

        if str(
            edge.get("type", "")
        ) != "method_to_semantic":
            continue

        src = str(edge["source"])
        dst = str(edge["target"])

        src_node = by_id.get(src)

        if src_node is None:
            continue

        if str(
            src_node.get("type", "")
        ) != "method":
            continue

        role = semantic_by_id.get(dst)

        if role:
            roles_by_method[src].add(role)

    # 整张局部核心里的其他语义
    core_roles = (
        set(semantic_by_id.values())
        - {anchor}
    )

    # TOP-5承载方法上直接挂的其他语义
    anchor_method_roles = set()

    for mid in anchor_carriers:
        anchor_method_roles.update(
            roles_by_method.get(
                mid,
                set(),
            )
            - {anchor}
        )

    return (
        anchor,
        core_roles,
        anchor_method_roles,
    )


def load_contexts():
    """
    返回：
      contexts[sha][anchor] = {
          graph_count,
          core_counts,
          anchor_method_counts
      }

    注意：
    这里先聚合一个APK同一anchor的全部局部图，
    后面才使用APK标签。
    """

    contexts = defaultdict(
        lambda: defaultdict(
            lambda: {
                "graph_count": 0,
                "core_counts": Counter(),
                "anchor_method_counts": Counter(),
            }
        )
    )

    roles = set()

    with GRAPH_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            x = json.loads(line)

            sha = norm_sha(
                x["sha256"]
            )

            anchor, core_roles, direct_roles = (
                extract_graph(x)
            )

            rec = contexts[
                sha
            ][anchor]

            rec["graph_count"] += 1

            # 每张图中同一语义只记一次，
            # 防止节点重复影响频率。
            for role in core_roles:
                rec[
                    "core_counts"
                ][role] += 1
                roles.add(role)

            for role in direct_roles:
                rec[
                    "anchor_method_counts"
                ][role] += 1
                roles.add(role)

            roles.add(anchor)

    roles = sorted(roles)

    print(
        "semantic role vocabulary =",
        len(roles),
    )

    print(
        "context APKs =",
        len(contexts),
    )

    return contexts, roles


def make_vector(
    rec,
    anchor,
    roles,
    variant,
):
    n = int(
        rec["graph_count"]
    )

    if n <= 0:
        raise RuntimeError(
            "graph_count <= 0"
        )

    core = np.asarray(
        [
            0.0
            if role == anchor
            else (
                rec[
                    "core_counts"
                ].get(role, 0)
                / n
            )
            for role in roles
        ],
        dtype=float,
    )

    direct = np.asarray(
        [
            0.0
            if role == anchor
            else (
                rec[
                    "anchor_method_counts"
                ].get(role, 0)
                / n
            )
            for role in roles
        ],
        dtype=float,
    )

    if variant == "local_core_roles":
        return core

    if variant == "anchor_method_roles":
        return direct

    if variant == "combined":

        # 将局部核心数量压到0~1，
        # 避免数量尺度压过语义频率。
        count_feature = (
            n / (n + 1.0)
        )

        return np.concatenate(
            [
                core,
                direct,
                np.asarray(
                    [count_feature],
                    dtype=float,
                ),
            ]
        )

    raise ValueError(
        variant
    )


def label_map(
    rows,
    compact_by_sha,
):
    labels = m.base.labels_for(
        rows,
        compact_by_sha,
    )

    result = {}

    for row, y in zip(
        rows,
        labels,
    ):
        sha = norm_sha(
            row["sha256"]
        )

        result[sha] = int(y)

    return result


def safe_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(
        score,
        dtype=float,
    )

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def weighted_rate(
    mask,
    weights,
):
    mask = np.asarray(
        mask,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    total = float(
        weights.sum()
    )

    if total <= 1e-12:
        return np.nan

    return float(
        np.sum(
            mask * weights
        )
        / total
    )


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. 图上下文
    # ========================================================

    contexts, roles = (
        load_contexts()
    )

    # ========================================================
    # 2. 2017滚动划分和APK标签
    # ========================================================

    compact_rows = (
        m.read_csv_rows(
            m.COMPACT_FEATURES
        )
    )

    compact_by_sha = (
        m.base.normalized_rows(
            compact_rows
        )
    )

    rolling_parts = (
        m.base.read_split_rows(
            m.ROLLING_SPLITS
        )
    )

    rolling_splits = {
        name: parts
        for name, parts
        in rolling_parts.items()
        if (
            "rolling" in name
            and parts["train"]
            and parts["test"]
        )
    }

    print(
        "rolling splits =",
        sorted(
            rolling_splits
        ),
    )

    # ========================================================
    # 3. 每个split、每个anchor：
    #    只用训练窗口建立良性/恶性两个原型
    # ========================================================

    score_rows = []
    split_metric_rows = []

    for split_name in sorted(
        rolling_splits
    ):

        train_rows = (
            rolling_splits[
                split_name
            ]["train"]
        )

        val_rows = (
            rolling_splits[
                split_name
            ]["test"]
        )

        train_y = label_map(
            train_rows,
            compact_by_sha,
        )

        val_y = label_map(
            val_rows,
            compact_by_sha,
        )

        print()
        print(
            "=" * 80
        )
        print(
            split_name
        )
        print(
            "=" * 80
        )

        for variant in VARIANTS:

            for anchor in TOP5:

                train_items = []

                for sha, y in (
                    train_y.items()
                ):

                    rec = (
                        contexts
                        .get(sha, {})
                        .get(anchor)
                    )

                    if rec is None:
                        continue

                    v = make_vector(
                        rec,
                        anchor,
                        roles,
                        variant,
                    )

                    train_items.append(
                        (
                            sha,
                            y,
                            v,
                        )
                    )

                benign = [
                    v
                    for _, y, v
                    in train_items
                    if y == 0
                ]

                malicious = [
                    v
                    for _, y, v
                    in train_items
                    if y == 1
                ]

                # 不是调参，只是防止一个样本形成所谓原型。
                if (
                    len(benign) < 2
                    or len(malicious) < 2
                ):
                    print(
                        "SKIP",
                        variant,
                        anchor,
                        "train benign=",
                        len(benign),
                        "malicious=",
                        len(malicious),
                    )
                    continue

                benign_proto = (
                    np.mean(
                        np.stack(
                            benign
                        ),
                        axis=0,
                    )
                )

                malicious_proto = (
                    np.mean(
                        np.stack(
                            malicious
                        ),
                        axis=0,
                    )
                )

                val_scores = []
                val_labels = []

                for sha, y in (
                    val_y.items()
                ):

                    rec = (
                        contexts
                        .get(sha, {})
                        .get(anchor)
                    )

                    if rec is None:
                        continue

                    v = make_vector(
                        rec,
                        anchor,
                        roles,
                        variant,
                    )

                    sim_m = cosine(
                        v,
                        malicious_proto,
                    )

                    sim_b = cosine(
                        v,
                        benign_proto,
                    )

                    # 正数：
                    # 局部语境更靠近恶意原型
                    #
                    # 负数：
                    # 更靠近良性原型
                    score = (
                        sim_m
                        - sim_b
                    )

                    score_rows.append({
                        "split_name":
                            split_name,

                        "sha256":
                            sha,

                        "label":
                            y,

                        "variant":
                            variant,

                        "anchor":
                            anchor,

                        "score":
                            score,

                        "sim_malicious":
                            sim_m,

                        "sim_benign":
                            sim_b,

                        "graph_count":
                            rec[
                                "graph_count"
                            ],

                        "train_benign":
                            len(
                                benign
                            ),

                        "train_malicious":
                            len(
                                malicious
                            ),
                    })

                    val_scores.append(
                        score
                    )

                    val_labels.append(
                        y
                    )

                auc = safe_auc(
                    val_labels,
                    val_scores,
                )

                split_metric_rows.append({
                    "split_name":
                        split_name,

                    "variant":
                        variant,

                    "anchor":
                        anchor,

                    "train_benign":
                        len(benign),

                    "train_malicious":
                        len(malicious),

                    "val_n":
                        len(val_labels),

                    "val_benign":
                        int(
                            np.sum(
                                np.asarray(
                                    val_labels
                                ) == 0
                            )
                        ),

                    "val_malicious":
                        int(
                            np.sum(
                                np.asarray(
                                    val_labels
                                ) == 1
                            )
                        ),

                    "auc":
                        auc,
                })

    scores = pd.DataFrame(
        score_rows
    )

    split_metrics = (
        pd.DataFrame(
            split_metric_rows
        )
    )

    scores.to_csv(
        OUTPUT
        / "anchor_scores.csv",
        index=False,
    )

    split_metrics.to_csv(
        OUTPUT
        / "per_split_anchor_auc.csv",
        index=False,
    )

    # ========================================================
    # 4. 将三个验证窗口合起来，看每个anchor的可分性
    # ========================================================

    overall_rows = []

    for (
        variant,
        anchor,
    ), d in scores.groupby(
        [
            "variant",
            "anchor",
        ]
    ):

        overall_rows.append({
            "variant":
                variant,

            "anchor":
                anchor,

            "n":
                len(d),

            "benign":
                int(
                    (
                        d["label"] == 0
                    ).sum()
                ),

            "malicious":
                int(
                    (
                        d["label"] == 1
                    ).sum()
                ),

            "auc":
                safe_auc(
                    d["label"],
                    d["score"],
                ),

            "malicious_mean_score":
                float(
                    d.loc[
                        d["label"] == 1,
                        "score",
                    ].mean()
                ),

            "benign_mean_score":
                float(
                    d.loc[
                        d["label"] == 0,
                        "score",
                    ].mean()
                ),
        })

    overall = pd.DataFrame(
        overall_rows
    )

    overall.to_csv(
        OUTPUT
        / "overall_anchor_auc.csv",
        index=False,
    )

    # ========================================================
    # 5. Compact错误频率
    #
    # 不把5个随机种子当5个APK。
    # ========================================================

    pred = pd.read_csv(
        PREDICTIONS
    )

    pred["sha256"] = (
        pred["sha256"]
        .map(norm_sha)
    )

    compact = (
        pred.groupby(
            [
                "split_name",
                "sha256",
            ],
            as_index=False,
        )
        .agg(
            label=(
                "label",
                "first",
            ),

            mean_base_probability=(
                "base_probability",
                "mean",
            ),

            positive_prediction_rate=(
                "base_prediction",
                "mean",
            ),

            seeds=(
                "seed",
                "nunique",
            ),
        )
    )

    if not (
        compact["seeds"] == 5
    ).all():
        raise RuntimeError(
            "Compact预测不是每个APK 5个seed"
        )

    compact["fn_rate"] = np.where(
        compact["label"] == 1,
        (
            1.0
            - compact[
                "positive_prediction_rate"
            ]
        ),
        0.0,
    )

    compact["tp_rate"] = np.where(
        compact["label"] == 1,
        compact[
            "positive_prediction_rate"
        ],
        0.0,
    )

    compact["fp_rate"] = np.where(
        compact["label"] == 0,
        compact[
            "positive_prediction_rate"
        ],
        0.0,
    )

    compact["tn_rate"] = np.where(
        compact["label"] == 0,
        (
            1.0
            - compact[
                "positive_prediction_rate"
            ]
        ),
        0.0,
    )

    total_fn_weight = float(
        compact[
            "fn_rate"
        ].sum()
    )

    total_fp_weight = float(
        compact[
            "fp_rate"
        ].sum()
    )

    # ========================================================
    # 6. 一个APK有多个TOP-5：
    #    先对其anchor分数取均值。
    #
    #    这里只测试方向，不做最终融合。
    # ========================================================

    direction_rows = []
    apk_score_rows = []

    for variant, d in scores.groupby(
        "variant"
    ):

        apk_score = (
            d.groupby(
                [
                    "split_name",
                    "sha256",
                ],
                as_index=False,
            )
            .agg(
                context_score=(
                    "score",
                    "mean",
                ),

                anchor_count=(
                    "anchor",
                    "nunique",
                ),
            )
        )

        joined = apk_score.merge(
            compact,
            on=[
                "split_name",
                "sha256",
            ],
            how="left",
            validate="one_to_one",
        )

        joined[
            "variant"
        ] = variant

        apk_score_rows.append(
            joined
        )

        fn_w = joined[
            "fn_rate"
        ].to_numpy(
            dtype=float
        )

        fp_w = joined[
            "fp_rate"
        ].to_numpy(
            dtype=float
        )

        tp_w = joined[
            "tp_rate"
        ].to_numpy(
            dtype=float
        )

        tn_w = joined[
            "tn_rate"
        ].to_numpy(
            dtype=float
        )

        s = joined[
            "context_score"
        ].to_numpy(
            dtype=float
        )

        covered_fn = float(
            fn_w.sum()
        )

        covered_fp = float(
            fp_w.sum()
        )

        direction_rows.append({
            "variant":
                variant,

            "scored_apks":
                len(joined),

            # Compact漏报中，有多少能得到局部语境分数
            "FN_coverage":
                (
                    covered_fn
                    / total_fn_weight
                )
                if total_fn_weight > 0
                else np.nan,

            # FN希望局部证据为正
            "FN_positive_direction":
                weighted_rate(
                    s > 0,
                    fn_w,
                ),

            "FN_mean_score":
                (
                    float(
                        np.average(
                            s,
                            weights=fn_w,
                        )
                    )
                    if fn_w.sum() > 0
                    else np.nan
                ),

            # Compact误报覆盖
            "FP_coverage":
                (
                    covered_fp
                    / total_fp_weight
                )
                if total_fp_weight > 0
                else np.nan,

            # FP希望局部证据为负
            "FP_negative_direction":
                weighted_rate(
                    s < 0,
                    fp_w,
                ),

            "FP_mean_score":
                (
                    float(
                        np.average(
                            s,
                            weights=fp_w,
                        )
                    )
                    if fp_w.sum() > 0
                    else np.nan
                ),

            # TP如果得到负方向，存在把正确恶意往下拉的风险
            "TP_wrong_negative":
                weighted_rate(
                    s < 0,
                    tp_w,
                ),

            # TN如果得到正方向，存在把正确良性往上推的风险
            "TN_wrong_positive":
                weighted_rate(
                    s > 0,
                    tn_w,
                ),
        })

    direction = pd.DataFrame(
        direction_rows
    )

    all_apk_scores = pd.concat(
        apk_score_rows,
        ignore_index=True,
    )

    all_apk_scores.to_csv(
        OUTPUT
        / "apk_context_scores.csv",
        index=False,
    )

    direction.to_csv(
        OUTPUT
        / "compact_error_direction.csv",
        index=False,
    )

    # ========================================================
    # 7. 每个anchor单独看 Compact 错误方向
    # ========================================================

    anchor_direction_rows = []

    joined_anchor = scores.merge(
        compact,
        on=[
            "split_name",
            "sha256",
        ],
        how="left",
        validate="many_to_one",
    )

    for (
        variant,
        anchor,
    ), d in joined_anchor.groupby(
        [
            "variant",
            "anchor",
        ]
    ):

        s = d[
            "score"
        ].to_numpy(
            dtype=float
        )

        fn_w = d[
            "fn_rate"
        ].to_numpy(
            dtype=float
        )

        fp_w = d[
            "fp_rate"
        ].to_numpy(
            dtype=float
        )

        anchor_direction_rows.append({
            "variant":
                variant,

            "anchor":
                anchor,

            "rows":
                len(d),

            "FN_weight":
                float(
                    fn_w.sum()
                ),

            "FN_positive_direction":
                weighted_rate(
                    s > 0,
                    fn_w,
                ),

            "FP_weight":
                float(
                    fp_w.sum()
                ),

            "FP_negative_direction":
                weighted_rate(
                    s < 0,
                    fp_w,
                ),
        })

    anchor_direction = pd.DataFrame(
        anchor_direction_rows
    )

    anchor_direction.to_csv(
        OUTPUT
        / "anchor_error_direction.csv",
        index=False,
    )

    # ========================================================
    # 8. 输出重点
    # ========================================================

    print()
    print("=" * 90)
    print("ANCHOR PROTOTYPE SEPARABILITY")
    print("=" * 90)

    print(
        overall.sort_values(
            [
                "variant",
                "auc",
            ],
            ascending=[
                True,
                False,
            ],
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("COMPACT ERROR DIRECTION")
    print("=" * 90)

    print(
        direction.to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("PER-ANCHOR ERROR DIRECTION")
    print("=" * 90)

    print(
        anchor_direction.sort_values(
            [
                "variant",
                "anchor",
            ]
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "saved =",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
