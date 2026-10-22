from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, "scripts")

import eval_top5_context_prototypes_2017 as proto
import run_behavior_core_v3_1_residual_allratio70_2017 as m


OUTPUT = Path(
    "artifacts/"
    "top5-context-local-neighbor-diagnosis-2017"
)

OLD_PROTO = Path(
    "artifacts/"
    "top5-context-prototype-2017/"
    "overall_anchor_auc.csv"
)

KS = [1, 3, 5]

# 三种表示都保留，避免人为挑结果。
VARIANTS = [
    "local_core_roles",
    "anchor_method_roles",
    "combined",
]


def norm_sha(x):
    return str(x).strip().lower()


def normalize_rows(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    n = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    n[n < 1e-12] = 1.0

    return x / n


def safe_auc(y, score):

    y = np.asarray(
        y,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    if len(np.unique(y)) < 2:
        return np.nan

    return proto.safe_auc(
        y,
        score,
    )


def weighted_rate(mask, weights):

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


def topk_class_score(
    similarity,
    labels,
    target_label,
    k,
):
    """
    每个验证样本：
    取训练集中指定类别最相似的k个APK，
    返回平均相似度。
    """

    mask = (
        labels == target_label
    )

    class_sim = similarity[
        :,
        mask,
    ]

    if class_sim.shape[1] < k:
        return None

    # 从小到大排序，取最后k个
    part = np.partition(
        class_sim,
        class_sim.shape[1] - k,
        axis=1,
    )

    topk = part[
        :,
        -k:,
    ]

    return np.mean(
        topk,
        axis=1,
    )


def build_compact_error_table():

    pred = pd.read_csv(
        proto.PREDICTIONS
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
            "存在不是5个seed的APK"
        )

    compact["fn_rate"] = np.where(
        compact["label"] == 1,
        1.0
        - compact[
            "positive_prediction_rate"
        ],
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
        1.0
        - compact[
            "positive_prediction_rate"
        ],
        0.0,
    )

    return compact


def train_neighborhood_diagnostic(
    X,
    y,
):
    """
    只在训练窗口内部做留一诊断：

    对每个APK：
      最近同类相似度
      最近异类相似度

    如果同一个类别本身存在多个局部模式，
    单中心可能不好，但局部最近邻仍可能很好。
    """

    Xn = normalize_rows(X)

    sim = Xn @ Xn.T

    np.fill_diagonal(
        sim,
        -np.inf,
    )

    same_scores = []
    other_scores = []
    margins = []
    wins = []

    y = np.asarray(
        y,
        dtype=int,
    )

    for i in range(len(y)):

        same = (
            y == y[i]
        )

        same[i] = False

        other = (
            y != y[i]
        )

        if (
            not np.any(same)
            or not np.any(other)
        ):
            continue

        s_same = float(
            np.max(
                sim[i, same]
            )
        )

        s_other = float(
            np.max(
                sim[i, other]
            )
        )

        same_scores.append(
            s_same
        )

        other_scores.append(
            s_other
        )

        margins.append(
            s_same - s_other
        )

        wins.append(
            int(
                s_same > s_other
            )
        )

    return {
        "train_n":
            len(y),

        "nearest_same_mean":
            float(
                np.mean(
                    same_scores
                )
            )
            if same_scores
            else np.nan,

        "nearest_other_mean":
            float(
                np.mean(
                    other_scores
                )
            )
            if other_scores
            else np.nan,

        "nearest_margin_mean":
            float(
                np.mean(
                    margins
                )
            )
            if margins
            else np.nan,

        "same_class_neighbor_win_rate":
            float(
                np.mean(
                    wins
                )
            )
            if wins
            else np.nan,
    }


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. 读取局部语境
    # ========================================================

    contexts, roles = (
        proto.load_contexts()
    )

    # ========================================================
    # 2. 2017滚动划分
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

    compact_error = (
        build_compact_error_table()
    )

    # ========================================================
    # 3. 每个split / TOP-5 / 表示
    # ========================================================

    score_rows = []
    split_auc_rows = []
    neighborhood_rows = []

    for split_name in sorted(
        rolling_splits
    ):

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

        train_y_map = (
            proto.label_map(
                train_rows,
                compact_by_sha,
            )
        )

        val_y_map = (
            proto.label_map(
                val_rows,
                compact_by_sha,
            )
        )

        for variant in VARIANTS:

            for anchor in proto.TOP5:

                train_items = []

                for sha, y in (
                    train_y_map.items()
                ):

                    rec = (
                        contexts
                        .get(sha, {})
                        .get(anchor)
                    )

                    if rec is None:
                        continue

                    v = proto.make_vector(
                        rec,
                        anchor,
                        roles,
                        variant,
                    )

                    train_items.append(
                        (
                            sha,
                            int(y),
                            v,
                        )
                    )

                val_items = []

                for sha, y in (
                    val_y_map.items()
                ):

                    rec = (
                        contexts
                        .get(sha, {})
                        .get(anchor)
                    )

                    if rec is None:
                        continue

                    v = proto.make_vector(
                        rec,
                        anchor,
                        roles,
                        variant,
                    )

                    val_items.append(
                        (
                            sha,
                            int(y),
                            v,
                        )
                    )

                if (
                    not train_items
                    or not val_items
                ):
                    continue

                X_train = np.stack(
                    [
                        x[2]
                        for x in train_items
                    ]
                )

                y_train = np.asarray(
                    [
                        x[1]
                        for x in train_items
                    ],
                    dtype=int,
                )

                X_val = np.stack(
                    [
                        x[2]
                        for x in val_items
                    ]
                )

                y_val = np.asarray(
                    [
                        x[1]
                        for x in val_items
                    ],
                    dtype=int,
                )

                # --------------------------------------------
                # 训练窗口内部：
                # 最近同类 vs 最近异类
                # --------------------------------------------

                diag = (
                    train_neighborhood_diagnostic(
                        X_train,
                        y_train,
                    )
                )

                neighborhood_rows.append({
                    "split_name":
                        split_name,

                    "variant":
                        variant,

                    "anchor":
                        anchor,

                    "train_benign":
                        int(
                            np.sum(
                                y_train == 0
                            )
                        ),

                    "train_malicious":
                        int(
                            np.sum(
                                y_train == 1
                            )
                        ),

                    **diag,
                })

                # --------------------------------------------
                # 验证集 vs 训练集
                # --------------------------------------------

                X_train_n = (
                    normalize_rows(
                        X_train
                    )
                )

                X_val_n = (
                    normalize_rows(
                        X_val
                    )
                )

                similarity = (
                    X_val_n
                    @ X_train_n.T
                )

                for k in KS:

                    score_m = (
                        topk_class_score(
                            similarity,
                            y_train,
                            1,
                            k,
                        )
                    )

                    score_b = (
                        topk_class_score(
                            similarity,
                            y_train,
                            0,
                            k,
                        )
                    )

                    if (
                        score_m is None
                        or score_b is None
                    ):
                        print(
                            "SKIP k=",
                            k,
                            variant,
                            anchor,
                            "benign=",
                            int(
                                np.sum(
                                    y_train == 0
                                )
                            ),
                            "malicious=",
                            int(
                                np.sum(
                                    y_train == 1
                                )
                            ),
                        )
                        continue

                    score = (
                        score_m
                        - score_b
                    )

                    auc = safe_auc(
                        y_val,
                        score,
                    )

                    split_auc_rows.append({
                        "split_name":
                            split_name,

                        "variant":
                            variant,

                        "anchor":
                            anchor,

                        "k":
                            k,

                        "train_benign":
                            int(
                                np.sum(
                                    y_train == 0
                                )
                            ),

                        "train_malicious":
                            int(
                                np.sum(
                                    y_train == 1
                                )
                            ),

                        "val_n":
                            len(
                                val_items
                            ),

                        "auc":
                            auc,
                    })

                    for (
                        sha,
                        y,
                        _
                    ), sm, sb, s in zip(
                        val_items,
                        score_m,
                        score_b,
                        score,
                    ):

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

                            "k":
                                k,

                            "malicious_neighbor_similarity":
                                float(sm),

                            "benign_neighbor_similarity":
                                float(sb),

                            "score":
                                float(s),
                        })

    scores = pd.DataFrame(
        score_rows
    )

    per_split_auc = pd.DataFrame(
        split_auc_rows
    )

    neighborhood = pd.DataFrame(
        neighborhood_rows
    )

    scores.to_csv(
        OUTPUT
        / "validation_neighbor_scores.csv",
        index=False,
    )

    per_split_auc.to_csv(
        OUTPUT
        / "per_split_neighbor_auc.csv",
        index=False,
    )

    neighborhood.to_csv(
        OUTPUT
        / "train_neighborhood_summary.csv",
        index=False,
    )

    # ========================================================
    # 4. 汇总验证AUC
    # ========================================================

    overall_rows = []

    for (
        variant,
        anchor,
        k,
    ), d in scores.groupby(
        [
            "variant",
            "anchor",
            "k",
        ]
    ):

        split_part = (
            per_split_auc[
                (
                    per_split_auc[
                        "variant"
                    ] == variant
                )
                & (
                    per_split_auc[
                        "anchor"
                    ] == anchor
                )
                & (
                    per_split_auc[
                        "k"
                    ] == k
                )
            ]
        )

        overall_rows.append({
            "variant":
                variant,

            "anchor":
                anchor,

            "k":
                k,

            "n":
                len(d),

            "pooled_auc":
                safe_auc(
                    d["label"],
                    d["score"],
                ),

            "mean_split_auc":
                float(
                    split_part[
                        "auc"
                    ].mean()
                ),

            "min_split_auc":
                float(
                    split_part[
                        "auc"
                    ].min()
                ),
        })

    overall = pd.DataFrame(
        overall_rows
    )

    # ========================================================
    # 5. 与上一轮“单中心原型”直接比较
    # ========================================================

    if OLD_PROTO.exists():

        old = pd.read_csv(
            OLD_PROTO
        )[
            [
                "variant",
                "anchor",
                "auc",
            ]
        ].rename(
            columns={
                "auc":
                    "single_prototype_auc"
            }
        )

        overall = overall.merge(
            old,
            on=[
                "variant",
                "anchor",
            ],
            how="left",
        )

        overall[
            "neighbor_minus_prototype"
        ] = (
            overall[
                "pooled_auc"
            ]
            - overall[
                "single_prototype_auc"
            ]
        )

    overall.to_csv(
        OUTPUT
        / "prototype_vs_neighbors.csv",
        index=False,
    )

    # ========================================================
    # 6. Compact错误方向
    #
    # 两种多锚点聚合：
    #
    # mean:
    #   和上一轮一样，所有TOP-5平均
    #
    # strongest:
    #   取绝对值最大的那个锚点，
    #   保留它自己的正负方向。
    #
    # strongest不使用真实标签。
    # ========================================================

    direction_rows = []
    apk_score_rows = []

    total_fn = float(
        compact_error[
            "fn_rate"
        ].sum()
    )

    total_fp = float(
        compact_error[
            "fp_rate"
        ].sum()
    )

    for (
        variant,
        k,
    ), d in scores.groupby(
        [
            "variant",
            "k",
        ]
    ):

        # --------------------------------------------
        # 平均所有锚点
        # --------------------------------------------

        mean_score = (
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

        mean_score[
            "aggregation"
        ] = "mean"

        # --------------------------------------------
        # 取最强锚点
        # --------------------------------------------

        strongest = d.copy()

        strongest[
            "abs_score"
        ] = strongest[
            "score"
        ].abs()

        strongest = (
            strongest
            .sort_values(
                "abs_score"
            )
            .groupby(
                [
                    "split_name",
                    "sha256",
                ],
                as_index=False,
            )
            .tail(1)
        )

        strongest = strongest[
            [
                "split_name",
                "sha256",
                "score",
                "anchor",
            ]
        ].rename(
            columns={
                "score":
                    "context_score",

                "anchor":
                    "strongest_anchor",
            }
        )

        strongest[
            "anchor_count"
        ] = np.nan

        strongest[
            "aggregation"
        ] = "strongest"

        for part in [
            mean_score,
            strongest,
        ]:

            joined = part.merge(
                compact_error,
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

            joined[
                "k"
            ] = k

            apk_score_rows.append(
                joined
            )

            s = joined[
                "context_score"
            ].to_numpy(
                dtype=float
            )

            fn_w = joined[
                "fn_rate"
            ].to_numpy(
                dtype=float
            )

            tp_w = joined[
                "tp_rate"
            ].to_numpy(
                dtype=float
            )

            fp_w = joined[
                "fp_rate"
            ].to_numpy(
                dtype=float
            )

            tn_w = joined[
                "tn_rate"
            ].to_numpy(
                dtype=float
            )

            direction_rows.append({
                "variant":
                    variant,

                "k":
                    k,

                "aggregation":
                    part[
                        "aggregation"
                    ].iloc[0],

                "scored_apks":
                    len(joined),

                "FN_coverage":
                    (
                        float(
                            fn_w.sum()
                        )
                        / total_fn
                    )
                    if total_fn > 0
                    else np.nan,

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

                "FP_coverage":
                    (
                        float(
                            fp_w.sum()
                        )
                        / total_fp
                    )
                    if total_fp > 0
                    else np.nan,

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

                "TP_wrong_negative":
                    weighted_rate(
                        s < 0,
                        tp_w,
                    ),

                "TN_wrong_positive":
                    weighted_rate(
                        s > 0,
                        tn_w,
                    ),
            })

    direction = pd.DataFrame(
        direction_rows
    )

    apk_scores = pd.concat(
        apk_score_rows,
        ignore_index=True,
    )

    direction.to_csv(
        OUTPUT
        / "compact_error_direction.csv",
        index=False,
    )

    apk_scores.to_csv(
        OUTPUT
        / "apk_neighbor_scores.csv",
        index=False,
    )

    # ========================================================
    # 7. 每个TOP-5单独看错误方向
    # ========================================================

    joined_anchor = scores.merge(
        compact_error,
        on=[
            "split_name",
            "sha256",
        ],
        how="left",
        validate="many_to_one",
    )

    anchor_direction_rows = []

    for (
        variant,
        anchor,
        k,
    ), d in joined_anchor.groupby(
        [
            "variant",
            "anchor",
            "k",
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

            "k":
                k,

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
    # 8. 控制台输出
    # ========================================================

    print()
    print("=" * 90)
    print("单中心原型 VS 局部邻域")
    print("=" * 90)

    show_cols = [
        "variant",
        "anchor",
        "k",
        "pooled_auc",
        "mean_split_auc",
        "min_split_auc",
    ]

    if (
        "single_prototype_auc"
        in overall.columns
    ):
        show_cols += [
            "single_prototype_auc",
            "neighbor_minus_prototype",
        ]

    print(
        overall[
            show_cols
        ]
        .sort_values(
            [
                "variant",
                "anchor",
                "k",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("训练窗口局部邻域")
    print("=" * 90)

    neighborhood_mean = (
        neighborhood.groupby(
            [
                "variant",
                "anchor",
            ],
            as_index=False,
        )
        .agg(
            nearest_same_mean=(
                "nearest_same_mean",
                "mean",
            ),

            nearest_other_mean=(
                "nearest_other_mean",
                "mean",
            ),

            nearest_margin_mean=(
                "nearest_margin_mean",
                "mean",
            ),

            same_class_neighbor_win_rate=(
                "same_class_neighbor_win_rate",
                "mean",
            ),
        )
    )

    print(
        neighborhood_mean
        .sort_values(
            [
                "variant",
                "same_class_neighbor_win_rate",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("Compact错误方向")
    print("=" * 90)

    print(
        direction
        .sort_values(
            [
                "variant",
                "k",
                "aggregation",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("每个TOP-5的错误方向")
    print("=" * 90)

    print(
        anchor_direction
        .sort_values(
            [
                "variant",
                "anchor",
                "k",
            ]
        )
        .to_string(
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
