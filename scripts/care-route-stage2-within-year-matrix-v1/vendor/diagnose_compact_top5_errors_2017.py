from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, "scripts")

import run_behavior_core_v3_1_residual_allratio70_2017 as m


# ============================================================
# 配置
# ============================================================

PREDICTIONS = Path(
    "artifacts/"
    "behavior-core-v3-1-residual-allratio70-2017/"
    "predictions.csv"
)

OUTPUT = Path(
    "artifacts/"
    "compact-top5-error-diagnosis-2017"
)

TOP5 = [
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
]

THRESHOLD = 0.5


def norm_sha(x):
    return str(x).strip().lower()


def safe_number(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return 0.0


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. 读取已经保存好的 Compact 验证集预测
    # ========================================================

    pred = pd.read_csv(
        PREDICTIONS
    )

    required = {
        "split_name",
        "seed",
        "sha256",
        "label",
        "base_probability",
        "base_prediction",
        "graph_enabled",
        "graph_count",
    }

    missing = (
        required
        - set(pred.columns)
    )

    if missing:
        raise RuntimeError(
            "predictions.csv 缺少字段: "
            + repr(
                sorted(missing)
            )
        )

    pred["sha256"] = (
        pred["sha256"]
        .map(norm_sha)
    )

    pred["label"] = (
        pred["label"]
        .astype(int)
    )

    pred["base_prediction"] = (
        pred["base_prediction"]
        .astype(int)
    )

    pred["base_probability"] = (
        pred["base_probability"]
        .astype(float)
    )

    # 再确认阈值确实是0.5
    check_prediction = (
        pred["base_probability"]
        >= THRESHOLD
    ).astype(int)

    if not np.array_equal(
        check_prediction.values,
        pred["base_prediction"].values,
    ):
        raise RuntimeError(
            "base_prediction 与阈值0.5不一致"
        )

    # ========================================================
    # 2. 读取原始 Compact 特征，只用于判断 TOP-5 是否存在
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

    all_columns = set()

    for row in compact_by_sha.values():
        all_columns.update(
            row.keys()
        )

    role_columns = {}

    for role in TOP5:

        candidates = [
            f"role_node__{role}",
            f"role_edge__{role}",
        ]

        existing = [
            c
            for c in candidates
            if c in all_columns
        ]

        if not existing:
            raise RuntimeError(
                "找不到TOP-5字段: "
                + role
            )

        role_columns[
            role
        ] = existing

    print(
        "TOP-5 source columns:"
    )

    for role in TOP5:
        print(
            role,
            "->",
            role_columns[role],
        )

    # ========================================================
    # 3. 给每个预测样本附加 TOP-5 信息
    # ========================================================

    top5_rows = []

    unique_shas = sorted(
        pred["sha256"].unique()
    )

    for sha in unique_shas:

        row = compact_by_sha.get(
            sha
        )

        if row is None:
            raise RuntimeError(
                "Compact表找不到SHA: "
                + sha
            )

        item = {
            "sha256": sha,
        }

        present_roles = []

        for role in TOP5:

            present = any(
                safe_number(
                    row.get(
                        col,
                        0,
                    )
                ) > 0
                for col in
                role_columns[role]
            )

            item[
                "top5__" + role
            ] = int(
                present
            )

            if present:
                present_roles.append(
                    role
                )

        item["top5_present"] = int(
            len(present_roles) > 0
        )

        item["top5_role_count"] = (
            len(
                present_roles
            )
        )

        item["top5_roles"] = (
            ";".join(
                present_roles
            )
        )

        top5_rows.append(
            item
        )

    top5_df = pd.DataFrame(
        top5_rows
    )

    pred = pred.merge(
        top5_df,
        on="sha256",
        how="left",
        validate="many_to_one",
    )

    # 图存在却没有TOP-5是不合理的
    inconsistent = pred[
        (pred["graph_enabled"] == 1)
        & (pred["top5_present"] == 0)
    ]

    if len(inconsistent):
        raise RuntimeError(
            "发现 graph_enabled=1 "
            "但 top5_present=0 的样本"
        )

    # ========================================================
    # 4. Compact 错误类型
    # ========================================================

    y = pred["label"]
    p = pred["base_prediction"]

    pred["error_type"] = "TN"

    pred.loc[
        (y == 1) & (p == 1),
        "error_type",
    ] = "TP"

    pred.loc[
        (y == 1) & (p == 0),
        "error_type",
    ] = "FN"

    pred.loc[
        (y == 0) & (p == 1),
        "error_type",
    ] = "FP"

    pred["is_error"] = (
        pred["label"]
        != pred["base_prediction"]
    ).astype(int)

    # 离0.5的距离
    pred["margin_to_05"] = (
        pred["base_probability"]
        - 0.5
    ).abs()

    # 当前残差公式真正允许的纠正强度系数
    pred["uncertainty_gate"] = (
        4.0
        * pred["base_probability"]
        * (
            1.0
            - pred["base_probability"]
        )
    )

    # ========================================================
    # 5. 概率分桶
    # ========================================================

    bins = [
        -np.inf,
        0.20,
        0.35,
        0.50,
        0.65,
        0.80,
        np.inf,
    ]

    names = [
        "[0,0.20)",
        "[0.20,0.35)",
        "[0.35,0.50)",
        "[0.50,0.65)",
        "[0.65,0.80)",
        "[0.80,1]",
    ]

    pred[
        "probability_bucket"
    ] = pd.cut(
        pred[
            "base_probability"
        ],
        bins=bins,
        labels=names,
        right=False,
    )

    # ========================================================
    # 6. 路由区域
    # ========================================================

    pred[
        "routing_group"
    ] = "no_top5"

    pred.loc[
        (pred["top5_present"] == 1)
        & (pred["graph_enabled"] == 0),
        "routing_group",
    ] = "top5_no_valid_graph"

    pred.loc[
        pred["graph_enabled"] == 1,
        "routing_group",
    ] = "top5_valid_graph"

    # ========================================================
    # 7. 保存完整逐预测表
    # ========================================================

    pred.to_csv(
        OUTPUT
        / "prediction_error_map.csv",
        index=False,
    )

    # ========================================================
    # 8. 每个 APK 聚合5个随机种子
    #
    # 不把5个随机种子误当5个独立APK。
    # ========================================================

    agg_dict = {
        "label": "first",
        "top5_present": "first",
        "top5_role_count": "first",
        "top5_roles": "first",
        "graph_enabled": "first",
        "graph_count": "first",
        "base_probability": "mean",
        "base_prediction": "mean",
        "is_error": "mean",
        "uncertainty_gate": "mean",
    }

    for role in TOP5:
        agg_dict[
            "top5__" + role
        ] = "first"

    apk = (
        pred.groupby(
            [
                "split_name",
                "sha256",
            ],
            as_index=False,
        )
        .agg(
            agg_dict
        )
    )

    apk = apk.rename(
        columns={
            "base_probability":
                "mean_base_probability",

            "base_prediction":
                "positive_prediction_rate",

            "is_error":
                "error_rate_across_seeds",

            "uncertainty_gate":
                "mean_uncertainty_gate",
        }
    )

    apk.to_csv(
        OUTPUT
        / "apk_error_stability.csv",
        index=False,
    )

    # ========================================================
    # 9. 总体错误
    # ========================================================

    error_counts = (
        pred.groupby(
            "error_type"
        )
        .size()
        .reset_index(
            name="prediction_rows"
        )
    )

    error_counts.to_csv(
        OUTPUT
        / "error_type_counts.csv",
        index=False,
    )

    # ========================================================
    # 10. 路由区域错误率
    # ========================================================

    routing_rows = []

    for group, d in pred.groupby(
        "routing_group"
    ):

        malicious = d[
            d["label"] == 1
        ]

        benign = d[
            d["label"] == 0
        ]

        routing_rows.append({
            "routing_group":
                group,

            "prediction_rows":
                len(d),

            "unique_apks":
                d[
                    [
                        "split_name",
                        "sha256",
                    ]
                ]
                .drop_duplicates()
                .shape[0],

            "error_rate":
                float(
                    d["is_error"].mean()
                ),

            "malicious_rows":
                len(malicious),

            "fn_rate":
                float(
                    (
                        malicious[
                            "error_type"
                        ]
                        == "FN"
                    ).mean()
                )
                if len(malicious)
                else np.nan,

            "benign_rows":
                len(benign),

            "fp_rate":
                float(
                    (
                        benign[
                            "error_type"
                        ]
                        == "FP"
                    ).mean()
                )
                if len(benign)
                else np.nan,
        })

    routing_summary = (
        pd.DataFrame(
            routing_rows
        )
    )

    routing_summary.to_csv(
        OUTPUT
        / "routing_summary.csv",
        index=False,
    )

    # ========================================================
    # 11. 每个TOP-5分别看
    #
    # 注意：
    # 一个APK可以有多个TOP-5，因此这里不是互斥分组。
    # ========================================================

    role_rows = []

    total_fn = int(
        (
            pred["error_type"]
            == "FN"
        ).sum()
    )

    total_fp = int(
        (
            pred["error_type"]
            == "FP"
        ).sum()
    )

    for role in TOP5:

        d = pred[
            pred[
                "top5__" + role
            ] == 1
        ]

        malicious = d[
            d["label"] == 1
        ]

        benign = d[
            d["label"] == 0
        ]

        fn = int(
            (
                d["error_type"]
                == "FN"
            ).sum()
        )

        fp = int(
            (
                d["error_type"]
                == "FP"
            ).sum()
        )

        role_rows.append({
            "role":
                role,

            "prediction_rows":
                len(d),

            "unique_apks":
                d[
                    [
                        "split_name",
                        "sha256",
                    ]
                ]
                .drop_duplicates()
                .shape[0],

            "error_rate":
                float(
                    d["is_error"].mean()
                )
                if len(d)
                else np.nan,

            "fn":
                fn,

            "fn_rate_within_malicious":
                float(
                    (
                        malicious[
                            "error_type"
                        ]
                        == "FN"
                    ).mean()
                )
                if len(malicious)
                else np.nan,

            "fn_share_of_all_fn":
                fn / total_fn
                if total_fn
                else np.nan,

            "fp":
                fp,

            "fp_rate_within_benign":
                float(
                    (
                        benign[
                            "error_type"
                        ]
                        == "FP"
                    ).mean()
                )
                if len(benign)
                else np.nan,

            "fp_share_of_all_fp":
                fp / total_fp
                if total_fp
                else np.nan,

            "mean_probability":
                float(
                    d[
                        "base_probability"
                    ].mean()
                )
                if len(d)
                else np.nan,
        })

    role_summary = (
        pd.DataFrame(
            role_rows
        )
    )

    role_summary.to_csv(
        OUTPUT
        / "top5_role_summary.csv",
        index=False,
    )

    # ========================================================
    # 12. FN / FP 的置信度
    # ========================================================

    confidence_rows = []

    for error_type in [
        "FN",
        "FP",
        "TP",
        "TN",
    ]:

        d = pred[
            pred[
                "error_type"
            ] == error_type
        ]

        if not len(d):
            continue

        q = (
            d[
                "base_probability"
            ]
            .quantile(
                [
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                ]
            )
        )

        confidence_rows.append({
            "error_type":
                error_type,

            "n":
                len(d),

            "prob_q10":
                float(q.loc[0.10]),

            "prob_q25":
                float(q.loc[0.25]),

            "prob_q50":
                float(q.loc[0.50]),

            "prob_q75":
                float(q.loc[0.75]),

            "prob_q90":
                float(q.loc[0.90]),

            "mean_margin_to_05":
                float(
                    d[
                        "margin_to_05"
                    ].mean()
                ),

            "mean_uncertainty_gate":
                float(
                    d[
                        "uncertainty_gate"
                    ].mean()
                ),
        })

    confidence_summary = (
        pd.DataFrame(
            confidence_rows
        )
    )

    confidence_summary.to_csv(
        OUTPUT
        / "confidence_summary.csv",
        index=False,
    )

    # ========================================================
    # 13. 概率桶中的错误构成
    # ========================================================

    bucket_summary = (
        pred.groupby(
            [
                "probability_bucket",
                "error_type",
            ],
            observed=True,
        )
        .size()
        .reset_index(
            name="prediction_rows"
        )
    )

    bucket_summary.to_csv(
        OUTPUT
        / "probability_bucket_summary.csv",
        index=False,
    )

    # ========================================================
    # 14. 控制台关键结果
    # ========================================================

    print()
    print("=" * 72)
    print("COMPACT ERROR MAP")
    print("=" * 72)

    print(
        "prediction rows =",
        len(pred),
    )

    print(
        "unique validation APKs =",
        apk.shape[0],
    )

    print()
    print("ERROR TYPE COUNTS")
    print(
        error_counts.to_string(
            index=False
        )
    )

    print()
    print("ROUTING SUMMARY")
    print(
        routing_summary.to_string(
            index=False
        )
    )

    print()
    print("TOP-5 ROLE SUMMARY")
    print(
        role_summary.to_string(
            index=False
        )
    )

    print()
    print("CONFIDENCE SUMMARY")
    print(
        confidence_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 15. 最关键：FN / FP 中有多少属于TOP-5
    # ========================================================

    print()
    print("=" * 72)
    print("ERROR COVERAGE BY TOP-5")
    print("=" * 72)

    for error_type in [
        "FN",
        "FP",
    ]:

        d = pred[
            pred[
                "error_type"
            ] == error_type
        ]

        if not len(d):
            continue

        top5_rate = float(
            d[
                "top5_present"
            ].mean()
        )

        graph_rate = float(
            d[
                "graph_enabled"
            ].mean()
        )

        print(
            error_type,
            "n=",
            len(d),
            "TOP5=",
            f"{top5_rate:.4f}",
            "valid_graph=",
            f"{graph_rate:.4f}",
        )

    print()
    print(
        "saved =",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
