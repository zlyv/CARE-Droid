import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS = Path(
    "artifacts/"
    "behavior-core-v3-1-residual-allratio70-2017/"
    "predictions.csv"
)

GRAPH_FILE = Path(
    "artifacts/"
    "top5-semantic-behavior-graph-v1-2017-trigger551/"
    "binding_cores_v3_1_semantic_junction_train_safe.jsonl"
)

OUTPUT = Path(
    "artifacts/"
    "top5-local-context-diagnosis-2017"
)

TOP5 = {
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
}


def norm_sha(x):
    return str(x).strip().lower()


# ============================================================
# 读取每张v3.1局部图，提取可解释上下文
# ============================================================

def extract_graph_context(x):

    nodes = x.get("nodes", [])
    edges = x.get("edges", [])

    by_id = {
        str(n["id"]): n
        for n in nodes
    }

    semantic_roles = {}

    anchor_roles = []

    carrier_methods = set()

    for n in nodes:

        nid = str(n["id"])
        kind = str(n.get("type", ""))

        if kind == "semantic":

            role = str(
                n.get("role", "")
            ).strip()

            semantic_roles[nid] = role

            if int(
                n.get(
                    "is_anchor_role",
                    0,
                )
            ):
                anchor_roles.append(
                    role
                )

        elif kind == "method":

            if int(
                n.get(
                    "is_anchor_carrier",
                    0,
                )
            ):
                carrier_methods.add(
                    nid
                )

    anchor_roles = sorted(
        set(anchor_roles)
    )

    if len(anchor_roles) != 1:
        raise RuntimeError(
            "预期每张图恰好一个anchor role，实际="
            + repr(anchor_roles)
        )

    anchor = anchor_roles[0]

    if anchor not in TOP5:
        raise RuntimeError(
            "未知anchor: "
            + anchor
        )

    # --------------------------------------------------------
    # method -> semantic role
    # --------------------------------------------------------

    roles_by_method = defaultdict(
        set
    )

    for e in edges:

        if str(
            e.get("type", "")
        ) != "method_to_semantic":
            continue

        source = str(e["source"])
        target = str(e["target"])

        source_node = by_id.get(source)
        target_node = by_id.get(target)

        if (
            source_node is None
            or target_node is None
        ):
            continue

        if (
            str(
                source_node.get(
                    "type",
                    "",
                )
            ) != "method"
        ):
            continue

        role = semantic_roles.get(
            target
        )

        if role:
            roles_by_method[
                source
            ].add(
                role
            )

    # ========================================================
    # 上下文1：
    # 整个局部核心里，与anchor共同出现的语义
    # ========================================================

    all_roles = set(
        semantic_roles.values()
    )

    core_context = (
        all_roles
        - {anchor}
    )

    # ========================================================
    # 上下文2：
    # TOP-5承载方法上直接挂载的其他语义
    # ========================================================

    anchor_method_context = set()

    anchor_method_role_counts = []

    for method_id in carrier_methods:

        roles = roles_by_method.get(
            method_id,
            set(),
        )

        anchor_method_role_counts.append(
            len(roles)
        )

        anchor_method_context.update(
            roles - {anchor}
        )

    # 所有方法中最多同时承载多少个语义
    max_roles_per_method = max(
        [
            len(v)
            for v in roles_by_method.values()
        ],
        default=0,
    )

    semantic_carriers = sum(
        int(len(v) > 0)
        for v in roles_by_method.values()
    )

    return {
        "anchor":
            anchor,

        "core_context":
            core_context,

        "anchor_method_context":
            anchor_method_context,

        "semantic_role_count":
            len(all_roles),

        "max_roles_per_method":
            max_roles_per_method,

        "semantic_carrier_count":
            semantic_carriers,

        "anchor_method_max_roles":
            max(
                anchor_method_role_counts,
                default=0,
            ),

        "anchor_method_has_extra":
            int(
                len(
                    anchor_method_context
                ) > 0
            ),
    }


def mean_or_nan(x):
    x = list(x)

    if not x:
        return np.nan

    return float(
        np.mean(x)
    )


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. Compact预测：
    #    五个seed聚合成一个APK错误频率
    # ========================================================

    pred = pd.read_csv(
        PREDICTIONS
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

    apk = (
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
        apk["seeds"] == 5
    ).all():
        raise RuntimeError(
            "有APK不是5个seed"
        )

    # 恶意样本：
    # 被预测为0的比例 = FN频率
    apk["fn_rate"] = np.where(
        apk["label"] == 1,
        1.0
        - apk[
            "positive_prediction_rate"
        ],
        np.nan,
    )

    # 良性样本：
    # 被预测为1的比例 = FP频率
    apk["fp_rate"] = np.where(
        apk["label"] == 0,
        apk[
            "positive_prediction_rate"
        ],
        np.nan,
    )

    pred_by_sha = {}

    for _, row in apk.iterrows():

        sha = row["sha256"]

        if sha in pred_by_sha:
            raise RuntimeError(
                "同一SHA出现在多个验证split: "
                + sha
            )

        pred_by_sha[sha] = row

    # ========================================================
    # 2. 提取v3.1上下文
    #    一个APK可有多个anchor、每个anchor可有多张图
    # ========================================================

    raw = defaultdict(
        lambda: defaultdict(
            list
        )
    )

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

            ctx = extract_graph_context(
                x
            )

            raw[sha][
                ctx["anchor"]
            ].append(
                ctx
            )

    # ========================================================
    # 3. APK + anchor聚合
    # ========================================================

    rows = []

    all_context_roles = set()

    for sha, anchor_map in raw.items():

        # 只分析2017滚动验证APK
        if sha not in pred_by_sha:
            continue

        p = pred_by_sha[sha]

        for anchor, graphs in (
            anchor_map.items()
        ):

            core_union = set()
            anchor_method_union = set()

            for g in graphs:

                core_union.update(
                    g[
                        "core_context"
                    ]
                )

                anchor_method_union.update(
                    g[
                        "anchor_method_context"
                    ]
                )

            all_context_roles.update(
                core_union
            )

            all_context_roles.update(
                anchor_method_union
            )

            rows.append({
                "split_name":
                    p["split_name"],

                "sha256":
                    sha,

                "label":
                    int(
                        p["label"]
                    ),

                "anchor":
                    anchor,

                "mean_base_probability":
                    float(
                        p[
                            "mean_base_probability"
                        ]
                    ),

                "fn_rate":
                    float(
                        p["fn_rate"]
                    )
                    if int(
                        p["label"]
                    ) == 1
                    else np.nan,

                "fp_rate":
                    float(
                        p["fp_rate"]
                    )
                    if int(
                        p["label"]
                    ) == 0
                    else np.nan,

                "graph_count":
                    len(graphs),

                "core_context_roles":
                    ";".join(
                        sorted(
                            core_union
                        )
                    ),

                "anchor_method_context_roles":
                    ";".join(
                        sorted(
                            anchor_method_union
                        )
                    ),

                "mean_semantic_role_count":
                    float(
                        np.mean([
                            g[
                                "semantic_role_count"
                            ]
                            for g in graphs
                        ])
                    ),

                "max_roles_per_method":
                    max(
                        g[
                            "max_roles_per_method"
                        ]
                        for g in graphs
                    ),

                "mean_semantic_carrier_count":
                    float(
                        np.mean([
                            g[
                                "semantic_carrier_count"
                            ]
                            for g in graphs
                        ])
                    ),

                "anchor_method_max_roles":
                    max(
                        g[
                            "anchor_method_max_roles"
                        ]
                        for g in graphs
                    ),

                "anchor_method_has_extra":
                    int(
                        any(
                            g[
                                "anchor_method_has_extra"
                            ]
                            for g in graphs
                        )
                    ),
            })

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUTPUT
        / "apk_anchor_context.csv",
        index=False,
    )

    # ========================================================
    # 4. 每个anchor总体错误率
    # ========================================================

    anchor_rows = []

    for anchor, d in df.groupby(
        "anchor"
    ):

        mal = d[
            d["label"] == 1
        ]

        ben = d[
            d["label"] == 0
        ]

        anchor_rows.append({
            "anchor":
                anchor,

            "apk_anchor_rows":
                len(d),

            "malicious_apks":
                len(mal),

            "mean_fn_rate":
                mean_or_nan(
                    mal["fn_rate"]
                ),

            "benign_apks":
                len(ben),

            "mean_fp_rate":
                mean_or_nan(
                    ben["fp_rate"]
                ),

            "mean_graph_count":
                float(
                    d[
                        "graph_count"
                    ].mean()
                ),

            "mean_core_role_count":
                float(
                    d[
                        "mean_semantic_role_count"
                    ].mean()
                ),

            "anchor_method_extra_rate":
                float(
                    d[
                        "anchor_method_has_extra"
                    ].mean()
                ),
        })

    anchor_summary = pd.DataFrame(
        anchor_rows
    )

    anchor_summary.to_csv(
        OUTPUT
        / "anchor_error_summary.csv",
        index=False,
    )

    # ========================================================
    # 5. 单个上下文语义的错误影响
    #
    # 同一个anchor内部比较：
    #
    # 恶意：
    #   有context role时的FN频率
    #   vs 没有context role时的FN频率
    #
    # 良性：
    #   有context role时的FP频率
    #   vs 没有context role时的FP频率
    # ========================================================

    effect_rows = []

    scopes = {
        "same_local_core":
            "core_context_roles",

        "same_anchor_method":
            "anchor_method_context_roles",
    }

    for anchor, d_anchor in df.groupby(
        "anchor"
    ):

        for scope, column in (
            scopes.items()
        ):

            role_sets = (
                d_anchor[column]
                .fillna("")
                .map(
                    lambda x: set(
                        r
                        for r in str(
                            x
                        ).split(";")
                        if r
                    )
                )
            )

            possible_roles = sorted(
                set().union(
                    *role_sets.tolist()
                )
                if len(role_sets)
                else set()
            )

            for role in possible_roles:

                has_role = (
                    role_sets.map(
                        lambda x:
                            role in x
                    )
                )

                # --------------------------------------------
                # 恶意：FN
                # --------------------------------------------

                mal = (
                    d_anchor[
                        d_anchor[
                            "label"
                        ] == 1
                    ]
                    .copy()
                )

                mal_has = (
                    role_sets.loc[
                        mal.index
                    ]
                    .map(
                        lambda x:
                            role in x
                    )
                )

                mal_with = mal[
                    mal_has
                ]

                mal_without = mal[
                    ~mal_has
                ]

                fn_with = (
                    mean_or_nan(
                        mal_with[
                            "fn_rate"
                        ]
                    )
                )

                fn_without = (
                    mean_or_nan(
                        mal_without[
                            "fn_rate"
                        ]
                    )
                )

                # --------------------------------------------
                # 良性：FP
                # --------------------------------------------

                ben = (
                    d_anchor[
                        d_anchor[
                            "label"
                        ] == 0
                    ]
                    .copy()
                )

                ben_has = (
                    role_sets.loc[
                        ben.index
                    ]
                    .map(
                        lambda x:
                            role in x
                    )
                )

                ben_with = ben[
                    ben_has
                ]

                ben_without = ben[
                    ~ben_has
                ]

                fp_with = (
                    mean_or_nan(
                        ben_with[
                            "fp_rate"
                        ]
                    )
                )

                fp_without = (
                    mean_or_nan(
                        ben_without[
                            "fp_rate"
                        ]
                    )
                )

                effect_rows.append({
                    "anchor":
                        anchor,

                    "scope":
                        scope,

                    "context_role":
                        role,

                    "malicious_with":
                        len(mal_with),

                    "malicious_without":
                        len(mal_without),

                    "fn_rate_with":
                        fn_with,

                    "fn_rate_without":
                        fn_without,

                    "delta_fn_rate":
                        (
                            fn_with
                            - fn_without
                        )
                        if (
                            np.isfinite(
                                fn_with
                            )
                            and np.isfinite(
                                fn_without
                            )
                        )
                        else np.nan,

                    "benign_with":
                        len(ben_with),

                    "benign_without":
                        len(ben_without),

                    "fp_rate_with":
                        fp_with,

                    "fp_rate_without":
                        fp_without,

                    "delta_fp_rate":
                        (
                            fp_with
                            - fp_without
                        )
                        if (
                            np.isfinite(
                                fp_with
                            )
                            and np.isfinite(
                                fp_without
                            )
                        )
                        else np.nan,
                })

    effects = pd.DataFrame(
        effect_rows
    )

    effects.to_csv(
        OUTPUT
        / "context_role_effects.csv",
        index=False,
    )

    # ========================================================
    # 6. 数值型局部结构：
    #    FN/TP、FP/TN分别比较
    # ========================================================

    numeric_features = [
        "graph_count",
        "mean_semantic_role_count",
        "max_roles_per_method",
        "mean_semantic_carrier_count",
        "anchor_method_max_roles",
        "anchor_method_has_extra",
    ]

    numeric_rows = []

    for anchor, d_anchor in df.groupby(
        "anchor"
    ):

        for feature in numeric_features:

            mal = d_anchor[
                d_anchor[
                    "label"
                ] == 1
            ]

            ben = d_anchor[
                d_anchor[
                    "label"
                ] == 0
            ]

            if len(mal):

                # 用seed错误频率做权重，
                # 不把5个seed当5个APK。
                fn_w = mal[
                    "fn_rate"
                ].values

                tp_w = (
                    1.0 - fn_w
                )

                x = mal[
                    feature
                ].values.astype(
                    float
                )

                fn_mean = (
                    np.average(
                        x,
                        weights=fn_w,
                    )
                    if fn_w.sum() > 0
                    else np.nan
                )

                tp_mean = (
                    np.average(
                        x,
                        weights=tp_w,
                    )
                    if tp_w.sum() > 0
                    else np.nan
                )

            else:
                fn_mean = np.nan
                tp_mean = np.nan

            if len(ben):

                fp_w = ben[
                    "fp_rate"
                ].values

                tn_w = (
                    1.0 - fp_w
                )

                x = ben[
                    feature
                ].values.astype(
                    float
                )

                fp_mean = (
                    np.average(
                        x,
                        weights=fp_w,
                    )
                    if fp_w.sum() > 0
                    else np.nan
                )

                tn_mean = (
                    np.average(
                        x,
                        weights=tn_w,
                    )
                    if tn_w.sum() > 0
                    else np.nan
                )

            else:
                fp_mean = np.nan
                tn_mean = np.nan

            numeric_rows.append({
                "anchor":
                    anchor,

                "feature":
                    feature,

                "FN_weighted_mean":
                    fn_mean,

                "TP_weighted_mean":
                    tp_mean,

                "FN_minus_TP":
                    (
                        fn_mean
                        - tp_mean
                    )
                    if (
                        np.isfinite(
                            fn_mean
                        )
                        and np.isfinite(
                            tp_mean
                        )
                    )
                    else np.nan,

                "FP_weighted_mean":
                    fp_mean,

                "TN_weighted_mean":
                    tn_mean,

                "FP_minus_TN":
                    (
                        fp_mean
                        - tn_mean
                    )
                    if (
                        np.isfinite(
                            fp_mean
                        )
                        and np.isfinite(
                            tn_mean
                        )
                    )
                    else np.nan,
            })

    numeric_summary = pd.DataFrame(
        numeric_rows
    )

    numeric_summary.to_csv(
        OUTPUT
        / "numeric_context_effects.csv",
        index=False,
    )

    # ========================================================
    # 7. 控制台重点输出
    # ========================================================

    print()
    print("=" * 80)
    print("ANCHOR ERROR SUMMARY")
    print("=" * 80)

    print(
        anchor_summary
        .sort_values(
            "mean_fn_rate",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    # 只打印有基本支持度的结果，防止单个APK误导。
    stable = effects[
        (
            (
                effects[
                    "malicious_with"
                ] >= 3
            )
            & (
                effects[
                    "malicious_without"
                ] >= 3
            )
        )
        |
        (
            (
                effects[
                    "benign_with"
                ] >= 3
            )
            & (
                effects[
                    "benign_without"
                ] >= 3
            )
        )
    ].copy()

    print()
    print("=" * 80)
    print("TOP CONTEXTS ASSOCIATED WITH FN")
    print("=" * 80)

    x = (
        stable[
            stable[
                "malicious_with"
            ] >= 3
        ]
        .sort_values(
            "delta_fn_rate",
            ascending=False,
        )
        .head(30)
    )

    print(
        x[
            [
                "anchor",
                "scope",
                "context_role",
                "malicious_with",
                "malicious_without",
                "fn_rate_with",
                "fn_rate_without",
                "delta_fn_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("TOP CONTEXTS ASSOCIATED WITH FP")
    print("=" * 80)

    x = (
        stable[
            stable[
                "benign_with"
            ] >= 3
        ]
        .sort_values(
            "delta_fp_rate",
            ascending=False,
        )
        .head(30)
    )

    print(
        x[
            [
                "anchor",
                "scope",
                "context_role",
                "benign_with",
                "benign_without",
                "fp_rate_with",
                "fp_rate_without",
                "delta_fp_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("NUMERIC LOCAL-CONTEXT EFFECTS")
    print("=" * 80)

    print(
        numeric_summary.to_string(
            index=False
        )
    )

    print()
    print("rows =", len(df))
    print(
        "saved =",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
