#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold


# ============================================================
# 确保可以导入 scripts/ 下现有程序
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )


# ============================================================
# 导入当前正式 count10 + 150 模型。
#
# 这个导入会自动：
#   1. 把 build_fold_data 换成 count10 + 150
#   2. 把残差网络换成当前正式网络
#   3. presence 仍然只来自150维
# ============================================================

import run_frozen_anchor_residual_count10_150_2017_2021 as current


base = current.base


YEARS = (
    2018,
    2019,
    2020,
    2021,
)

N_SPLITS = 5

# 当前2017源域选择得到的正式训练轮数。
FINAL_EPOCHS = 86

THRESHOLD = 0.5


# ============================================================
# 现有“2017训练 -> 各年份测试”结果。
#
# 只用于最后打印对照。
# 不参与任何训练、选参或测试。
# ============================================================

CROSS_YEAR_COMPARISON = {
    2018: {
        "compact_BA": 0.892064,
        "residual_BA": 0.922608,
    },
    2019: {
        "compact_BA": 0.882437,
        "residual_BA": 0.902940,
    },
    2020: {
        "compact_BA": 0.876516,
        "residual_BA": 0.878547,
    },
    2021: {
        "compact_BA": 0.855000,
        "residual_BA": 0.866000,
    },
}


def resolve_device(value):
    if value == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA requested but "
                "torch.cuda.is_available() is false"
            )

        return torch.device(
            "cuda"
        )

    if value == "cpu":
        return torch.device(
            "cpu"
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


def summarize_metric_rows(
    rows,
):
    """
    rows:
        同一个 year/model 下，
        每个 seed 的整年 OOF 指标。

    正式汇总：
        对5个seed指标取平均。
    """

    return {
        "BA":
            float(
                np.mean(
                    [
                        r["BA"]
                        for r in rows
                    ]
                )
            ),

        "BA_std_seed":
            float(
                np.std(
                    [
                        r["BA"]
                        for r in rows
                    ]
                )
            ),

        "AUC":
            float(
                np.mean(
                    [
                        r["AUC"]
                        for r in rows
                    ]
                )
            ),

        "ACC":
            float(
                np.mean(
                    [
                        r["ACC"]
                        for r in rows
                    ]
                )
            ),

        "F1":
            float(
                np.mean(
                    [
                        r["F1"]
                        for r in rows
                    ]
                )
            ),

        "FP":
            float(
                np.mean(
                    [
                        r["FP"]
                        for r in rows
                    ]
                )
            ),

        "FN":
            float(
                np.mean(
                    [
                        r["FN"]
                        for r in rows
                    ]
                )
            ),
    }


def calculate_metrics(
    labels,
    probability,
):
    m = base.metrics(
        np.asarray(
            labels,
            dtype=int,
        ),
        np.asarray(
            probability,
            dtype=float,
        ),
        threshold=THRESHOLD,
    )

    return {
        "BA":
            float(
                m[
                    "balanced_accuracy"
                ]
            ),

        "AUC":
            float(
                m[
                    "auc"
                ]
            ),

        "ACC":
            float(
                m[
                    "accuracy"
                ]
            ),

        "F1":
            float(
                m[
                    "f1"
                ]
            ),

        "FP":
            int(
                m[
                    "fp"
                ]
            ),

        "FN":
            int(
                m[
                    "fn"
                ]
            ),
    }


def get_year_rows(
    future_parts,
    year,
):
    split_name = (
        f"train_fixed_2017__test_{year}"
    )

    if split_name not in future_parts:
        raise SystemExit(
            "缺少未来划分: "
            + split_name
        )

    rows = list(
        future_parts[
            split_name
        ][
            "test"
        ]
    )

    if not rows:
        raise SystemExit(
            f"{year} 没有测试样本"
        )

    sha = [
        str(
            row[
                "sha256"
            ]
        )
        .strip()
        .casefold()
        for row in rows
    ]

    if len(
        sha
    ) != len(
        set(
            sha
        )
    ):
        raise SystemExit(
            f"{year} 存在重复SHA"
        )

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compact-features",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-features/"
            "features.csv"
        ),
    )

    parser.add_argument(
        "--compact-schema",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-features/"
            "feature_schema.json"
        ),
    )

    parser.add_argument(
        "--semantic-2017",
        type=Path,
        default=Path(
            "artifacts/"
            "clean-local-context-2017/"
            "semantic_relation_features.csv.gz"
        ),
    )

    parser.add_argument(
        "--semantic-future",
        type=Path,
        default=Path(
            "artifacts/"
            "clean-local-context-future/"
            "semantic_relation_features.csv.gz"
        ),
    )

    parser.add_argument(
        "--future-splits",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-splits-fixed-2017-test-2018-2021/"
            "split_manifest.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "within-year-2018-2021-cv"
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            0,
            1,
            2,
            3,
            4,
        ],
    )

    parser.add_argument(
        "--fold-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        args.device
    )

    if device.type == "cuda":
        torch.backends.cudnn.deterministic = (
            True
        )
        torch.backends.cudnn.benchmark = (
            False
        )

    # ========================================================
    # 数据读取
    # ========================================================

    print(
        "======================================"
    )
    print(
        "2018-2021 同年份内部5折实验"
    )
    print(
        "======================================"
    )

    print(
        "device =",
        device,
    )

    print(
        "seeds =",
        args.seeds,
    )

    print(
        "fold seed =",
        args.fold_seed,
    )

    print(
        "final epochs =",
        FINAL_EPOCHS,
    )

    print(
        "threshold =",
        THRESHOLD,
    )

    compact_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                args.compact_features
            )
        )
    )

    compact_schema = json.loads(
        args.compact_schema.read_text(
            encoding="utf-8"
        )
    )

    semantic_2017 = (
        base.read_gz_csv(
            args.semantic_2017
        )
    )

    semantic_future = (
        base.read_gz_csv(
            args.semantic_future
        )
    )

    # ========================================================
    # 150维定义继续沿用当前正式方法。
    #
    # 这里只使用2017文件确定“哪些列属于冻结150维”，
    # 不拿2017样本参与本实验任何模型训练。
    # ========================================================

    names150 = (
        base.support150_names(
            semantic_2017[
                0
            ].keys()
        )
    )

    future_names150 = (
        base.support150_names(
            semantic_future[
                0
            ].keys()
        )
    )

    if future_names150 != names150:
        raise SystemExit(
            "2017/future support150列不一致"
        )

    semantic_by_sha = (
        base.normalized_rows(
            semantic_2017
            + semantic_future
        )
    )

    future_parts = (
        base.read_split_rows(
            args.future_splits
        )
    )

    # ========================================================
    # 当前正式配置。
    # ========================================================

    if len(
        base.CONFIGS
    ) != 1:
        raise SystemExit(
            "预期当前模型只有1个配置，实际="
            + str(
                len(
                    base.CONFIGS
                )
            )
        )

    config = dict(
        base.CONFIGS[
            0
        ]
    )

    fixed_config = {
        **config,

        "max_epochs":
            FINAL_EPOCHS,

        "patience":
            FINAL_EPOCHS + 1,

        "fixed_epochs":
            True,
    }

    print(
        "config =",
        config[
            "name"
        ],
    )

    print(
        "hidden =",
        config[
            "hidden"
        ],
    )

    print(
        "dropout =",
        config[
            "dropout"
        ],
    )

    # ========================================================
    # 输出容器
    # ========================================================

    fold_seed_rows = []
    prediction_rows = []

    # ========================================================
    # 四个年份完全独立
    # ========================================================

    for year in YEARS:
        rows = get_year_rows(
            future_parts,
            year,
        )

        y_all = (
            base.labels_for(
                rows,
                compact_by_sha,
            )
            .astype(int)
        )

        unique_y, counts_y = (
            np.unique(
                y_all,
                return_counts=True,
            )
        )

        class_text = {
            int(k): int(v)
            for k, v in zip(
                unique_y,
                counts_y,
            )
        }

        print()
        print(
            "=" * 78
        )
        print(
            year,
            "N =",
            len(
                rows
            ),
            "class =",
            class_text,
        )
        print(
            "=" * 78
        )

        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=args.fold_seed,
        )

        dummy_x = np.zeros(
            len(
                rows
            ),
            dtype=np.int8,
        )

        for fold_index, (
            train_idx,
            test_idx,
        ) in enumerate(
            splitter.split(
                dummy_x,
                y_all,
            ),
            start=1,
        ):
            train_rows = [
                rows[
                    int(i)
                ]
                for i in train_idx
            ]

            test_rows = [
                rows[
                    int(i)
                ]
                for i in test_idx
            ]

            y_train = (
                base.labels_for(
                    train_rows,
                    compact_by_sha,
                )
                .astype(int)
            )

            y_test = (
                base.labels_for(
                    test_rows,
                    compact_by_sha,
                )
                .astype(int)
            )

            # =================================================
            # 这里是关键：
            #
            # build_fold_data 内部所有标准化均只使用
            # train_rows 的统计量。
            #
            # current模块已经把它替换成：
            # count10 + 150 的正式构造方式。
            # =================================================

            data = (
                base.build_fold_data(
                    train_rows,
                    test_rows,
                    compact_by_sha,
                    compact_schema,
                    semantic_by_sha,
                    names150,
                )
            )

            print()
            print(
                f"{year} fold={fold_index} "
                f"train={len(train_rows)} "
                f"test={len(test_rows)} "
                f"compact_dim="
                f"{data['compact_train'].shape[1]} "
                f"aux_dim="
                f"{data['semantic_train'].shape[1]}",
                flush=True,
            )

            for seed in args.seeds:
                # =============================================
                # 固定86轮。
                #
                # 为了训练阶段完全不访问外层测试折，
                # fit_one 的 val 输入这里故意传训练折本身。
                #
                # fixed_epochs=True，
                # 所以根本不会根据val选择epoch。
                # =============================================

                (
                    model,
                    best_epoch,
                    _,
                ) = base.fit_one(
                    data[
                        "compact_train"
                    ],
                    data[
                        "semantic_train"
                    ],
                    data[
                        "coverage_train"
                    ],
                    data[
                        "presence_train"
                    ],
                    y_train,

                    data[
                        "compact_train"
                    ],
                    data[
                        "semantic_train"
                    ],
                    data[
                        "coverage_train"
                    ],
                    data[
                        "presence_train"
                    ],
                    y_train,

                    fixed_config,
                    seed,
                    device,
                )

                if (
                    best_epoch
                    != FINAL_EPOCHS
                ):
                    raise RuntimeError(
                        "固定训练轮数异常："
                        f"{best_epoch}"
                    )

                (
                    residual_result,
                    probability,
                    base_probability,
                    gate,
                    delta,
                ) = base.evaluate_model(
                    model,
                    data[
                        "compact_test"
                    ],
                    data[
                        "semantic_test"
                    ],
                    data[
                        "coverage_test"
                    ],
                    data[
                        "presence_test"
                    ],
                    y_test,
                    device,
                )

                compact_result = (
                    base.metrics(
                        y_test,
                        base_probability,
                        threshold=THRESHOLD,
                    )
                )

                # =============================================
                # 每折每seed结果
                # =============================================

                fold_seed_rows.append({
                    "year":
                        year,

                    "fold":
                        fold_index,

                    "seed":
                        seed,

                    "train_n":
                        len(
                            train_rows
                        ),

                    "test_n":
                        len(
                            test_rows
                        ),

                    "model":
                        "compact",

                    "BA":
                        float(
                            compact_result[
                                "balanced_accuracy"
                            ]
                        ),

                    "AUC":
                        float(
                            compact_result[
                                "auc"
                            ]
                        ),

                    "ACC":
                        float(
                            compact_result[
                                "accuracy"
                            ]
                        ),

                    "F1":
                        float(
                            compact_result[
                                "f1"
                            ]
                        ),

                    "FP":
                        int(
                            compact_result[
                                "fp"
                            ]
                        ),

                    "FN":
                        int(
                            compact_result[
                                "fn"
                            ]
                        ),
                })

                fold_seed_rows.append({
                    "year":
                        year,

                    "fold":
                        fold_index,

                    "seed":
                        seed,

                    "train_n":
                        len(
                            train_rows
                        ),

                    "test_n":
                        len(
                            test_rows
                        ),

                    "model":
                        "count10_150",

                    "BA":
                        float(
                            residual_result[
                                "balanced_accuracy"
                            ]
                        ),

                    "AUC":
                        float(
                            residual_result[
                                "auc"
                            ]
                        ),

                    "ACC":
                        float(
                            residual_result[
                                "accuracy"
                            ]
                        ),

                    "F1":
                        float(
                            residual_result[
                                "f1"
                            ]
                        ),

                    "FP":
                        int(
                            residual_result[
                                "fp"
                            ]
                        ),

                    "FN":
                        int(
                            residual_result[
                                "fn"
                            ]
                        ),
                })

                # =============================================
                # 保存逐样本OOF预测。
                #
                # 后面如果想测试“最小跨界”，
                # 不需要重新训练。
                # =============================================

                for (
                    row,
                    label,
                    prob,
                    base_p,
                    gate_value,
                    delta_value,
                ) in zip(
                    test_rows,
                    y_test,
                    probability,
                    base_probability,
                    gate,
                    delta,
                ):
                    prediction_rows.append({
                        "year":
                            year,

                        "fold":
                            fold_index,

                        "sha256":
                            row[
                                "sha256"
                            ],

                        "seed":
                            seed,

                        "label":
                            int(
                                label
                            ),

                        "probability":
                            float(
                                prob
                            ),

                        "base_probability":
                            float(
                                base_p
                            ),

                        "gate":
                            float(
                                gate_value
                            ),

                        "semantic_delta":
                            float(
                                delta_value
                            ),
                    })

                print(
                    f"  seed={seed} "
                    f"CompactBA="
                    f"{compact_result['balanced_accuracy']:.6f} "
                    f"ResidualBA="
                    f"{residual_result['balanced_accuracy']:.6f} "
                    f"ResidualAUC="
                    f"{residual_result['auc']:.6f}",
                    flush=True,
                )

    # ========================================================
    # 保存逐折结果和预测
    # ========================================================

    fold_seed_df = pd.DataFrame(
        fold_seed_rows
    )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    fold_seed_df.to_csv(
        args.output
        / "per_fold_seed.csv",
        index=False,
    )

    predictions_df.to_csv(
        args.output
        / "oof_predictions.csv",
        index=False,
    )

    # ========================================================
    # 整年 OOF：
    #
    # 对每一个year × seed，
    # 拼接5个fold的测试预测。
    #
    # 每个APK此时恰好出现一次。
    #
    # 这是与“2017 -> 整年测试”最直接可比的指标。
    # ========================================================

    oof_seed_rows = []

    for (
        year,
        seed,
    ), g in predictions_df.groupby(
        [
            "year",
            "seed",
        ],
        sort=True,
    ):
        if (
            g[
                "sha256"
            ].duplicated().any()
        ):
            raise RuntimeError(
                f"{year} seed={seed} "
                "OOF中出现重复APK"
            )

        expected_n = len(
            get_year_rows(
                future_parts,
                int(
                    year
                ),
            )
        )

        if len(
            g
        ) != expected_n:
            raise RuntimeError(
                f"{year} seed={seed} "
                f"OOF数量={len(g)}, "
                f"预期={expected_n}"
            )

        y = g[
            "label"
        ].to_numpy(
            dtype=int
        )

        # Compact
        compact_m = calculate_metrics(
            y,
            g[
                "base_probability"
            ].to_numpy(
                dtype=float
            ),
        )

        oof_seed_rows.append({
            "year":
                int(
                    year
                ),

            "seed":
                int(
                    seed
                ),

            "model":
                "compact",

            "N":
                len(
                    g
                ),

            **compact_m,
        })

        # count10 + 150
        residual_m = calculate_metrics(
            y,
            g[
                "probability"
            ].to_numpy(
                dtype=float
            ),
        )

        oof_seed_rows.append({
            "year":
                int(
                    year
                ),

            "seed":
                int(
                    seed
                ),

            "model":
                "count10_150",

            "N":
                len(
                    g
                ),

            **residual_m,
        })

    oof_seed_df = pd.DataFrame(
        oof_seed_rows
    )

    oof_seed_df.to_csv(
        args.output
        / "oof_per_seed.csv",
        index=False,
    )

    # ========================================================
    # 每年份最终汇总：
    # 对5个seed的整年OOF指标取平均。
    # ========================================================

    year_summary_rows = []

    for year in YEARS:
        for model_name in (
            "compact",
            "count10_150",
        ):
            rows_current = (
                oof_seed_df[
                    (
                        oof_seed_df[
                            "year"
                        ] == year
                    )
                    &
                    (
                        oof_seed_df[
                            "model"
                        ] == model_name
                    )
                ]
                .to_dict(
                    orient="records"
                )
            )

            if len(
                rows_current
            ) != len(
                args.seeds
            ):
                raise RuntimeError(
                    f"{year} {model_name} "
                    "seed数量异常"
                )

            s = summarize_metric_rows(
                rows_current
            )

            year_summary_rows.append({
                "year":
                    year,

                "model":
                    model_name,

                "N":
                    int(
                        rows_current[
                            0
                        ][
                            "N"
                        ]
                    ),

                **s,
            })

    year_summary = pd.DataFrame(
        year_summary_rows
    )

    year_summary.to_csv(
        args.output
        / "within_year_summary.csv",
        index=False,
    )

    # ========================================================
    # 同年 vs 2017跨年对照
    # ========================================================

    comparison_rows = []

    for year in YEARS:
        for model_name in (
            "compact",
            "count10_150",
        ):
            r = year_summary[
                (
                    year_summary[
                        "year"
                    ] == year
                )
                &
                (
                    year_summary[
                        "model"
                    ] == model_name
                )
            ].iloc[
                0
            ]

            if model_name == "compact":
                cross_ba = (
                    CROSS_YEAR_COMPARISON[
                        year
                    ][
                        "compact_BA"
                    ]
                )
            else:
                cross_ba = (
                    CROSS_YEAR_COMPARISON[
                        year
                    ][
                        "residual_BA"
                    ]
                )

            comparison_rows.append({
                "year":
                    year,

                "model":
                    model_name,

                "within_year_BA":
                    float(
                        r[
                            "BA"
                        ]
                    ),

                "cross_year_2017_to_year_BA":
                    float(
                        cross_ba
                    ),

                "within_minus_cross_BA":
                    float(
                        r[
                            "BA"
                        ]
                        -
                        cross_ba
                    ),

                "within_year_AUC":
                    float(
                        r[
                            "AUC"
                        ]
                    ),

                "within_year_FP":
                    float(
                        r[
                            "FP"
                        ]
                    ),

                "within_year_FN":
                    float(
                        r[
                            "FN"
                        ]
                    ),
            })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        args.output
        / "within_vs_cross_year.csv",
        index=False,
    )

    # ========================================================
    # 打印最终结果
    # ========================================================

    print()
    print(
        "=" * 86
    )
    print(
        "同年份整年OOF结果"
    )
    print(
        "=" * 86
    )

    for year in YEARS:
        print()
        print(
            year
        )

        for model_name in (
            "compact",
            "count10_150",
        ):
            r = year_summary[
                (
                    year_summary[
                        "year"
                    ] == year
                )
                &
                (
                    year_summary[
                        "model"
                    ] == model_name
                )
            ].iloc[
                0
            ]

            print(
                f"  {model_name:12s} "
                f"BA={r['BA']:.6f} "
                f"seedStd={r['BA_std_seed']:.6f} "
                f"AUC={r['AUC']:.6f} "
                f"ACC={r['ACC']:.6f} "
                f"FP={r['FP']:.2f} "
                f"FN={r['FN']:.2f}"
            )

    print()
    print(
        "=" * 86
    )
    print(
        "同年训练 vs 2017跨年训练"
    )
    print(
        "=" * 86
    )

    for year in YEARS:
        r = comparison_df[
            (
                comparison_df[
                    "year"
                ] == year
            )
            &
            (
                comparison_df[
                    "model"
                ] == "count10_150"
            )
        ].iloc[
            0
        ]

        print(
            f"{year}: "
            f"同年BA="
            f"{r['within_year_BA']:.6f} "
            f"2017→{year}="
            f"{r['cross_year_2017_to_year_BA']:.6f} "
            f"差值="
            f"{r['within_minus_cross_BA']:+.6f}"
        )

    # ========================================================
    # 四年平均
    # ========================================================

    print()
    print(
        "=" * 86
    )
    print(
        "四年平均"
    )
    print(
        "=" * 86
    )

    for model_name in (
        "compact",
        "count10_150",
    ):
        g = year_summary[
            year_summary[
                "model"
            ] == model_name
        ]

        print(
            f"{model_name:12s} "
            f"meanBA="
            f"{g['BA'].mean():.6f} "
            f"minBA="
            f"{g['BA'].min():.6f} "
            f"meanAUC="
            f"{g['AUC'].mean():.6f}"
        )

    print()
    print(
        "结果目录:",
        args.output,
    )


if __name__ == "__main__":
    main()
