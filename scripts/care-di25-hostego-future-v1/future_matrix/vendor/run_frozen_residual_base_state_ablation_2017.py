#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import run_frozen_anchor_residual_2017_2021 as base
import run_frozen_anchor_residual_20_150_2017_2021 as aug


OUTPUT_DIR = Path(
    "artifacts/"
    "frozen-residual-base-state-ablation-2017"
)

SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

ROLES = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

COUNT_METRICS = (
    "caller_log",
    "caller_ratio",
)

COUNT10_NAMES = [
    f"up_role__{role}__{metric}"
    for role in ROLES
    for metric in COUNT_METRICS
]

if len(COUNT10_NAMES) != 10:
    raise RuntimeError(
        "数量块不是10维"
    )


# ============================================================
# 强制正式数量10
# ============================================================

aug.STRUCTURE_DIM = 10
aug.OLD20_NAMES = list(
    COUNT10_NAMES
)

base.build_fold_data = (
    aug.build_fold_data_20_150
)


# ============================================================
# 四个残差头输入版本
#
# 注意：
# correction_gate始终完全一样：
#
# presence * 4p(1-p)
# ============================================================

VARIANTS = (
    {
        "name":
            "current_base_uncertainty",

        "use_base_logit":
            True,

        "use_uncertainty":
            True,
    },
    {
        "name":
            "no_base_logit",

        "use_base_logit":
            False,

        "use_uncertainty":
            True,
    },
    {
        "name":
            "no_uncertainty_input",

        "use_base_logit":
            True,

        "use_uncertainty":
            False,
    },
    {
        "name":
            "semantic_structure_coverage_only",

        "use_base_logit":
            False,

        "use_uncertainty":
            False,
    },
)


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(
                    key
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# 网络
# ============================================================

class BaseStateAblationNet(
    nn.Module
):
    def __init__(
        self,
        compact_dim,
        semantic_dim,
        coverage_dim,
        hidden,
        dropout,
        base_weight,
        base_bias,
        use_base_logit,
        use_uncertainty,
    ):
        super().__init__()

        self.structure_dim = 10

        self.use_base_logit = bool(
            use_base_logit
        )

        self.use_uncertainty = bool(
            use_uncertainty
        )

        relation_dim = (
            semantic_dim
            - self.structure_dim
        )

        if relation_dim <= 0:
            raise RuntimeError(
                "关系维数异常"
            )

        base_weight = (
            torch.as_tensor(
                base_weight,
                dtype=torch.float32,
            )
            .reshape(
                compact_dim,
                1,
            )
        )

        base_bias = (
            torch.as_tensor(
                base_bias,
                dtype=torch.float32,
            )
            .reshape(1)
        )

        self.register_buffer(
            "base_weight",
            base_weight,
        )

        self.register_buffer(
            "base_bias",
            base_bias,
        )

        # 与正式网络完全相同。
        self.semantic_encoder = (
            nn.Sequential(
                nn.Linear(
                    relation_dim,
                    hidden,
                ),
                nn.LayerNorm(
                    hidden
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
            )
        )

        delta_input_dim = (
            self.structure_dim
            + hidden
            + coverage_dim
        )

        if self.use_base_logit:
            delta_input_dim += 1

        if self.use_uncertainty:
            delta_input_dim += 1

        self.delta_head = nn.Sequential(
            nn.Linear(
                delta_input_dim,
                hidden,
            ),
            nn.LayerNorm(
                hidden
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden,
                1,
            ),
        )

    def forward(
        self,
        compact,
        semantic,
        coverage,
        presence,
    ):
        # ====================================================
        # Frozen Compact
        # ====================================================

        base_logit = (
            compact
            @ self.base_weight
            + self.base_bias
        )

        base_probability = (
            torch.sigmoid(
                base_logit
            )
        )

        uncertainty = (
            4.0
            * base_probability
            * (
                1.0
                - base_probability
            )
        )

        # ====================================================
        # 这个门所有消融完全相同。
        # ====================================================

        correction_gate = (
            presence
            * uncertainty
        )

        # ====================================================
        # 辅助表示
        # ====================================================

        structure = semantic[
            :,
            :self.structure_dim,
        ]

        relation = semantic[
            :,
            self.structure_dim:,
        ]

        semantic_hidden = (
            self.semantic_encoder(
                relation
            )
        )

        parts = [
            structure,
            semantic_hidden,
            coverage,
        ]

        if self.use_base_logit:
            parts.append(
                base_logit
            )

        if self.use_uncertainty:
            parts.append(
                uncertainty
            )

        delta_input = torch.cat(
            parts,
            dim=1,
        )

        delta = torch.tanh(
            self.delta_head(
                delta_input
            )
        )

        final_logit = (
            base_logit
            + correction_gate
            * delta
        )

        return (
            final_logit,
            base_logit,
            correction_gate,
            delta,
        )


# ============================================================
# 使用base.fit_one接口
#
# 每个variant运行前替换类。
# ============================================================

CURRENT_VARIANT = None


def make_variant_class(
    use_base_logit,
    use_uncertainty,
):
    class CurrentNet(
        BaseStateAblationNet
    ):
        def __init__(
            self,
            compact_dim,
            semantic_dim,
            coverage_dim,
            hidden,
            dropout,
            base_weight,
            base_bias,
        ):
            super().__init__(
                compact_dim=compact_dim,
                semantic_dim=semantic_dim,
                coverage_dim=coverage_dim,
                hidden=hidden,
                dropout=dropout,
                base_weight=base_weight,
                base_bias=base_bias,
                use_base_logit=use_base_logit,
                use_uncertainty=use_uncertainty,
            )

    return CurrentNet


# ============================================================
# 汇总
# ============================================================

def summarize(
    rows,
):
    by_split = defaultdict(
        list
    )

    for row in rows:
        by_split[
            row[
                "split_name"
            ]
        ].append(
            row
        )

    split_rows = []

    for split_name in sorted(
        by_split
    ):
        rs = by_split[
            split_name
        ]

        split_rows.append({
            "split_name":
                split_name,

            "mean_BA":
                float(
                    np.mean([
                        r[
                            "balanced_accuracy"
                        ]
                        for r in rs
                    ])
                ),

            "mean_AUC":
                float(
                    np.mean([
                        r[
                            "auc"
                        ]
                        for r in rs
                    ])
                ),

            "mean_ACC":
                float(
                    np.mean([
                        r[
                            "accuracy"
                        ]
                        for r in rs
                    ])
                ),
        })

    ba = np.asarray(
        [
            r[
                "mean_BA"
            ]
            for r in split_rows
        ],
        dtype=float,
    )

    auc = np.asarray(
        [
            r[
                "mean_AUC"
            ]
            for r in split_rows
        ],
        dtype=float,
    )

    acc = np.asarray(
        [
            r[
                "mean_ACC"
            ]
            for r in split_rows
        ],
        dtype=float,
    )

    return (
        {
            "mean_BA":
                float(
                    ba.mean()
                ),

            "min_split_BA":
                float(
                    ba.min()
                ),

            "std_split_BA":
                float(
                    ba.std()
                ),

            "mean_AUC":
                float(
                    auc.mean()
                ),

            "mean_ACC":
                float(
                    acc.mean()
                ),
        },
        split_rows,
    )


def main():

    compact_path = Path(
        "artifacts/p1-features/"
        "features.csv"
    )

    schema_path = Path(
        "artifacts/p1-features/"
        "feature_schema.json"
    )

    semantic_path = Path(
        "artifacts/clean-local-context-2017/"
        "semantic_relation_features.csv.gz"
    )

    split_path = Path(
        "artifacts/p9-2017-rolling-splits/"
        "split_manifest.csv"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device,
    )

    if device.type == "cuda":
        torch.backends.cudnn.deterministic = (
            True
        )
        torch.backends.cudnn.benchmark = (
            False
        )

    compact_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                compact_path
            )
        )
    )

    compact_schema = json.loads(
        schema_path.read_text(
            encoding="utf-8"
        )
    )

    semantic_2017 = (
        base.read_gz_csv(
            semantic_path
        )
    )

    semantic_by_sha = (
        base.normalized_rows(
            semantic_2017
        )
    )

    names150 = (
        base.support150_names(
            semantic_2017[
                0
            ].keys()
        )
    )

    if len(
        names150
    ) != 150:
        raise SystemExit(
            "final150不是150维"
        )

    rolling_parts = (
        base.read_split_rows(
            split_path
        )
    )

    rolling_splits = {
        name:
            parts
        for name, parts
        in rolling_parts.items()
        if (
            "rolling" in name
            and parts[
                "train"
            ]
            and parts[
                "test"
            ]
        )
    }

    if len(
        rolling_splits
    ) != 3:
        raise SystemExit(
            "滚动划分不是3个"
        )

    # 强制载入数量结构。
    aug.load_old20_rows()

    if len(
        base.CONFIGS
    ) != 1:
        raise SystemExit(
            "Frozen配置不是1个"
        )

    config_base = copy.deepcopy(
        base.CONFIGS[
            0
        ]
    )

    # --------------------------------------------------------
    # 每折数据完全相同。
    # --------------------------------------------------------

    fold_data = {}

    for split_name in sorted(
        rolling_splits
    ):

        train_rows = (
            rolling_splits[
                split_name
            ][
                "train"
            ]
        )

        val_rows = (
            rolling_splits[
                split_name
            ][
                "test"
            ]
        )

        data = (
            aug.build_fold_data_20_150(
                train_rows,
                val_rows,
                compact_by_sha,
                compact_schema,
                semantic_by_sha,
                names150,
            )
        )

        y_train = (
            base.labels_for(
                train_rows,
                compact_by_sha,
            )
        )

        y_val = (
            base.labels_for(
                val_rows,
                compact_by_sha,
            )
        )

        fold_data[
            split_name
        ] = (
            data,
            y_train,
            y_val,
        )

        print(
            split_name,
            "semantic_dim=",
            data[
                "semantic_train"
            ].shape[
                1
            ],
        )

    all_runs = []
    summaries = []
    split_summaries = []

    # ========================================================
    # 四个variant
    # ========================================================

    for variant in VARIANTS:

        print()
        print(
            "======================================"
        )
        print(
            variant[
                "name"
            ]
        )
        print(
            "base_logit input =",
            variant[
                "use_base_logit"
            ]
        )
        print(
            "uncertainty input =",
            variant[
                "use_uncertainty"
            ]
        )
        print(
            "======================================"
        )

        base.FrozenAnchorResidualNet = (
            make_variant_class(
                variant[
                    "use_base_logit"
                ],
                variant[
                    "use_uncertainty"
                ],
            )
        )

        config = copy.deepcopy(
            config_base
        )

        config[
            "name"
        ] = variant[
            "name"
        ]

        current = []

        for split_name in sorted(
            fold_data
        ):
            (
                data,
                y_train,
                y_val,
            ) = fold_data[
                split_name
            ]

            print()
            print(
                split_name
            )

            for seed in SEEDS:

                (
                    model,
                    best_epoch,
                    history,
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
                    y_val,

                    config,
                    seed,
                    device,
                )

                best = max(
                    history,
                    key=lambda x:
                        x[
                            "balanced_accuracy"
                        ],
                )

                row = {
                    "variant":
                        variant[
                            "name"
                        ],

                    "split_name":
                        split_name,

                    "seed":
                        seed,

                    "balanced_accuracy":
                        float(
                            best[
                                "balanced_accuracy"
                            ]
                        ),

                    "auc":
                        float(
                            best[
                                "auc"
                            ]
                        ),

                    "accuracy":
                        float(
                            best[
                                "accuracy"
                            ]
                        ),

                    "epoch":
                        int(
                            best[
                                "epoch"
                            ]
                        ),
                }

                current.append(
                    row
                )

                all_runs.append(
                    row
                )

                print(
                    f" seed={seed} "
                    f"BA={row['balanced_accuracy']:.6f} "
                    f"AUC={row['auc']:.6f} "
                    f"ACC={row['accuracy']:.6f} "
                    f"epoch={row['epoch']}"
                )

        (
            summary,
            split_rows,
        ) = summarize(
            current
        )

        summary[
            "variant"
        ] = variant[
            "name"
        ]

        summaries.append(
            summary
        )

        for row in split_rows:
            row[
                "variant"
            ] = variant[
                "name"
            ]

            split_summaries.append(
                row
            )

        print()
        print(
            f"SUMMARY {variant['name']} "
            f"BA={summary['mean_BA']:.6f} "
            f"minBA={summary['min_split_BA']:.6f} "
            f"stdBA={summary['std_split_BA']:.6f} "
            f"AUC={summary['mean_AUC']:.6f} "
            f"ACC={summary['mean_ACC']:.6f}"
        )

    # ========================================================
    # 与当前正式模型配对
    # ========================================================

    baseline_name = (
        "current_base_uncertainty"
    )

    baseline = {
        (
            r[
                "split_name"
            ],
            r[
                "seed"
            ],
        ):
            r
        for r in all_runs
        if r[
            "variant"
        ] == baseline_name
    }

    pair_rows = []
    pair_summary = []

    for variant in VARIANTS[
        1:
    ]:

        name = variant[
            "name"
        ]

        candidate = {
            (
                r[
                    "split_name"
                ],
                r[
                    "seed"
                ],
            ):
                r
            for r in all_runs
            if r[
                "variant"
            ] == name
        }

        deltas = []
        auc_deltas = []

        wins = 0
        ties = 0
        losses = 0

        for key in sorted(
            baseline
        ):
            b = baseline[
                key
            ]

            c = candidate[
                key
            ]

            d = (
                c[
                    "balanced_accuracy"
                ]
                -
                b[
                    "balanced_accuracy"
                ]
            )

            da = (
                c[
                    "auc"
                ]
                -
                b[
                    "auc"
                ]
            )

            deltas.append(
                d
            )

            auc_deltas.append(
                da
            )

            if d > 1e-12:
                result = "win"
                wins += 1
            elif d < -1e-12:
                result = "loss"
                losses += 1
            else:
                result = "tie"
                ties += 1

            pair_rows.append({
                "variant":
                    name,

                "split_name":
                    key[
                        0
                    ],

                "seed":
                    key[
                        1
                    ],

                "delta_BA":
                    d,

                "delta_AUC":
                    da,

                "result":
                    result,
            })

        pair_summary.append({
            "variant":
                name,

            "wins":
                wins,

            "ties":
                ties,

            "losses":
                losses,

            "mean_delta_BA":
                float(
                    np.mean(
                        deltas
                    )
                ),

            "median_delta_BA":
                float(
                    np.median(
                        deltas
                    )
                ),

            "mean_delta_AUC":
                float(
                    np.mean(
                        auc_deltas
                    )
                ),
        })

    write_csv(
        OUTPUT_DIR
        / "per_run.csv",
        all_runs,
    )

    write_csv(
        OUTPUT_DIR
        / "summary.csv",
        summaries,
    )

    write_csv(
        OUTPUT_DIR
        / "split_summary.csv",
        split_summaries,
    )

    write_csv(
        OUTPUT_DIR
        / "paired_comparison.csv",
        pair_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "paired_summary.csv",
        pair_summary,
    )

    print()
    print(
        "======================================"
    )
    print(
        "2017-only最终结果"
    )
    print(
        "======================================"
    )

    for row in summaries:
        print(
            f"{row['variant']:36s} "
            f"BA={row['mean_BA']:.6f} "
            f"minBA={row['min_split_BA']:.6f} "
            f"stdBA={row['std_split_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "相对当前模型配对结果"
    )
    print(
        "======================================"
    )

    for row in pair_summary:
        print(
            f"{row['variant']:36s} "
            f"胜/平/负="
            f"{row['wins']}/"
            f"{row['ties']}/"
            f"{row['losses']} "
            f"meanΔBA="
            f"{row['mean_delta_BA']:+.6f} "
            f"medianΔBA="
            f"{row['median_delta_BA']:+.6f} "
            f"meanΔAUC="
            f"{row['mean_delta_AUC']:+.6f}"
        )


if __name__ == "__main__":
    main()
