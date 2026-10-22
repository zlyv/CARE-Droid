import argparse
import copy
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


sys.path.insert(0, "scripts")

import run_behavior_core_v3_1_no_structural_no_junction_2017 as m


DEFAULT_OUTPUT = Path(
    "artifacts/"
    "global-conditioned-local-residual-2x2-2017"
)

THRESHOLD = 0.5


VARIANTS = {
    # A：原来的局部残差 + 原来的不确定性门控
    "A_local_oldgate": {
        "conditioned": False,
        "combine_mode": "old_gate",
    },

    # B：Compact条件化局部残差 + 原来的不确定性门控
    "B_conditioned_oldgate": {
        "conditioned": True,
        "combine_mode": "old_gate",
    },

    # C：原来的局部残差 + 直接加到logit
    "C_local_additive": {
        "conditioned": False,
        "combine_mode": "additive",
    },

    # D：Compact条件化局部残差 + 直接加到logit
    "D_conditioned_additive": {
        "conditioned": True,
        "combine_mode": "additive",
    },
}


def norm_sha(x):
    return str(x).strip().lower()


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path, rows):
    m.write_csv(
        path,
        rows,
    )


# ============================================================
# Compact条件化局部残差
#
# 关键约束：
#
# 1. local_model仍然只看局部语义表示。
#
# 2. Compact logit不能单独产生delta。
#    它只能调节local_delta。
#
# 3. 初始gamma=1，因此初始状态等价于原模型。
#
# 4. gamma允许变负：
#    相同局部语境在不同Compact状态下可以改变修正方向。
# ============================================================

class CompactConditionedResidualNet(
    nn.Module
):

    def __init__(
        self,
        role_count,
        edge_type_count,
    ):
        super().__init__()

        self.local_model = (
            m.BehaviorResidualNet(
                role_count=role_count,
                edge_type_count=edge_type_count,
            )
        )

        self.gamma_net = nn.Sequential(
            nn.Linear(
                1,
                8,
            ),
            nn.Tanh(),
            nn.Linear(
                8,
                1,
            ),
        )

        # --------------------------------------------
        # 初始化gamma=1
        #
        # gamma = 2*tanh(raw)
        #
        # 令raw=atanh(0.5)
        # 则gamma=1
        # --------------------------------------------

        last = self.gamma_net[-1]

        nn.init.zeros_(
            last.weight
        )

        nn.init.constant_(
            last.bias,
            math.atanh(0.5),
        )

    def forward(
        self,
        batch,
    ):

        local_delta = (
            self.local_model(
                batch
            )
            .reshape(-1)
        )

        base_logit = (
            batch[
                "base_logits"
            ]
            .reshape(-1, 1)
        )

        raw_gamma = (
            self.gamma_net(
                base_logit
            )
            .reshape(-1)
        )

        gamma = (
            2.0
            * torch.tanh(
                raw_gamma
            )
        )

        delta = (
            local_delta
            * gamma
        )

        return delta


# ============================================================
# 两种融合方式
# ============================================================

def combine_logits(
    base_logits,
    delta,
    mode,
):

    if mode == "old_gate":

        p = torch.sigmoid(
            base_logits
        )

        uncertainty = (
            4.0
            * p
            * (
                1.0 - p
            )
        )

        return (
            base_logits
            + uncertainty
            * torch.tanh(
                delta
            )
        )

    if mode == "additive":

        # --------------------------------------------
        # 将局部模块解释成：
        # 对Compact logit增加一项局部证据。
        #
        # 不使用4p(1-p)。
        # 不使用tanh人为限制到+-1。
        # --------------------------------------------

        return (
            base_logits
            + delta
        )

    raise ValueError(
        "unknown mode: "
        + repr(mode)
    )


# ============================================================
# 预测delta
#
# 不调用原evaluate_full_split，
# 因为原函数把旧公式写死了。
# ============================================================

def predict_delta_map(
    model,
    enabled,
    graphs_by_sha,
    base_logit_by_sha,
    label_by_sha,
    device,
    batch_size,
):

    result = {}

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(enabled),
            batch_size,
        ):

            part = enabled[
                start:
                start + batch_size
            ]

            if not part:
                continue

            batch = m.collate_apks(
                part,
                graphs_by_sha,
                base_logit_by_sha,
                label_by_sha,
                device,
            )

            delta = (
                model(
                    batch
                )
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            if len(delta) != len(part):
                raise RuntimeError(
                    "delta数量与APK数量不一致"
                )

            for sha, value in zip(
                part,
                delta,
            ):
                result[
                    sha
                ] = float(value)

    return result


def sigmoid_numpy(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    # 防止exp溢出
    x = np.clip(
        x,
        -60.0,
        60.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-x)
        )
    )


def evaluate_full_split(
    model,
    rows,
    labels,
    base_logits,
    graphs_by_sha,
    base_logit_by_sha,
    label_by_sha,
    device,
    batch_size,
    combine_mode,
):

    shas = [
        norm_sha(
            row["sha256"]
        )
        for row in rows
    ]

    enabled = [
        sha
        for sha in shas
        if sha in graphs_by_sha
    ]

    delta_map = (
        predict_delta_map(
            model,
            enabled,
            graphs_by_sha,
            base_logit_by_sha,
            label_by_sha,
            device,
            batch_size,
        )
    )

    base_logits_np = np.asarray(
        base_logits,
        dtype=float,
    ).reshape(-1)

    final_logits = (
        base_logits_np.copy()
    )

    delta_values = np.zeros(
        len(shas),
        dtype=float,
    )

    actual_correction = np.zeros(
        len(shas),
        dtype=float,
    )

    for i, sha in enumerate(
        shas
    ):

        if sha not in delta_map:
            continue

        delta = float(
            delta_map[sha]
        )

        z = float(
            base_logits_np[i]
        )

        if combine_mode == "old_gate":

            p = (
                1.0
                / (
                    1.0
                    + math.exp(
                        -max(
                            min(z, 60.0),
                            -60.0,
                        )
                    )
                )
            )

            correction = (
                4.0
                * p
                * (
                    1.0 - p
                )
                * math.tanh(
                    delta
                )
            )

        elif combine_mode == "additive":

            correction = delta

        else:
            raise ValueError(
                combine_mode
            )

        final_logits[i] = (
            z
            + correction
        )

        delta_values[i] = delta
        actual_correction[i] = (
            correction
        )

    base_probability = (
        sigmoid_numpy(
            base_logits_np
        )
    )

    final_probability = (
        sigmoid_numpy(
            final_logits
        )
    )

    y = labels.astype(int)

    base_metrics = (
        m.base.metrics(
            y,
            base_probability,
            threshold=THRESHOLD,
        )
    )

    final_metrics = (
        m.base.metrics(
            y,
            final_probability,
            threshold=THRESHOLD,
        )
    )

    base_pred = (
        base_probability
        >= THRESHOLD
    ).astype(int)

    final_pred = (
        final_probability
        >= THRESHOLD
    ).astype(int)

    changed = (
        base_pred
        != final_pred
    )

    corrected = (
        changed
        & (
            base_pred != y
        )
        & (
            final_pred == y
        )
    )

    regressed = (
        changed
        & (
            base_pred == y
        )
        & (
            final_pred != y
        )
    )

    # --------------------------------------------
    # 分开看FN和FP究竟纠正了多少
    # --------------------------------------------

    base_fn = (
        (y == 1)
        & (base_pred == 0)
    )

    base_fp = (
        (y == 0)
        & (base_pred == 1)
    )

    corrected_fn = (
        base_fn
        & (
            final_pred == 1
        )
    )

    corrected_fp = (
        base_fp
        & (
            final_pred == 0
        )
    )

    tp_regressed = (
        (y == 1)
        & (base_pred == 1)
        & (final_pred == 0)
    )

    tn_regressed = (
        (y == 0)
        & (base_pred == 0)
        & (final_pred == 1)
    )

    enabled_mask = np.asarray(
        [
            sha in delta_map
            for sha in shas
        ],
        dtype=bool,
    )

    enabled_delta = (
        delta_values[
            enabled_mask
        ]
    )

    enabled_correction = (
        actual_correction[
            enabled_mask
        ]
    )

    if len(enabled_delta):

        abs_delta = np.abs(
            enabled_delta
        )

        abs_correction = np.abs(
            enabled_correction
        )

        delta_diag = {
            "mean_abs_delta":
                float(
                    np.mean(
                        abs_delta
                    )
                ),

            "p95_abs_delta":
                float(
                    np.quantile(
                        abs_delta,
                        0.95,
                    )
                ),

            "max_abs_delta":
                float(
                    np.max(
                        abs_delta
                    )
                ),

            "mean_abs_correction":
                float(
                    np.mean(
                        abs_correction
                    )
                ),

            "p95_abs_correction":
                float(
                    np.quantile(
                        abs_correction,
                        0.95,
                    )
                ),

            "max_abs_correction":
                float(
                    np.max(
                        abs_correction
                    )
                ),
        }

    else:

        delta_diag = {
            "mean_abs_delta":
                0.0,

            "p95_abs_delta":
                0.0,

            "max_abs_delta":
                0.0,

            "mean_abs_correction":
                0.0,

            "p95_abs_correction":
                0.0,

            "max_abs_correction":
                0.0,
        }

    diagnostics = {
        "enabled":
            len(enabled),

        "changed":
            int(
                np.sum(
                    changed
                )
            ),

        "corrected":
            int(
                np.sum(
                    corrected
                )
            ),

        "regressed":
            int(
                np.sum(
                    regressed
                )
            ),

        "base_FN":
            int(
                np.sum(
                    base_fn
                )
            ),

        "corrected_FN":
            int(
                np.sum(
                    corrected_fn
                )
            ),

        "base_FP":
            int(
                np.sum(
                    base_fp
                )
            ),

        "corrected_FP":
            int(
                np.sum(
                    corrected_fp
                )
            ),

        "TP_regressed":
            int(
                np.sum(
                    tp_regressed
                )
            ),

        "TN_regressed":
            int(
                np.sum(
                    tn_regressed
                )
            ),

        **delta_diag,
    }

    detail = []

    for i, sha in enumerate(
        shas
    ):

        detail.append({
            "sha256":
                sha,

            "label":
                int(y[i]),

            "graph_enabled":
                int(
                    sha in delta_map
                ),

            "graph_count":
                len(
                    graphs_by_sha.get(
                        sha,
                        [],
                    )
                ),

            "base_probability":
                float(
                    base_probability[i]
                ),

            "final_probability":
                float(
                    final_probability[i]
                ),

            "base_logit":
                float(
                    base_logits_np[i]
                ),

            "final_logit":
                float(
                    final_logits[i]
                ),

            "delta":
                float(
                    delta_values[i]
                ),

            "actual_correction":
                float(
                    actual_correction[i]
                ),

            "base_prediction":
                int(
                    base_pred[i]
                ),

            "final_prediction":
                int(
                    final_pred[i]
                ),

            "changed":
                int(
                    changed[i]
                ),

            "corrected":
                int(
                    corrected[i]
                ),

            "regressed":
                int(
                    regressed[i]
                ),
        })

    return (
        base_metrics,
        final_metrics,
        diagnostics,
        detail,
    )


# ============================================================
# 训练一个残差变体
# ============================================================

def fit_variant(
    train_rows,
    val_rows,
    y_train,
    y_val,
    train_base_logits,
    val_base_logits,
    graphs_by_sha,
    role_count,
    edge_type_count,
    seed,
    device,
    max_epochs,
    patience,
    batch_size,
    learning_rate,
    weight_decay,
    conditioned,
    combine_mode,
):

    set_seed(seed)

    train_shas = [
        norm_sha(
            row["sha256"]
        )
        for row in train_rows
    ]

    val_shas = [
        norm_sha(
            row["sha256"]
        )
        for row in val_rows
    ]

    train_base_logit_by_sha = {
        sha: float(z)
        for sha, z
        in zip(
            train_shas,
            train_base_logits,
        )
    }

    val_base_logit_by_sha = {
        sha: float(z)
        for sha, z
        in zip(
            val_shas,
            val_base_logits,
        )
    }

    train_label_by_sha = {
        sha: int(y)
        for sha, y
        in zip(
            train_shas,
            y_train,
        )
    }

    val_label_by_sha = {
        sha: int(y)
        for sha, y
        in zip(
            val_shas,
            y_val,
        )
    }

    enabled_train = [
        sha
        for sha in train_shas
        if sha in graphs_by_sha
    ]

    enabled_val = [
        sha
        for sha in val_shas
        if sha in graphs_by_sha
    ]

    if not enabled_train:
        raise RuntimeError(
            "训练集没有有效局部图"
        )

    positive = sum(
        train_label_by_sha[sha]
        for sha in enabled_train
    )

    negative = (
        len(enabled_train)
        - positive
    )

    pos_weight_value = (
        negative / positive
        if positive > 0
        else 1.0
    )

    pos_weight = torch.tensor(
        [
            pos_weight_value
        ],
        dtype=torch.float32,
        device=device,
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    if conditioned:

        model = (
            CompactConditionedResidualNet(
                role_count=role_count,
                edge_type_count=edge_type_count,
            )
            .to(device)
        )

    else:

        model = (
            m.BehaviorResidualNet(
                role_count=role_count,
                edge_type_count=edge_type_count,
            )
            .to(device)
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state = copy.deepcopy(
        model.state_dict()
    )

    best_ba = -1.0
    best_epoch = 0
    patience_left = patience

    history = []

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        model.train()

        shuffled = list(
            enabled_train
        )

        random.shuffle(
            shuffled
        )

        epoch_losses = []

        for start in range(
            0,
            len(shuffled),
            batch_size,
        ):

            part = shuffled[
                start:
                start + batch_size
            ]

            batch = m.collate_apks(
                part,
                graphs_by_sha,
                train_base_logit_by_sha,
                train_label_by_sha,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            delta = model(
                batch
            )

            final_logits = (
                combine_logits(
                    batch[
                        "base_logits"
                    ],
                    delta,
                    combine_mode,
                )
            )

            # 标签仍然只在APK聚合之后使用
            loss = loss_fn(
                final_logits,
                batch[
                    "labels"
                ],
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss
                    .detach()
                    .cpu()
                )
            )

        (
            _,
            val_metrics,
            diagnostics,
            _,
        ) = evaluate_full_split(
            model,
            val_rows,
            y_val,
            val_base_logits,
            graphs_by_sha,
            val_base_logit_by_sha,
            val_label_by_sha,
            device,
            batch_size,
            combine_mode,
        )

        ba = float(
            val_metrics[
                "balanced_accuracy"
            ]
        )

        history.append({
            "epoch":
                epoch,

            "loss":
                float(
                    np.mean(
                        epoch_losses
                    )
                ),

            "BA":
                ba,

            "ACC":
                float(
                    val_metrics[
                        "accuracy"
                    ]
                ),

            "AUC":
                float(
                    val_metrics[
                        "auc"
                    ]
                ),

            "changed":
                diagnostics[
                    "changed"
                ],

            "corrected":
                diagnostics[
                    "corrected"
                ],

            "regressed":
                diagnostics[
                    "regressed"
                ],

            "corrected_FN":
                diagnostics[
                    "corrected_FN"
                ],

            "corrected_FP":
                diagnostics[
                    "corrected_FP"
                ],

            "mean_abs_correction":
                diagnostics[
                    "mean_abs_correction"
                ],

            "max_abs_correction":
                diagnostics[
                    "max_abs_correction"
                ],
        })

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"epoch={epoch} "
                f"loss={history[-1]['loss']:.6f} "
                f"BA={ba:.6f} "
                f"AUC={history[-1]['AUC']:.6f} "
                f"changed={diagnostics['changed']} "
                f"corrected={diagnostics['corrected']} "
                f"regressed={diagnostics['regressed']} "
                f"fixFN={diagnostics['corrected_FN']} "
                f"fixFP={diagnostics['corrected_FP']} "
                f"|corr|={diagnostics['mean_abs_correction']:.4f}",
                flush=True,
            )

        if ba > (
            best_ba + 1e-8
        ):

            best_ba = ba
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_left = patience

        else:

            patience_left -= 1

            if patience_left <= 0:
                break

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_epoch,
        history,
        train_base_logit_by_sha,
        train_label_by_sha,
        val_base_logit_by_sha,
        val_label_by_sha,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
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
        "--device",
        default="auto",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.device == "cpu":

        device = torch.device(
            "cpu"
        )

    elif args.device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA不可用"
            )

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(
        "=" * 80
    )

    print(
        "2017 2x2 residual mechanism experiment"
    )

    print(
        "Graph representation = "
        "v3.1 no structural edge "
        "+ no junction/recursive flags"
    )

    print(
        "Threshold = 0.5"
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
        "variants =",
        list(
            VARIANTS
        ),
    )

    print(
        "=" * 80
    )

    # ========================================================
    # 图
    #
    # 这里直接复用
    # no_structural_no_junction
    # 的图构造逻辑。
    # ========================================================

    (
        role_to_idx,
        edge_to_idx,
    ) = m.scan_vocabulary(
        m.GRAPH_FILE
    )

    graphs_by_sha = (
        m.load_graphs(
            m.GRAPH_FILE,
            role_to_idx,
            edge_to_idx,
        )
    )

    # ========================================================
    # Compact和滚动划分
    # ========================================================

    compact_rows = (
        m.read_csv_rows(
            m.COMPACT_FEATURES
        )
    )

    compact_schema = json.loads(
        m.COMPACT_SCHEMA.read_text(
            encoding="utf-8"
        )
    )

    compact_by_sha = (
        m.base.normalized_rows(
            compact_rows
        )
    )

    semantic_2017 = (
        m.base.read_gz_csv(
            m.SEMANTIC_2017
        )
    )

    names150 = (
        m.base.support150_names(
            semantic_2017[
                0
            ].keys()
        )
    )

    semantic_by_sha = (
        m.base.normalized_rows(
            semantic_2017
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

    if not rolling_splits:
        raise RuntimeError(
            "没有2017滚动划分"
        )

    if len(
        m.base.CONFIGS
    ) != 1:
        raise RuntimeError(
            "预期基础配置只有1个"
        )

    old_config = copy.deepcopy(
        m.base.CONFIGS[0]
    )

    per_seed_rows = []
    prediction_rows = []
    history_rows = []

    # ========================================================
    # 关键：
    #
    # 每个split+seed只训练一次Compact，
    # 四个残差变体共享完全相同的base logits。
    # ========================================================

    for split_name in sorted(
        rolling_splits
    ):

        print()
        print(
            "=" * 80
        )

        print(
            "split =",
            split_name,
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

        data = (
            m.base.build_fold_data(
                train_rows,
                val_rows,
                compact_by_sha,
                compact_schema,
                semantic_by_sha,
                names150,
            )
        )

        if (
            data[
                "compact_train"
            ].shape[1]
            != 70
        ):
            raise RuntimeError(
                "Compact不是70维"
            )

        y_train = (
            m.base.labels_for(
                train_rows,
                compact_by_sha,
            )
        )

        y_val = (
            m.base.labels_for(
                val_rows,
                compact_by_sha,
            )
        )

        for seed in args.seeds:

            print()
            print(
                "#" * 80
            )

            print(
                "seed =",
                seed,
            )

            print(
                "#" * 80
            )

            # --------------------------------------------
            # 只训练一次Compact
            # --------------------------------------------

            set_seed(seed)

            (
                base_model,
                _,
                _,
            ) = m.base.fit_one(
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
                old_config,
                seed,
                device,
            )

            train_base_logits = (
                m.get_base_logits(
                    base_model,
                    data,
                    "train",
                    device,
                )
            )

            val_base_logits = (
                m.get_base_logits(
                    base_model,
                    data,
                    "test",
                    device,
                )
            )

            # --------------------------------------------
            # 四个机制
            # --------------------------------------------

            for (
                variant_name,
                cfg,
            ) in VARIANTS.items():

                print()
                print(
                    "-" * 80
                )

                print(
                    "variant =",
                    variant_name,
                )

                print(
                    "conditioned =",
                    cfg[
                        "conditioned"
                    ],
                )

                print(
                    "combine_mode =",
                    cfg[
                        "combine_mode"
                    ],
                )

                print(
                    "-" * 80
                )

                (
                    residual_model,
                    best_epoch,
                    history,
                    train_base_logit_by_sha,
                    train_label_by_sha,
                    val_base_logit_by_sha,
                    val_label_by_sha,
                ) = fit_variant(
                    train_rows,
                    val_rows,
                    y_train,
                    y_val,
                    train_base_logits,
                    val_base_logits,
                    graphs_by_sha,
                    len(
                        role_to_idx
                    ),
                    len(
                        edge_to_idx
                    ),
                    seed,
                    device,
                    args.max_epochs,
                    args.patience,
                    args.batch_size,
                    args.learning_rate,
                    args.weight_decay,
                    cfg[
                        "conditioned"
                    ],
                    cfg[
                        "combine_mode"
                    ],
                )

                (
                    base_metrics,
                    final_metrics,
                    diagnostics,
                    detail,
                ) = evaluate_full_split(
                    residual_model,
                    val_rows,
                    y_val,
                    val_base_logits,
                    graphs_by_sha,
                    val_base_logit_by_sha,
                    val_label_by_sha,
                    device,
                    args.batch_size,
                    cfg[
                        "combine_mode"
                    ],
                )

                row = {
                    "variant":
                        variant_name,

                    "split_name":
                        split_name,

                    "seed":
                        seed,

                    "conditioned":
                        int(
                            cfg[
                                "conditioned"
                            ]
                        ),

                    "combine_mode":
                        cfg[
                            "combine_mode"
                        ],

                    "best_epoch":
                        best_epoch,

                    "base_BA":
                        float(
                            base_metrics[
                                "balanced_accuracy"
                            ]
                        ),

                    "final_BA":
                        float(
                            final_metrics[
                                "balanced_accuracy"
                            ]
                        ),

                    "delta_BA":
                        float(
                            final_metrics[
                                "balanced_accuracy"
                            ]
                            - base_metrics[
                                "balanced_accuracy"
                            ]
                        ),

                    "base_ACC":
                        float(
                            base_metrics[
                                "accuracy"
                            ]
                        ),

                    "final_ACC":
                        float(
                            final_metrics[
                                "accuracy"
                            ]
                        ),

                    "base_AUC":
                        float(
                            base_metrics[
                                "auc"
                            ]
                        ),

                    "final_AUC":
                        float(
                            final_metrics[
                                "auc"
                            ]
                        ),

                    "base_FP":
                        int(
                            base_metrics[
                                "fp"
                            ]
                        ),

                    "final_FP":
                        int(
                            final_metrics[
                                "fp"
                            ]
                        ),

                    "base_FN":
                        int(
                            base_metrics[
                                "fn"
                            ]
                        ),

                    "final_FN":
                        int(
                            final_metrics[
                                "fn"
                            ]
                        ),

                    **diagnostics,
                }

                per_seed_rows.append(
                    row
                )

                for h in history:

                    history_rows.append({
                        "variant":
                            variant_name,

                        "split_name":
                            split_name,

                        "seed":
                            seed,

                        **h,
                    })

                for item in detail:

                    prediction_rows.append({
                        "variant":
                            variant_name,

                        "split_name":
                            split_name,

                        "seed":
                            seed,

                        **item,
                    })

                print(
                    "RESULT "
                    f"{variant_name} "
                    f"base_BA={row['base_BA']:.6f} "
                    f"final_BA={row['final_BA']:.6f} "
                    f"delta={row['delta_BA']*100:+.3f}pp "
                    f"FP={row['base_FP']}->{row['final_FP']} "
                    f"FN={row['base_FN']}->{row['final_FN']} "
                    f"fixFN={row['corrected_FN']} "
                    f"fixFP={row['corrected_FP']} "
                    f"regressed={row['regressed']} "
                    f"|delta|={row['mean_abs_delta']:.4f} "
                    f"max|delta|={row['max_abs_delta']:.4f}",
                    flush=True,
                )

    # ========================================================
    # split级汇总
    # ========================================================

    by_variant_split = defaultdict(
        list
    )

    for row in per_seed_rows:

        by_variant_split[
            (
                row[
                    "variant"
                ],
                row[
                    "split_name"
                ],
            )
        ].append(
            row
        )

    split_rows = []

    for (
        variant,
        split_name,
    ), rows in sorted(
        by_variant_split.items()
    ):

        split_rows.append({
            "variant":
                variant,

            "split_name":
                split_name,

            "base_BA":
                float(
                    np.mean([
                        x["base_BA"]
                        for x in rows
                    ])
                ),

            "final_BA":
                float(
                    np.mean([
                        x["final_BA"]
                        for x in rows
                    ])
                ),

            "delta_BA":
                float(
                    np.mean([
                        x["delta_BA"]
                        for x in rows
                    ])
                ),

            "base_ACC":
                float(
                    np.mean([
                        x["base_ACC"]
                        for x in rows
                    ])
                ),

            "final_ACC":
                float(
                    np.mean([
                        x["final_ACC"]
                        for x in rows
                    ])
                ),

            "base_AUC":
                float(
                    np.mean([
                        x["base_AUC"]
                        for x in rows
                    ])
                ),

            "final_AUC":
                float(
                    np.mean([
                        x["final_AUC"]
                        for x in rows
                    ])
                ),

            "base_FP":
                float(
                    np.mean([
                        x["base_FP"]
                        for x in rows
                    ])
                ),

            "final_FP":
                float(
                    np.mean([
                        x["final_FP"]
                        for x in rows
                    ])
                ),

            "base_FN":
                float(
                    np.mean([
                        x["base_FN"]
                        for x in rows
                    ])
                ),

            "final_FN":
                float(
                    np.mean([
                        x["final_FN"]
                        for x in rows
                    ])
                ),

            "corrected":
                float(
                    np.mean([
                        x["corrected"]
                        for x in rows
                    ])
                ),

            "regressed":
                float(
                    np.mean([
                        x["regressed"]
                        for x in rows
                    ])
                ),

            "corrected_FN":
                float(
                    np.mean([
                        x["corrected_FN"]
                        for x in rows
                    ])
                ),

            "corrected_FP":
                float(
                    np.mean([
                        x["corrected_FP"]
                        for x in rows
                    ])
                ),

            "TP_regressed":
                float(
                    np.mean([
                        x["TP_regressed"]
                        for x in rows
                    ])
                ),

            "TN_regressed":
                float(
                    np.mean([
                        x["TN_regressed"]
                        for x in rows
                    ])
                ),

            "mean_abs_delta":
                float(
                    np.mean([
                        x["mean_abs_delta"]
                        for x in rows
                    ])
                ),

            "max_abs_delta":
                float(
                    np.max([
                        x["max_abs_delta"]
                        for x in rows
                    ])
                ),
        })

    # ========================================================
    # 最终按split平均
    # ========================================================

    summary_rows = []

    for variant in VARIANTS:

        rows = [
            x
            for x in split_rows
            if x[
                "variant"
            ] == variant
        ]

        if not rows:
            continue

        summary_rows.append({
            "variant":
                variant,

            "base_mean_BA":
                float(
                    np.mean([
                        x["base_BA"]
                        for x in rows
                    ])
                ),

            "final_mean_BA":
                float(
                    np.mean([
                        x["final_BA"]
                        for x in rows
                    ])
                ),

            "delta_BA":
                float(
                    np.mean([
                        x["final_BA"]
                        - x["base_BA"]
                        for x in rows
                    ])
                ),

            "final_min_BA":
                float(
                    np.min([
                        x["final_BA"]
                        for x in rows
                    ])
                ),

            "base_mean_ACC":
                float(
                    np.mean([
                        x["base_ACC"]
                        for x in rows
                    ])
                ),

            "final_mean_ACC":
                float(
                    np.mean([
                        x["final_ACC"]
                        for x in rows
                    ])
                ),

            "base_mean_AUC":
                float(
                    np.mean([
                        x["base_AUC"]
                        for x in rows
                    ])
                ),

            "final_mean_AUC":
                float(
                    np.mean([
                        x["final_AUC"]
                        for x in rows
                    ])
                ),

            "base_mean_FP":
                float(
                    np.mean([
                        x["base_FP"]
                        for x in rows
                    ])
                ),

            "final_mean_FP":
                float(
                    np.mean([
                        x["final_FP"]
                        for x in rows
                    ])
                ),

            "base_mean_FN":
                float(
                    np.mean([
                        x["base_FN"]
                        for x in rows
                    ])
                ),

            "final_mean_FN":
                float(
                    np.mean([
                        x["final_FN"]
                        for x in rows
                    ])
                ),

            "mean_corrected":
                float(
                    np.mean([
                        x["corrected"]
                        for x in rows
                    ])
                ),

            "mean_regressed":
                float(
                    np.mean([
                        x["regressed"]
                        for x in rows
                    ])
                ),

            "mean_corrected_FN":
                float(
                    np.mean([
                        x["corrected_FN"]
                        for x in rows
                    ])
                ),

            "mean_corrected_FP":
                float(
                    np.mean([
                        x["corrected_FP"]
                        for x in rows
                    ])
                ),

            "mean_TP_regressed":
                float(
                    np.mean([
                        x["TP_regressed"]
                        for x in rows
                    ])
                ),

            "mean_TN_regressed":
                float(
                    np.mean([
                        x["TN_regressed"]
                        for x in rows
                    ])
                ),

            "mean_abs_delta":
                float(
                    np.mean([
                        x["mean_abs_delta"]
                        for x in rows
                    ])
                ),

            "max_abs_delta":
                float(
                    np.max([
                        x["max_abs_delta"]
                        for x in rows
                    ])
                ),
        })

    write_csv(
        args.output
        / "per_seed.csv",
        per_seed_rows,
    )

    write_csv(
        args.output
        / "per_split.csv",
        split_rows,
    )

    write_csv(
        args.output
        / "summary.csv",
        summary_rows,
    )

    write_csv(
        args.output
        / "predictions.csv",
        prediction_rows,
    )

    write_csv(
        args.output
        / "history.csv",
        history_rows,
    )

    # ========================================================
    # 控制台最终结果
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL 2x2 SUMMARY"
    )

    print(
        "=" * 100
    )

    for row in summary_rows:

        print()

        print(
            row[
                "variant"
            ]
        )

        print(
            "  BA       : "
            f"{row['base_mean_BA']:.6f}"
            " -> "
            f"{row['final_mean_BA']:.6f}"
            "  "
            f"({row['delta_BA']*100:+.3f}pp)"
        )

        print(
            "  min BA   : "
            f"{row['final_min_BA']:.6f}"
        )

        print(
            "  ACC      : "
            f"{row['base_mean_ACC']:.6f}"
            " -> "
            f"{row['final_mean_ACC']:.6f}"
        )

        print(
            "  AUC      : "
            f"{row['base_mean_AUC']:.6f}"
            " -> "
            f"{row['final_mean_AUC']:.6f}"
        )

        print(
            "  FP       : "
            f"{row['base_mean_FP']:.3f}"
            " -> "
            f"{row['final_mean_FP']:.3f}"
        )

        print(
            "  FN       : "
            f"{row['base_mean_FN']:.3f}"
            " -> "
            f"{row['final_mean_FN']:.3f}"
        )

        print(
            "  corrected: "
            f"{row['mean_corrected']:.3f}"
            "  regressed: "
            f"{row['mean_regressed']:.3f}"
        )

        print(
            "  fix FN   : "
            f"{row['mean_corrected_FN']:.3f}"
            "  fix FP   : "
            f"{row['mean_corrected_FP']:.3f}"
        )

        print(
            "  TP regress: "
            f"{row['mean_TP_regressed']:.3f}"
            "  TN regress: "
            f"{row['mean_TN_regressed']:.3f}"
        )

        print(
            "  mean|delta|: "
            f"{row['mean_abs_delta']:.4f}"
            "  max|delta|: "
            f"{row['max_abs_delta']:.4f}"
        )

    # A应当接近旧实验
    a = next(
        (
            x
            for x in summary_rows
            if x[
                "variant"
            ] == "A_local_oldgate"
        ),
        None,
    )

    if a is not None:

        expected = 0.913554

        diff = (
            a[
                "final_mean_BA"
            ]
            - expected
        )

        print()
        print(
            "=" * 100
        )

        print(
            "BASELINE REPRODUCTION CHECK"
        )

        print(
            "=" * 100
        )

        print(
            "expected old no-structural/no-junction BA ~",
            expected,
        )

        print(
            "current A BA =",
            f"{a['final_mean_BA']:.6f}",
        )

        print(
            "difference =",
            f"{diff*100:+.4f}pp",
        )

        if abs(diff) > 0.0005:
            print(
                "WARNING: A与旧实验差异偏大，"
                "先不要解释B/C/D。"
            )
        else:
            print(
                "A复现正常，可以比较B/C/D。"
            )

    print()
    print(
        "saved =",
        args.output,
    )


if __name__ == "__main__":
    main()
