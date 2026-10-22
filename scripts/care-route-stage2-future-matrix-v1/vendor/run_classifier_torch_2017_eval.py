#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_graph_pool_diagnostic import (
    read_csv_rows,
    read_split_rows,
)

from run_dual_view_error_attribution import (
    compact_matrix_named,
)

from run_sensitive_upstream_structural_eval import (
    label_value,
    standardize_named,
)

from run_pilot_compact_eval import (
    fit_logistic,
    sigmoid,
    metrics,
)


# =========================================================
# 冻结实验配置
# =========================================================

ALPHA = 0.25
THRESHOLD = 0.5

SEEDS = [0, 1, 2, 3, 4]

LR_ITERATIONS = 6000
LR_RATE = 0.05
LR_L2 = 0.03

NN_MAX_EPOCHS = 500
NN_PATIENCE = 50
NN_BATCH_SIZE = 64
NN_LEARNING_RATE = 1e-3
NN_WEIGHT_DECAY = 1e-3
NN_VALID_RATIO = 0.15

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


COMPACT_PATH = Path(
    "artifacts/p1-features/features.csv"
)

SCHEMA_PATH = Path(
    "artifacts/p1-features/feature_schema.json"
)

SEMANTIC_PATH = Path(
    "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

SPLIT_PATH = Path(
    "artifacts/p9-2017-rolling-splits/"
    "split_manifest.csv"
)

OUTPUT_DIR = Path(
    "artifacts/classifier-torch-2017"
)


# =========================================================
# 基础IO
# =========================================================

def read_gz_csv(path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def support150_names(header):
    support = sorted(
        name
        for name in header
        if name.startswith(
            "semrel_support__"
        )
    )

    if len(support) != 155:
        raise SystemExit(
            f"support维数不是155：{len(support)}"
        )

    selected = []

    for name in support:
        parts = name.split("__")

        if len(parts) != 3:
            raise SystemExit(
                "异常特征名：" + name
            )

        anchor = parts[1]
        context = parts[2]

        if anchor != context:
            selected.append(name)

    if len(selected) != 150:
        raise SystemExit(
            f"最终不是150维：{len(selected)}"
        )

    return selected


# =========================================================
# 可重复性
# =========================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 训练集内部验证划分
# 只用于提前停止，不碰滚动测试折
# =========================================================

def stratified_inner_split(
    y,
    seed,
    valid_ratio=0.15,
):
    rng = np.random.default_rng(seed)

    train_indices = []
    valid_indices = []

    for label in (0, 1):
        indices = np.flatnonzero(
            y == label
        ).copy()

        rng.shuffle(indices)

        n_valid = max(
            1,
            int(
                round(
                    len(indices)
                    * valid_ratio
                )
            ),
        )

        valid_indices.extend(
            indices[:n_valid]
        )

        train_indices.extend(
            indices[n_valid:]
        )

    train_indices = np.asarray(
        train_indices,
        dtype=int,
    )

    valid_indices = np.asarray(
        valid_indices,
        dtype=int,
    )

    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)

    return (
        train_indices,
        valid_indices,
    )


# =========================================================
# 网络
# =========================================================

class CompactNet(nn.Module):
    def __init__(self, compact_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                compact_dim,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(
                32,
                16,
            ),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(self, compact):
        return self.net(
            compact
        ).squeeze(1)


class TwoBranchNet(nn.Module):
    def __init__(
        self,
        compact_dim,
        semantic_dim,
    ):
        super().__init__()

        self.compact_branch = nn.Sequential(
            nn.Linear(
                compact_dim,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(
                32,
                16,
            ),
            nn.ReLU(),
        )

        self.semantic_branch = nn.Sequential(
            nn.Linear(
                semantic_dim,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(
                32,
                8,
            ),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                16 + 8,
                16,
            ),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(
        self,
        compact,
        semantic,
    ):
        compact_z = (
            self.compact_branch(
                compact
            )
        )

        semantic_z = (
            self.semantic_branch(
                semantic
            )
        )

        merged = torch.cat(
            [
                compact_z,
                semantic_z,
            ],
            dim=1,
        )

        return self.fusion(
            merged
        ).squeeze(1)


# =========================================================
# PyTorch训练
# =========================================================

def fit_compact_net(
    compact_train,
    y_train,
    seed,
):
    seed_everything(seed)

    fit_idx, val_idx = (
        stratified_inner_split(
            y_train,
            seed,
            NN_VALID_RATIO,
        )
    )

    x_fit = torch.tensor(
        compact_train[fit_idx],
        dtype=torch.float32,
    )

    y_fit = torch.tensor(
        y_train[fit_idx],
        dtype=torch.float32,
    )

    x_val = torch.tensor(
        compact_train[val_idx],
        dtype=torch.float32,
        device=DEVICE,
    )

    y_val = torch.tensor(
        y_train[val_idx],
        dtype=torch.float32,
        device=DEVICE,
    )

    dataset = TensorDataset(
        x_fit,
        y_fit,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=min(
            NN_BATCH_SIZE,
            len(dataset),
        ),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = CompactNet(
        compact_train.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NN_LEARNING_RATE,
        weight_decay=NN_WEIGHT_DECAY,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(
        1,
        NN_MAX_EPOCHS + 1,
    ):
        model.train()

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits = model(
                x_val
            )

            val_loss = float(
                criterion(
                    val_logits,
                    y_val,
                ).item()
            )

        if (
            val_loss
            < best_loss - 1e-6
        ):
            best_loss = val_loss
            best_epoch = epoch

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }

            wait = 0

        else:
            wait += 1

        if wait >= NN_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(
            "Compact网络没有得到有效状态"
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_epoch,
        best_loss,
    )


def fit_two_branch_net(
    compact_train,
    semantic_train,
    y_train,
    seed,
):
    seed_everything(seed)

    fit_idx, val_idx = (
        stratified_inner_split(
            y_train,
            seed,
            NN_VALID_RATIO,
        )
    )

    compact_fit = torch.tensor(
        compact_train[fit_idx],
        dtype=torch.float32,
    )

    semantic_fit = torch.tensor(
        semantic_train[fit_idx],
        dtype=torch.float32,
    )

    y_fit = torch.tensor(
        y_train[fit_idx],
        dtype=torch.float32,
    )

    compact_val = torch.tensor(
        compact_train[val_idx],
        dtype=torch.float32,
        device=DEVICE,
    )

    semantic_val = torch.tensor(
        semantic_train[val_idx],
        dtype=torch.float32,
        device=DEVICE,
    )

    y_val = torch.tensor(
        y_train[val_idx],
        dtype=torch.float32,
        device=DEVICE,
    )

    dataset = TensorDataset(
        compact_fit,
        semantic_fit,
        y_fit,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=min(
            NN_BATCH_SIZE,
            len(dataset),
        ),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = TwoBranchNet(
        compact_train.shape[1],
        semantic_train.shape[1],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NN_LEARNING_RATE,
        weight_decay=NN_WEIGHT_DECAY,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(
        1,
        NN_MAX_EPOCHS + 1,
    ):
        model.train()

        for (
            compact_b,
            semantic_b,
            yb,
        ) in loader:

            compact_b = (
                compact_b.to(DEVICE)
            )

            semantic_b = (
                semantic_b.to(DEVICE)
            )

            yb = yb.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                compact_b,
                semantic_b,
            )

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits = model(
                compact_val,
                semantic_val,
            )

            val_loss = float(
                criterion(
                    val_logits,
                    y_val,
                ).item()
            )

        if (
            val_loss
            < best_loss - 1e-6
        ):
            best_loss = val_loss
            best_epoch = epoch

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }

            wait = 0

        else:
            wait += 1

        if wait >= NN_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(
            "两分支网络没有得到有效状态"
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_epoch,
        best_loss,
    )


def predict_compact_net(
    model,
    compact_test,
):
    model.eval()

    x = torch.tensor(
        compact_test,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        logits = model(x)

        probability = (
            torch.sigmoid(
                logits
            )
            .detach()
            .cpu()
            .numpy()
        )

    return probability


def predict_two_branch_net(
    model,
    compact_test,
    semantic_test,
):
    model.eval()

    compact = torch.tensor(
        compact_test,
        dtype=torch.float32,
        device=DEVICE,
    )

    semantic = torch.tensor(
        semantic_test,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        logits = model(
            compact,
            semantic,
        )

        probability = (
            torch.sigmoid(
                logits
            )
            .detach()
            .cpu()
            .numpy()
        )

    return probability


# =========================================================
# 汇总
# =========================================================

def summarize(metric_rows):
    grouped = defaultdict(list)

    for row in metric_rows:
        grouped[
            (
                row["variant"],
                row["split_name"],
            )
        ].append(row)

    per_split = []

    for (
        variant,
        split_name,
    ), rows in sorted(
        grouped.items()
    ):
        out = {
            "variant":
                variant,

            "split_name":
                split_name,

            "seed_count":
                len(rows),
        }

        for field in (
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auc",
            "fp",
            "fn",
        ):
            out[
                field + "_mean"
            ] = float(
                np.mean([
                    float(row[field])
                    for row in rows
                ])
            )

        per_split.append(out)

    by_variant = defaultdict(list)

    for row in per_split:
        by_variant[
            row["variant"]
        ].append(row)

    summary = []

    for variant, rows in (
        by_variant.items()
    ):
        ba = np.asarray(
            [
                row[
                    "balanced_accuracy_mean"
                ]
                for row in rows
            ],
            dtype=float,
        )

        summary.append({
            "variant":
                variant,

            "mean_BA":
                float(ba.mean()),

            "min_BA":
                float(ba.min()),

            "std_BA":
                float(ba.std()),

            "mean_AUC":
                float(np.mean([
                    row["auc_mean"]
                    for row in rows
                ])),

            "mean_ACC":
                float(np.mean([
                    row["accuracy_mean"]
                    for row in rows
                ])),

            "mean_F1":
                float(np.mean([
                    row["f1_mean"]
                    for row in rows
                ])),

            "mean_FP":
                float(np.mean([
                    row["fp_mean"]
                    for row in rows
                ])),

            "mean_FN":
                float(np.mean([
                    row["fn_mean"]
                    for row in rows
                ])),
        })

    summary.sort(
        key=lambda row: (
            row["mean_BA"],
            row["min_BA"],
            row["mean_AUC"],
        ),
        reverse=True,
    )

    return per_split, summary


# =========================================================
# 主程序
# =========================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    compact_rows = read_csv_rows(
        COMPACT_PATH
    )

    compact_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in compact_rows
    }

    schema = json.loads(
        SCHEMA_PATH.read_text(
            encoding="utf-8"
        )
    )

    semantic_rows = read_gz_csv(
        SEMANTIC_PATH
    )

    semantic_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in semantic_rows
    }

    names150 = support150_names(
        semantic_rows[0].keys()
    )

    splits = read_split_rows(
        SPLIT_PATH
    )

    all_shas = set()

    for parts in splits.values():
        for partition in (
            "train",
            "test",
        ):
            for row in parts[
                partition
            ]:
                all_shas.add(
                    row["sha256"]
                )

    if (
        all_shas
        - set(compact_by_sha)
    ):
        raise SystemExit(
            "Compact人口不完整"
        )

    if (
        all_shas
        - set(semantic_by_sha)
    ):
        raise SystemExit(
            "语义人口不完整"
        )

    print(
        "======================================"
    )
    print(
        "2017 分类器对照："
        "逻辑回归 vs 小网络"
    )
    print(
        "======================================"
    )

    print(
        "唯一SHA =",
        len(all_shas),
    )

    print(
        "设备 =",
        DEVICE,
    )

    if torch.cuda.is_available():
        print(
            "GPU =",
            torch.cuda.get_device_name(
                DEVICE
            ),
        )

    print(
        "冻结150维 =",
        len(names150),
    )

    print(
        "alpha =",
        ALPHA,
    )

    print(
        "threshold =",
        THRESHOLD,
    )

    print(
        "最大训练轮数 =",
        NN_MAX_EPOCHS,
    )

    print(
        "提前停止耐心 =",
        NN_PATIENCE,
    )

    metric_rows = []
    training_rows = []

    for split_name in sorted(
        splits
    ):
        train_rows = (
            splits[
                split_name
            ]["train"]
        )

        test_rows = (
            splits[
                split_name
            ]["test"]
        )

        y_train = np.asarray(
            [
                label_value(
                    compact_by_sha[
                        row["sha256"]
                    ].get(
                        "metadata_label",
                        "",
                    )
                )
                for row in train_rows
            ],
            dtype=np.int64,
        )

        y_test = np.asarray(
            [
                label_value(
                    compact_by_sha[
                        row["sha256"]
                    ].get(
                        "metadata_label",
                        "",
                    )
                )
                for row in test_rows
            ],
            dtype=np.int64,
        )

        (
            compact_train,
            compact_test,
            compact_names,
        ) = compact_matrix_named(
            train_rows,
            test_rows,
            compact_by_sha,
            schema,
        )

        (
            semantic_train,
            semantic_test,
            semantic_names,
        ) = standardize_named(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )

        semantic_train = (
            ALPHA
            * semantic_train
        )

        semantic_test = (
            ALPHA
            * semantic_test
        )

        joint_train = np.concatenate(
            [
                compact_train,
                semantic_train,
            ],
            axis=1,
        )

        joint_test = np.concatenate(
            [
                compact_test,
                semantic_test,
            ],
            axis=1,
        )

        print()
        print(
            split_name
        )

        print(
            "train=",
            len(train_rows),
            "test=",
            len(test_rows),
            "Compact=",
            len(compact_names),
            "语义可用=",
            len(semantic_names),
        )

        for seed in SEEDS:
            # ---------------------------------
            # 逻辑回归 Compact
            # ---------------------------------

            w, b = fit_logistic(
                compact_train,
                y_train,
                seed,
                iterations=LR_ITERATIONS,
                learning_rate=LR_RATE,
                l2=LR_L2,
            )

            p = sigmoid(
                compact_test @ w + b
            )

            result = metrics(
                y_test,
                p,
                threshold=THRESHOLD,
            )

            metric_rows.append({
                "variant":
                    "lr_compact",

                "split_name":
                    split_name,

                "seed":
                    seed,

                **result,
            })

            # ---------------------------------
            # 逻辑回归 Compact + 150
            # ---------------------------------

            w, b = fit_logistic(
                joint_train,
                y_train,
                seed,
                iterations=LR_ITERATIONS,
                learning_rate=LR_RATE,
                l2=LR_L2,
            )

            p = sigmoid(
                joint_test @ w + b
            )

            result = metrics(
                y_test,
                p,
                threshold=THRESHOLD,
            )

            metric_rows.append({
                "variant":
                    "lr_compact_plus150",

                "split_name":
                    split_name,

                "seed":
                    seed,

                **result,
            })

            # ---------------------------------
            # 小网络 Compact
            # ---------------------------------

            (
                model,
                best_epoch,
                best_loss,
            ) = fit_compact_net(
                compact_train,
                y_train,
                seed,
            )

            p = predict_compact_net(
                model,
                compact_test,
            )

            result = metrics(
                y_test,
                p,
                threshold=THRESHOLD,
            )

            metric_rows.append({
                "variant":
                    "nn_compact",

                "split_name":
                    split_name,

                "seed":
                    seed,

                **result,
            })

            training_rows.append({
                "variant":
                    "nn_compact",

                "split_name":
                    split_name,

                "seed":
                    seed,

                "best_epoch":
                    best_epoch,

                "best_validation_loss":
                    best_loss,
            })

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ---------------------------------
            # 两分支 Compact + 150
            # ---------------------------------

            (
                model,
                best_epoch,
                best_loss,
            ) = fit_two_branch_net(
                compact_train,
                semantic_train,
                y_train,
                seed,
            )

            p = (
                predict_two_branch_net(
                    model,
                    compact_test,
                    semantic_test,
                )
            )

            result = metrics(
                y_test,
                p,
                threshold=THRESHOLD,
            )

            metric_rows.append({
                "variant":
                    "nn_two_branch_plus150",

                "split_name":
                    split_name,

                "seed":
                    seed,

                **result,
            })

            training_rows.append({
                "variant":
                    "nn_two_branch_plus150",

                "split_name":
                    split_name,

                "seed":
                    seed,

                "best_epoch":
                    best_epoch,

                "best_validation_loss":
                    best_loss,
            })

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(
                " seed",
                seed,
                "完成",
                flush=True,
            )

    per_split, summary = (
        summarize(
            metric_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "per_seed_metrics.csv",
        metric_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_split.csv",
        per_split,
    )

    write_csv(
        OUTPUT_DIR
        / "summary.csv",
        summary,
    )

    write_csv(
        OUTPUT_DIR
        / "training_diagnostics.csv",
        training_rows,
    )

    print()
    print(
        "======================================"
    )
    print(
        "2017结果"
    )
    print(
        "======================================"
    )

    for row in summary:
        print(
            f"{row['variant']:28s} "
            f"BA={row['mean_BA']:.6f} "
            f"min={row['min_BA']:.6f} "
            f"std={row['std_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"FP={row['mean_FP']:.2f} "
            f"FN={row['mean_FN']:.2f}"
        )

    print()
    print(
        "当前正式线性+150参考："
    )
    print(
        "BA≈0.906712 "
        "min≈0.890250 "
        "AUC≈0.950344"
    )


if __name__ == "__main__":
    main()
