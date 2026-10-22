#!/usr/bin/env python3

from __future__ import annotations

import copy
import csv
import inspect
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = ROOT / "scripts"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import run_bounded_additive_residual_3way_2017 as bounded

# E使用no-structural/no-junction图表示。
exp = bounded.m


YEARS = (
    2018,
    2019,
    2020,
    2021,
)

SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

THRESHOLD = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cpu"
)


SOURCE_BASE = (
    ROOT
    / "artifacts/allratio70-source-base-capture/"
    "source_base_predictions.csv"
)

SOURCE_EPOCH_FILE = (
    ROOT
    / "artifacts/bounded-additive-residual-3way-2017/"
    "per_seed.csv"
)

FUTURE_BASE = (
    ROOT
    / "artifacts/allratio70-compact-2017-2021/"
    "future_predictions.csv"
)

SPLIT_FILE = (
    ROOT
    / "artifacts/p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

SOURCE_GRAPH = (
    ROOT
    / "artifacts/top5-semantic-behavior-graph-v1-2017-trigger551/"
    "binding_cores_v3_1_semantic_junction_train_safe.jsonl"
)

FUTURE_GRAPH = {
    year:
    ROOT
    / f"artifacts/top5-semantic-behavior-graph-v1-{year}-triggered/"
    "binding_cores_v3_1_semantic_junction_train_safe.jsonl"
    for year in YEARS
}

OUTPUT = (
    ROOT
    / "artifacts/bounded-additive-residual-3way-2017-2021-formal"
)


EXPECTED_BASE_MEAN_BA = {
    2018: 0.921432,
    2019: 0.914925,
    2020: 0.876593,
    2021: 0.900000,
}


def norm_sha(x):
    return str(x).strip().lower()


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_label(x):
    s = str(x).strip().lower()

    if s in {
        "1",
        "1.0",
        "malware",
        "malicious",
        "true",
    }:
        return 1

    if s in {
        "0",
        "0.0",
        "benign",
        "false",
    }:
        return 0

    raise ValueError(
        "无法解析标签: "
        + repr(x)
    )


def sigmoid(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = np.clip(
        x,
        -50.0,
        50.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-x)
        )
    )


def probability_to_logit(p):
    p = float(p)

    p = min(
        max(
            p,
            1e-7,
        ),
        1.0 - 1e-7,
    )

    return float(
        np.log(
            p / (1.0 - p)
        )
    )


def combine(
    base_logit,
    delta,
):
    """Frozen E bounded-additive rule selected on 2017 only."""

    alpha = float(bounded.BOUNDED_ALPHA)
    correction = alpha * np.tanh(float(delta) / alpha)
    return float(base_logit + correction)


def metric_dict(
    y,
    p,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    pred = (
        p >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[
                0,
                1,
            ],
        ).ravel()
    )

    return {
        "BA":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "ACC":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "AUC":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),
    }


def scalar(x):
    if torch.is_tensor(x):
        return float(
            x.detach()
            .cpu()
            .reshape(-1)[0]
            .item()
        )

    arr = np.asarray(x)

    return float(
        arr.reshape(-1)[0]
    )


# ============================================================
# 训练轮数：只能来自2017
# ============================================================

def source_selected_epoch():

    rows = [
        row
        for row in read_csv(SOURCE_EPOCH_FILE)
        if row.get("variant") == "E_local_bounded_additive"
    ]

    epochs = [
        int(row["best_epoch"])
        for row in rows
    ]

    if len(epochs) != 15:
        raise RuntimeError(
            "2017 E内部最佳轮数应为15条，实际="
            + str(len(epochs))
        )

    median = int(statistics.median(epochs))

    print("2017 E best epochs =", sorted(epochs))
    print("source-only E median epoch =", median)

    if median != 26:
        raise RuntimeError(
            "E源域固定轮数不再是26，停止实验"
        )

    return median


def load_source_base():

    rows = read_csv(
        SOURCE_BASE
    )

    by_seed = defaultdict(
        list
    )

    for r in rows:
        seed = int(
            r["seed"]
        )

        by_seed[
            seed
        ].append({
            "sha256":
                norm_sha(
                    r["sha256"]
                ),

            "label":
                int(
                    r["label"]
                ),

            "base_logit":
                float(
                    r[
                        "base_logit"
                    ]
                ),

            "base_probability":
                float(
                    r[
                        "base_probability"
                    ]
                ),

            "order_index":
                int(
                    r[
                        "order_index"
                    ]
                ),
        })

    for seed in SEEDS:

        by_seed[
            seed
        ].sort(
            key=lambda r:
            r["order_index"]
        )

        if len(
            by_seed[
                seed
            ]
        ) != 799:
            raise RuntimeError(
                f"seed={seed} "
                f"2017 Compact数量="
                f"{len(by_seed[seed])}, "
                "预期799"
            )

    return by_seed


# ============================================================
# 未来Compact：严格保留随机种子
# ============================================================

def load_future_base():

    rows = read_csv(
        FUTURE_BASE
    )

    out = {}

    for r in rows:

        year = int(
            r["year"]
        )

        if year not in YEARS:
            continue

        seed = int(
            r["seed"]
        )

        sha = norm_sha(
            r["sha256"]
        )

        key = (
            year,
            seed,
            sha,
        )

        if key in out:
            raise RuntimeError(
                "重复未来Compact记录: "
                + repr(key)
            )

        out[key] = float(
            r[
                "base_probability"
            ]
        )

    return out


# ============================================================
# 未来标签：
# 只用于预测完成后的指标计算
# ============================================================

def load_future_labels():

    rows = read_csv(
        SPLIT_FILE
    )

    labels = {
        year: {}
        for year in YEARS
    }

    order = {
        year: []
        for year in YEARS
    }

    for year in YEARS:

        split_name = (
            f"train_fixed_2017__test_{year}"
        )

        for r in rows:

            if (
                r.get(
                    "split_name"
                )
                != split_name
            ):
                continue

            if (
                r.get(
                    "partition"
                )
                != "test"
            ):
                continue

            if int(
                r["year"]
            ) != year:
                continue

            sha = norm_sha(
                r["sha256"]
            )

            if sha in labels[
                year
            ]:
                continue

            labels[
                year
            ][sha] = parse_label(
                r["label"]
            )

            order[
                year
            ].append(
                sha
            )

    return (
        labels,
        order,
    )


# ============================================================
# 图
# ============================================================

def unpack_vocab(obj):

    if not isinstance(
        obj,
        (tuple, list),
    ):
        raise RuntimeError(
            "scan_vocabulary返回值异常"
        )

    if len(obj) < 2:
        raise RuntimeError(
            "scan_vocabulary返回值不足2项"
        )

    return (
        obj[0],
        obj[1],
    )


def unpack_graphs(obj):

    if isinstance(
        obj,
        dict,
    ):
        result = obj

    elif isinstance(
        obj,
        (tuple, list),
    ):
        candidates = [
            x
            for x in obj
            if isinstance(
                x,
                dict,
            )
        ]

        if not candidates:
            raise RuntimeError(
                "load_graphs没有返回字典"
            )

        result = candidates[0]

    else:
        raise RuntimeError(
            "load_graphs返回值异常"
        )

    return {
        norm_sha(k): v
        for k, v in result.items()
    }


def load_graph_sets():

    (
        role_to_idx,
        edge_to_idx,
    ) = unpack_vocab(
        exp.scan_vocabulary(
            SOURCE_GRAPH
        )
    )

    print(
        "2017 role vocabulary =",
        len(role_to_idx),
    )

    print(
        "2017 edge vocabulary =",
        len(edge_to_idx),
    )

    # 未来只能使用2017已有词表。
    # 如果出现2017完全没有的新语义类别，
    # 不允许偷偷扩词表。
    for year in YEARS:

        (
            future_roles,
            future_edges,
        ) = unpack_vocab(
            exp.scan_vocabulary(
                FUTURE_GRAPH[
                    year
                ]
            )
        )

        unknown_roles = (
            set(future_roles)
            - set(role_to_idx)
        )

        unknown_edges = (
            set(future_edges)
            - set(edge_to_idx)
        )

        if unknown_roles:
            raise RuntimeError(
                f"{year}出现2017未见语义角色: "
                + repr(
                    sorted(
                        unknown_roles
                    )
                )
            )

        if unknown_edges:
            raise RuntimeError(
                f"{year}出现2017未见边类型: "
                + repr(
                    sorted(
                        unknown_edges
                    )
                )
            )

    source_graphs = (
        unpack_graphs(
            exp.load_graphs(
                SOURCE_GRAPH,
                role_to_idx,
                edge_to_idx,
            )
        )
    )

    future_graphs = {}

    for year in YEARS:

        future_graphs[
            year
        ] = unpack_graphs(
            exp.load_graphs(
                FUTURE_GRAPH[
                    year
                ],
                role_to_idx,
                edge_to_idx,
            )
        )

    return (
        role_to_idx,
        edge_to_idx,
        source_graphs,
        future_graphs,
    )


# ============================================================
# 固定18轮训练
# ============================================================

def extract_model(obj):

    if isinstance(
        obj,
        exp.BehaviorResidualNet,
    ):
        return obj

    if isinstance(
        obj,
        (tuple, list),
    ):
        for x in obj:
            if isinstance(
                x,
                exp.BehaviorResidualNet,
            ):
                return x

    raise RuntimeError(
        "无法从训练函数返回值中取得残差模型"
    )


def fit_fixed_epochs(
    rows,
    labels,
    base_logits,
    graphs_by_sha,
    role_count,
    edge_type_count,
    seed,
    fixed_epochs,
):

    source = inspect.getsource(bounded.fit_variant)

    if source.count("load_state_dict") != 1:
        raise RuntimeError(
            "bounded.fit_variant代码结构发生变化，请先检查"
        )

    original = exp.BehaviorResidualNet.load_state_dict

    def no_reload(self, state_dict, strict=True):
        # 阻止训练函数重新加载“训练集同时作为验证集”的最佳参数；
        # 保留完整第26轮参数。
        return None

    exp.BehaviorResidualNet.load_state_dict = no_reload

    try:
        result = bounded.fit_variant(
            rows,
            rows,
            labels,
            labels,
            base_logits,
            base_logits,
            graphs_by_sha,
            role_count,
            edge_type_count,
            seed,
            DEVICE,
            fixed_epochs,
            fixed_epochs + 100,
            BATCH_SIZE,
            LEARNING_RATE,
            WEIGHT_DECAY,
            False,
            "bounded_additive",
        )
    finally:
        exp.BehaviorResidualNet.load_state_dict = original

    model = result[0]
    model.eval()
    return model


def extract_delta_map(obj):

    if isinstance(
        obj,
        dict,
    ):
        result = obj

    elif isinstance(
        obj,
        (tuple, list),
    ):
        candidates = [
            x
            for x in obj
            if isinstance(
                x,
                dict,
            )
        ]

        if not candidates:
            raise RuntimeError(
                "predict_delta_map未返回字典"
            )

        result = candidates[0]

    else:
        raise RuntimeError(
            "predict_delta_map返回值异常"
        )

    return {
        norm_sha(k):
        scalar(v)
        for k, v in result.items()
    }


def predict_one_year(
    model,
    year,
    seed,
    sha_order,
    base_prob_map,
    graphs_by_sha,
):

    base_logit_by_sha = {}

    for sha in sha_order:

        key = (
            year,
            seed,
            sha,
        )

        if key not in base_prob_map:
            raise RuntimeError(
                "缺少未来Compact概率: "
                + repr(key)
            )

        base_logit_by_sha[
            sha
        ] = probability_to_logit(
            base_prob_map[
                key
            ]
        )

    # ========================================================
    # 只有真正拥有有效v2行为核心的APK进入局部残差模型。
    #
    # 没有图的APK绝不能传给predict_delta_map，
    # 它们后面直接令 delta=0，
    # 因而最终输出严格等于Compact。
    # ========================================================

    enabled_shas = [
        sha
        for sha in sha_order
        if (
            sha in graphs_by_sha
            and bool(
                graphs_by_sha[sha]
            )
        )
    ]

    # 未来真实标签不进入残差预测。
    # label参数只是旧接口要求，这里统一传0占位。
    dummy_labels = {
        sha: 0
        for sha in enabled_shas
    }

    if enabled_shas:

        raw_delta = (
            bounded.predict_delta_map(
                model,
                enabled_shas,
                graphs_by_sha,
                base_logit_by_sha,
                dummy_labels,
                DEVICE,
                BATCH_SIZE,
            )
        )

        delta_map = extract_delta_map(
            raw_delta
        )

    else:
        delta_map = {}

    rows = []

    for sha in sha_order:

        base_prob = float(
            base_prob_map[
                (
                    year,
                    seed,
                    sha,
                )
            ]
        )

        base_logit = (
            base_logit_by_sha[
                sha
            ]
        )

        enabled = (
            sha in graphs_by_sha
            and bool(
                graphs_by_sha[
                    sha
                ]
            )
        )

        delta = (
            float(
                delta_map.get(
                    sha,
                    0.0,
                )
            )
            if enabled
            else 0.0
        )

        final_logit = combine(
            base_logit,
            delta,
        )

        final_prob = float(
            sigmoid(
                final_logit
            )
        )

        rows.append({
            "year":
                year,

            "seed":
                seed,

            "sha256":
                sha,

            "graph_enabled":
                int(enabled),

            "base_probability":
                base_prob,

            "delta":
                delta,

            "final_probability":
                final_prob,
        })

    return rows


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixed_epochs = (
        source_selected_epoch()
    )

    source_by_seed = (
        load_source_base()
    )

    future_base = (
        load_future_base()
    )

    (
        future_labels,
        future_order,
    ) = load_future_labels()

    (
        role_to_idx,
        edge_to_idx,
        source_graphs,
        future_graphs,
    ) = load_graph_sets()

    print()
    print(
        "===== GRAPH COVERAGE ====="
    )

    print(
        "2017 graph-enabled APK =",
        len(source_graphs),
    )

    for year in YEARS:
        enabled = len(
            set(
                future_order[
                    year
                ]
            )
            & set(
                future_graphs[
                    year
                ]
            )
        )

        print(
            year,
            "graph-enabled APK =",
            enabled,
            "/",
            len(
                future_order[
                    year
                ]
            ),
        )

    # --------------------------------------------------------
    # 先复现未来Compact基线
    # --------------------------------------------------------

    print()
    print(
        "===== COMPACT BASELINE CHECK ====="
    )

    base_check_rows = []

    for year in YEARS:

        seed_metrics = []

        for seed in SEEDS:

            shas = (
                future_order[
                    year
                ]
            )

            y = np.asarray(
                [
                    future_labels[
                        year
                    ][sha]
                    for sha in shas
                ],
                dtype=int,
            )

            p = np.asarray(
                [
                    future_base[
                        (
                            year,
                            seed,
                            sha,
                        )
                    ]
                    for sha in shas
                ],
                dtype=float,
            )

            m = metric_dict(
                y,
                p,
            )

            seed_metrics.append(
                m["BA"]
            )

        mean_ba = float(
            np.mean(
                seed_metrics
            )
        )

        print(
            f"{year}: "
            f"mean BA={mean_ba:.6f}"
        )

        expected = (
            EXPECTED_BASE_MEAN_BA[
                year
            ]
        )

        if abs(
            mean_ba
            - expected
        ) > 5e-6:
            raise RuntimeError(
                f"{year} Compact基线不一致: "
                f"{mean_ba:.9f} "
                f"vs {expected:.9f}"
            )

    print(
        "Compact baseline reproduced."
    )

    # --------------------------------------------------------
    # 五个随机种子：
    # 同一个种子的Compact -> 同一个种子的残差
    # --------------------------------------------------------

    per_seed_rows = []
    prediction_rows = []

    for seed in SEEDS:

        print()
        print(
            "=" * 72
        )

        print(
            f"RANDOM SEED {seed} | "
            f"FIXED EPOCHS {fixed_epochs}"
        )

        print(
            "=" * 72
        )

        source = (
            source_by_seed[
                seed
            ]
        )

        train_rows = [
            {
                "sha256":
                    r["sha256"]
            }
            for r in source
        ]

        y_train = np.asarray(
            [
                r["label"]
                for r in source
            ],
            dtype=int,
        )

        base_logits = np.asarray(
            [
                r[
                    "base_logit"
                ]
                for r in source
            ],
            dtype=float,
        )

        enabled_train = sum(
            1
            for r in train_rows
            if r[
                "sha256"
            ] in source_graphs
        )

        print(
            "2017 train =",
            len(train_rows),
            "graph-enabled =",
            enabled_train,
        )

        model = fit_fixed_epochs(
            train_rows,
            y_train,
            base_logits,
            source_graphs,
            len(role_to_idx),
            len(edge_to_idx),
            seed,
            fixed_epochs,
        )

        torch.save(
            {
                "seed":
                    seed,

                "fixed_epochs":
                    fixed_epochs,

                "state_dict":
                    model.state_dict(),

                "role_to_idx":
                    role_to_idx,

                "edge_to_idx":
                    edge_to_idx,
            },
            OUTPUT
            / f"residual_seed{seed}.pt",
        )

        for year in YEARS:

            pred_rows = predict_one_year(
                model,
                year,
                seed,
                future_order[
                    year
                ],
                future_base,
                future_graphs[
                    year
                ],
            )

            # 到这里预测已经全部完成，
            # 现在才加入未来真实标签计算指标。
            y = np.asarray(
                [
                    future_labels[
                        year
                    ][
                        r[
                            "sha256"
                        ]
                    ]
                    for r in pred_rows
                ],
                dtype=int,
            )

            bp = np.asarray(
                [
                    r[
                        "base_probability"
                    ]
                    for r in pred_rows
                ],
                dtype=float,
            )

            fp = np.asarray(
                [
                    r[
                        "final_probability"
                    ]
                    for r in pred_rows
                ],
                dtype=float,
            )

            bm = metric_dict(
                y,
                bp,
            )

            fm = metric_dict(
                y,
                fp,
            )

            bpred = (
                bp >= THRESHOLD
            ).astype(int)

            fpred = (
                fp >= THRESHOLD
            ).astype(int)

            changed_mask = (
                bpred
                != fpred
            )

            changed = int(
                np.sum(
                    changed_mask
                )
            )

            corrected = int(
                np.sum(
                    changed_mask
                    & (
                        fpred == y
                    )
                )
            )

            regressed = int(
                np.sum(
                    changed_mask
                    & (
                        bpred == y
                    )
                )
            )

            enabled = sum(
                r[
                    "graph_enabled"
                ]
                for r in pred_rows
            )

            per_seed_rows.append({
                "year":
                    year,

                "seed":
                    seed,

                "fixed_epoch":
                    fixed_epochs,

                "n":
                    len(y),

                "enabled":
                    enabled,

                "base_BA":
                    bm["BA"],

                "final_BA":
                    fm["BA"],

                "delta_BA":
                    fm["BA"]
                    - bm["BA"],

                "base_ACC":
                    bm["ACC"],

                "final_ACC":
                    fm["ACC"],

                "base_AUC":
                    bm["AUC"],

                "final_AUC":
                    fm["AUC"],

                "base_FP":
                    bm["FP"],

                "final_FP":
                    fm["FP"],

                "base_FN":
                    bm["FN"],

                "final_FN":
                    fm["FN"],

                "changed":
                    changed,

                "corrected":
                    corrected,

                "regressed":
                    regressed,
            })

            print(
                f"{year}: "
                f"BA {bm['BA']:.6f}"
                f" -> {fm['BA']:.6f} "
                f"({(fm['BA']-bm['BA'])*100:+.3f}pp) "
                f"| AUC {bm['AUC']:.6f}"
                f" -> {fm['AUC']:.6f} "
                f"| FP {bm['FP']}->{fm['FP']} "
                f"| FN {bm['FN']}->{fm['FN']} "
                f"| changed={changed} "
                f"corrected={corrected} "
                f"regressed={regressed}"
            )

            for i, r in enumerate(
                pred_rows
            ):
                prediction_rows.append({
                    **r,

                    "label":
                        int(
                            y[i]
                        ),
                })

    # --------------------------------------------------------
    # 年份 × 随机种子后再平均
    # --------------------------------------------------------

    per_year_rows = []

    print()
    print(
        "=" * 72
    )

    print(
        "FORMAL FUTURE SUMMARY"
    )

    print(
        "=" * 72
    )

    for year in YEARS:

        rs = [
            r
            for r in per_seed_rows
            if r[
                "year"
            ] == year
        ]

        def mean(name):
            return float(
                np.mean(
                    [
                        r[name]
                        for r in rs
                    ]
                )
            )

        row = {
            "year":
                year,

            "base_BA":
                mean(
                    "base_BA"
                ),

            "final_BA":
                mean(
                    "final_BA"
                ),

            "delta_BA":
                mean(
                    "final_BA"
                )
                - mean(
                    "base_BA"
                ),

            "base_ACC":
                mean(
                    "base_ACC"
                ),

            "final_ACC":
                mean(
                    "final_ACC"
                ),

            "base_AUC":
                mean(
                    "base_AUC"
                ),

            "final_AUC":
                mean(
                    "final_AUC"
                ),

            "base_FP":
                mean(
                    "base_FP"
                ),

            "final_FP":
                mean(
                    "final_FP"
                ),

            "base_FN":
                mean(
                    "base_FN"
                ),

            "final_FN":
                mean(
                    "final_FN"
                ),

            "changed":
                mean(
                    "changed"
                ),

            "corrected":
                mean(
                    "corrected"
                ),

            "regressed":
                mean(
                    "regressed"
                ),

            "enabled":
                mean(
                    "enabled"
                ),
        }

        per_year_rows.append(
            row
        )

        print(
            f"{year}: "
            f"BA {row['base_BA']:.6f}"
            f" -> {row['final_BA']:.6f} "
            f"({row['delta_BA']*100:+.3f}pp) "
            f"| AUC {row['base_AUC']:.6f}"
            f" -> {row['final_AUC']:.6f} "
            f"| FP {row['base_FP']:.2f}"
            f"->{row['final_FP']:.2f} "
            f"| FN {row['base_FN']:.2f}"
            f"->{row['final_FN']:.2f} "
            f"| corrected={row['corrected']:.2f} "
            f"regressed={row['regressed']:.2f}"
        )

    base_mean_ba = float(
        np.mean(
            [
                r[
                    "base_BA"
                ]
                for r
                in per_year_rows
            ]
        )
    )

    final_mean_ba = float(
        np.mean(
            [
                r[
                    "final_BA"
                ]
                for r
                in per_year_rows
            ]
        )
    )

    base_min_ba = float(
        np.min(
            [
                r[
                    "base_BA"
                ]
                for r
                in per_year_rows
            ]
        )
    )

    final_min_ba = float(
        np.min(
            [
                r[
                    "final_BA"
                ]
                for r
                in per_year_rows
            ]
        )
    )

    base_mean_auc = float(
        np.mean(
            [
                r[
                    "base_AUC"
                ]
                for r
                in per_year_rows
            ]
        )
    )

    final_mean_auc = float(
        np.mean(
            [
                r[
                    "final_AUC"
                ]
                for r
                in per_year_rows
            ]
        )
    )

    print()
    print(
        "future mean BA = "
        f"{base_mean_ba:.6f}"
        f" -> {final_mean_ba:.6f} "
        f"({(final_mean_ba-base_mean_ba)*100:+.3f}pp)"
    )

    print(
        "future min yearly mean BA = "
        f"{base_min_ba:.6f}"
        f" -> {final_min_ba:.6f}"
    )

    print(
        "future mean AUC = "
        f"{base_mean_auc:.6f}"
        f" -> {final_mean_auc:.6f}"
    )

    print(
        "worst year-seed final BA = "
        f"{min(r['final_BA'] for r in per_seed_rows):.6f}"
    )

    write_csv(
        OUTPUT
        / "per_seed.csv",
        per_seed_rows,
    )

    write_csv(
        OUTPUT
        / "per_year.csv",
        per_year_rows,
    )

    write_csv(
        OUTPUT
        / "predictions.csv",
        prediction_rows,
    )

    with (
        OUTPUT
        / "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "compact":
                    "allratio70",

                "source_year":
                    2017,

                "future_years":
                    list(
                        YEARS
                    ),

                "seeds":
                    list(
                        SEEDS
                    ),

                "fixed_epoch":
                    fixed_epochs,

                "epoch_selection":
                    (
                        "median of 15 source-only "
                        "2017 rolling-validation best epochs"
                    ),

                "threshold":
                    THRESHOLD,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "future_labels_used_for_training":
                    False,

                "future_labels_used_for_prediction":
                    False,

                "future_labels_used_for_metrics_only":
                    True,
            },
            f,
            indent=2,
        )

    print(
        "saved =",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
