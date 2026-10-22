#!/usr/bin/env python3

from __future__ import annotations

import csv
import inspect
import json
import math
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


import run_behavior_core_v2_residual_allratio70_2017 as exp


# ============================================================
# 冻结协议
# ============================================================

YEARS = (2018, 2019, 2020, 2021)
SEEDS = (0, 1, 2, 3, 4)

FIXED_EPOCHS = 18
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# 固定0.5
THRESHOLD = 0.5

DEVICE = torch.device("cpu")


SPLIT_FILE = (
    ROOT
    / "artifacts/p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

BASE_PRED_ROOT = (
    ROOT
    / "artifacts/allratio70-compact-2017-2021"
)

TRAIN_GRAPH = (
    ROOT
    / "artifacts/top5-semantic-behavior-graph-v1-2017-trigger551/"
    "binding_cores_v2_train_safe.jsonl"
)

FUTURE_GRAPH = {
    year:
    ROOT
    / f"artifacts/top5-semantic-behavior-graph-v1-{year}-triggered/"
    "binding_cores_v2_train_safe.jsonl"
    for year in YEARS
}

OUTPUT = (
    ROOT
    / "artifacts/behavior-core-v2-residual-allratio70-2017-2021"
)


# 已经正式得到的全比例 Compact 基线。
# 用于防止误读了别的 prediction 文件。
EXPECTED_BASE_BA = {
    2018: 0.921432,
    2019: 0.914925,
    2020: 0.876593,
    2021: 0.900000,
}


# ============================================================
# 基础工具
# ============================================================

def norm_sha(x):
    return str(x).strip().lower()


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
        "cannot parse label: "
        + repr(x)
    )


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        w.writeheader()
        w.writerows(rows)


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

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def probability_to_logit(p):
    p = np.asarray(
        p,
        dtype=np.float64,
    )

    p = np.clip(
        p,
        1e-7,
        1.0 - 1e-7,
    )

    return np.log(
        p / (1.0 - p)
    )


def residual_combine(
    base_logit,
    delta,
):
    """
    z_final =
        z_global
        + 4*p*(1-p)*tanh(delta)
    """

    p = sigmoid(base_logit)

    return (
        base_logit
        + 4.0
        * p
        * (1.0 - p)
        * np.tanh(delta)
    )


def metrics(
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
            labels=[0, 1],
        ).ravel()
    )

    return {
        "BA":
            balanced_accuracy_score(
                y,
                pred,
            ),

        "ACC":
            accuracy_score(
                y,
                pred,
            ),

        "AUC":
            roc_auc_score(
                y,
                p,
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

    if isinstance(
        x,
        np.ndarray,
    ):
        return float(
            x.reshape(-1)[0]
        )

    return float(x)


# ============================================================
# 固定划分
# ============================================================

def load_manifest():
    rows = read_csv(
        SPLIT_FILE
    )

    train_map = {}
    future_map = {
        y: {}
        for y in YEARS
    }

    for r in rows:
        sha = norm_sha(
            r["sha256"]
        )

        year = int(
            r["year"]
        )

        part = str(
            r["partition"]
        ).strip().lower()

        label = parse_label(
            r["label"]
        )

        if (
            year == 2017
            and part == "train"
        ):
            if sha in train_map:
                old = parse_label(
                    train_map[sha]["label"]
                )

                if old != label:
                    raise RuntimeError(
                        "2017 label conflict: "
                        + sha
                    )

            else:
                train_map[sha] = dict(r)

        if (
            year in YEARS
            and part == "test"
        ):
            if sha in future_map[year]:
                old = parse_label(
                    future_map[
                        year
                    ][sha]["label"]
                )

                if old != label:
                    raise RuntimeError(
                        "future label conflict: "
                        + sha
                    )

            else:
                future_map[
                    year
                ][sha] = dict(r)

    return (
        train_map,
        future_map,
    )


# ============================================================
# 读取已经冻结的全比例 Compact 概率
# ============================================================

def infer_year(row):
    raw = str(
        row.get(
            "year",
            "",
        )
    ).strip()

    if raw:
        try:
            y = int(
                float(raw)
            )

            if y in (
                2017,
                2018,
                2019,
                2020,
                2021,
            ):
                return y
        except Exception:
            pass

    text = " ".join(
        str(
            row.get(
                k,
                "",
            )
        )
        for k in (
            "split_name",
            "test_year",
            "dataset",
        )
    )

    for y in (
        2017,
        2018,
        2019,
        2020,
        2021,
    ):
        if str(y) in text:
            return y

    return None


def load_base_probabilities():
    values = defaultdict(list)
    used_files = []

    for path in sorted(
        BASE_PRED_ROOT.rglob(
            "*.csv"
        )
    ):
        try:
            with path.open(
                newline="",
                encoding="utf-8",
            ) as f:
                reader = csv.DictReader(
                    f
                )

                fields = (
                    reader.fieldnames
                    or []
                )

                if (
                    "base_probability"
                    not in fields
                ):
                    continue

                sha_field = None

                for candidate in (
                    "sha256",
                    "sha",
                    "sample_id",
                ):
                    if candidate in fields:
                        sha_field = candidate
                        break

                if sha_field is None:
                    continue

                n = 0

                for r in reader:
                    sha = norm_sha(
                        r.get(
                            sha_field,
                            "",
                        )
                    )

                    if not sha:
                        continue

                    year = infer_year(
                        r
                    )

                    if year is None:
                        continue

                    try:
                        prob = float(
                            r[
                                "base_probability"
                            ]
                        )
                    except Exception:
                        continue

                    values[
                        (
                            year,
                            sha,
                        )
                    ].append(
                        prob
                    )

                    n += 1

                if n:
                    used_files.append(
                        (
                            path,
                            n,
                        )
                    )

        except Exception:
            continue

    if not values:
        raise RuntimeError(
            "No base_probability rows found under "
            + str(
                BASE_PRED_ROOT
            )
        )

    print(
        "\n===== Compact prediction files ====="
    )

    for path, n in used_files:
        print(
            path,
            "rows=",
            n,
        )

    out = {}

    for key, xs in values.items():
        arr = np.asarray(
            xs,
            dtype=float,
        )

        # 同一个APK如果因4个未来split重复出现，
        # Compact概率应该完全一致。
        if (
            len(arr) > 1
            and float(
                np.max(arr)
                - np.min(arr)
            ) > 1e-7
        ):
            raise RuntimeError(
                "Conflicting Compact probabilities "
                f"for {key}: "
                f"min={arr.min()} "
                f"max={arr.max()} "
                f"n={len(arr)}"
            )

        out[key] = float(
            arr[0]
        )

    return out


# ============================================================
# 图读取
# ============================================================

def unwrap_graphs(obj):
    if isinstance(
        obj,
        dict,
    ):
        return obj

    if isinstance(
        obj,
        tuple,
    ):
        candidates = [
            x
            for x in obj
            if isinstance(
                x,
                dict,
            )
        ]

        if candidates:
            return candidates[0]

    raise RuntimeError(
        "Unexpected load_graphs return type: "
        + repr(
            type(obj)
        )
    )


def load_all_graphs():

    vocab = exp.scan_vocabulary(
        TRAIN_GRAPH
    )

    if (
        not isinstance(
            vocab,
            tuple,
        )
        or len(vocab) < 2
    ):
        raise RuntimeError(
            "Unexpected scan_vocabulary result: "
            + repr(vocab)
        )

    role_to_idx = vocab[0]
    edge_to_idx = vocab[1]

    if not isinstance(
        role_to_idx,
        dict,
    ):
        raise RuntimeError(
            "role vocabulary is not dict"
        )

    if not isinstance(
        edge_to_idx,
        dict,
    ):
        raise RuntimeError(
            "edge vocabulary is not dict"
        )

    print()
    print(
        "role vocabulary =",
        len(role_to_idx),
    )

    print(
        "edge vocabulary =",
        len(edge_to_idx),
    )

    train_graphs = unwrap_graphs(
        exp.load_graphs(
            TRAIN_GRAPH,
            role_to_idx,
            edge_to_idx,
        )
    )

    future_graphs = {}

    for year in YEARS:
        future_graphs[
            year
        ] = unwrap_graphs(
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
        train_graphs,
        future_graphs,
    )


# ============================================================
# 固定18轮训练
# ============================================================

def extract_model(result):
    if isinstance(
        result,
        exp.BehaviorResidualNet,
    ):
        return result

    if isinstance(
        result,
        (tuple, list),
    ):
        for item in result:
            if isinstance(
                item,
                exp.BehaviorResidualNet,
            ):
                return item

    raise RuntimeError(
        "Could not find BehaviorResidualNet "
        "in fit result: "
        + repr(
            type(result)
        )
    )


def fit_fixed_epochs(
    train_rows,
    y_train,
    train_base_logits,
    graphs_by_sha,
    role_count,
    edge_type_count,
    seed,
):

    source = inspect.getsource(
        exp.fit_behavior_residual
    )

    # 如果以后训练函数结构被改过，
    # 这里宁可停下来，也不要偷偷跑错协议。
    n_reload = source.count(
        "load_state_dict"
    )

    if n_reload != 1:
        raise RuntimeError(
            "fit_behavior_residual changed: "
            f"load_state_dict count={n_reload}. "
            "Stop and inspect before running."
        )

    for forbidden in (
        "ReduceLROnPlateau",
        "lr_scheduler",
        "scheduler.step",
    ):
        if forbidden in source:
            raise RuntimeError(
                "Training function now contains "
                f"{forbidden}; "
                "fixed-epoch wrapper must be reviewed."
            )

    # 原函数最后会把“验证最佳轮次”的参数重新装回来。
    # 最终时间迁移实验不允许再选checkpoint。
    #
    # 因此这里只屏蔽最后那一次 reload。
    # patience设置得远大于18，保证18轮完整训练。
    original_load = (
        exp.BehaviorResidualNet
        .load_state_dict
    )

    def keep_final_epoch(
        self,
        state_dict,
        strict=True,
    ):
        return None

    exp.BehaviorResidualNet.load_state_dict = (
        keep_final_epoch
    )

    try:
        result = (
            exp.fit_behavior_residual(
                train_rows,
                train_rows,
                y_train,
                y_train,
                train_base_logits,
                train_base_logits,
                graphs_by_sha,
                role_count,
                edge_type_count,
                seed,
                DEVICE,
                FIXED_EPOCHS,
                FIXED_EPOCHS + 100,
                BATCH_SIZE,
                LEARNING_RATE,
                WEIGHT_DECAY,
            )
        )

    finally:
        (
            exp.BehaviorResidualNet
            .load_state_dict
        ) = original_load

    model = extract_model(
        result
    )

    model.eval()

    return model


# ============================================================
# 预测
# ============================================================

def extract_delta_map(obj):
    if isinstance(
        obj,
        dict,
    ):
        return obj

    if isinstance(
        obj,
        (tuple, list),
    ):
        for item in obj:
            if isinstance(
                item,
                dict,
            ):
                return item

    raise RuntimeError(
        "Unexpected predict_delta_map result: "
        + repr(
            type(obj)
        )
    )


def predict_future(
    model,
    rows,
    graphs_by_sha,
    base_prob_by_sha,
):

    shas = [
        norm_sha(
            r["sha256"]
        )
        for r in rows
    ]

    base_logit_by_sha = {
        sha:
        float(
            probability_to_logit(
                base_prob_by_sha[
                    sha
                ]
            )
        )
        for sha in shas
    }

    # 非常重要：
    # 未来预测阶段不给模型真实标签。
    # predict_delta_map接口要求label map，
    # 所以统一传0作为占位符。
    dummy_label_by_sha = {
        sha: 0
        for sha in shas
    }

    raw = exp.predict_delta_map(
        model,
        shas,
        graphs_by_sha,
        base_logit_by_sha,
        dummy_label_by_sha,
        DEVICE,
        BATCH_SIZE,
    )

    delta_map = extract_delta_map(
        raw
    )

    out = []

    for r in rows:
        sha = norm_sha(
            r["sha256"]
        )

        base_prob = float(
            base_prob_by_sha[
                sha
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

        if enabled:
            delta = scalar(
                delta_map.get(
                    sha,
                    0.0,
                )
            )
        else:
            delta = 0.0

        final_logit = (
            residual_combine(
                base_logit,
                delta,
            )
        )

        final_prob = float(
            sigmoid(
                final_logit
            )
        )

        out.append({
            "sha256":
                sha,

            "label":
                parse_label(
                    r["label"]
                ),

            "graph_enabled":
                int(enabled),

            "base_probability":
                base_prob,

            "delta":
                delta,

            "final_probability":
                final_prob,
        })

    return out


# ============================================================
# main
# ============================================================

def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "============================================================"
    )
    print(
        "Formal temporal experiment"
    )
    print(
        "2017 train -> 2018-2021 test"
    )
    print(
        "Compact = frozen all-ratio70"
    )
    print(
        "Behavior residual = frozen v2 representation"
    )
    print(
        "Epoch source = 2017 rolling validation only"
    )
    print(
        "Fixed epochs =",
        FIXED_EPOCHS,
    )
    print(
        "Threshold =",
        THRESHOLD,
    )
    print(
        "Device =",
        DEVICE,
    )
    print(
        "============================================================"
    )

    train_manifest, future_manifest = (
        load_manifest()
    )

    compact_prob = (
        load_base_probabilities()
    )

    # --------------------------------------------------------
    # 2017训练集
    # --------------------------------------------------------

    train_rows = []

    missing_train_prob = []

    for sha, r in sorted(
        train_manifest.items()
    ):
        key = (
            2017,
            sha,
        )

        if key not in compact_prob:
            missing_train_prob.append(
                sha
            )
            continue

        train_rows.append(
            r
        )

    print()
    print(
        "2017 manifest train =",
        len(train_manifest),
    )

    print(
        "2017 Compact-covered =",
        len(train_rows),
    )

    print(
        "2017 missing Compact =",
        len(
            missing_train_prob
        ),
    )

    if not train_rows:
        raise RuntimeError(
            "No 2017 training Compact "
            "probabilities were found. "
            "Stop here."
        )

    y_train = np.asarray(
        [
            parse_label(
                r["label"]
            )
            for r in train_rows
        ],
        dtype=int,
    )

    train_base_prob = np.asarray(
        [
            compact_prob[
                (
                    2017,
                    norm_sha(
                        r["sha256"]
                    ),
                )
            ]
            for r in train_rows
        ],
        dtype=float,
    )

    train_base_logits = (
        probability_to_logit(
            train_base_prob
        )
    )

    train_base_logit_by_sha = {
        norm_sha(
            r["sha256"]
        ):
        float(logit)
        for r, logit
        in zip(
            train_rows,
            train_base_logits,
        )
    }

    train_label_by_sha = {
        norm_sha(
            r["sha256"]
        ):
        int(y)
        for r, y
        in zip(
            train_rows,
            y_train,
        )
    }

    # --------------------------------------------------------
    # 图
    # --------------------------------------------------------

    (
        role_to_idx,
        edge_to_idx,
        train_graphs,
        future_graphs,
    ) = load_all_graphs()

    train_enabled = sum(
        1
        for r in train_rows
        if norm_sha(
            r["sha256"]
        ) in train_graphs
    )

    print()
    print(
        "2017 residual-enabled =",
        train_enabled,
        "/",
        len(train_rows),
    )

    print(
        "\nFuture graph-enabled APK:"
    )

    for year in YEARS:
        target_sha = set(
            future_manifest[
                year
            ]
        )

        enabled = len(
            target_sha
            & set(
                future_graphs[
                    year
                ]
            )
        )

        print(
            year,
            enabled,
            "/",
            len(target_sha),
        )

    # --------------------------------------------------------
    # 准备未来测试数据
    # --------------------------------------------------------

    future_rows = {}
    future_base_prob = {}

    print(
        "\n===== BASE SANITY CHECK ====="
    )

    for year in YEARS:

        rows = []

        missing = []

        probs = {}

        for sha, r in sorted(
            future_manifest[
                year
            ].items()
        ):
            key = (
                year,
                sha,
            )

            if key not in compact_prob:
                missing.append(
                    sha
                )
                continue

            rows.append(
                r
            )

            probs[
                sha
            ] = compact_prob[
                key
            ]

        if not rows:
            raise RuntimeError(
                f"No Compact predictions for {year}"
            )

        y = np.asarray(
            [
                parse_label(
                    r["label"]
                )
                for r in rows
            ],
            dtype=int,
        )

        p = np.asarray(
            [
                probs[
                    norm_sha(
                        r["sha256"]
                    )
                ]
                for r in rows
            ],
            dtype=float,
        )

        m = metrics(
            y,
            p,
        )

        print(
            f"{year}: "
            f"n={len(rows)} "
            f"BA={m['BA']:.6f} "
            f"ACC={m['ACC']:.6f} "
            f"AUC={m['AUC']:.6f} "
            f"FP={m['FP']} "
            f"FN={m['FN']} "
            f"missing={len(missing)}"
        )

        expected = (
            EXPECTED_BASE_BA[
                year
            ]
        )

        if abs(
            m["BA"]
            - expected
        ) > 5e-6:
            raise RuntimeError(
                f"{year} Compact BA mismatch: "
                f"got={m['BA']:.9f}, "
                f"expected~={expected:.9f}. "
                "Stop before residual evaluation."
            )

        future_rows[
            year
        ] = rows

        future_base_prob[
            year
        ] = probs

    print(
        "\nCompact baseline reproduced exactly."
    )

    # --------------------------------------------------------
    # 正式训练：5个随机种子 × 固定18轮
    # --------------------------------------------------------

    per_seed_rows = []
    prediction_rows = []

    for seed in SEEDS:

        print()
        print(
            "=" * 72
        )
        print(
            f"SEED {seed} | "
            f"fixed epochs = {FIXED_EPOCHS}"
        )
        print(
            "=" * 72
        )

        model = fit_fixed_epochs(
            train_rows,
            y_train,
            train_base_logits,
            train_graphs,
            len(role_to_idx),
            len(edge_to_idx),
            seed,
        )

        torch.save(
            {
                "seed":
                    seed,

                "fixed_epochs":
                    FIXED_EPOCHS,

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

            pred_rows = (
                predict_future(
                    model,
                    future_rows[
                        year
                    ],
                    future_graphs[
                        year
                    ],
                    future_base_prob[
                        year
                    ],
                )
            )

            y = np.asarray(
                [
                    r["label"]
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

            bm = metrics(
                y,
                bp,
            )

            fm = metrics(
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

            corrected = int(
                np.sum(
                    changed_mask
                    & (
                        fpred
                        == y
                    )
                )
            )

            regressed = int(
                np.sum(
                    changed_mask
                    & (
                        bpred
                        == y
                    )
                )
            )

            changed = int(
                np.sum(
                    changed_mask
                )
            )

            enabled = sum(
                int(
                    r[
                        "graph_enabled"
                    ]
                )
                for r in pred_rows
            )

            row = {
                "year":
                    year,

                "seed":
                    seed,

                "fixed_epoch":
                    FIXED_EPOCHS,

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
            }

            per_seed_rows.append(
                row
            )

            print(
                f"{year}: "
                f"base_BA={bm['BA']:.6f} "
                f"final_BA={fm['BA']:.6f} "
                f"delta={(fm['BA']-bm['BA'])*100:+.3f}pp "
                f"FP={bm['FP']}->{fm['FP']} "
                f"FN={bm['FN']}->{fm['FN']} "
                f"changed={changed} "
                f"corrected={corrected} "
                f"regressed={regressed} "
                f"enabled={enabled}"
            )

            for r in pred_rows:
                prediction_rows.append({
                    "year":
                        year,

                    "seed":
                        seed,

                    "sha256":
                        r["sha256"],

                    "label":
                        r["label"],

                    "graph_enabled":
                        r[
                            "graph_enabled"
                        ],

                    "base_probability":
                        r[
                            "base_probability"
                        ],

                    "delta":
                        r["delta"],

                    "final_probability":
                        r[
                            "final_probability"
                        ],
                })

    # --------------------------------------------------------
    # 正式汇总：先按年份平均5个随机种子
    # --------------------------------------------------------

    year_summary = []

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
            if r["year"] == year
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

        year_summary.append(
            row
        )

        print(
            f"{year}: "
            f"BA {row['base_BA']:.6f}"
            f" -> {row['final_BA']:.6f} "
            f"({row['delta_BA']*100:+.3f}pp) | "
            f"AUC {row['base_AUC']:.6f}"
            f" -> {row['final_AUC']:.6f} | "
            f"FP {row['base_FP']:.2f}"
            f" -> {row['final_FP']:.2f} | "
            f"FN {row['base_FN']:.2f}"
            f" -> {row['final_FN']:.2f}"
        )

    base_mean_ba = float(
        np.mean(
            [
                r["base_BA"]
                for r in year_summary
            ]
        )
    )

    final_mean_ba = float(
        np.mean(
            [
                r["final_BA"]
                for r in year_summary
            ]
        )
    )

    base_min_ba = float(
        np.min(
            [
                r["base_BA"]
                for r in year_summary
            ]
        )
    )

    final_min_ba = float(
        np.min(
            [
                r["final_BA"]
                for r in year_summary
            ]
        )
    )

    base_mean_auc = float(
        np.mean(
            [
                r["base_AUC"]
                for r in year_summary
            ]
        )
    )

    final_mean_auc = float(
        np.mean(
            [
                r["final_AUC"]
                for r in year_summary
            ]
        )
    )

    worst_seed_ba = float(
        np.min(
            [
                r["final_BA"]
                for r in per_seed_rows
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
        "future min yearly-mean BA = "
        f"{base_min_ba:.6f}"
        f" -> {final_min_ba:.6f}"
    )

    print(
        "future mean AUC = "
        f"{base_mean_auc:.6f}"
        f" -> {final_mean_auc:.6f}"
    )

    print(
        "worst final year-seed BA = "
        f"{worst_seed_ba:.6f}"
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    write_csv(
        OUTPUT
        / "per_seed.csv",
        per_seed_rows,
    )

    write_csv(
        OUTPUT
        / "per_year.csv",
        year_summary,
    )

    write_csv(
        OUTPUT
        / "predictions.csv",
        prediction_rows,
    )

    config = {
        "compact":
            "stable70_allratio46",

        "compact_dim":
            70,

        "residual_representation":
            "binding_core_v2_train_safe",

        "source_year":
            2017,

        "target_years":
            list(YEARS),

        "seeds":
            list(SEEDS),

        "fixed_epochs":
            FIXED_EPOCHS,

        "epoch_selection":
            (
                "median of 15 best epochs "
                "from 2017 rolling validation only"
            ),

        "source_best_epochs":
            [
                1,
                5,
                6,
                15,
                16,
                16,
                18,
                18,
                18,
                19,
                20,
                26,
                31,
                35,
                49,
            ],

        "threshold":
            THRESHOLD,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "future_label_use":
            (
                "metrics only after predictions; "
                "dummy labels passed to graph prediction helper"
            ),
    }

    with (
        OUTPUT
        / "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print()
    print(
        "saved =",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
