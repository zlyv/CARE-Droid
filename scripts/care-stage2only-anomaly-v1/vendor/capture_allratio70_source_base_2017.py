#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = ROOT / "scripts"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import run_frozen_anchor_stable70_count10_150_2017_2021 as stable
import run_stable70_vs_allratio70_2017 as allratio


OUT = (
    ROOT
    / "artifacts/allratio70-source-base-capture"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def norm_sha(x):
    return str(x).strip().lower()


# ------------------------------------------------------------
# 保存“某个训练矩阵 -> SHA顺序”的对应关系
# ------------------------------------------------------------

matrix_sha_map = {}
last_full_train_shas = None

captured = {}


original_matrix = (
    allratio.compact_matrix_allratio70
)

original_fit_one = (
    stable.base.fit_one
)


def register_matrix_objects(
    obj,
    train_n,
    shas,
):
    """
    尽可能把返回值中的训练矩阵对象id
    和SHA顺序对应起来。
    """

    if isinstance(
        obj,
        np.ndarray,
    ):
        if (
            obj.ndim == 2
            and obj.shape[0] == train_n
        ):
            matrix_sha_map[
                id(obj)
            ] = list(shas)

        return

    if isinstance(
        obj,
        (tuple, list),
    ):
        for x in obj:
            register_matrix_objects(
                x,
                train_n,
                shas,
            )


def capture_matrix(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
):

    global last_full_train_shas

    result = original_matrix(
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
    )

    shas = [
        norm_sha(
            r["sha256"]
        )
        for r in train_rows
    ]

    register_matrix_objects(
        result,
        len(train_rows),
        shas,
    )

    # 2017全部训练实际应为799个Compact覆盖样本。
    if len(train_rows) >= 790:
        last_full_train_shas = list(
            shas
        )

    return result


def capture_fit_one(
    x_train,
    s_train,
    c_train,
    p_train,
    y_train,
    x_val,
    s_val,
    c_val,
    p_val,
    y_val,
    config,
    seed,
    device,
):

    result = original_fit_one(
        x_train,
        s_train,
        c_train,
        p_train,
        y_train,
        x_val,
        s_val,
        c_val,
        p_val,
        y_val,
        config,
        seed,
        device,
    )

    model = result[0]

    # 只有最终“全部2017训练”使用 fixed_epochs。
    if not bool(
        config.get(
            "fixed_epochs",
            False,
        )
    ):
        return result

    shas = matrix_sha_map.get(
        id(x_train)
    )

    if shas is None:
        shas = last_full_train_shas

    if shas is None:
        raise RuntimeError(
            "无法恢复最终2017训练SHA顺序"
        )

    if len(shas) != len(x_train):
        raise RuntimeError(
            "SHA数量与训练矩阵不一致: "
            f"{len(shas)} vs {len(x_train)}"
        )

    model.eval()

    tx = stable.base.tensor(
        x_train,
        device,
    )

    ts = stable.base.tensor(
        s_train,
        device,
    )

    tc = stable.base.tensor(
        c_train,
        device,
    )

    tp = stable.base.tensor(
        p_train,
        device,
    )

    with torch.no_grad():
        (
            _,
            base_logits,
            _,
            _,
        ) = model(
            tx,
            ts,
            tc,
            tp,
        )

    logits = (
        base_logits
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    probs = (
        torch.sigmoid(
            base_logits
        )
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    labels = (
        np.asarray(
            y_train
        )
        .reshape(-1)
        .astype(int)
    )

    if not (
        len(shas)
        == len(logits)
        == len(probs)
        == len(labels)
    ):
        raise RuntimeError(
            "最终训练输出长度不一致"
        )

    for i, (
        sha,
        label,
        logit,
        prob,
    ) in enumerate(
        zip(
            shas,
            labels,
            logits,
            probs,
        )
    ):

        key = (
            int(seed),
            sha,
        )

        row = {
            "seed":
                int(seed),

            "order_index":
                i,

            "sha256":
                sha,

            "label":
                int(label),

            "base_logit":
                float(logit),

            "base_probability":
                float(prob),
        }

        if key in captured:
            old = captured[key]

            if abs(
                old[
                    "base_logit"
                ]
                - row[
                    "base_logit"
                ]
            ) > 1e-6:
                raise RuntimeError(
                    "同一随机种子重复训练产生明显不同的"
                    "Compact基础分数: "
                    f"seed={seed} sha={sha}"
                )

            continue

        captured[
            key
        ] = row

    print(
        f"[CAPTURE] seed={seed} "
        f"source_n={len(shas)}"
    )

    return result


# ------------------------------------------------------------
# 替换成全比例70维，并挂载抓取器
# ------------------------------------------------------------

stable.base.compact_matrix_named = (
    capture_matrix
)

stable.base.fit_one = (
    capture_fit_one
)


if __name__ == "__main__":

    # stable.base.main仍照常完成原来的源域选择和未来预测，
    # 但输出放进独立目录，不覆盖已有正式结果。
    if "--output" not in sys.argv:
        sys.argv.extend([
            "--output",
            str(OUT),
        ])

    print(
        "============================================================"
    )
    print(
        "Capture exact 2017 all-ratio70 Compact base outputs"
    )
    print(
        "No future information is used for Compact training"
    )
    print(
        "============================================================"
    )

    stable.base.main()

    rows = sorted(
        captured.values(),
        key=lambda r: (
            r["seed"],
            r["order_index"],
        ),
    )

    if not rows:
        raise RuntimeError(
            "没有抓取到最终2017 Compact输出"
        )

    path = (
        OUT
        / "source_base_predictions.csv"
    )

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
        writer.writerows(
            rows
        )

    print()
    print(
        "===== SOURCE CAPTURE SUMMARY ====="
    )

    for seed in range(5):

        rs = [
            r
            for r in rows
            if r["seed"] == seed
        ]

        print(
            f"seed={seed} n={len(rs)}"
        )

    print(
        "total rows =",
        len(rows),
    )

    print(
        "saved =",
        path,
    )
