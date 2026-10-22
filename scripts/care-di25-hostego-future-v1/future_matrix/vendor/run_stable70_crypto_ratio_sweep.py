#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# 复用刚刚已经跑通的密码学成分实验。
import run_stable70_crypto_component_sweep as comp


old = comp.old
r3 = comp.r3


OUT = Path(
    "artifacts/"
    "stable70-crypto-ratio-sweep"
)


NODE = "role_node__cryptography"
EDGE = "role_edge__cryptography"

NODE_DEN = "sensitive_node_count"
EDGE_DEN = "sensitive_edge_count"


def raw_ratio(
    rows,
    feature_by_sha,
    numerator,
    denominator,
):
    values = []

    for row in rows:
        source = comp.get_raw_row(
            feature_by_sha,
            row,
        )

        num = comp.safe_float(
            source.get(
                numerator,
                0.0,
            )
        )

        den = comp.safe_float(
            source.get(
                denominator,
                0.0,
            )
        )

        if den > 0.0:
            value = num / den
        else:
            value = 0.0

        values.append(
            value
        )

    return np.asarray(
        values,
        dtype=float,
    )


def standardized_ratio(
    train_rows,
    test_rows,
    feature_by_sha,
    numerator,
    denominator,
):
    train = raw_ratio(
        train_rows,
        feature_by_sha,
        numerator,
        denominator,
    )

    test = raw_ratio(
        test_rows,
        feature_by_sha,
        numerator,
        denominator,
    )

    mean = float(
        np.mean(train)
    )

    std = float(
        np.std(train)
    )

    if (
        not np.isfinite(std)
        or std < 1e-12
    ):
        raise RuntimeError(
            "ratio在训练折为常数: "
            + numerator
            + "/"
            + denominator
        )

    return (
        (train - mean) / std,
        (test - mean) / std,
    )


def make_ratio_variants(
    train_rows,
    test_rows,
    feature_by_sha,
    schema,
):
    bases = (
        comp.parent
        .build_base_variants_stable70(
            train_rows,
            test_rows,
            feature_by_sha,
            schema,
        )
    )

    (
        base_train,
        base_test,
        base_names,
    ) = bases[
        "stable70"
    ]

    base_train = np.asarray(
        base_train,
        dtype=float,
    )

    base_test = np.asarray(
        base_test,
        dtype=float,
    )

    base_names = list(
        base_names
    )

    if NODE not in base_names:
        raise RuntimeError(
            "Stable70找不到 "
            + NODE
        )

    if EDGE not in base_names:
        raise RuntimeError(
            "Stable70找不到 "
            + EDGE
        )

    node_idx = base_names.index(
        NODE
    )

    edge_idx = base_names.index(
        EDGE
    )

    (
        node_train_ratio,
        node_test_ratio,
    ) = standardized_ratio(
        train_rows,
        test_rows,
        feature_by_sha,
        NODE,
        NODE_DEN,
    )

    (
        edge_train_ratio,
        edge_test_ratio,
    ) = standardized_ratio(
        train_rows,
        test_rows,
        feature_by_sha,
        EDGE,
        EDGE_DEN,
    )

    variants = {}

    # ========================================================
    # Stable70基准
    # ========================================================

    variants[
        "stable70"
    ] = (
        base_train.copy(),
        base_test.copy(),
        list(base_names),
        0,
    )

    # ========================================================
    # 只比例化node
    # ========================================================

    a = base_train.copy()
    b = base_test.copy()
    n = list(base_names)

    a[:, node_idx] = (
        node_train_ratio
    )

    b[:, node_idx] = (
        node_test_ratio
    )

    n[node_idx] = (
        "stable_ratio__cryptography__node"
    )

    variants[
        "ratio_crypto_node"
    ] = (
        a,
        b,
        n,
        0,
    )

    # ========================================================
    # 只比例化edge
    # ========================================================

    a = base_train.copy()
    b = base_test.copy()
    n = list(base_names)

    a[:, edge_idx] = (
        edge_train_ratio
    )

    b[:, edge_idx] = (
        edge_test_ratio
    )

    n[edge_idx] = (
        "stable_ratio__cryptography__edge"
    )

    variants[
        "ratio_crypto_edge"
    ] = (
        a,
        b,
        n,
        0,
    )

    # ========================================================
    # node + edge全部比例化
    # ========================================================

    a = base_train.copy()
    b = base_test.copy()
    n = list(base_names)

    a[:, node_idx] = (
        node_train_ratio
    )

    b[:, node_idx] = (
        node_test_ratio
    )

    a[:, edge_idx] = (
        edge_train_ratio
    )

    b[:, edge_idx] = (
        edge_test_ratio
    )

    n[node_idx] = (
        "stable_ratio__cryptography__node"
    )

    n[edge_idx] = (
        "stable_ratio__cryptography__edge"
    )

    variants[
        "ratio_crypto_both"
    ] = (
        a,
        b,
        n,
        0,
    )

    # ========================================================
    # 已知的完全删除版本，作为诊断参照。
    # ========================================================

    (
        a,
        b,
        n,
    ) = comp.remove_columns(
        base_train,
        base_test,
        base_names,
        {
            NODE,
            EDGE,
        },
    )

    variants[
        "drop_crypto_both"
    ] = (
        a,
        b,
        n,
        2,
    )

    return variants


# 直接替换上一版中的变体构造。
comp.make_variants = (
    make_ratio_variants
)

comp.OUT = OUT
old.OUT = OUT

old.run_one_population = (
    comp.run_one_population
)

old.compute_marginal_effects = (
    comp.simple_effects
)


if __name__ == "__main__":
    print(
        "============================================================"
    )

    print(
        "Stable70 cryptography ratio sweep"
    )

    print(
        "绝对数量 -> 相对敏感行为占比"
    )

    print(
        "============================================================"
    )

    old.main()

    comp.print_final_table()
