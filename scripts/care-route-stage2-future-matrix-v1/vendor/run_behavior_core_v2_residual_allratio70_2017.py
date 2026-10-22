#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================
# 复用刚刚确认的全比例70维 Compact
# ============================================================

import run_stable70_vs_allratio70_2017 as ratio_exp


base = ratio_exp.base

base.compact_matrix_named = (
    ratio_exp.compact_matrix_allratio70
)


COMPACT_FEATURES = (
    ROOT
    / "artifacts/p1-features/features.csv"
)

COMPACT_SCHEMA = (
    ROOT
    / "artifacts/p1-features/feature_schema.json"
)

SEMANTIC_2017 = (
    ROOT
    / "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

ROLLING_SPLITS = (
    ROOT
    / "artifacts/p9-2017-rolling-splits/"
    "split_manifest.csv"
)

GRAPH_FILE = (
    ROOT
    / "artifacts/"
    "top5-semantic-behavior-graph-v1-2017-trigger551/"
    "binding_cores_v2_train_safe.jsonl"
)

DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/"
    "behavior-core-v2-residual-allratio70-2017"
)


THRESHOLD = 0.5


# ============================================================
# 基础工具
# ============================================================

def norm_sha(value):
    return str(value).strip().casefold()


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def read_csv_rows(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        w.writeheader()
        w.writerows(rows)


# ============================================================
# 读取训练安全 v2 图
#
# 严格忽略：
# label
# binary_label
# behavior_id
# 方法名
# 包名
# raw_id
#
# sha256 只用于 APK 聚合。
# ============================================================

def scan_vocabulary(path):
    roles = set()
    edge_types = set()
    count = 0
    shas = set()

    with path.open(
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

            shas.add(sha)
            count += 1

            for node in x.get(
                "nodes",
                [],
            ):
                if (
                    str(
                        node.get(
                            "type",
                            ""
                        )
                    )
                    == "semantic"
                ):
                    role = str(
                        node.get(
                            "role",
                            ""
                        )
                    ).strip()

                    if role:
                        roles.add(role)

            for edge in x.get(
                "edges",
                [],
            ):
                edge_types.add(
                    str(
                        edge.get(
                            "type",
                            "unknown",
                        )
                    )
                )

    role_to_idx = {
        role: i + 1
        for i, role
        in enumerate(
            sorted(roles)
        )
    }

    edge_to_idx = {
        name: i
        for i, name
        in enumerate(
            sorted(edge_types)
        )
    }

    print(
        "v2 graph count =",
        count,
    )

    print(
        "v2 APK count =",
        len(shas),
    )

    print(
        "semantic roles =",
        len(role_to_idx),
    )

    print(
        "edge types =",
        sorted(edge_to_idx),
    )

    return (
        role_to_idx,
        edge_to_idx,
    )


def make_graph_record(
    x,
    role_to_idx,
    edge_to_idx,
):
    nodes = x.get(
        "nodes",
        [],
    )

    edges = x.get(
        "edges",
        [],
    )

    id_to_idx = {
        str(node["id"]): i
        for i, node
        in enumerate(nodes)
    }

    node_type = []
    role_idx = []
    node_scalar = []

    for node in nodes:

        kind = str(
            node.get(
                "type",
                "",
            )
        )

        is_semantic = int(
            kind == "semantic"
        )

        node_type.append(
            is_semantic
        )

        if is_semantic:
            role = str(
                node.get(
                    "role",
                    "",
                )
            )

            rid = role_to_idx.get(
                role,
                0,
            )
        else:
            rid = 0

        role_idx.append(
            rid
        )

        # ====================================================
        # 白名单字段。
        #
        # 方法节点：
        # 是否锚点承载方法
        # 是否最近共同组织点
        # 距组织点距离
        # 局部入度 / 出度
        #
        # 语义节点：
        # 是否Top-5锚点语义
        # ====================================================

        carrier = safe_float(
            node.get(
                "is_anchor_carrier",
                node.get(
                    "is_carrier",
                    0,
                ),
            )
        )

        binding_root = safe_float(
            node.get(
                "is_binding_root",
                0,
            )
        )

        distance = safe_float(
            node.get(
                "distance_from_root",
                node.get(
                    "organizer_distance",
                    0,
                ),
            )
        )

        local_in = safe_float(
            node.get(
                "local_in_degree",
                0,
            )
        )

        local_out = safe_float(
            node.get(
                "local_out_degree",
                0,
            )
        )

        anchor_role = safe_float(
            node.get(
                "is_anchor_role",
                0,
            )
        )

        node_scalar.append([
            carrier,
            binding_root,
            math.log1p(
                max(0.0, distance)
            ),
            math.log1p(
                max(0.0, local_in)
            ),
            math.log1p(
                max(0.0, local_out)
            ),
            anchor_role,
        ])

    edge_src = []
    edge_dst = []
    edge_type = []
    edge_scalar = []

    for edge in edges:

        source = str(
            edge["source"]
        )

        target = str(
            edge["target"]
        )

        if (
            source not in id_to_idx
            or target not in id_to_idx
        ):
            continue

        edge_src.append(
            id_to_idx[source]
        )

        edge_dst.append(
            id_to_idx[target]
        )

        edge_type.append(
            edge_to_idx[
                str(
                    edge.get(
                        "type",
                        "unknown",
                    )
                )
            ]
        )

        raw_hops = safe_float(
            edge.get(
                "raw_hops",
                1,
            ),
            1,
        )

        neutral_hops = safe_float(
            edge.get(
                "neutral_hops",
                0,
            )
        )

        edge_scalar.append([
            math.log1p(
                max(
                    0.0,
                    raw_hops,
                )
            ),
            math.log1p(
                max(
                    0.0,
                    neutral_hops,
                )
            ),
        ])

    if edge_src:
        edge_index = torch.tensor(
            [
                edge_src,
                edge_dst,
            ],
            dtype=torch.long,
        )

        edge_type_tensor = torch.tensor(
            edge_type,
            dtype=torch.long,
        )

        edge_scalar_tensor = torch.tensor(
            edge_scalar,
            dtype=torch.float32,
        )

    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        edge_type_tensor = torch.empty(
            (0,),
            dtype=torch.long,
        )

        edge_scalar_tensor = torch.empty(
            (0, 2),
            dtype=torch.float32,
        )

    return {
        "node_type":
            torch.tensor(
                node_type,
                dtype=torch.long,
            ),

        "role_idx":
            torch.tensor(
                role_idx,
                dtype=torch.long,
            ),

        "node_scalar":
            torch.tensor(
                node_scalar,
                dtype=torch.float32,
            ),

        "edge_index":
            edge_index,

        "edge_type":
            edge_type_tensor,

        "edge_scalar":
            edge_scalar_tensor,
    }


def load_graphs(
    path,
    role_to_idx,
    edge_to_idx,
):
    graphs_by_sha = defaultdict(
        list
    )

    total = 0

    with path.open(
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

            graph = make_graph_record(
                x,
                role_to_idx,
                edge_to_idx,
            )

            graphs_by_sha[
                sha
            ].append(
                graph
            )

            total += 1

    print(
        "loaded graphs =",
        total,
    )

    print(
        "enabled APKs =",
        len(graphs_by_sha),
    )

    if total != 2995:
        print(
            "WARNING: expected 2995 v2 cores, got",
            total,
        )

    return dict(
        graphs_by_sha
    )


# ============================================================
# APK图批处理
# ============================================================

def collate_apks(
    shas,
    graphs_by_sha,
    base_logit_by_sha,
    label_by_sha,
    device,
):
    node_types = []
    role_idxs = []
    node_scalars = []

    edge_indices = []
    edge_types = []
    edge_scalars = []

    node_graph_index = []
    graph_apk_index = []

    node_offset = 0
    graph_offset = 0

    for apk_i, sha in enumerate(
        shas
    ):

        graphs = graphs_by_sha[
            sha
        ]

        for graph in graphs:

            n = int(
                graph[
                    "node_type"
                ].shape[0]
            )

            node_types.append(
                graph[
                    "node_type"
                ]
            )

            role_idxs.append(
                graph[
                    "role_idx"
                ]
            )

            node_scalars.append(
                graph[
                    "node_scalar"
                ]
            )

            node_graph_index.append(
                torch.full(
                    (n,),
                    graph_offset,
                    dtype=torch.long,
                )
            )

            graph_apk_index.append(
                apk_i
            )

            eidx = graph[
                "edge_index"
            ]

            if eidx.shape[1] > 0:
                edge_indices.append(
                    eidx
                    + node_offset
                )

                edge_types.append(
                    graph[
                        "edge_type"
                    ]
                )

                edge_scalars.append(
                    graph[
                        "edge_scalar"
                    ]
                )

            node_offset += n
            graph_offset += 1

    node_type = torch.cat(
        node_types,
        dim=0,
    ).to(device)

    role_idx = torch.cat(
        role_idxs,
        dim=0,
    ).to(device)

    node_scalar = torch.cat(
        node_scalars,
        dim=0,
    ).to(device)

    node_graph_index = torch.cat(
        node_graph_index,
        dim=0,
    ).to(device)

    graph_apk_index = torch.tensor(
        graph_apk_index,
        dtype=torch.long,
        device=device,
    )

    if edge_indices:
        edge_index = torch.cat(
            edge_indices,
            dim=1,
        ).to(device)

        edge_type = torch.cat(
            edge_types,
            dim=0,
        ).to(device)

        edge_scalar = torch.cat(
            edge_scalars,
            dim=0,
        ).to(device)

    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
            device=device,
        )

        edge_type = torch.empty(
            (0,),
            dtype=torch.long,
            device=device,
        )

        edge_scalar = torch.empty(
            (0, 2),
            dtype=torch.float32,
            device=device,
        )

    base_logits = torch.tensor(
        [
            base_logit_by_sha[sha]
            for sha in shas
        ],
        dtype=torch.float32,
        device=device,
    )

    labels = torch.tensor(
        [
            label_by_sha[sha]
            for sha in shas
        ],
        dtype=torch.float32,
        device=device,
    )

    return {
        "node_type":
            node_type,

        "role_idx":
            role_idx,

        "node_scalar":
            node_scalar,

        "edge_index":
            edge_index,

        "edge_type":
            edge_type,

        "edge_scalar":
            edge_scalar,

        "node_graph_index":
            node_graph_index,

        "graph_apk_index":
            graph_apk_index,

        "graph_count":
            graph_offset,

        "apk_count":
            len(shas),

        "base_logits":
            base_logits,

        "labels":
            labels,
    }


# ============================================================
# 分组池化
# ============================================================

def group_mean(
    x,
    index,
    groups,
):
    out = torch.zeros(
        (
            groups,
            x.shape[1],
        ),
        dtype=x.dtype,
        device=x.device,
    )

    out.index_add_(
        0,
        index,
        x,
    )

    count = torch.zeros(
        groups,
        dtype=x.dtype,
        device=x.device,
    )

    count.index_add_(
        0,
        index,
        torch.ones(
            index.shape[0],
            dtype=x.dtype,
            device=x.device,
        ),
    )

    return out / count.clamp_min(
        1.0
    ).unsqueeze(1)


def group_max(
    x,
    index,
    groups,
):
    d = x.shape[1]

    out = torch.full(
        (
            groups,
            d,
        ),
        -1e9,
        dtype=x.dtype,
        device=x.device,
    )

    if hasattr(
        out,
        "scatter_reduce_",
    ):
        expanded = (
            index
            .unsqueeze(1)
            .expand(
                -1,
                d,
            )
        )

        out.scatter_reduce_(
            0,
            expanded,
            x,
            reduce="amax",
            include_self=True,
        )

        return out

    # 旧版本 PyTorch 备用逻辑
    for i in range(groups):
        mask = (
            index == i
        )

        out[i] = x[
            mask
        ].max(
            dim=0
        ).values

    return out


# ============================================================
# 有方向的图消息传递层
# ============================================================

class MessageLayer(
    nn.Module
):
    def __init__(
        self,
        hidden,
        edge_dim,
        dropout,
    ):
        super().__init__()

        self.self_linear = nn.Linear(
            hidden,
            hidden,
        )

        self.in_message = nn.Linear(
            hidden + edge_dim,
            hidden,
        )

        self.out_message = nn.Linear(
            hidden + edge_dim,
            hidden,
        )

        self.norm = nn.LayerNorm(
            hidden
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        h,
        edge_index,
        edge_feat,
    ):
        base_h = self.self_linear(
            h
        )

        if edge_index.shape[1] == 0:
            update = F.relu(
                base_h
            )

            return self.norm(
                h
                + self.dropout(
                    update
                )
            )

        src = edge_index[0]
        dst = edge_index[1]

        in_input = torch.cat(
            [
                h[src],
                edge_feat,
            ],
            dim=1,
        )

        out_input = torch.cat(
            [
                h[dst],
                edge_feat,
            ],
            dim=1,
        )

        msg_in = F.relu(
            self.in_message(
                in_input
            )
        )

        msg_out = F.relu(
            self.out_message(
                out_input
            )
        )

        agg_in = torch.zeros_like(
            h
        )

        agg_out = torch.zeros_like(
            h
        )

        agg_in.index_add_(
            0,
            dst,
            msg_in,
        )

        agg_out.index_add_(
            0,
            src,
            msg_out,
        )

        in_count = torch.zeros(
            h.shape[0],
            dtype=h.dtype,
            device=h.device,
        )

        out_count = torch.zeros(
            h.shape[0],
            dtype=h.dtype,
            device=h.device,
        )

        ones = torch.ones(
            src.shape[0],
            dtype=h.dtype,
            device=h.device,
        )

        in_count.index_add_(
            0,
            dst,
            ones,
        )

        out_count.index_add_(
            0,
            src,
            ones,
        )

        agg_in = (
            agg_in
            / in_count
            .clamp_min(1.0)
            .unsqueeze(1)
        )

        agg_out = (
            agg_out
            / out_count
            .clamp_min(1.0)
            .unsqueeze(1)
        )

        update = F.relu(
            base_h
            + agg_in
            + agg_out
        )

        return self.norm(
            h
            + self.dropout(
                update
            )
        )


# ============================================================
# 行为图 -> APK -> 单一残差
# ============================================================

class BehaviorResidualNet(
    nn.Module
):
    def __init__(
        self,
        role_count,
        edge_type_count,
        hidden=48,
        dropout=0.10,
    ):
        super().__init__()

        self.node_type_embedding = (
            nn.Embedding(
                2,
                4,
            )
        )

        self.role_embedding = (
            nn.Embedding(
                role_count + 1,
                12,
            )
        )

        self.edge_type_embedding = (
            nn.Embedding(
                max(
                    1,
                    edge_type_count,
                ),
                8,
            )
        )

        self.node_input = nn.Linear(
            4 + 12 + 6,
            hidden,
        )

        edge_dim = 8 + 2

        self.layer1 = MessageLayer(
            hidden,
            edge_dim,
            dropout,
        )

        self.layer2 = MessageLayer(
            hidden,
            edge_dim,
            dropout,
        )

        # 单图 = mean + max
        graph_dim = (
            hidden * 2
        )

        # APK = 行为图 mean + max
        apk_dim = (
            graph_dim * 2
        )

        self.delta_head = nn.Sequential(
            nn.Linear(
                apk_dim,
                hidden,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden,
                hidden // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden // 2,
                1,
            ),
        )

    def forward(
        self,
        batch,
    ):
        node_type_embedding = (
            self.node_type_embedding(
                batch[
                    "node_type"
                ]
            )
        )

        role_embedding = (
            self.role_embedding(
                batch[
                    "role_idx"
                ]
            )
        )

        node_input = torch.cat(
            [
                node_type_embedding,
                role_embedding,
                batch[
                    "node_scalar"
                ],
            ],
            dim=1,
        )

        h = F.relu(
            self.node_input(
                node_input
            )
        )

        edge_embedding = (
            self.edge_type_embedding(
                batch[
                    "edge_type"
                ]
            )
        )

        edge_feat = torch.cat(
            [
                edge_embedding,
                batch[
                    "edge_scalar"
                ],
            ],
            dim=1,
        )

        h = self.layer1(
            h,
            batch[
                "edge_index"
            ],
            edge_feat,
        )

        h = self.layer2(
            h,
            batch[
                "edge_index"
            ],
            edge_feat,
        )

        graph_mean = group_mean(
            h,
            batch[
                "node_graph_index"
            ],
            batch[
                "graph_count"
            ],
        )

        graph_max = group_max(
            h,
            batch[
                "node_graph_index"
            ],
            batch[
                "graph_count"
            ],
        )

        graph_repr = torch.cat(
            [
                graph_mean,
                graph_max,
            ],
            dim=1,
        )

        apk_mean = group_mean(
            graph_repr,
            batch[
                "graph_apk_index"
            ],
            batch[
                "apk_count"
            ],
        )

        apk_max = group_max(
            graph_repr,
            batch[
                "graph_apk_index"
            ],
            batch[
                "apk_count"
            ],
        )

        apk_repr = torch.cat(
            [
                apk_mean,
                apk_max,
            ],
            dim=1,
        )

        delta = (
            self.delta_head(
                apk_repr
            )
            .squeeze(1)
        )

        return delta


# ============================================================
# 冻结 Compact 的基础输出
# ============================================================

def get_base_logits(
    model,
    data,
    part,
    device,
):
    if part == "train":
        suffix = "train"
    else:
        suffix = "test"

    model.eval()

    with torch.no_grad():

        (
            _,
            base_logits,
            _,
            _,
        ) = model(
            base.tensor(
                data[
                    "compact_" + suffix
                ],
                device,
            ),
            base.tensor(
                data[
                    "semantic_" + suffix
                ],
                device,
            ),
            base.tensor(
                data[
                    "coverage_" + suffix
                ],
                device,
            ),
            base.tensor(
                data[
                    "presence_" + suffix
                ],
                device,
            ),
        )

    return (
        base_logits
        .detach()
        .cpu()
        .numpy()
        .ravel()
    )


# ============================================================
# 有界残差公式
#
# z_final =
# z_G + 4*p_G*(1-p_G)*tanh(delta)
# ============================================================

def combine_logits(
    base_logits,
    delta,
):
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

    final_logits = (
        base_logits
        + uncertainty
        * torch.tanh(
            delta
        )
    )

    return final_logits


def predict_delta_map(
    model,
    shas,
    graphs_by_sha,
    base_logit_by_sha,
    label_by_sha,
    device,
    batch_size,
):
    model.eval()

    result = {}

    with torch.no_grad():

        for start in range(
            0,
            len(shas),
            batch_size,
        ):

            part = shas[
                start:
                start + batch_size
            ]

            batch = collate_apks(
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
                .ravel()
            )

            for sha, value in zip(
                part,
                delta,
            ):
                result[sha] = float(
                    value
                )

    return result


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

    delta_map = predict_delta_map(
        model,
        enabled,
        graphs_by_sha,
        base_logit_by_sha,
        label_by_sha,
        device,
        batch_size,
    )

    final_logits = np.asarray(
        base_logits,
        dtype=float,
    ).copy()

    delta_values = np.zeros(
        len(shas),
        dtype=float,
    )

    for i, sha in enumerate(
        shas
    ):
        if sha not in delta_map:
            continue

        delta = delta_map[sha]

        z = float(
            base_logits[i]
        )

        p = 1.0 / (
            1.0
            + math.exp(
                -z
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

        final_logits[i] = (
            z
            + correction
        )

        delta_values[i] = delta

    base_probability = (
        1.0
        / (
            1.0
            + np.exp(
                -np.asarray(
                    base_logits,
                    dtype=float,
                )
            )
        )
    )

    final_probability = (
        1.0
        / (
            1.0
            + np.exp(
                -final_logits
            )
        )
    )

    base_metrics = base.metrics(
        labels.astype(int),
        base_probability,
        threshold=THRESHOLD,
    )

    final_metrics = base.metrics(
        labels.astype(int),
        final_probability,
        threshold=THRESHOLD,
    )

    base_pred = (
        base_probability
        >= THRESHOLD
    ).astype(int)

    final_pred = (
        final_probability
        >= THRESHOLD
    ).astype(int)

    y = labels.astype(int)

    changed = (
        base_pred
        != final_pred
    )

    corrected = (
        changed
        & (
            base_pred
            != y
        )
        & (
            final_pred
            == y
        )
    )

    regressed = (
        changed
        & (
            base_pred
            == y
        )
        & (
            final_pred
            != y
        )
    )

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
                    sha in graphs_by_sha
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

            "delta":
                float(
                    delta_values[i]
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
    }

    return (
        base_metrics,
        final_metrics,
        diagnostics,
        detail,
    )


# ============================================================
# 训练单个图残差模型
# ============================================================

def fit_behavior_residual(
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
):
    set_seed(
        seed
    )

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

    print(
        "enabled train =",
        len(enabled_train),
        "/",
        len(train_shas),
    )

    print(
        "enabled val =",
        len(enabled_val),
        "/",
        len(val_shas),
    )

    if not enabled_train:
        raise RuntimeError(
            "训练集没有有效行为图"
        )

    positive = sum(
        train_label_by_sha[sha]
        for sha in enabled_train
    )

    negative = (
        len(enabled_train)
        - positive
    )

    if positive > 0:
        pos_weight_value = (
            negative
            / positive
        )
    else:
        pos_weight_value = 1.0

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

    model = BehaviorResidualNet(
        role_count=role_count,
        edge_type_count=edge_type_count,
    ).to(device)

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

            batch = collate_apks(
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

            final_logits = combine_logits(
                batch[
                    "base_logits"
                ],
                delta,
            )

            # ================================================
            # 注意：
            # 标签只在 APK 聚合之后使用。
            # 没有任何行为子图继承 APK 标签。
            # ================================================

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
                    loss.detach().cpu()
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
                f"regressed={diagnostics['regressed']}",
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


# ============================================================
# 主流程
# ============================================================

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

    if abs(
        THRESHOLD - 0.5
    ) > 1e-12:
        raise RuntimeError(
            "阈值必须固定为0.5"
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
        "============================================================"
    )

    print(
        "All-ratio70 Compact + V2 semantic behavior residual"
    )

    print(
        "Compact = 24 global + 46 semantic ratios"
    )

    print(
        "Graph residual = APK-level only"
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
        "============================================================"
    )

    # ========================================================
    # 图数据
    # ========================================================

    (
        role_to_idx,
        edge_to_idx,
    ) = scan_vocabulary(
        GRAPH_FILE
    )

    graphs_by_sha = load_graphs(
        GRAPH_FILE,
        role_to_idx,
        edge_to_idx,
    )

    # ========================================================
    # Compact / 2017语义辅助表 / 2017滚动划分
    #
    # 注意：
    # semantic_2017 这里只用于复用现有基础分类器训练流程；
    # 新的局部残差完全不使用 Stable15。
    # ========================================================

    compact_rows = read_csv_rows(
        COMPACT_FEATURES
    )

    compact_schema = json.loads(
        COMPACT_SCHEMA.read_text(
            encoding="utf-8"
        )
    )

    compact_by_sha = (
        base.normalized_rows(
            compact_rows
        )
    )

    semantic_2017 = (
        base.read_gz_csv(
            SEMANTIC_2017
        )
    )

    names150 = (
        base.support150_names(
            semantic_2017[
                0
            ].keys()
        )
    )

    semantic_by_sha = (
        base.normalized_rows(
            semantic_2017
        )
    )

    rolling_parts = (
        base.read_split_rows(
            ROLLING_SPLITS
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
            "没有找到2017滚动划分"
        )

    print(
        "rolling splits =",
        sorted(
            rolling_splits
        ),
    )

    if len(
        base.CONFIGS
    ) != 1:
        raise RuntimeError(
            "预期现有基础配置只有1个"
        )

    old_config = copy.deepcopy(
        base.CONFIGS[0]
    )

    per_seed_rows = []
    prediction_rows = []

    for split_name in sorted(
        rolling_splits
    ):

        print()
        print(
            "=" * 72
        )

        print(
            "split =",
            split_name,
        )

        print(
            "=" * 72
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

        data = base.build_fold_data(
            train_rows,
            val_rows,
            compact_by_sha,
            compact_schema,
            semantic_by_sha,
            names150,
        )

        if (
            data[
                "compact_train"
            ].shape[1]
            != 70
        ):
            raise RuntimeError(
                "全比例Compact不是70维"
            )

        y_train = base.labels_for(
            train_rows,
            compact_by_sha,
        )

        y_val = base.labels_for(
            val_rows,
            compact_by_sha,
        )

        for seed in args.seeds:

            print()
            print(
                "seed =",
                seed,
                flush=True,
            )

            set_seed(
                seed
            )

            # ================================================
            # 先按现有正式协议训练基础 Compact。
            #
            # 旧局部分支虽然会被现有 fit_one 构造，
            # 但后面完全丢弃；
            # 我们只取冻结的 base_logits。
            # ================================================

            (
                old_model,
                _,
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
                get_base_logits(
                    old_model,
                    data,
                    "train",
                    device,
                )
            )

            val_base_logits = (
                get_base_logits(
                    old_model,
                    data,
                    "test",
                    device,
                )
            )

            (
                graph_model,
                best_epoch,
                history,
                train_base_logit_by_sha,
                train_label_by_sha,
                val_base_logit_by_sha,
                val_label_by_sha,
            ) = fit_behavior_residual(
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
            )

            (
                base_metrics,
                final_metrics,
                diagnostics,
                detail,
            ) = evaluate_full_split(
                graph_model,
                val_rows,
                y_val,
                val_base_logits,
                graphs_by_sha,
                val_base_logit_by_sha,
                val_label_by_sha,
                device,
                args.batch_size,
            )

            row = {
                "split_name":
                    split_name,

                "seed":
                    seed,

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

                "enabled":
                    diagnostics[
                        "enabled"
                    ],

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
            }

            per_seed_rows.append(
                row
            )

            for item in detail:
                prediction_rows.append({
                    "split_name":
                        split_name,

                    "seed":
                        seed,

                    **item,
                })

            print(
                "RESULT "
                f"base_BA={row['base_BA']:.6f} "
                f"final_BA={row['final_BA']:.6f} "
                f"delta={row['delta_BA']*100:+.3f}pp "
                f"base_AUC={row['base_AUC']:.6f} "
                f"final_AUC={row['final_AUC']:.6f} "
                f"changed={row['changed']} "
                f"corrected={row['corrected']} "
                f"regressed={row['regressed']}",
                flush=True,
            )

    # ========================================================
    # 汇总：
    # 仍然先按 split 对 seed 平均，
    # 再对 split 平均。
    # ========================================================

    by_split = defaultdict(
        list
    )

    for row in per_seed_rows:
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
        rows = by_split[
            split_name
        ]

        split_rows.append({
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

            "changed":
                float(
                    np.mean([
                        x["changed"]
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
        })

    summary = {
        "base_mean_BA":
            float(
                np.mean([
                    x["base_BA"]
                    for x in split_rows
                ])
            ),

        "final_mean_BA":
            float(
                np.mean([
                    x["final_BA"]
                    for x in split_rows
                ])
            ),

        "delta_BA":
            float(
                np.mean([
                    x["final_BA"]
                    - x["base_BA"]
                    for x in split_rows
                ])
            ),

        "base_min_BA":
            float(
                np.min([
                    x["base_BA"]
                    for x in split_rows
                ])
            ),

        "final_min_BA":
            float(
                np.min([
                    x["final_BA"]
                    for x in split_rows
                ])
            ),

        "base_mean_ACC":
            float(
                np.mean([
                    x["base_ACC"]
                    for x in split_rows
                ])
            ),

        "final_mean_ACC":
            float(
                np.mean([
                    x["final_ACC"]
                    for x in split_rows
                ])
            ),

        "base_mean_AUC":
            float(
                np.mean([
                    x["base_AUC"]
                    for x in split_rows
                ])
            ),

        "final_mean_AUC":
            float(
                np.mean([
                    x["final_AUC"]
                    for x in split_rows
                ])
            ),

        "base_mean_FP":
            float(
                np.mean([
                    x["base_FP"]
                    for x in split_rows
                ])
            ),

        "final_mean_FP":
            float(
                np.mean([
                    x["final_FP"]
                    for x in split_rows
                ])
            ),

        "base_mean_FN":
            float(
                np.mean([
                    x["base_FN"]
                    for x in split_rows
                ])
            ),

        "final_mean_FN":
            float(
                np.mean([
                    x["final_FN"]
                    for x in split_rows
                ])
            ),

        "mean_changed":
            float(
                np.mean([
                    x["changed"]
                    for x in split_rows
                ])
            ),

        "mean_corrected":
            float(
                np.mean([
                    x["corrected"]
                    for x in split_rows
                ])
            ),

        "mean_regressed":
            float(
                np.mean([
                    x["regressed"]
                    for x in split_rows
                ])
            ),
    }

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
        / "predictions.csv",
        prediction_rows,
    )

    write_csv(
        args.output
        / "summary.csv",
        [
            summary
        ],
    )

    print()
    print(
        "=" * 72
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        "base mean BA =",
        f"{summary['base_mean_BA']:.6f}",
    )

    print(
        "final mean BA =",
        f"{summary['final_mean_BA']:.6f}",
    )

    print(
        "delta BA =",
        f"{summary['delta_BA']*100:+.3f}",
        "pp",
    )

    print(
        "base min BA =",
        f"{summary['base_min_BA']:.6f}",
    )

    print(
        "final min BA =",
        f"{summary['final_min_BA']:.6f}",
    )

    print(
        "base mean ACC =",
        f"{summary['base_mean_ACC']:.6f}",
    )

    print(
        "final mean ACC =",
        f"{summary['final_mean_ACC']:.6f}",
    )

    print(
        "base mean AUC =",
        f"{summary['base_mean_AUC']:.6f}",
    )

    print(
        "final mean AUC =",
        f"{summary['final_mean_AUC']:.6f}",
    )

    print(
        "FP:",
        f"{summary['base_mean_FP']:.3f}",
        "->",
        f"{summary['final_mean_FP']:.3f}",
    )

    print(
        "FN:",
        f"{summary['base_mean_FN']:.3f}",
        "->",
        f"{summary['final_mean_FN']:.3f}",
    )

    print(
        "mean changed =",
        f"{summary['mean_changed']:.3f}",
    )

    print(
        "mean corrected =",
        f"{summary['mean_corrected']:.3f}",
    )

    print(
        "mean regressed =",
        f"{summary['mean_regressed']:.3f}",
    )

    print(
        "saved =",
        args.output,
    )


if __name__ == "__main__":
    main()
