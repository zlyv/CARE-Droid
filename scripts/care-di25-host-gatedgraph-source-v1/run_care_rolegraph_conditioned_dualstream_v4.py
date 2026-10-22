#!/usr/bin/env python3
"""CARE-Droid v4: 31-role graph conditioned local dual-stream evidence.

This source-only experiment keeps the frozen first layer and routing protocol:

* layer 1: role-free Global24;
* router: frozen A25 (25% budget);
* source evaluation: fixed strict 2017 outer folds;
* future years: deliberately not touched by this script.

It replaces flat S7/local concatenation with a faithful graph interpretation of
S7.  Stats120 is split into 31x3 role-node features plus 27 graph-level
features; PairRaw465 becomes the upper triangle of an undirected role graph;
InterZ961 becomes a 31x31 directed edge channel.  A lightweight edge-biased
Graph Transformer yields one global structural token.  The token either is
only fused at the end (RGConcat), conditions local attention (RGCondAttn), or
conditions a critical-instance/context dual stream (RGCondDual).  Every expert
is a bounded correction of a strictly cross-fitted Global24 probability and is
applied only to the frozen A25 route.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required: " + str(exc))

try:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
except Exception as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required: " + str(exc))

try:
    import run_care_bounded_fixkeep_ablation as v3
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing CARE v3 dependencies in scripts/: " + str(exc)
    )


v2 = v3.v2
care_v1 = v3.care_v1
g24 = v3.g24
audit = v3.audit
evidence = v3.evidence
acn = v3.acn
VARIANT = v3.VARIANT
ROLE_COUNT = 31
ROLE_NODE_WIDTH = 3
ROLE_NODE_DIM = ROLE_COUNT * ROLE_NODE_WIDTH
GRAPH_GLOBAL_DIM = 27
PAIR_DIM = ROLE_COUNT * (ROLE_COUNT - 1) // 2
INTER_DIM = ROLE_COUNT * ROLE_COUNT
S7_DIM = 120 + PAIR_DIM + INTER_DIM
AUGMENTED_REGION_DIM = v3.AUGMENTED_REGION_DIM
THRESHOLD = 0.5

MODEL_VARIANTS = (
    "RGConcat",
    "RGCondAttn",
    "RGCondDual",
)
METHOD_FOR_VARIANT = {
    "RGConcat": "Global24__A25__CARE_RGConcat_FixKeep",
    "RGCondAttn": "Global24__A25__CARE_RGCondAttn_FixKeep",
    "RGCondDual": "Global24__A25__CARE_RGCondDual_FixKeep",
}
DIAGNOSTIC_NAMES = (
    "bounded_delta",
    "critical_score",
    "mean_instance_score",
    "attention_entropy",
    "max_attention",
    "context_entropy",
    "mean_external_distance",
    "mean_internal_distance",
    "mean_shared_commonness",
    "max_local_residual",
    "graph_pool_entropy",
)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    return device


def probability_to_logit(probability, epsilon: float = 1e-5) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(value / (1.0 - value)).astype(np.float32)


def parse_variants(text: str) -> list[str]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token and token not in values:
            if token not in MODEL_VARIANTS:
                raise SystemExit("unknown variant: " + token)
            values.append(token)
    if not values:
        raise SystemExit("at least one model variant is required")
    return values


def parse_ints(text: str) -> list[int]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            value = int(token)
            if value not in values:
                values.append(value)
    if not values:
        raise SystemExit("at least one seed is required")
    return values


def read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm_sha(value) -> str:
    return str(value).strip().lower()


def optional_reference_probability(path: Path, manifest, column: str):
    path = Path(path)
    if not path.exists():
        return None
    by_sha = {norm_sha(row["sha256"]): row for row in read_csv(path)}
    result = []
    for row in manifest:
        sha = norm_sha(row["sha256"])
        if sha not in by_sha or column not in by_sha[sha]:
            return None
        result.append(float(by_sha[sha][column]))
    return np.asarray(result, dtype=np.float64)


def get_source_host_bags(rows, endpoint_cache, args):
    """Load a source prefix from the existing 2017--2021 region cache safely.

    CARE v1 requires an exact cache/cohort match and would otherwise rebuild
    and overwrite the combined cache when this source-only script requests the
    first 799 rows.  The established cache is ordered source first, so a strict
    SHA-prefix check is sufficient and avoids that destructive rewrite.
    """
    path = Path(args.region_cache)
    expected = np.asarray([norm_sha(row["sha256"]) for row in rows])
    if path.exists() and not bool(args.rebuild_region_cache):
        saved = np.load(path, allow_pickle=False)
        saved_sha = saved["sha256"].astype(str)
        if len(saved_sha) >= len(expected) and np.array_equal(
            saved_sha[:len(expected)], expected
        ):
            host_values = np.asarray(saved["host_values"], dtype=np.float32)
            host_offsets = np.asarray(saved["host_offsets"], dtype=np.int64)
            host_bags = [
                host_values[host_offsets[index]:host_offsets[index + 1]]
                for index in range(len(expected))
            ]
            audits = [{
                "sha256": expected[index],
                "metadata_year": str(rows[index].get("metadata_year", "")).strip(),
                "host_region_n": int(len(host_bags[index])),
                "real_host_region_n": "cached_prefix",
                "ego_region_n": "not_loaded",
                "host_was_empty": "cached_prefix",
                "hostego_total_region_n": int(len(host_bags[index])),
            } for index in range(len(expected))]
            print("Loaded source prefix from local-region cache:", path, flush=True)
            return host_bags, audits, True
    source_cache = path.with_name(path.stem + "_source_only" + path.suffix)
    original = args.region_cache
    args.region_cache = source_cache
    try:
        host_bags, _, audits, loaded = care_v1.get_region_bags(
            rows, endpoint_cache, args
        )
    finally:
        args.region_cache = original
    return host_bags, audits, loaded


def inner_oof_global24(
    rows,
    labels,
    compact_by_sha,
    compact_schema,
    original_names,
    folds,
    split_seed,
    model_seed,
) -> np.ndarray:
    """Strict cross-fitted Global24 probabilities for correction training."""
    labels = np.asarray(labels, dtype=np.int64)
    result = np.full(len(labels), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(
        n_splits=int(folds), shuffle=True, random_state=int(split_seed)
    )
    for inner_fold, (fit_index, val_index) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        fit_rows = [rows[index] for index in fit_index]
        val_rows = [rows[index] for index in val_index]
        x_fit, x_val, names = audit.compact_matrix_variant(
            VARIANT,
            fit_rows,
            val_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        global_index, _ = evidence.split_view_indices(names)
        result[val_index] = g24.fit_view(
            x_fit,
            x_val,
            labels[fit_index],
            global_index,
            int(model_seed) + inner_fold,
        )
    if np.any(~np.isfinite(result)):
        raise RuntimeError("incomplete inner OOF Global24 probabilities")
    return result


def decompose_s7(s7: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s7 = np.asarray(s7, dtype=np.float32)
    if s7.ndim != 2 or s7.shape[1] != S7_DIM:
        raise RuntimeError("bad S7 shape: " + repr(s7.shape))
    node = s7[:, :ROLE_NODE_DIM].reshape(-1, ROLE_COUNT, ROLE_NODE_WIDTH)
    graph_global = s7[:, ROLE_NODE_DIM:120]
    pair_values = s7[:, 120:120 + PAIR_DIM]
    inter = s7[:, 120 + PAIR_DIM:].reshape(-1, ROLE_COUNT, ROLE_COUNT)
    pair = np.zeros((len(s7), ROLE_COUNT, ROLE_COUNT), dtype=np.float32)
    left, right = np.triu_indices(ROLE_COUNT, 1)
    pair[:, left, right] = pair_values
    pair[:, right, left] = pair_values
    return node, graph_global, pair, inter


def fit_role_graph_scaler(s7: np.ndarray) -> dict:
    node, graph_global, pair, _ = decompose_s7(s7)
    node_mean = np.mean(node, axis=(0, 1), keepdims=True)
    node_std = np.std(node, axis=(0, 1), keepdims=True)
    graph_mean = np.mean(graph_global, axis=0, keepdims=True)
    graph_std = np.std(graph_global, axis=0, keepdims=True)
    pair_log = np.log1p(np.maximum(pair, 0.0))
    mask = ~np.eye(ROLE_COUNT, dtype=bool)
    pair_flat = pair_log[:, mask]
    pair_mean = float(np.mean(pair_flat))
    pair_std = float(np.std(pair_flat))
    node_std[node_std < 1e-6] = 1.0
    graph_std[graph_std < 1e-6] = 1.0
    if pair_std < 1e-6:
        pair_std = 1.0
    return {
        "node_mean": node_mean.astype(np.float32),
        "node_std": node_std.astype(np.float32),
        "graph_mean": graph_mean.astype(np.float32),
        "graph_std": graph_std.astype(np.float32),
        "pair_mean": pair_mean,
        "pair_std": pair_std,
    }


def transform_role_graphs(s7: np.ndarray, scaler: dict) -> dict[str, np.ndarray]:
    node, graph_global, pair, inter = decompose_s7(s7)
    node = (node - scaler["node_mean"]) / scaler["node_std"]
    graph_global = (graph_global - scaler["graph_mean"]) / scaler["graph_std"]
    pair_log = np.log1p(np.maximum(pair, 0.0))
    pair_z = (pair_log - float(scaler["pair_mean"])) / float(scaler["pair_std"])
    diagonal = np.arange(ROLE_COUNT)
    pair_z[:, diagonal, diagonal] = 0.0
    inter = np.clip(inter, -5.0, 5.0) / 5.0
    edge = np.stack([pair_z, inter, np.swapaxes(inter, 1, 2)], axis=-1)
    if not (np.isfinite(node).all() and np.isfinite(edge).all() and np.isfinite(graph_global).all()):
        raise RuntimeError("non-finite role graph tensor")
    return {
        "node": node.astype(np.float32),
        "edge": edge.astype(np.float32),
        "global": graph_global.astype(np.float32),
    }


def graph_tensors(values: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in values.items()
    }


class EdgeBiasedGraphBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("graph dimension must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = int(dim // heads)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(3, max(8, heads * 2)),
            nn.ReLU(),
            nn.Linear(max(8, heads * 2), heads),
        )
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        n = nodes.shape[0]
        normalized = self.norm1(nodes)
        qkv = self.qkv(normalized).reshape(n, 3, self.heads, self.head_dim)
        q = qkv[:, 0].permute(1, 0, 2)
        k = qkv[:, 1].permute(1, 0, 2)
        value = qkv[:, 2].permute(1, 0, 2)
        score = torch.einsum("hid,hjd->hij", q, k) / math.sqrt(self.head_dim)
        score = score + self.edge_bias(edges).permute(2, 0, 1)
        attention = torch.softmax(score, dim=-1)
        mixed = torch.einsum("hij,hjd->hid", attention, value)
        mixed = mixed.permute(1, 0, 2).reshape(n, -1)
        nodes = nodes + self.dropout(self.out(mixed))
        nodes = nodes + self.dropout(self.ffn(self.norm2(nodes)))
        return nodes


class RoleGraphEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = int(args.graph_dim)
        self.role_embedding = nn.Embedding(ROLE_COUNT, dim)
        self.node_input = nn.Linear(ROLE_NODE_WIDTH, dim)
        self.blocks = nn.ModuleList([
            EdgeBiasedGraphBlock(dim, int(args.graph_heads), float(args.dropout))
            for _ in range(int(args.graph_layers))
        ])
        self.pool_score = nn.Linear(dim, 1)
        self.global_encoder = nn.Sequential(
            nn.Linear(GRAPH_GLOBAL_DIM, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, node, edge, graph_global):
        role_index = torch.arange(ROLE_COUNT, device=node.device)
        hidden = self.node_input(node) + self.role_embedding(role_index)
        for block in self.blocks:
            hidden = block(hidden, edge)
        pool_weight = torch.softmax(self.pool_score(hidden).reshape(-1), dim=0)
        pooled = torch.sum(pool_weight[:, None] * hidden, dim=0)
        global_hidden = self.global_encoder(graph_global.reshape(-1))
        token = self.fuse(torch.cat([pooled, global_hidden]))
        entropy = -torch.sum(pool_weight * torch.log(pool_weight.clamp_min(1e-8)))
        entropy = entropy / math.log(ROLE_COUNT)
        return token, entropy


class RoleGraphConditionedEvidence(nn.Module):
    def __init__(self, global_dim: int, variant: str, args):
        super().__init__()
        self.variant = str(variant)
        self.conditioned = self.variant in ("RGCondAttn", "RGCondDual")
        self.dual_stream = self.variant == "RGCondDual"
        self.max_delta = float(args.max_delta)
        graph_dim = int(args.graph_dim)
        local_dim = int(args.embedding_dim)
        self.graph_encoder = RoleGraphEncoder(args)
        self.region_encoder = nn.Sequential(
            nn.Linear(AUGMENTED_REGION_DIM, int(args.hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(int(args.hidden_dim), local_dim),
            nn.ReLU(),
        )
        self.local_risk = nn.Linear(local_dim, 1)
        self.graph_query = nn.Linear(graph_dim, local_dim)
        self.local_key = nn.Linear(local_dim, local_dim)
        self.critical_query = nn.Linear(local_dim, local_dim)
        self.context_key = nn.Linear(local_dim, local_dim)
        self.residual_beta_raw = nn.Parameter(torch.tensor(0.0))
        summary_dim = 8
        fused_dim = global_dim + 1 + graph_dim + 3 * local_dim + summary_dim
        self.delta_head = nn.Sequential(
            nn.Linear(fused_dim, int(args.hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(int(args.hidden_dim), 1),
        )

    def forward(self, global_features, base_logit, graph, regions):
        graph_token, graph_entropy = self.graph_encoder(
            graph["node"], graph["edge"], graph["global"]
        )
        local = self.region_encoder(regions)
        local_score = self.local_risk(local).reshape(-1)
        residual = torch.log1p(torch.clamp(regions[:, -1], min=0.0))
        score = local_score + nn.functional.softplus(self.residual_beta_raw) * residual
        if self.conditioned:
            query = self.graph_query(graph_token)
            key = self.local_key(local)
            score = score + torch.mv(key, query) / math.sqrt(local.shape[1])
        attention = torch.softmax(score, dim=0)
        attended = torch.sum(attention[:, None] * local, dim=0)
        critical_index = torch.argmax(score)
        critical = local[critical_index]
        if self.dual_stream:
            query = self.critical_query(critical)
            key = self.context_key(local)
            context_score = torch.mv(key, query) / math.sqrt(local.shape[1])
            context_score = context_score + score
            context_weight = torch.softmax(context_score, dim=0)
            context = torch.sum(context_weight[:, None] * local, dim=0)
        else:
            context_weight = attention
            context = attended
        attention_entropy = -torch.sum(
            attention * torch.log(attention.clamp_min(1e-8))
        ) / math.log(max(int(len(attention)), 2))
        context_entropy = -torch.sum(
            context_weight * torch.log(context_weight.clamp_min(1e-8))
        ) / math.log(max(int(len(context_weight)), 2))
        prototype_summary = torch.stack([
            torch.mean(regions[:, -4]),
            torch.mean(regions[:, -3]),
            torch.mean(regions[:, -2]),
            torch.max(regions[:, -1]),
        ])
        scalar_summary = torch.cat([
            score[critical_index].reshape(1),
            torch.mean(score).reshape(1),
            attention_entropy.reshape(1),
            torch.max(attention).reshape(1),
            prototype_summary,
        ])
        fused = torch.cat([
            global_features.reshape(-1),
            base_logit.reshape(1),
            graph_token,
            critical,
            attended,
            context,
            scalar_summary,
        ])
        raw_delta = self.delta_head(fused).reshape(())
        bounded_delta = self.max_delta * torch.tanh(raw_delta)
        final_logit = base_logit.reshape(()) + bounded_delta
        diagnostic = torch.cat([
            bounded_delta.reshape(1),
            score[critical_index].reshape(1),
            torch.mean(score).reshape(1),
            attention_entropy.reshape(1),
            torch.max(attention).reshape(1),
            context_entropy.reshape(1),
            prototype_summary,
            graph_entropy.reshape(1),
        ])
        return final_logit, diagnostic


def fixkeep_weights(labels, base_probability, args):
    labels = np.asarray(labels, dtype=np.int64)
    base_correct = (np.asarray(base_probability) >= THRESHOLD) == labels
    weights = np.where(
        base_correct, float(args.keep_weight), float(args.fix_weight)
    ).astype(np.float32)
    for label in (0, 1):
        selected = labels == label
        weights[selected] *= float(np.sum(selected)) / float(np.sum(weights[selected]))
    return weights, base_correct


def batch_loss(logits, labels, weights, base_probability, base_correct, args):
    element = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    loss = torch.sum(element * weights) / torch.sum(weights)
    if float(args.keep_consistency) > 0.0 and bool(torch.any(base_correct)):
        final_probability = torch.sigmoid(logits[base_correct])
        keep_loss = torch.mean(torch.square(
            final_probability - base_probability[base_correct]
        ))
        loss = loss + float(args.keep_consistency) * keep_loss
    return loss


def model_forward_batch(model, global_tensor, base_logit, graph, bags, indices):
    logits = []
    for index in indices:
        i = int(index)
        one_graph = {key: value[i] for key, value in graph.items()}
        logit, _ = model(global_tensor[i], base_logit[i], one_graph, bags[i])
        logits.append(logit)
    return torch.stack(logits)


def train_epochs(model, optimizer, global_tensor, base_probability, graph, bags,
                 labels, weights, correct, epochs, args, seed):
    base_logit = torch.as_tensor(
        probability_to_logit(base_probability), dtype=torch.float32,
        device=global_tensor.device,
    )
    base_probability_t = torch.as_tensor(
        base_probability, dtype=torch.float32, device=global_tensor.device
    )
    labels_t = torch.as_tensor(labels, dtype=torch.float32, device=global_tensor.device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=global_tensor.device)
    correct_t = torch.as_tensor(correct, dtype=torch.bool, device=global_tensor.device)
    rng = np.random.RandomState(int(seed) + 811)
    final_loss = float("nan")
    for _ in range(int(epochs)):
        model.train()
        order = rng.permutation(len(labels))
        total = 0.0
        for start in range(0, len(order), int(args.batch_size)):
            batch = order[start:start + int(args.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = model_forward_batch(
                model, global_tensor, base_logit, graph, bags, batch
            )
            index_t = torch.as_tensor(batch, dtype=torch.long, device=global_tensor.device)
            loss = batch_loss(
                logits,
                labels_t[index_t],
                weights_t[index_t],
                base_probability_t[index_t],
                correct_t[index_t],
                args,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
        final_loss = total / len(labels)
    return final_loss


def evaluate_loss(model, global_values, base_probability, graph_values, bags,
                  labels, args, device):
    global_tensor = torch.as_tensor(global_values, dtype=torch.float32, device=device)
    base_logit = torch.as_tensor(
        probability_to_logit(base_probability), dtype=torch.float32, device=device
    )
    graph = graph_tensors(graph_values, device)
    bag_tensors = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in bags]
    weights, correct = fixkeep_weights(labels, base_probability, args)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    correct_t = torch.as_tensor(correct, dtype=torch.bool, device=device)
    labels_t = torch.as_tensor(labels, dtype=torch.float32, device=device)
    base_probability_t = torch.as_tensor(base_probability, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model_forward_batch(
            model, global_tensor, base_logit, graph, bag_tensors,
            np.arange(len(labels)),
        )
        loss = batch_loss(
            logits, labels_t, weights_t, base_probability_t, correct_t, args
        )
    return float(loss.cpu())


def select_epochs(global_values, s7_values, raw_bags, base_probability, labels,
                  variant, args, device, seed):
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(args.early_stop_fraction),
        random_state=int(seed) + int(args.split_seed),
    )
    fit_index, val_index = next(splitter.split(np.zeros(len(labels)), labels))
    fit_raw = [raw_bags[index] for index in fit_index]
    val_raw = [raw_bags[index] for index in val_index]
    fit_bags, val_bags, _ = v3.fit_scaled_dualproto_bags(
        fit_raw, val_raw, np.asarray(labels)[fit_index], args, int(seed) + 707
    )
    scaler = fit_role_graph_scaler(np.asarray(s7_values)[fit_index])
    fit_graph = transform_role_graphs(np.asarray(s7_values)[fit_index], scaler)
    val_graph = transform_role_graphs(np.asarray(s7_values)[val_index], scaler)
    fit_global = torch.as_tensor(
        np.asarray(global_values)[fit_index], dtype=torch.float32, device=device
    )
    fit_graph_t = graph_tensors(fit_graph, device)
    fit_bag_t = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in fit_bags]
    fit_base = np.asarray(base_probability)[fit_index]
    fit_labels = np.asarray(labels)[fit_index]
    weights, correct = fixkeep_weights(fit_labels, fit_base, args)
    seed_everything(seed)
    model = RoleGraphConditionedEvidence(global_values.shape[1], variant, args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    best_loss = float("inf")
    best_epoch = 1
    patience = 0
    for epoch in range(1, int(args.max_epochs) + 1):
        train_epochs(
            model, optimizer, fit_global, fit_base, fit_graph_t, fit_bag_t,
            fit_labels, weights, correct, 1, args, int(seed) + epoch * 17,
        )
        value = evaluate_loss(
            model,
            np.asarray(global_values)[val_index],
            np.asarray(base_probability)[val_index],
            val_graph,
            val_bags,
            np.asarray(labels)[val_index],
            args,
            device,
        )
        if value < best_loss - float(args.early_stop_min_delta):
            best_loss = value
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= int(args.early_stop_patience):
            break
    del model, fit_graph_t, fit_bag_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return int(best_epoch), float(best_loss)


def predict_model(model, global_values, base_probability, graph_values, bags, device):
    global_tensor = torch.as_tensor(global_values, dtype=torch.float32, device=device)
    base_logit = torch.as_tensor(
        probability_to_logit(base_probability), dtype=torch.float32, device=device
    )
    graph = graph_tensors(graph_values, device)
    bag_tensors = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in bags]
    probability = []
    diagnostic = []
    model.eval()
    with torch.no_grad():
        for index in range(len(bags)):
            one_graph = {key: value[index] for key, value in graph.items()}
            logit, diag = model(
                global_tensor[index], base_logit[index], one_graph, bag_tensors[index]
            )
            probability.append(float(torch.sigmoid(logit).cpu()))
            diagnostic.append(diag.detach().cpu().numpy())
    return np.asarray(probability), np.asarray(diagnostic)


def fit_predict_variant(global_train, global_test, s7_train, s7_test,
                        raw_train_bags, raw_test_bags, base_train, base_test,
                        labels, variant, seeds, args, device):
    probabilities = []
    diagnostics = []
    audits = []
    for seed in seeds:
        selected_epochs, selection_loss = select_epochs(
            global_train, s7_train, raw_train_bags, base_train, labels,
            variant, args, device, int(seed),
        )
        train_bags, test_bags, prototype_audit = v3.fit_scaled_dualproto_bags(
            raw_train_bags, raw_test_bags, labels, args, int(seed) + 1707
        )
        scaler = fit_role_graph_scaler(s7_train)
        train_graph = transform_role_graphs(s7_train, scaler)
        test_graph = transform_role_graphs(s7_test, scaler)
        global_tensor = torch.as_tensor(
            global_train, dtype=torch.float32, device=device
        )
        train_graph_t = graph_tensors(train_graph, device)
        train_bag_t = [
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in train_bags
        ]
        weights, correct = fixkeep_weights(labels, base_train, args)
        seed_everything(seed)
        model = RoleGraphConditionedEvidence(global_train.shape[1], variant, args).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        final_loss = train_epochs(
            model, optimizer, global_tensor, base_train, train_graph_t,
            train_bag_t, labels, weights, correct, selected_epochs, args,
            int(seed) + 3301,
        )
        probability, diagnostic = predict_model(
            model, global_test, base_test, test_graph, test_bags, device
        )
        probabilities.append(probability)
        diagnostics.append(diagnostic)
        parameter_n = int(sum(value.numel() for value in model.parameters()))
        audits.append({
            "seed": int(seed),
            "variant": variant,
            "selected_epochs": int(selected_epochs),
            "selection_loss": float(selection_loss),
            "final_training_loss": float(final_loss),
            "parameter_n": parameter_n,
            **prototype_audit,
        })
        del model, train_graph_t, train_bag_t
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (
        np.mean(np.asarray(probabilities), axis=0),
        np.mean(np.asarray(diagnostics), axis=0),
        audits,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--cache", type=Path, default=Path(
        "artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz"
    ))
    parser.add_argument("--p72-oof", type=Path, default=Path(
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"
    ))
    parser.add_argument("--source-multiscale-features", type=Path, default=Path(
        "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"
    ))
    parser.add_argument("--previous-result", type=Path, default=Path(
        "artifacts/global24-a25-route-conditioned-joints7-v1-full"
    ))
    parser.add_argument("--care-reference", type=Path, default=Path(
        "artifacts/care-bounded-fixkeep-v3-full/source_predictions.csv"
    ))
    parser.add_argument("--region-cache", type=Path, default=Path(
        "artifacts/care-local-region-cache-v1/regions_2017_2021.npz"
    ))
    parser.add_argument("--rebuild-region-cache", action="store_true")
    parser.add_argument("--variants", default=",".join(MODEL_VARIANTS))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260901)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stop-fraction", type=float, default=0.18)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=24)
    parser.add_argument("--graph-dim", type=int, default=32)
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-delta", type=float, default=1.5)
    parser.add_argument("--keep-weight", type=float, default=2.0)
    parser.add_argument("--fix-weight", type=float, default=1.0)
    parser.add_argument("--keep-consistency", type=float, default=0.5)
    parser.add_argument("--shared-prototypes", type=int, default=32)
    parser.add_argument("--benign-prototypes", type=int, default=24)
    parser.add_argument("--prototype-regions-per-apk", type=int, default=16)
    parser.add_argument("--internal-background-coverage", type=float, default=0.70)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-rolegraph-conditioned-dualstream-v4"
    ))
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    seeds = parse_ints(args.seeds)
    if args.graph_dim % args.graph_heads:
        raise SystemExit("--graph-dim must be divisible by --graph-heads")
    if args.inner_folds < 2 or args.max_epochs < 1:
        raise SystemExit("invalid folds/epochs")
    if not 0.0 < args.early_stop_fraction < 0.5:
        raise SystemExit("--early-stop-fraction must be in (0,0.5)")
    if args.max_delta <= 0.0:
        raise SystemExit("--max-delta must be positive")
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    print("Device:", device, "seeds:", seeds, "variants:", variants, flush=True)

    (
        manifest,
        source_rows,
        _,
        compact_by_sha,
        compact_schema,
        original_names,
    ) = audit.load_inputs(args)
    source_y = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    source_fold = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    source_reference = care_v1.reference_arrays(
        args.previous_result / "source_predictions.csv", manifest, True
    )
    endpoint_cache = audit.exp.read_cache(args.cache)
    source_s7, _ = v3.load_source_s7(source_rows, endpoint_cache, args)
    if source_s7.shape != (len(source_rows), S7_DIM):
        raise RuntimeError("unexpected source S7 shape")
    host_bags, bag_audit, region_cache_loaded = get_source_host_bags(
        source_rows, endpoint_cache, args
    )

    source_expert = {
        variant: np.full(len(source_y), np.nan, dtype=np.float64)
        for variant in variants
    }
    source_diagnostic = {
        variant: np.full((len(source_y), len(DIAGNOSTIC_NAMES)), np.nan, dtype=np.float64)
        for variant in variants
    }
    fit_audit = []
    print("Running strict 2017 role-graph conditioned ablation...", flush=True)
    for fold in sorted(np.unique(source_fold)):
        train_index = np.flatnonzero(source_fold != fold)
        val_index = np.flatnonzero(source_fold == fold)
        train_rows = [source_rows[index] for index in train_index]
        val_rows = [source_rows[index] for index in val_index]
        x_train, x_val, names = audit.compact_matrix_variant(
            VARIANT, train_rows, val_rows,
            compact_by_sha, compact_schema, original_names,
        )
        global_index, _ = evidence.split_view_indices(names)
        global_train = np.asarray(x_train[:, global_index], dtype=np.float32)
        global_val = np.asarray(x_val[:, global_index], dtype=np.float32)
        train_y = source_y[train_index]
        print("  fold", int(fold), "inner OOF Global24...", flush=True)
        base_train = inner_oof_global24(
            train_rows,
            train_y,
            compact_by_sha,
            compact_schema,
            original_names,
            args.inner_folds,
            args.split_seed + 1000 * int(fold),
            args.split_seed + 10000 * int(fold),
        )
        base_val = source_reference["coarse"][val_index]
        train_bags = [host_bags[index] for index in train_index]
        val_bags = [host_bags[index] for index in val_index]
        for variant in variants:
            print("  fold", int(fold), variant, flush=True)
            fold_seeds = [100000 * int(fold) + seed for seed in seeds]
            probability, diagnostic, audits = fit_predict_variant(
                global_train,
                global_val,
                source_s7[train_index],
                source_s7[val_index],
                train_bags,
                val_bags,
                base_train,
                base_val,
                train_y,
                variant,
                fold_seeds,
                args,
                device,
            )
            source_expert[variant][val_index] = probability
            source_diagnostic[variant][val_index] = diagnostic
            for row in audits:
                fit_audit.append({
                    "fold": int(fold),
                    "train_n": int(len(train_index)),
                    "validation_n": int(len(val_index)),
                    **row,
                })
        print("fold", int(fold), "done", flush=True)

    if any(np.any(~np.isfinite(value)) for value in source_expert.values()):
        raise RuntimeError("incomplete source expert predictions")
    probabilities = {
        "Global24": source_reference["coarse"],
        "Global24__A25__JointS7_ordinary": source_reference["ordinary"],
    }
    care_reference = optional_reference_probability(
        args.care_reference,
        manifest,
        "Global24__A25__CARE_DualProto_ES_FixKeep__prob",
    )
    if care_reference is not None:
        probabilities["Global24__A25__CARE_DualProto_ES_FixKeep_reference"] = care_reference
    for variant in variants:
        method = METHOD_FOR_VARIANT[variant]
        probabilities[method] = acn.routed_probability(
            source_reference["coarse"],
            source_expert[variant],
            source_reference["route"],
        )
    summary = v2.metric_rows(
        source_y, probabilities, source_reference["coarse"], source_reference["route"]
    )
    per_fold = []
    for fold in sorted(np.unique(source_fold)):
        selected = source_fold == fold
        for row in v2.metric_rows(
            source_y[selected],
            {name: value[selected] for name, value in probabilities.items()},
            source_reference["coarse"][selected],
            source_reference["route"][selected],
        ):
            per_fold.append({"fold": int(fold), **row})
    prediction_rows = []
    for index, row in enumerate(manifest):
        output = {
            "sha256": norm_sha(row["sha256"]),
            "label": int(source_y[index]),
            "fold": int(source_fold[index]),
            "Global24__prob": float(source_reference["coarse"][index]),
            "A25__score": float(source_reference["score"][index]),
            "A25__routed": int(source_reference["route"][index]),
            "Global24__A25__JointS7_ordinary__prob": float(
                source_reference["ordinary"][index]
            ),
        }
        if care_reference is not None:
            output["CARE_FixKeep_reference__prob"] = float(care_reference[index])
        for variant in variants:
            method = METHOD_FOR_VARIANT[variant]
            output[method + "__expert_prob"] = float(source_expert[variant][index])
            output[method + "__prob"] = float(probabilities[method][index])
            for column, name in enumerate(DIAGNOSTIC_NAMES):
                output[method + "__" + name] = float(
                    source_diagnostic[variant][index, column]
                )
        prediction_rows.append(output)

    g24.write_csv(args.output / "source_summary.csv", summary)
    g24.write_csv(args.output / "source_per_fold.csv", per_fold)
    g24.write_csv(args.output / "source_predictions.csv", prediction_rows)
    g24.write_csv(args.output / "fit_audit.csv", fit_audit)
    g24.write_csv(args.output / "region_bag_audit.csv", bag_audit)
    architecture = {
        "experiment": "CARE role-graph conditioned dual-stream v4",
        "evaluation": "strict fixed 2017 outer folds only",
        "first_layer": "frozen role-free Global24",
        "router": "frozen A25",
        "candidate_base": "strictly cross-fitted Global24 logit",
        "bounded_correction": float(args.max_delta),
        "s7_mapping": {
            "Stats120": "31x3 role-node features + 27 graph-global features",
            "PairRaw465": "31 choose 2 undirected edge channel",
            "InterZ961": "31x31 directed edge channel clipped to [-5,5]",
        },
        "variants": {
            "RGConcat": "role token fused at the end; no global conditioning of local attention",
            "RGCondAttn": "role token conditions local attention; single attention pool",
            "RGCondDual": "role-conditioned critical instance plus critical-context stream",
        },
        "dual_prototype": True,
        "fixkeep": True,
        "region_cache_loaded": bool(region_cache_loaded),
        "seeds": seeds,
        "parameters": vars(args),
    }
    architecture["parameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in architecture["parameters"].items()
    }
    with (args.output / "architecture.json").open("w", encoding="utf-8") as handle:
        json.dump(architecture, handle, ensure_ascii=False, indent=2)

    by_method = {row["method"]: row for row in summary}
    lines = [
        "CARE-Droid v4: role-graph conditioned local evidence",
        "",
        "Strict source-only evaluation: fixed 2017 outer folds.",
        "Global24 and A25 are frozen. Future years are not loaded.",
        "S7 is interpreted as a 31-node role graph rather than a flat 1546-D vector.",
        "All candidates are bounded corrections of cross-fitted Global24 and are applied only on A25.",
        "",
        "Source summary:",
    ]
    for name in probabilities:
        row = by_method[name]
        lines.append(
            f"{name}: BA={float(row['BA']):.6f} AUC={float(row['AUC']):.6f} "
            f"F1={float(row['F1']):.6f} fixed={int(row['fixed'])} "
            f"broken={int(row['broken'])} net={int(row['net'])}"
        )
    lines.extend([
        "",
        "Decision rules:",
        "1. RGCondAttn vs RGConcat isolates whether global-to-local conditioning helps.",
        "2. RGCondDual vs RGCondAttn isolates the critical-instance/context dual stream.",
        "3. The full method must beat ordinary Joint-S7 and not merely compress S7.",
        "4. Compare Fix/Break and per-fold BA, not only aggregate AUC.",
        "5. Only after a source-side decision may a separate frozen future diagnostic run.",
    ])
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
