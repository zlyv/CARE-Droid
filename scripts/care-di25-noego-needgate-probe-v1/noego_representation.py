#!/usr/bin/env python3
"""Exact Host-only Anomaly structural representation extraction (88-D)."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from probe_common import REPRESENTATION_WIDTH, blocked, validate_representation


def encode_one(model, graph: dict, regions: torch.Tensor) -> torch.Tensor:
    """Mirror DedupBoundedRGCondAttn.forward up to, but excluding, its delta head."""
    graph_token, _ = model.graph_encoder(graph["node"], graph["edge"], graph["global"])
    local = model.region_encoder(regions)
    local_score = model.local_risk(local).reshape(-1)
    residual = torch.log1p(torch.clamp(regions[:, -1], min=0.0))
    score = local_score + nn.functional.softplus(model.residual_beta_raw) * residual
    query = model.graph_query(graph_token)
    key = model.local_key(local)
    score = score + torch.mv(key, query) / math.sqrt(local.shape[1])
    attention = torch.softmax(score, dim=0)
    attended = torch.sum(attention[:, None] * local, dim=0)
    critical = local[torch.argmax(score)]
    attention_entropy = -torch.sum(attention * torch.log(attention.clamp_min(1e-8))) / math.log(max(int(len(attention)), 2))
    prototype_summary = torch.stack([
        torch.mean(regions[:, -4]), torch.mean(regions[:, -3]),
        torch.mean(regions[:, -2]), torch.max(regions[:, -1]),
    ])
    scalar_summary = torch.cat([
        score[torch.argmax(score)].reshape(1), torch.mean(score).reshape(1),
        attention_entropy.reshape(1), torch.max(attention).reshape(1), prototype_summary,
    ])
    structural = torch.cat([graph_token, critical, attended, scalar_summary])
    if structural.numel() != REPRESENTATION_WIDTH:
        raise blocked(f"Host-only structural width changed: {structural.numel()} != 88")
    return structural


def encode_many(model, graph_values: dict, region_values, device) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        graph_on_device = {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in graph_values.items()
        }
        for index, regions in enumerate(region_values):
            one_graph = {key: value[index] for key, value in graph_on_device.items()}
            region_tensor = torch.as_tensor(regions, dtype=torch.float32, device=device)
            rows.append(encode_one(model, one_graph, region_tensor).detach().cpu().numpy())
    return validate_representation(np.asarray(rows, dtype=np.float32), len(rows))


def train_seed_and_encode(v4, anomaly, data, train, validation, global_train, global_validation,
                          base_train, base_validation, args, device, seed: int):
    """Train the historical Host-only model on outer-train and extract fold-local 88-D vectors."""
    train_raw = [data.bags[int(i)] for i in train]
    validation_raw = [data.bags[int(i)] for i in validation]
    selected_epochs, selection_loss = v4.select_epochs(
        global_train, data.s7[train], train_raw, base_train, data.labels[train],
        "RGCondAttn", args, device, int(seed),
    )
    train_bags, validation_bags, bag_audit = v4.v3.fit_scaled_dualproto_bags(
        train_raw, validation_raw, data.labels[train], args, int(seed) + 707
    )
    graph_scaler = v4.fit_role_graph_scaler(data.s7[train])
    train_graph_values = v4.transform_role_graphs(data.s7[train], graph_scaler)
    validation_graph_values = v4.transform_role_graphs(data.s7[validation], graph_scaler)
    train_regions = [np.asarray(x, dtype=np.float32) for x in train_bags]
    validation_regions = [np.asarray(x, dtype=np.float32) for x in validation_bags]
    weights, correct = v4.fixkeep_weights(data.labels[train], base_train, args)
    v4.seed_everything(seed)
    model = anomaly.DedupBoundedRGCondAttn(global_train.shape[1], "RGCondAttn", args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    global_tensor = torch.as_tensor(global_train, dtype=torch.float32, device=device)
    train_graphs = v4.graph_tensors(train_graph_values, device)
    train_region_tensors = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in train_regions]
    v4.train_epochs(
        model, optimizer, global_tensor, base_train, train_graphs, train_region_tensors,
        data.labels[train], weights, correct, selected_epochs, args, int(seed) + 909,
    )
    train_features = encode_many(model, train_graph_values, train_regions, device)
    validation_features = encode_many(model, validation_graph_values, validation_regions, device)
    audit = {
        "seed": int(seed), "selected_epochs": int(selected_epochs), "selection_loss": float(selection_loss),
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_n": int(len(train)), "validation_n": int(len(validation)),
        "train_routed_n": int(np.sum(data.route[train])),
        "validation_routed_n": int(np.sum(data.route[validation])),
        "representation_width": int(train_features.shape[1]),
        "bag_audit": str(bag_audit),
    }
    return train_features, validation_features, audit
