#!/usr/bin/env python3
"""Train and extract the audited true Host-only Anomaly 88-D representation."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from probe_protocol import validate_noego_88


def encode_one(model, graph: dict, regions: torch.Tensor) -> torch.Tensor:
    graph_token, _ = model.graph_encoder(graph["node"], graph["edge"], graph["global"])
    local = model.region_encoder(regions)
    local_score = model.local_risk(local).reshape(-1)
    residual = torch.log1p(torch.clamp(regions[:, -1], min=0.0))
    score = local_score + nn.functional.softplus(model.residual_beta_raw) * residual
    score = score + torch.mv(model.local_key(local), model.graph_query(graph_token)) / math.sqrt(local.shape[1])
    attention = torch.softmax(score, dim=0)
    critical_index = torch.argmax(score)
    critical = local[critical_index]
    attended = torch.sum(attention[:, None] * local, dim=0)
    attention_entropy = -torch.sum(attention * torch.log(attention.clamp_min(1e-8))) / math.log(max(int(len(attention)), 2))
    prototype_summary = torch.stack([
        torch.mean(regions[:, -4]), torch.mean(regions[:, -3]),
        torch.mean(regions[:, -2]), torch.max(regions[:, -1]),
    ])
    scalar_summary = torch.cat([
        score[critical_index].reshape(1), torch.mean(score).reshape(1),
        attention_entropy.reshape(1), torch.max(attention).reshape(1), prototype_summary,
    ])
    # Deliberately excludes Global24, Stage-I logit, Ego and delta-head output.
    result = torch.cat([graph_token, critical, attended, scalar_summary])
    if result.numel() != 88:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host-only representation width={result.numel()} expected=88")
    return result


def encode_many(model, graph_values: dict, bags, device) -> np.ndarray:
    graph = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in graph_values.items()}
    output = []
    model.eval()
    with torch.no_grad():
        for index, bag in enumerate(bags):
            one_graph = {key: value[index] for key, value in graph.items()}
            regions = torch.as_tensor(bag, dtype=torch.float32, device=device)
            output.append(encode_one(model, one_graph, regions).detach().cpu().numpy())
    return validate_noego_88(np.asarray(output, dtype=np.float32), len(bags))


def fit_seed(v4, anomaly, global_train, s7_train, s7_validation, raw_train_bags,
             raw_validation_bags, base_train, labels_train, args, device, seed):
    epochs, selection_loss = v4.select_epochs(
        global_train, s7_train, raw_train_bags, base_train, labels_train,
        "RGCondAttn", args, device, int(seed),
    )
    train_bags, validation_bags, prototype_audit = v4.v3.fit_scaled_dualproto_bags(
        raw_train_bags, raw_validation_bags, labels_train, args, int(seed) + 1707
    )
    graph_scaler = v4.fit_role_graph_scaler(s7_train)
    train_graph_values = v4.transform_role_graphs(s7_train, graph_scaler)
    validation_graph_values = v4.transform_role_graphs(s7_validation, graph_scaler)
    train_graph = v4.graph_tensors(train_graph_values, device)
    train_bag_tensors = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in train_bags]
    global_tensor = torch.as_tensor(global_train, dtype=torch.float32, device=device)
    weights, correct = v4.fixkeep_weights(labels_train, base_train, args)
    v4.seed_everything(int(seed))
    model = anomaly.DedupBoundedRGCondAttn(global_train.shape[1], "RGCondAttn", args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    final_loss = v4.train_epochs(
        model, optimizer, global_tensor, base_train, train_graph, train_bag_tensors,
        labels_train, weights, correct, int(epochs), args, int(seed) + 3301,
    )
    train_representation = encode_many(model, train_graph_values, train_bags, device)
    validation_representation = encode_many(model, validation_graph_values, validation_bags, device)
    audit = {
        "seed": int(seed), "selected_epochs": int(epochs),
        "selection_loss": float(selection_loss), "final_training_loss": float(final_loss),
        "parameter_n": int(sum(value.numel() for value in model.parameters())),
        "representation_dim": 88, "ego_consumed": False,
        **dict(prototype_audit),
    }
    del model, train_graph, train_bag_tensors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return train_representation, validation_representation, audit
