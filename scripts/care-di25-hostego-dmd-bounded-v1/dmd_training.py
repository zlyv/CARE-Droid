#!/usr/bin/env python3
"""Fold-safe DMD-BC training, inference logging, and checkpointing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit

LAMBDA_ACTION = 1.0
LAMBDA_MAGNITUDE = 1.0
MARGIN_EPSILON = 1e-3


def sha_list(values):
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def targets(labels, base_probability, action_classes, device):
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    p = torch.as_tensor(base_probability, dtype=torch.float32, device=device)
    pred = (p >= .5).long()
    wrong = pred != y
    # Full: NEG=0, KEEP=1, POS=2. NoKeep: NEG=0, POS=1.
    if action_classes == 3:
        action = torch.where(wrong, torch.where(y == 1, 2, 0), 1)
        action_mask = torch.ones_like(wrong)
    else:
        action = torch.where(y == 1, 1, 0)
        action_mask = wrong
    logit = torch.logit(p.clamp(1e-6, 1 - 1e-6))
    magnitude = torch.clamp(torch.abs(logit) + MARGIN_EPSILON, max=2.25)
    return action, action_mask, magnitude, wrong


def forward_aux(model, global_tensor, base_logit, graph, bags, indices, capture=False):
    logits, action, magnitude, rows = [], [], [], []
    for value in indices:
        i = int(value)
        one_graph = {key: tensor[i] for key, tensor in graph.items()}
        logit, _ = model(global_tensor[i], base_logit[i], one_graph, bags[i])
        aux = model.last_dmd
        logits.append(logit)
        action.append(aux["action_logits"])
        magnitude.append(aux["magnitude"])
        if capture:
            internal = model.last_internal_capture
            rows.append({
                "action_logits": aux["action_logits"].detach().cpu().numpy(),
                "action_probability": aux["action_probability"].detach().cpu().numpy(),
                "direction": float(aux["direction"].detach().cpu()),
                "magnitude": float(aux["magnitude"].detach().cpu()),
                "delta": float(aux["delta"].detach().cpu()),
                "gamma": internal["gamma"], "anomaly": internal["anomaly"],
                "attention": internal["attention"],
                "critical_host_index": internal["critical_host_index"],
            })
    return torch.stack(logits), torch.stack(action), torch.stack(magnitude), rows


def total_loss(v4, model, logits, action_logits, magnitude, labels, weights,
               base_probability, base_correct, action_classes, args):
    final = v4.batch_loss(logits, labels.float(), weights, base_probability,
                          base_correct, args)
    action_target, action_mask, magnitude_target, wrong = targets(
        labels, base_probability, action_classes, logits.device)
    if bool(action_mask.any()):
        action = nn.functional.cross_entropy(action_logits[action_mask], action_target[action_mask])
    else:
        action = logits.sum() * 0.0
    if bool(wrong.any()):
        magnitude_loss = nn.functional.smooth_l1_loss(
            magnitude[wrong], magnitude_target[wrong])
    else:
        magnitude_loss = logits.sum() * 0.0
    total = final + LAMBDA_ACTION * action + LAMBDA_MAGNITUDE * magnitude_loss
    return total, {"final": final, "action": action, "magnitude": magnitude_loss}


def train_epochs(v4, model, optimizer, global_tensor, base_probability, graph, bags,
                 labels, weights, correct, epochs, args, seed, action_classes):
    device = global_tensor.device
    base_logit = torch.as_tensor(v4.probability_to_logit(base_probability), dtype=torch.float32, device=device)
    base_p = torch.as_tensor(base_probability, dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    correct_t = torch.as_tensor(correct, dtype=torch.bool, device=device)
    rng = np.random.RandomState(int(seed) + 811)
    means = {}
    for _ in range(int(epochs)):
        model.train(); order = rng.permutation(len(labels)); totals = {k: 0.0 for k in ("total","final","action","magnitude")}
        for start in range(0, len(order), int(args.batch_size)):
            batch = order[start:start + int(args.batch_size)]
            index = torch.as_tensor(batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, action, magnitude, _ = forward_aux(model, global_tensor, base_logit, graph, bags, batch)
            loss, parts = total_loss(v4, model, logits, action, magnitude, labels_t[index],
                weights_t[index], base_p[index], correct_t[index], action_classes, args)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)); optimizer.step()
            totals["total"] += float(loss.detach().cpu()) * len(batch)
            for key in parts: totals[key] += float(parts[key].detach().cpu()) * len(batch)
        means = {key: value / len(labels) for key, value in totals.items()}
    return means


def evaluate(v4, model, global_values, base_probability, graph_values, bags, labels,
             args, device, action_classes):
    gt = torch.as_tensor(global_values, dtype=torch.float32, device=device)
    bl = torch.as_tensor(v4.probability_to_logit(base_probability), dtype=torch.float32, device=device)
    graph = v4.graph_tensors(graph_values, device)
    bt = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in bags]
    weights, correct = v4.fixkeep_weights(labels, base_probability, args)
    model.eval()
    with torch.no_grad():
        logits, action, magnitude, _ = forward_aux(model, gt, bl, graph, bt, np.arange(len(labels)))
        loss, parts = total_loss(v4, model, logits, action, magnitude,
            torch.as_tensor(labels, dtype=torch.long, device=device),
            torch.as_tensor(weights, dtype=torch.float32, device=device),
            torch.as_tensor(base_probability, dtype=torch.float32, device=device),
            torch.as_tensor(correct, dtype=torch.bool, device=device), action_classes, args)
    return float(loss.cpu()), {key: float(value.cpu()) for key, value in parts.items()}


def select_epochs(v4, model_class, global_values, s7_values, raw_bags, base_probability,
                  labels, variant, args, device, seed, action_classes, fit_bags):
    split = StratifiedShuffleSplit(n_splits=1, test_size=float(args.early_stop_fraction),
        random_state=int(seed) + int(args.split_seed))
    fit_idx, val_idx = next(split.split(np.zeros(len(labels)), labels))
    fit_aug, val_aug, _, _ = fit_bags([raw_bags[i] for i in fit_idx],
        [raw_bags[i] for i in val_idx], np.asarray(labels)[fit_idx], args, int(seed)+707)
    scaler = v4.fit_role_graph_scaler(np.asarray(s7_values)[fit_idx])
    fg = v4.transform_role_graphs(np.asarray(s7_values)[fit_idx], scaler)
    vg = v4.transform_role_graphs(np.asarray(s7_values)[val_idx], scaler)
    gt = torch.as_tensor(np.asarray(global_values)[fit_idx], dtype=torch.float32, device=device)
    graph = v4.graph_tensors(fg, device)
    bags = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in fit_aug]
    bp = np.asarray(base_probability)[fit_idx]; ly = np.asarray(labels)[fit_idx]
    weights, correct = v4.fixkeep_weights(ly, bp, args)
    v4.seed_everything(seed); model = model_class(global_values.shape[1], variant, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    best, best_epoch, patience = float("inf"), 1, 0
    for epoch in range(1, int(args.max_epochs)+1):
        train_epochs(v4, model, optimizer, gt, bp, graph, bags, ly, weights, correct,
            1, args, int(seed)+epoch*17, action_classes)
        value, _ = evaluate(v4, model, np.asarray(global_values)[val_idx],
            np.asarray(base_probability)[val_idx], vg, val_aug, np.asarray(labels)[val_idx],
            args, device, action_classes)
        if value < best - float(args.early_stop_min_delta): best, best_epoch, patience = value, epoch, 0
        else: patience += 1
        if patience >= int(args.early_stop_patience): break
    del model, graph, bags
    if device.type == "cuda": torch.cuda.empty_cache()
    return int(best_epoch), float(best)

