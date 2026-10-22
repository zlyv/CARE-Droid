#!/usr/bin/env python3
from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def pad_bags(bags, device):
    lengths = torch.as_tensor([len(bag) for bag in bags], dtype=torch.long, device=device)
    width = max(int(lengths.max().item()), 1)
    value = torch.zeros((len(bags), width, 43), dtype=torch.float32, device=device)
    mask = torch.zeros((len(bags), width), dtype=torch.bool, device=device)
    for index, bag in enumerate(bags):
        tensor = torch.as_tensor(bag, dtype=torch.float32, device=device)
        value[index, :len(tensor)] = tensor; mask[index, :len(tensor)] = True
    return value, mask, lengths


class ABMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(43, 16), nn.ReLU(), nn.Dropout(.10))
        self.attention = nn.Sequential(nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, 1))
        self.classifier = nn.Linear(16, 1)

    def forward(self, x, mask, labels=None):
        z = self.encoder(x); attention = self.attention(z).squeeze(-1)
        attention = attention.masked_fill(~mask, -1e9); weight = torch.softmax(attention, dim=1)
        pooled = torch.sum(weight.unsqueeze(-1) * z, dim=1)
        logit = self.classifier(pooled).squeeze(-1)
        return {"logit": logit, "malware": F.softplus(logit), "benign": F.softplus(-logit),
                "host_signed": attention, "instance_loss": logit.new_zeros(())}


class SignedMIL(nn.Module):
    def __init__(self, clam: bool):
        super().__init__(); self.clam = clam
        self.encoder = nn.Sequential(nn.Linear(43, 16), nn.ReLU(), nn.Dropout(.10))
        self.malware_head = nn.Linear(16, 1); self.benign_head = nn.Linear(16, 1)
        self.bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def aggregate(raw, mask, lengths):
        positive = F.softplus(raw); masked = positive.masked_fill(~mask, -1e9)
        maximum = masked.max(dim=1).values; coverage = (torch.sigmoid(raw) * mask).sum(dim=1) / lengths
        top_mean = []
        for index, length in enumerate(lengths.tolist()):
            k = max(1, min(5, int(math.ceil(math.sqrt(length)))))
            top_mean.append(torch.topk(positive[index, :length], k=k).values.mean())
        top_mean = torch.stack(top_mean)
        return (maximum + top_mean + coverage) / 3.0

    def forward(self, x, mask, labels=None):
        z = self.encoder(x); malware_raw = self.malware_head(z).squeeze(-1)
        benign_raw = self.benign_head(z).squeeze(-1); lengths = mask.sum(dim=1).clamp_min(1)
        malware = self.aggregate(malware_raw, mask, lengths); benign = self.aggregate(benign_raw, mask, lengths)
        logit = malware - benign + self.bias; instance_loss = logit.new_zeros(())
        if self.clam and labels is not None:
            losses = []
            for index, (length, label) in enumerate(zip(lengths.tolist(), labels.tolist())):
                k = max(1, min(4, int(math.ceil(math.sqrt(length)))))
                m = malware_raw[index, :length]; b = benign_raw[index, :length]
                if int(label) == 1:
                    bottom_m = torch.topk(-m, k).indices
                    losses.extend([F.binary_cross_entropy_with_logits(torch.topk(m, k).values, torch.ones(k, device=x.device)),
                                   F.binary_cross_entropy_with_logits(m[bottom_m], torch.zeros(k, device=x.device)),
                                   F.binary_cross_entropy_with_logits(torch.topk(b, k).values, torch.zeros(k, device=x.device))])
                else:
                    bottom_b = torch.topk(-b, k).indices
                    losses.extend([F.binary_cross_entropy_with_logits(torch.topk(b, k).values, torch.ones(k, device=x.device)),
                                   F.binary_cross_entropy_with_logits(b[bottom_b], torch.zeros(k, device=x.device)),
                                   F.binary_cross_entropy_with_logits(torch.topk(m, k).values, torch.zeros(k, device=x.device))])
            if losses: instance_loss = torch.stack(losses).mean()
        return {"logit": logit, "malware": malware, "benign": benign,
                "host_signed": malware_raw - benign_raw, "instance_loss": instance_loss}


def make_model(arm):
    if arm == "M1_ABMIL": return ABMIL()
    if arm == "M2_SIGNED_MIL": return SignedMIL(False)
    if arm == "M3_SIGNED_CLAM": return SignedMIL(True)
    raise ValueError(arm)


def standardize(train_bags, other_bags):
    values = np.concatenate(train_bags, axis=0).astype(np.float64)
    mean = values.mean(axis=0); scale = values.std(axis=0); scale[scale < 1e-6] = 1.0
    convert = lambda bags: [np.asarray((bag - mean) / scale, np.float32) for bag in bags]
    return convert(train_bags), convert(other_bags), mean.astype(np.float32), scale.astype(np.float32)


def fit_model(arm, train_bags, train_y, seed, device, epochs=25, batch_size=16):
    seed_all(seed); model = make_model(arm).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_y = np.asarray(train_y, int); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train(); order = rng.permutation(len(train_bags))
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]; batch = [train_bags[i] for i in index]
            x, mask, _ = pad_bags(batch, device); labels = torch.as_tensor(train_y[index], dtype=torch.float32, device=device)
            output = model(x, mask, labels); bag_loss = F.binary_cross_entropy_with_logits(output["logit"], labels)
            loss = bag_loss + (.25 * output["instance_loss"] if arm == "M3_SIGNED_CLAM" else 0.0)
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    return model


@torch.no_grad()
def predict_model(model, bags, device, batch_size=32):
    model.eval(); result = {"logit": [], "malware": [], "benign": [], "host_signed": []}
    for start in range(0, len(bags), batch_size):
        batch = bags[start:start + batch_size]; x, mask, lengths = pad_bags(batch, device); output = model(x, mask)
        for key in ("logit", "malware", "benign"):
            result[key].extend(output[key].detach().cpu().tolist())
        signed = output["host_signed"].detach().cpu().tolist()
        for index, length in enumerate(lengths.cpu().tolist()):
            result["host_signed"].append(np.asarray(signed[index][:length], float))
    for key in ("logit", "malware", "benign"): result[key] = np.asarray(result[key], float)
    return result

