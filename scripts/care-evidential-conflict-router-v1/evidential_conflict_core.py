#!/usr/bin/env python3
"""Evidential view heads and conflict-aware evidence fusion for CARE-Droid."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for evidential heads: " + str(exc))


AUX_VIEW_NAMES = ("manifest", "code_call", "sensitive", "graph")


def resolve_device(requested: str) -> torch.device:
    text = str(requested).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(text)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EvidentialViewMLP(nn.Module):
    """Small per-view network whose two nonnegative outputs are class evidence."""

    def __init__(self, input_dimension: int, hidden_dimension: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dimension), int(hidden_dimension)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dimension), int(hidden_dimension)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dimension), 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return functional.softplus(self.network(features))


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1)) for every row."""
    classes = int(alpha.shape[1])
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first = (
        torch.lgamma(sum_alpha)
        - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        - torch.lgamma(torch.tensor(float(classes), device=alpha.device))
    )
    second = torch.sum(
        (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(sum_alpha)),
        dim=1,
        keepdim=True,
    )
    return first + second


def evidential_loss_terms(
    evidence: torch.Tensor,
    labels: torch.Tensor,
    annealing: float,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sensoy-style squared-error Bayes risk plus annealed KL regularization."""
    alpha = evidence + 1.0
    strength = torch.sum(alpha, dim=1, keepdim=True)
    probability = alpha / strength
    target = functional.one_hot(labels, num_classes=2).to(probability.dtype)
    data = torch.sum(
        (target - probability) ** 2
        + probability * (1.0 - probability) / (strength + 1.0),
        dim=1,
    ).mean()
    adjusted_alpha = target + (1.0 - target) * alpha
    kl = dirichlet_kl_to_uniform(adjusted_alpha).mean()
    total = data + float(annealing) * float(kl_weight) * kl
    return total, data, kl


@dataclass
class EvidentialFit:
    train_evidence: np.ndarray
    target_evidence: np.ndarray
    diagnostic: dict


def fit_evidential_view(
    train_x: np.ndarray,
    train_y: np.ndarray,
    target_x: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    hidden_dimension: int = 32,
    dropout: float = 0.10,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    kl_weight: float = 0.01,
    annealing_epochs: int = 100,
) -> EvidentialFit:
    """Fit one view without using target labels and return train/target evidence."""
    seed_everything(seed)
    x_train = np.asarray(train_x, dtype=np.float32)
    x_target = np.asarray(target_x, dtype=np.float32)
    y_train = np.asarray(train_y, dtype=np.int64)
    if x_train.ndim != 2 or x_target.ndim != 2:
        raise ValueError("feature matrices must be two-dimensional")
    if x_train.shape[1] != x_target.shape[1] or y_train.shape != (len(x_train),):
        raise ValueError("training and target matrices are not aligned")
    if not np.isfinite(x_train).all() or not np.isfinite(x_target).all():
        raise ValueError("feature matrix contains non-finite values")
    if not np.isin(y_train, (0, 1)).all():
        raise ValueError("training labels must be binary")

    mean = np.mean(x_train, axis=0, keepdims=True)
    scale = np.std(x_train, axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    x_train = (x_train - mean) / scale
    x_target = (x_target - mean) / scale

    model = EvidentialViewMLP(x_train.shape[1], hidden_dimension, dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    train_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    label_tensor = torch.as_tensor(y_train, dtype=torch.long, device=device)
    last_total = last_data = last_kl = math.nan
    model.train()
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        evidence = model(train_tensor)
        annealing = min(1.0, epoch / max(1.0, float(annealing_epochs)))
        total, data, kl = evidential_loss_terms(
            evidence, label_tensor, annealing, kl_weight
        )
        if not torch.isfinite(total):
            raise RuntimeError("non-finite evidential training loss")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        last_total = float(total.detach().cpu())
        last_data = float(data.detach().cpu())
        last_kl = float(kl.detach().cpu())

    model.eval()
    with torch.no_grad():
        train_evidence_tensor = model(train_tensor)
        target_tensor = torch.as_tensor(x_target, dtype=torch.float32, device=device)
        target_evidence_tensor = model(target_tensor)
    train_evidence = train_evidence_tensor.detach().cpu().numpy().astype(np.float64)
    target_evidence = target_evidence_tensor.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(train_evidence).all() or not np.isfinite(target_evidence).all():
        raise RuntimeError("non-finite learned evidence")
    train_alpha = train_evidence + 1.0
    train_probability = train_alpha[:, 1] / np.sum(train_alpha, axis=1)
    diagnostic = {
        "input_dimension": int(x_train.shape[1]),
        "train_n": int(len(x_train)),
        "epochs": int(epochs),
        "final_loss": last_total,
        "final_data_loss": last_data,
        "final_kl": last_kl,
        "train_accuracy": float(np.mean((train_probability >= 0.5) == y_train)),
        "train_mean_total_evidence": float(np.mean(np.sum(train_evidence, axis=1))),
        "target_mean_total_evidence": float(np.mean(np.sum(target_evidence, axis=1))),
        "device": str(device),
    }
    return EvidentialFit(train_evidence, target_evidence, diagnostic)


def fit_four_evidential_views(
    train_x: np.ndarray,
    train_y: np.ndarray,
    target_x: np.ndarray,
    groups: dict[str, np.ndarray],
    *,
    seed: int,
    device: torch.device,
    hidden_dimension: int = 32,
    dropout: float = 0.10,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    kl_weight: float = 0.01,
    annealing_epochs: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fit the four disjoint semantic evidence heads with identical settings."""
    train_evidence = np.zeros(
        (len(train_x), len(AUX_VIEW_NAMES), 2), dtype=np.float64
    )
    target_evidence = np.zeros(
        (len(target_x), len(AUX_VIEW_NAMES), 2), dtype=np.float64
    )
    audit: list[dict] = []
    for offset, view in enumerate(AUX_VIEW_NAMES):
        if view not in groups:
            raise RuntimeError("missing semantic feature group: " + view)
        index = np.asarray(groups[view], dtype=np.int64)
        result = fit_evidential_view(
            np.asarray(train_x)[:, index],
            train_y,
            np.asarray(target_x)[:, index],
            seed=int(seed + 1000 * offset),
            device=device,
            hidden_dimension=hidden_dimension,
            dropout=dropout,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            kl_weight=kl_weight,
            annealing_epochs=annealing_epochs,
        )
        train_evidence[:, offset, :] = result.train_evidence
        target_evidence[:, offset, :] = result.target_evidence
        audit.append({"view": view, **result.diagnostic})
    return train_evidence, target_evidence, audit


def opinion_from_evidence(evidence: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(evidence, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] != 2 or not np.isfinite(value).all():
        raise ValueError("evidence must be finite with two classes on the last axis")
    value = np.maximum(value, 0.0)
    alpha = value + 1.0
    strength = np.sum(alpha, axis=-1)
    return {
        "evidence": value,
        "alpha": alpha,
        "probability": alpha / strength[..., None],
        "belief": value / strength[..., None],
        "uncertainty": 2.0 / strength,
    }


def conflict_aware_fusion(
    evidence: np.ndarray,
    *,
    conflict_aggregator: str = "mean",
    reliability_gamma: float = 1.0,
) -> dict[str, np.ndarray]:
    """Fuse [sample, view, class] evidence and contract it under conflict.

    Per-view ignorance u controls a sample-specific reliability discount
    exp(-gamma*u).  Pairwise belief conflict K is then computed after discount.
    The pooled evidence concentration is contracted so that final uncertainty is
    U = 1 - (1-U_ignorance)*(1-K).  This is an evidence-space construction: the
    contracted class evidence is reconstructed before the final Dirichlet
    opinion is formed.
    """
    value = np.asarray(evidence, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (len(AUX_VIEW_NAMES), 2):
        raise ValueError("evidence must have shape [samples, four views, two classes]")
    view = opinion_from_evidence(value)
    reliability = np.exp(-float(reliability_gamma) * view["uncertainty"])
    discounted = reliability[..., None] * value
    discounted_view = opinion_from_evidence(discounted)

    pair_conflicts = []
    for left in range(len(AUX_VIEW_NAMES)):
        for right in range(left + 1, len(AUX_VIEW_NAMES)):
            belief_left = discounted_view["belief"][:, left, :]
            belief_right = discounted_view["belief"][:, right, :]
            pair_conflicts.append(
                belief_left[:, 0] * belief_right[:, 1]
                + belief_left[:, 1] * belief_right[:, 0]
            )
    pair_conflict = np.column_stack(pair_conflicts)
    if conflict_aggregator == "mean":
        conflict = np.mean(pair_conflict, axis=1)
    elif conflict_aggregator == "max":
        conflict = np.max(pair_conflict, axis=1)
    else:
        raise ValueError("unknown conflict aggregator: " + conflict_aggregator)
    conflict = np.clip(conflict, 0.0, 1.0)

    pooled = np.sum(discounted, axis=1)
    pooled_total = np.sum(pooled, axis=1)
    ignorance = 2.0 / (pooled_total + 2.0)
    uncertainty = 1.0 - (1.0 - ignorance) * (1.0 - conflict)
    uncertainty = np.clip(uncertainty, 1e-8, 1.0)

    effective_total = 2.0 * (1.0 - uncertainty) / uncertainty
    direction = np.divide(
        pooled,
        pooled_total[:, None],
        out=np.full_like(pooled, 0.5),
        where=pooled_total[:, None] > 1e-12,
    )
    fused_evidence = direction * effective_total[:, None]
    fused = opinion_from_evidence(fused_evidence)
    if not np.allclose(fused["uncertainty"], uncertainty, atol=1e-8):
        raise RuntimeError("evidence contraction failed to preserve uncertainty")
    return {
        "uncertainty": uncertainty,
        "ignorance": ignorance,
        "conflict": conflict,
        "pair_conflict": pair_conflict,
        "view_uncertainty": view["uncertainty"],
        "view_probability": view["probability"][..., 1],
        "view_reliability": reliability,
        "fused_evidence": fused_evidence,
        "fused_probability": fused["probability"][:, 1],
    }


def exact_top_fraction(score: np.ndarray, budget: float) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("score must be a finite one-dimensional array")
    if not 0.0 < float(budget) < 1.0:
        raise ValueError("budget must be in (0,1)")
    count = max(1, min(len(value), int(round(float(budget) * len(value)))))
    order = np.argsort(value, kind="mergesort")
    selected = np.zeros(len(value), dtype=bool)
    selected[order[-count:]] = True
    return selected


def groupwise_top_fraction(
    score: np.ndarray, group: np.ndarray, budget: float
) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64)
    groups = np.asarray(group)
    if value.shape != groups.shape:
        raise ValueError("score and group must have the same shape")
    selected = np.zeros(len(value), dtype=bool)
    for item in np.unique(groups):
        index = np.flatnonzero(groups == item)
        selected[index] = exact_top_fraction(value[index], budget)
    return selected


def source_quantile_threshold(score: np.ndarray, budget: float) -> float:
    value = np.asarray(score, dtype=np.float64)
    selected = exact_top_fraction(value, budget)
    return float(np.min(value[selected]))


def routing_diagnostics(
    method: str,
    labels: np.ndarray,
    base_probability: np.ndarray,
    route: np.ndarray,
    score: np.ndarray,
    high_confidence_threshold: float = 0.8,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    base = np.asarray(base_probability, dtype=np.float64)
    selected = np.asarray(route, dtype=bool)
    risk = np.asarray(score, dtype=np.float64)
    if not (y.shape == base.shape == selected.shape == risk.shape):
        raise ValueError("routing diagnostics inputs are not aligned")
    prediction = base >= 0.5
    error = prediction != y
    confidence = np.maximum(base, 1.0 - base)
    high_error = error & (confidence >= float(high_confidence_threshold))
    false_positive = prediction & (y == 0)
    false_negative = ~prediction & (y == 1)
    accepted = ~selected
    captured = int(np.sum(selected & error))
    high_captured = int(np.sum(selected & high_error))
    return {
        "method": method,
        "n": int(len(y)),
        "routed_n": int(np.sum(selected)),
        "routed_ratio": float(np.mean(selected)),
        "base_error_n": int(np.sum(error)),
        "routed_base_error_n": captured,
        "error_recall": float(captured / np.sum(error)) if np.any(error) else 0.0,
        "route_error_precision": float(captured / np.sum(selected)) if np.any(selected) else 0.0,
        "accepted_n": int(np.sum(accepted)),
        "accepted_error_n": int(np.sum(accepted & error)),
        "accepted_risk": float(np.mean(error[accepted])) if np.any(accepted) else 0.0,
        "high_confidence_threshold": float(high_confidence_threshold),
        "high_confidence_error_n": int(np.sum(high_error)),
        "routed_high_confidence_error_n": high_captured,
        "high_confidence_error_recall": (
            float(high_captured / np.sum(high_error)) if np.any(high_error) else 0.0
        ),
        "routed_FP_n": int(np.sum(selected & false_positive)),
        "routed_FN_n": int(np.sum(selected & false_negative)),
        "score_mean": float(np.mean(risk)),
        "score_min": float(np.min(risk)),
        "score_max": float(np.max(risk)),
    }
