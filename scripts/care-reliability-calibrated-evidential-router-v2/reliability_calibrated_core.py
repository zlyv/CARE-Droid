#!/usr/bin/env python3
"""Continuous-risk calibration and ranking for evidential uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evidential_conflict_core import (
    AUX_VIEW_NAMES,
    EvidentialViewMLP,
    evidential_loss_terms,
    seed_everything,
)


class JointEvidentialRouter(nn.Module):
    def __init__(
        self,
        dimensions: dict[str, int],
        hidden_dimension: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleDict({
            view: EvidentialViewMLP(dimensions[view], hidden_dimension, dropout)
            for view in AUX_VIEW_NAMES
        })

    def forward(self, view_features: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [self.heads[view](view_features[view]) for view in AUX_VIEW_NAMES],
            dim=1,
        )


def standardized_views(
    train_x: np.ndarray,
    target_x: np.ndarray,
    groups: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    train: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    dimensions: dict[str, int] = {}
    for view in AUX_VIEW_NAMES:
        index = np.asarray(groups[view], dtype=np.int64)
        train_value = np.asarray(train_x[:, index], dtype=np.float32)
        target_value = np.asarray(target_x[:, index], dtype=np.float32)
        mean = np.mean(train_value, axis=0, keepdims=True)
        scale = np.std(train_value, axis=0, keepdims=True)
        scale[scale < 1e-8] = 1.0
        train[view] = (train_value - mean) / scale
        target[view] = (target_value - mean) / scale
        dimensions[view] = int(len(index))
    return train, target, dimensions


def torch_conflict_fusion(
    evidence: torch.Tensor,
    reliability_gamma: float,
) -> dict[str, torch.Tensor]:
    if evidence.ndim != 3 or evidence.shape[1:] != (len(AUX_VIEW_NAMES), 2):
        raise ValueError("evidence must be [samples, four views, two classes]")
    alpha = evidence + 1.0
    view_uncertainty = 2.0 / torch.sum(alpha, dim=2)
    reliability = torch.exp(-float(reliability_gamma) * view_uncertainty)
    discounted = reliability.unsqueeze(2) * evidence
    discounted_strength = torch.sum(discounted, dim=2) + 2.0
    belief = discounted / discounted_strength.unsqueeze(2)
    conflicts = []
    for left in range(len(AUX_VIEW_NAMES)):
        for right in range(left + 1, len(AUX_VIEW_NAMES)):
            conflicts.append(
                belief[:, left, 0] * belief[:, right, 1]
                + belief[:, left, 1] * belief[:, right, 0]
            )
    conflict = torch.clamp(
        torch.mean(torch.stack(conflicts, dim=1), dim=1), 0.0, 1.0
    )
    ignorance = 2.0 / (torch.sum(discounted, dim=(1, 2)) + 2.0)
    uncertainty = 1.0 - (1.0 - ignorance) * (1.0 - conflict)
    return {
        "uncertainty": torch.clamp(uncertainty, 1e-6, 1.0 - 1e-6),
        "ignorance": ignorance,
        "conflict": conflict,
    }


def continuous_stage1_risk(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels and Stage-I probability are not aligned binary arrays")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Stage-I probability must be finite in [0,1]")
    return np.abs(y - p)


def make_ranking_pairs(
    risk: np.ndarray,
    *,
    seed: int,
    max_pairs: int = 8192,
    minimum_gap: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic sampled pairs where left risk exceeds right risk."""
    value = np.asarray(risk, dtype=np.float64)
    left, right = np.where(value[:, None] > value[None, :] + float(minimum_gap))
    if len(left) == 0:
        raise RuntimeError("no valid continuous-risk ranking pairs")
    if len(left) > int(max_pairs):
        rng = np.random.default_rng(int(seed))
        selected = rng.choice(len(left), size=int(max_pairs), replace=False)
        left, right = left[selected], right[selected]
    gap = value[left] - value[right]
    weight = gap / max(float(np.mean(gap)), 1e-8)
    return left.astype(np.int64), right.astype(np.int64), weight.astype(np.float32)


def weighted_pairwise_rank_loss(
    uncertainty: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    margin = uncertainty[left] - uncertainty[right]
    return torch.sum(weight * F.softplus(-margin)) / torch.sum(weight)


@dataclass
class ReliabilityFitResult:
    train_evidence: np.ndarray
    target_evidence: np.ndarray
    diagnostic: dict


def fit_reliability_calibrated_router(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_risk: np.ndarray,
    target_x: np.ndarray,
    groups: dict[str, np.ndarray],
    *,
    seed: int,
    device: torch.device,
    calibration_weight: float,
    ranking_weight: float,
    reliability_gamma: float = 1.0,
    huber_beta: float = 0.10,
    ranking_pairs: int = 8192,
    ranking_minimum_gap: float = 0.05,
    hidden_dimension: int = 32,
    dropout: float = 0.10,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    kl_weight: float = 0.01,
    annealing_epochs: int = 100,
) -> ReliabilityFitResult:
    seed_everything(seed)
    y = np.asarray(train_y, dtype=np.int64)
    risk = np.asarray(train_risk, dtype=np.float32)
    if y.shape != (len(train_x),) or risk.shape != y.shape:
        raise ValueError("training arrays are not aligned")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(risk).all():
        raise ValueError("invalid labels or risk target")
    if np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("continuous risk must lie in [0,1]")
    train_view, target_view, dimensions = standardized_views(
        np.asarray(train_x), np.asarray(target_x), groups
    )
    model = JointEvidentialRouter(dimensions, hidden_dimension, dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    train_tensor = {
        view: torch.as_tensor(value, dtype=torch.float32, device=device)
        for view, value in train_view.items()
    }
    target_tensor = {
        view: torch.as_tensor(value, dtype=torch.float32, device=device)
        for view, value in target_view.items()
    }
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
    risk_tensor = torch.as_tensor(risk, dtype=torch.float32, device=device)
    left_np, right_np, pair_weight_np = make_ranking_pairs(
        risk,
        seed=seed + 991,
        max_pairs=ranking_pairs,
        minimum_gap=ranking_minimum_gap,
    )
    left = torch.as_tensor(left_np, dtype=torch.long, device=device)
    right = torch.as_tensor(right_np, dtype=torch.long, device=device)
    pair_weight = torch.as_tensor(pair_weight_np, dtype=torch.float32, device=device)

    last: dict = {}
    model.train()
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        evidence = model(train_tensor)
        annealing = min(1.0, epoch / max(1.0, float(annealing_epochs)))
        view_losses = []
        data_losses = []
        kl_losses = []
        for column in range(len(AUX_VIEW_NAMES)):
            total, data, kl = evidential_loss_terms(
                evidence[:, column, :], y_tensor, annealing, kl_weight
            )
            view_losses.append(total)
            data_losses.append(data)
            kl_losses.append(kl)
        evidential_loss = torch.mean(torch.stack(view_losses))
        data_loss = torch.mean(torch.stack(data_losses))
        kl_loss = torch.mean(torch.stack(kl_losses))
        uncertainty = torch_conflict_fusion(
            evidence, reliability_gamma
        )["uncertainty"]
        calibration_loss = F.smooth_l1_loss(
            uncertainty, risk_tensor, beta=float(huber_beta)
        )
        rank_loss = weighted_pairwise_rank_loss(
            uncertainty, left, right, pair_weight
        )
        loss = (
            evidential_loss
            + float(calibration_weight) * calibration_loss
            + float(ranking_weight) * rank_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite reliability-calibrated loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        last = {
            "loss": float(loss.detach().cpu()),
            "evidential_loss": float(evidential_loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "calibration_loss": float(calibration_loss.detach().cpu()),
            "ranking_loss": float(rank_loss.detach().cpu()),
        }

    model.eval()
    with torch.no_grad():
        train_evidence_t = model(train_tensor)
        target_evidence_t = model(target_tensor)
        train_u = torch_conflict_fusion(
            train_evidence_t, reliability_gamma
        )["uncertainty"].detach().cpu().numpy()
    train_evidence = train_evidence_t.detach().cpu().numpy().astype(np.float64)
    target_evidence = target_evidence_t.detach().cpu().numpy().astype(np.float64)
    correlation = float(np.corrcoef(train_u, risk)[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    diagnostic = {
        **last,
        "train_n": int(len(y)),
        "target_n": int(len(target_x)),
        "risk_mean": float(np.mean(risk)),
        "risk_max": float(np.max(risk)),
        "high_risk_n_ge_0_8": int(np.sum(risk >= 0.8)),
        "calibration_weight": float(calibration_weight),
        "ranking_weight": float(ranking_weight),
        "huber_beta": float(huber_beta),
        "ranking_pair_n": int(len(left_np)),
        "ranking_minimum_gap": float(ranking_minimum_gap),
        "train_uncertainty_risk_pearson": correlation,
        "train_uncertainty_risk_mae": float(np.mean(np.abs(train_u - risk))),
        "mean_total_evidence": float(np.mean(np.sum(train_evidence, axis=(1, 2)))),
        "device": str(device),
        "dimensions": ";".join(
            f"{view}:{dimensions[view]}" for view in AUX_VIEW_NAMES
        ),
    }
    return ReliabilityFitResult(train_evidence, target_evidence, diagnostic)
