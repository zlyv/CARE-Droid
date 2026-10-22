#!/usr/bin/env python3
"""Joint evidential heads calibrated toward frozen Stage-I error risk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

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


def torch_conflict_fusion(
    evidence: torch.Tensor,
    reliability_gamma: float,
) -> dict[str, torch.Tensor]:
    """Differentiable counterpart of mean conflict-aware evidence fusion."""
    if evidence.ndim != 3 or evidence.shape[1:] != (len(AUX_VIEW_NAMES), 2):
        raise ValueError("evidence must be [samples, four views, two classes]")
    alpha = evidence + 1.0
    view_strength = torch.sum(alpha, dim=2)
    view_uncertainty = 2.0 / view_strength
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
    conflict = torch.mean(torch.stack(conflicts, dim=1), dim=1)
    conflict = torch.clamp(conflict, 0.0, 1.0)
    pooled_total = torch.sum(discounted, dim=(1, 2))
    ignorance = 2.0 / (pooled_total + 2.0)
    uncertainty = 1.0 - (1.0 - ignorance) * (1.0 - conflict)
    return {
        "uncertainty": torch.clamp(uncertainty, 1e-6, 1.0 - 1e-6),
        "ignorance": ignorance,
        "conflict": conflict,
        "view_uncertainty": view_uncertainty,
        "view_reliability": reliability,
    }


def balanced_route_bce(
    uncertainty: torch.Tensor,
    stage1_error: torch.Tensor,
) -> torch.Tensor:
    """Class-balanced BCE because Stage-I errors are a minority in the source."""
    target = stage1_error.to(dtype=uncertainty.dtype)
    positive = torch.sum(target)
    negative = torch.sum(1.0 - target)
    if positive.item() < 1 or negative.item() < 1:
        raise RuntimeError("route calibration requires both correct and error rows")
    count = float(len(target))
    positive_weight = count / (2.0 * positive)
    negative_weight = count / (2.0 * negative)
    loss = -(
        positive_weight * target * torch.log(uncertainty)
        + negative_weight * (1.0 - target) * torch.log(1.0 - uncertainty)
    )
    return torch.mean(loss)


@dataclass
class JointFitResult:
    train_evidence: np.ndarray
    target_evidence: np.ndarray
    diagnostic: dict


def _standardized_views(
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


def fit_joint_evidential_router(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_stage1_error: np.ndarray,
    target_x: np.ndarray,
    groups: dict[str, np.ndarray],
    *,
    seed: int,
    device: torch.device,
    route_loss_weight: float,
    reliability_gamma: float = 1.0,
    hidden_dimension: int = 32,
    dropout: float = 0.10,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    kl_weight: float = 0.01,
    annealing_epochs: int = 100,
) -> JointFitResult:
    """Train four evidence heads jointly with optional error-aware calibration."""
    seed_everything(seed)
    y = np.asarray(train_y, dtype=np.int64)
    error = np.asarray(train_stage1_error, dtype=np.int64)
    if y.shape != (len(train_x),) or error.shape != y.shape:
        raise ValueError("training labels/error target are not aligned")
    if not np.isin(y, (0, 1)).all() or not np.isin(error, (0, 1)).all():
        raise ValueError("training targets must be binary")
    train_view, target_view, dimensions = _standardized_views(
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
    error_tensor = torch.as_tensor(error, dtype=torch.float32, device=device)
    last = {}
    model.train()
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        evidence = model(train_tensor)
        annealing = min(1.0, epoch / max(1.0, float(annealing_epochs)))
        total_view = []
        data_view = []
        kl_view = []
        for column in range(len(AUX_VIEW_NAMES)):
            total, data, kl = evidential_loss_terms(
                evidence[:, column, :], y_tensor, annealing, kl_weight
            )
            total_view.append(total)
            data_view.append(data)
            kl_view.append(kl)
        evidential_loss = torch.mean(torch.stack(total_view))
        data_loss = torch.mean(torch.stack(data_view))
        kl_loss = torch.mean(torch.stack(kl_view))
        fusion = torch_conflict_fusion(evidence, reliability_gamma)
        route_loss = balanced_route_bce(fusion["uncertainty"], error_tensor)
        loss = evidential_loss + float(route_loss_weight) * route_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite joint evidential loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        last = {
            "loss": float(loss.detach().cpu()),
            "evidential_loss": float(evidential_loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "route_loss": float(route_loss.detach().cpu()),
        }

    model.eval()
    with torch.no_grad():
        train_evidence_tensor = model(train_tensor)
        target_evidence_tensor = model(target_tensor)
        train_fusion = torch_conflict_fusion(
            train_evidence_tensor, reliability_gamma
        )
    train_evidence = train_evidence_tensor.detach().cpu().numpy().astype(np.float64)
    target_evidence = target_evidence_tensor.detach().cpu().numpy().astype(np.float64)
    train_uncertainty = train_fusion["uncertainty"].detach().cpu().numpy()
    diagnostic = {
        **last,
        "train_n": int(len(y)),
        "target_n": int(len(target_x)),
        "stage1_error_n": int(np.sum(error)),
        "stage1_error_ratio": float(np.mean(error)),
        "route_loss_weight": float(route_loss_weight),
        "mean_uncertainty_stage1_correct": float(np.mean(train_uncertainty[error == 0])),
        "mean_uncertainty_stage1_error": float(np.mean(train_uncertainty[error == 1])),
        "mean_total_evidence": float(np.mean(np.sum(train_evidence, axis=(1, 2)))),
        "device": str(device),
        "dimensions": json_dimensions(dimensions),
    }
    return JointFitResult(train_evidence, target_evidence, diagnostic)


def json_dimensions(dimensions: dict[str, int]) -> str:
    return ";".join(f"{view}:{dimensions[view]}" for view in AUX_VIEW_NAMES)
