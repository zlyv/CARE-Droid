#!/usr/bin/env python3
"""Capacity-matched branch-isolation models for the frozen Q-Query method."""

from __future__ import annotations

from collections import OrderedDict

import torch

from graphglobal_path_models import make_graphglobal_path_class


BLOCKS = OrderedDict([
    ("global24", (0, 24)),
    ("stage1_logit", (24, 25)),
    ("role_token", (25, 57)),
    ("critical_host", (57, 81)),
    ("attended_host", (81, 105)),
    ("summary", (105, 113)),
])

# FULL_Q is a locked external reference and is not retrained by this package.
# Every trained arm still instantiates the complete 33,036-parameter network;
# only a fixed, parameter-free mask controls the correction head's inputs.
ARM_SPECS = OrderedDict([
    ("FULL_Q", ("Q_QUERY_ONLY", tuple(BLOCKS))),
    ("ROLE_ONLY", ("N0_NO_GRAPHGLOBAL", ("role_token",))),
    ("HOST_ONLY_Q", ("Q_QUERY_ONLY", ("critical_host", "attended_host", "summary"))),
    ("HOST_ONLY_N0", ("N0_NO_GRAPHGLOBAL", ("critical_host", "attended_host", "summary"))),
    ("STRUCTURE_ONLY_Q", ("Q_QUERY_ONLY", ("role_token", "critical_host", "attended_host", "summary"))),
    ("STAGE1_CONTEXT_ONLY", ("N0_NO_GRAPHGLOBAL", ("global24", "stage1_logit"))),
])
TRAIN_ARMS = tuple(a for a in ARM_SPECS if a != "FULL_Q")


def correction_mask(arm: str) -> torch.Tensor:
    if arm not in ARM_SPECS:
        raise KeyError(arm)
    keep = set(ARM_SPECS[arm][1])
    mask = torch.zeros(113, dtype=torch.float32)
    for name, (left, right) in BLOCKS.items():
        if name in keep:
            mask[left:right] = 1.0
    return mask


def make_qbranch_class(v4, arm: str):
    if arm not in TRAIN_ARMS:
        raise KeyError(f"{arm} is not a trainable branch arm")
    path_arm, _ = ARM_SPECS[arm]
    Base = make_graphglobal_path_class(v4, path_arm)
    fixed_mask = correction_mask(arm)

    class QBranchIsolation(Base):
        qbranch_arm = arm

        def __init__(self, global_dim, variant, args):
            super().__init__(global_dim, variant, args)
            self.register_buffer("correction_mask", fixed_mask.clone(), persistent=True)

        def delta_from_selected_evidence(self, global_features, base_logit, cache, graphglobal):
            # D is disabled in all branch arms. GraphGlobal can affect Host
            # selection only in the explicitly named *_Q arms.
            role_slot = cache.role_token
            fused = torch.cat([
                global_features.reshape(-1),
                base_logit.reshape(1),
                role_slot,
                cache.critical,
                cache.attended,
                cache.scalar_summary,
            ])
            if fused.numel() != 113:
                raise RuntimeError(f"Correction input width {fused.numel()} != 113")
            masked = fused * self.correction_mask.to(device=fused.device, dtype=fused.dtype)
            raw_delta = self.delta_head(masked).reshape(())
            bounded_delta = self.max_delta * torch.tanh(raw_delta)
            return raw_delta, bounded_delta, masked

    QBranchIsolation.__name__ = "QBranchIsolation_" + arm
    return QBranchIsolation
