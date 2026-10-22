#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace

import torch

import run_care_malben_role_source_v2 as experiment


def main() -> None:
    args = SimpleNamespace(
        max_delta=2.25,
        graph_dim=32,
        embedding_dim=24,
        hidden_dim=48,
        dropout=0.20,
        graph_heads=4,
        graph_layers=2,
    )
    graph = {
        "node": torch.randn(31, 3),
        "edge": torch.randn(31, 31, 3),
        "global": torch.randn(27),
    }
    regions = torch.randn(7, experiment.FULL_REGION_DIM)
    regions[:, -9] = torch.sigmoid(regions[:, -9])
    regions[:, -8] = torch.sigmoid(regions[:, -8])
    regions[:, -7] = regions[:, -8] - regions[:, -9]
    regions[:, -6] = torch.maximum(regions[:, -9], regions[:, -8])
    regions[:, -5] = 3
    regions[:, -4] = 30
    regions[:, -3] = 31
    regions[:, -2] = 0
    regions[:, -1] = 0
    counts = []
    for variant in experiment.VARIANTS:
        model = experiment.RoleConditionedEvidenceModel(24, variant, args)
        output, diagnostic = model(
            torch.randn(24), torch.tensor(0.2), graph, regions
        )
        assert output.ndim == 0
        assert tuple(diagnostic.shape) == (len(experiment.DIAGNOSTIC_NAMES),)
        assert tuple(model.last_region_capture.shape) == (
            7, len(experiment.REGION_CAPTURE_NAMES)
        )
        assert torch.isfinite(output) and torch.isfinite(diagnostic).all()
        counts.append(sum(value.numel() for value in model.parameters()))
    assert len(set(counts)) == 1, counts
    print("model contract self-test passed; parameters:", counts[0])


if __name__ == "__main__":
    main()
