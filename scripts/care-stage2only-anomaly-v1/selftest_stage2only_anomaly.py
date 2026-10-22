#!/usr/bin/env python3
"""Executable equal-width/no-Stage-I contract for the anomaly control."""

import inspect
from types import SimpleNamespace

import torch

from run_stage2only_anomaly_v1 import (
    BASE_REGION_DIM,
    OLD_AUGMENTED_DIM,
    StageIIOnlyAnomalyModel,
)


def main():
    args = SimpleNamespace(
        graph_dim=32, embedding_dim=24, hidden_dim=48, dropout=0.0,
        graph_heads=4, graph_layers=2,
    )
    model = StageIIOnlyAnomalyModel(args).eval()
    assert tuple(inspect.signature(model.forward).parameters) == ("graph", "regions")
    names = [name.lower() for name, _ in model.named_parameters()]
    assert not any(
        token in name for name in names
        for token in ("global24", "stage1", "base_logit", "malben")
    )
    graph = {
        "node": torch.zeros(31, 3),
        "edge": torch.zeros(31, 31, 3),
        "global": torch.zeros(27),
    }
    regions = torch.zeros(2, OLD_AUGMENTED_DIM)
    regions[:, BASE_REGION_DIM + 3] = torch.tensor([0.1, 0.8])
    with torch.no_grad():
        first = model(graph, regions)
        feature = model.last_feature.clone()
        second = model(graph, regions)
    assert torch.allclose(first, second)
    assert feature.numel() == 81
    print("STAGE2ONLY_ANOMALY_CONTRACT_PASS")


if __name__ == "__main__":
    main()
