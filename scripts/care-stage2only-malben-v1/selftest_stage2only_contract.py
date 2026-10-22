#!/usr/bin/env python3
"""Executable no-Global24/no-p1 contract for StageIIOnlyModel."""

import inspect
from types import SimpleNamespace
import torch
from run_stage2only_malben_v1 import StageIIOnlyModel, malben


def main():
    args = SimpleNamespace(
        graph_dim=32, embedding_dim=24, hidden_dim=48, dropout=0.0,
        graph_heads=4, graph_layers=2,
    )
    model = StageIIOnlyModel(args).eval()
    signature = tuple(inspect.signature(model.forward).parameters)
    assert signature == ("graph", "regions"), signature
    names = [name.lower() for name, _ in model.named_parameters()]
    assert not any("global24" in name or "stage1" in name or "base_logit" in name for name in names)
    graph = {
        "node": torch.zeros(31, 3), "edge": torch.zeros(31, 31, 3),
        "global": torch.zeros(27),
    }
    regions = torch.zeros(2, malben.FULL_REGION_DIM)
    regions[:, malben.OLD_AUGMENTED_DIM + 0] = 0.4
    regions[:, malben.OLD_AUGMENTED_DIM + 1] = 0.6
    regions[:, malben.OLD_AUGMENTED_DIM + 2] = 0.2
    with torch.no_grad():
        first = model(graph, regions)
        feature = model.last_feature.clone()
        second = model(graph, regions)
    assert torch.allclose(first, second)
    assert feature.numel() == 81
    print("independent Stage-II-only no-Global24/no-p1 contract passed")


if __name__ == "__main__":
    main()
