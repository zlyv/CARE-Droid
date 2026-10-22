#!/usr/bin/env python3
"""Assert the predeclared selective MalBen capacity (33,388 parameters)."""

from types import SimpleNamespace
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT, ROOT / "care-malben-contrast-v2",
             ROOT / "care-rolegraph-conditioned-v4-script"):
    sys.path.insert(0, str(path))

import run_care_malben_role_source_v2 as old
from shared_role_prototypes import evidence_columns, fit_role_prototypes


def main():
    old.fit_role_prototypes = fit_role_prototypes
    old.evidence_columns = evidence_columns
    args = SimpleNamespace(max_delta=2.25, graph_dim=32, embedding_dim=24,
                           hidden_dim=48, dropout=0.0, graph_heads=4, graph_layers=2)
    model = old.RoleConditionedEvidenceModel(24, "MalBenContrast", args).eval()
    count = sum(value.numel() for value in model.parameters())
    assert count == 33388, count
    graph = {"node": torch.zeros(31, 3), "edge": torch.zeros(31, 31, 3),
             "global": torch.zeros(27)}
    regions = torch.zeros(2, old.FULL_REGION_DIM)
    regions[:, old.OLD_AUGMENTED_DIM + 0] = 0.4
    regions[:, old.OLD_AUGMENTED_DIM + 1] = 0.6
    regions[:, old.OLD_AUGMENTED_DIM + 2] = 0.2
    with torch.no_grad():
        output, diagnostic = model(torch.zeros(24), torch.tensor(0.0), graph, regions)
    assert torch.isfinite(output) and torch.isfinite(diagnostic).all()
    print("SELECTIVE_MALBEN_PARAMETER_CONTRACT_PASS parameters=33388")


if __name__ == "__main__":
    main()
