#!/usr/bin/env python3
"""Small deterministic self-test without APK artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import uncertainty_router_core as uut


def main() -> None:
    probability = np.asarray([
        [0.10, 0.10, 0.10, 0.10, 0.10],
        [0.10, 0.20, 0.50, 0.80, 0.90],
        [0.80, 0.75, 0.70, 0.65, 0.60],
        [0.45, 0.48, 0.50, 0.52, 0.55],
    ])
    paper, paper_names = uut.router_features(probability, "paper9")
    code, code_names = uut.router_features(probability, "official7")
    assert paper.shape == (4, 10)
    assert code.shape == (4, 8)
    assert paper_names[0] == code_names[0] == "base_probability"
    paper_index = {name: index for index, name in enumerate(paper_names)}
    assert abs(paper[0, paper_index["std"]]) < 1e-12
    assert abs(paper[0, paper_index["kld_to_mean"]]) < 1e-12
    assert paper[1, paper_index["std"]] > paper[3, paper_index["std"]]

    rng = np.random.default_rng(17)
    repeated = rng.uniform(0.02, 0.98, size=(160, 5))
    x, _ = uut.router_features(repeated, "paper9")
    error = ((np.max(repeated, axis=1) - np.min(repeated, axis=1)) > 0.67).astype(int)
    assert np.min(np.bincount(error, minlength=2)) >= 5
    model = uut.fit_error_router(x, error, 17)
    score = uut.error_risk_score(model, x)
    route = uut.top_fraction(score, 0.25)
    assert int(np.sum(route)) == 40
    assert float(np.mean(error[route])) > float(np.mean(error[~route]))
    print("self-test passed")


if __name__ == "__main__":
    main()
