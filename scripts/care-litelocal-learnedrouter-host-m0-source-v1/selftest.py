#!/usr/bin/env python3
import numpy as np
from run_source_integration import (
    stage1_probability,
    mean_opposition,
    foldwise_top_budget,
    router_features,
    choose_threshold,
    fit_router_model,
    predict_router_risk,
)


def main():
    members = np.array([
        [.10,.12,.08,.11],
        [.90,.88,.92,.91],
        [.40,.60,.42,.58],
        [.48,.52,.47,.53],
        [.20,.25,.30,.22],
        [.70,.72,.68,.71],
        [.45,.46,.44,.43],
        [.55,.54,.56,.57],
    ])
    p = stage1_probability(members, "MEAN_COMMITTEE")
    X = router_features(members, p)
    assert X.shape == (8, 12)
    assert np.isfinite(X).all()
    assert np.allclose(mean_opposition(members, p)[:2], [p[0], 1-p[1]])

    folds = np.repeat([1,2],4)
    route = foldwise_top_budget(np.arange(8), folds, np.array(list("abcdefgh")), .25)
    assert route.sum() == 2

    # Synthetic learned-router target: ambiguous rows are errors.
    e = np.array([0,0,1,1,0,0,1,1])
    model = fit_router_model(X, e, seed=1)
    risk = predict_router_risk(model, X)
    tau, stats = choose_threshold(e, risk)
    learned = risk >= tau
    assert 0 <= stats["balanced_accuracy"] <= 1
    assert learned.dtype == bool
    assert np.isfinite(tau)
    print("LITELOCAL_LEARNED_ROUTER_HOST_M0_SELFTEST_PASS")


if __name__ == "__main__":
    main()
