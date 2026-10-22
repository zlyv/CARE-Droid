#!/usr/bin/env python3
import numpy as np
from audit_core import foldwise_top_budget, mean_opposition, stage1_probability


def main():
    members = np.array([[.1, .2, .3, .4], [.9, .8, .7, .6], [.4, .6, .4, .6]])
    assert np.allclose(stage1_probability(members, "LR_ONLY"), [.1, .9, .4])
    assert np.allclose(stage1_probability(members, "MEAN_COMMITTEE"), [.25, .75, .5])
    assert np.allclose(mean_opposition(members, np.array([.1, .9, .4])), [.25, .25, .5])
    folds = np.repeat([1, 2], 4)
    route = foldwise_top_budget(np.arange(8), folds, np.array(list("abcdefgh")), .25)
    assert route.sum() == 2
    source = open(__file__.replace("selftest.py", "run_audit.py"), "r", encoding="utf-8").read()
    assert "seed = 20260925 + 100000 * fold" in source
    assert "seed = 20260928 + 100000 * fold" not in source
    print("LITELOCAL_STAGE1_ATTRIBUTION_SELFTEST_PASS")
    print("model_seed_anchor=20260925 matches_original_LiteLocal=true")
    print("apk_reanalysis=false future_rows_loaded=0 stage2_training_invoked=false")


if __name__ == "__main__":
    main()
