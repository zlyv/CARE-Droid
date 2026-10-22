#!/usr/bin/env python3
import numpy as np
from run_source_integration import stage1_probability, mean_opposition, foldwise_top_budget


def main():
    members = np.array([[.1,.2,.3,.4],[.9,.8,.7,.6],[.4,.6,.4,.6],[.2,.3,.4,.5]])
    assert np.allclose(stage1_probability(members,"LR_ONLY"), [.1,.9,.4,.2])
    assert np.allclose(stage1_probability(members,"MEAN_COMMITTEE"), [.25,.75,.5,.35])
    assert np.allclose(mean_opposition(members, np.array([.1,.9,.4,.2])), [.25,.25,.5,.35])
    folds = np.repeat([1,2],4)
    route = foldwise_top_budget(np.arange(8), folds, np.array(list("abcdefgh")), .25)
    assert route.sum() == 2
    print("LITELOCAL_DI25_HOST_M0_SELFTEST_PASS")


if __name__ == "__main__":
    main()
