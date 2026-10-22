#!/usr/bin/env python3
import numpy as np
from common import cap_route, threshold_higher, top_k


def main():
    score = np.asarray([.1, .4, .3, .2, .8, .5, .7, .6]); sha = np.asarray(list("abcdefgh")); group = np.asarray([1]*4 + [2]*4)
    route = top_k(score, sha, group, .25)
    assert route.tolist() == [False, True, False, False, True, False, False, False]
    candidate = score >= .25; capped = cap_route(candidate, score, sha, group, .25)
    assert np.array_equal(route, capped)
    assert threshold_higher(np.arange(8), .75) == 6.0
    print("ELASTIC_MEANOPPOSITION_SELFTEST_PASS")


if __name__ == "__main__": main()
