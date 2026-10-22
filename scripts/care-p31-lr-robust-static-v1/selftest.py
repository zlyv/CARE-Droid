#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import ARMS, comparison_metrics, grouped_top25, mean_opposition
from transforms import feature_names, transform_p31


def main():
    x = np.zeros((5, 31), float)
    x[0, :2] = [.75, .25]; x[1, 1:3] = [.2, .8]
    x[2, 2:5] = [1, 2, 1]; x[3, 0] = 1
    expected_width = {"L0_RAW": 31, "L1_HELLINGER": 31, "L2_LOG": 31,
        "L3_PRESENCE": 31, "L4_RAW_PRESENCE": 62, "L5_CLR_PRESENCE": 62,
        "L6_HELLINGER_PRESENCE": 62}
    for arm in ARMS:
        transformed = transform_p31(x, arm)
        assert transformed.shape == (5, expected_width[arm])
        assert len(feature_names(arm)) == expected_width[arm]
        assert np.isfinite(transformed).all()
    assert np.allclose(transform_p31(x, "L0_RAW").sum(1), [1, 1, 1, 1, 0])
    assert np.allclose(transform_p31(x, "L5_CLR_PRESENCE")[-1], 0)
    member = np.asarray([[.1, .2, .3, .4], [.8, .7, .6, .9], [.4, .51, .45, .49], [.6, .4, .7, .3]])
    assert mean_opposition(member, member[:, 0]).shape == (4,)
    score = np.arange(20, dtype=float); sha = np.asarray([f"{index:02d}" for index in range(20)])
    route = grouped_top25(score, sha, np.repeat([1, 2], 10))
    assert int(route.sum()) == 4 and int(np.sum(route[:10])) == 2 and int(np.sum(route[10:])) == 2
    y = np.asarray([0, 0, 1, 1]); base = np.asarray([.1, .9, .1, .9]); final = np.asarray([.1, .1, .9, .1])
    value = comparison_metrics(y, base, final, np.ones(4, bool))
    assert (value["fixed"], value["broken"], value["net"]) == (2, 1, 1)
    print("P31_LR_ROBUST_STATIC_SELFTEST_PASS arms=" + ",".join(ARMS))


if __name__ == "__main__":
    main()
