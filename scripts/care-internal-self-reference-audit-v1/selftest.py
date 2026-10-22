#!/usr/bin/env python3
"""Pure numerical self-tests; reads no APK and writes nothing."""
from __future__ import annotations

import numpy as np

from run_audit import self_exclusion


def main():
    # One compact cluster: removing self must increase the distance for endpoints.
    x = np.asarray([[0., 0.], [2., 0.], [4., 0.]], np.float32)
    a = np.zeros(3, dtype=int)
    result = self_exclusion(x, a, coverage=1.0, repeats=5, random_seed=7)
    assert result['exact_replay_max_abs'] < 1e-7
    assert np.all(result['evaluable'])
    assert result['loo_internal'][0] > result['original_internal'][0]
    assert result['loo_internal'][2] > result['original_internal'][2]
    assert np.isfinite(result['random_internal']).all()

    # An unselected singleton must not change merely because it exists.
    x = np.asarray([[0.], [0.1], [10.]], np.float32)
    a = np.asarray([0, 0, 1])
    result = self_exclusion(x, a, coverage=.5, repeats=3, random_seed=3)
    assert result['self_in_reference'][2] == 0
    assert result['loo_internal'][2] == result['original_internal'][2]
    print('INTERNAL_SELF_REFERENCE_SELFTEST_PASS')


if __name__ == '__main__':
    main()

