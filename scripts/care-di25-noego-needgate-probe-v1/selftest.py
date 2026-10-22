#!/usr/bin/env python3
"""Synthetic contract tests. This file does not read CARE-Droid artifacts."""
from __future__ import annotations

import inspect
import numpy as np
from sklearn.preprocessing import StandardScaler

from probe_common import (
    ARMS, REPRESENTATION_WIDTH, classification_metrics, fit_probe, need_correct,
    noego_contract, paired_bootstrap, stage1_logit, validate_partition,
    validate_representation,
)


class RecordingScaler(StandardScaler):
    fitted_rows = None

    def fit(self, x, y=None, **kwargs):
        type(self).fitted_rows = np.asarray(x).copy()
        return super().fit(x, y, **kwargs)


def expect_block(function, *args):
    try:
        function(*args)
    except RuntimeError as error:
        assert str(error).startswith("BLOCKED_BY_CONTRACT")
        return
    raise AssertionError("expected BLOCKED_BY_CONTRACT")


def main() -> None:
    # 1/2: target and fixed Stage-I logit.
    y = np.asarray([0, 0, 1, 1])
    p = np.asarray([0.1, 0.9, 0.8, 0.2])
    assert need_correct(y, p).tolist() == [0, 1, 0, 1]
    assert np.all(np.diff(stage1_logit([0.1, 0.5, 0.9])) > 0)
    route = np.asarray([0, 1, 1, 0], dtype=bool)
    assert need_correct(y, p)[route].tolist() == [1, 0]

    # 3: the no-Ego representation is exact and cannot silently change width.
    validate_representation(np.zeros((4, REPRESENTATION_WIDTH), dtype=np.float32), 4)
    expect_block(validate_representation, np.zeros((4, 89), dtype=np.float32), 4)
    contract = noego_contract()
    assert contract["ego_consumed"] is False and contract["ego_zero_filled"] is False
    assert sum(item["dimension"] for item in contract["representation"]["components"]) == 88

    # 4/5: split disjointness and scaler fit scope.
    validate_partition(["a", "b"], ["c", "d"])
    expect_block(validate_partition, ["a", "b"], ["b", "c"])
    x_train = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    x_validation = np.asarray([[100.0], [101.0]])
    fit_probe(x_train, np.asarray([0, 0, 1, 1]), x_validation, scaler_factory=RecordingScaler)
    assert np.array_equal(RecordingScaler.fitted_rows, x_train)
    assert "y_validation" not in inspect.signature(fit_probe).parameters

    # 6/7: all four arms are distinct contracts and no validation row can be omitted.
    assert len(ARMS) == len(set(ARMS)) == 4
    expected = {("S0", value) for value in ("x", "y")}
    assigned = {("S0", value) for value in ("x", "y")}
    assert assigned == expected

    # 8: single-class training is blocked rather than silently fitted.
    expect_block(fit_probe, x_train, np.zeros(4, dtype=int), x_validation)

    # 7: metric positive class is explicitly need_correct=1.
    metric = classification_metrics(np.asarray([0, 1, 1, 0]), np.asarray([0.1, 0.9, 0.8, 0.2]))
    assert metric["TP"] == 2 and metric["TN"] == 2 and metric["Recall"] == 1.0

    # 9: paired bootstrap preserves pairing and reports requested seed/count.
    outcome = paired_bootstrap(
        np.asarray([0, 0, 1, 1, 0, 1]),
        np.asarray([0.1, 0.4, 0.6, 0.9, 0.3, 0.7]),
        np.asarray([0.1, 0.2, 0.8, 0.9, 0.2, 0.8]),
        "ROC_AUC", iterations=100, seed=7,
    )
    assert outcome["iterations_requested"] == 100 and outcome["bootstrap_seed"] == 7

    # 10: missing/invalid data is never repaired with zero fill.
    expect_block(validate_representation, np.asarray([[np.nan] * 88]), 1)
    print("NOEGO_NEEDGATE_SELFTEST_PASS")


if __name__ == "__main__":
    main()
