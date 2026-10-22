#!/usr/bin/env python3
"""Synthetic source-only protocol tests; no CARE-Droid artifact is read."""
from __future__ import annotations

import numpy as np

from fold_protocol import inner_oof_stage1, outer_fold_indices
from probe_protocol import ARMS, need_correct, stage1_logit, validate_noego_88
from source_contract import (
    EXPECTED_FOLD_SIZES, EXPECTED_ROUTE_BY_FOLD, HOST_KEYS, blocked,
    validate_router,
)


def synthetic_router():
    rows = []
    counter = 0
    for fold, size in EXPECTED_FOLD_SIZES.items():
        for local in range(size):
            rows.append({
                "sha256": f"{counter:064x}", "year": "2017",
                "label": str((counter + fold) % 2), "fold": str(fold),
                "p_stage1": "0.8" if counter % 3 else "0.2",
                "route_DI": "1" if local < EXPECTED_ROUTE_BY_FOLD[fold] else "0",
            })
            counter += 1
    return rows


def expect_block(function, *args):
    try:
        function(*args)
    except RuntimeError as error:
        assert str(error).startswith("BLOCKED_BY_CONTRACT")
        return
    raise AssertionError("expected BLOCKED_BY_CONTRACT")


def main() -> None:
    rows = synthetic_router()
    audit = validate_router(rows)
    assert audit["rows"] == 799 and audit["route_DI"] == 200 and audit["year"] == [2017]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    p_stage1 = np.asarray([float(row["p_stage1"]) for row in rows])
    route = np.asarray([int(row["route_DI"]) == 1 for row in rows])
    need = need_correct(labels, p_stage1)
    assert len(need[route]) == 200 and set(np.unique(need).tolist()).issubset({0, 1})
    assert np.all(np.diff(stage1_logit([0.1, 0.5, 0.9])) > 0)
    assert {name: value["dimension"] for name, value in ARMS.items()} == {"S0": 1, "S1": 25, "S2": 88, "S3": 89}
    validate_noego_88(np.zeros((3, 88), dtype=np.float32), 3)
    expect_block(validate_noego_88, np.zeros((3, 89), dtype=np.float32), 3)

    train, validation = outer_fold_indices(folds, 1)
    assert len(validation) == 160 and not (set(train.tolist()) & set(validation.tolist()))
    frozen_outer_values = p_stage1.copy()
    seen = []

    def fake_fit_predict(fit_global, validation_global, seed):
        assert not (set(fit_global.tolist()) & set(validation_global.tolist()))
        seen.extend(validation_global.tolist())
        # Deliberately differs from frozen outer OOF values.
        return np.full(len(validation_global), 0.25 + 0.01 * (seed % 2))

    class SyntheticFourFold:
        def __init__(self, n_splits, seed):
            assert n_splits == 4 and seed == 20260901

        def split(self, x, y):
            index = np.arange(len(y))
            for fold in range(4):
                validation_local = index[index % 4 == fold]
                fit_local = index[index % 4 != fold]
                yield fit_local, validation_local

    inner = inner_oof_stage1(
        train, labels, fake_fit_predict,
        splitter_factory=lambda n_splits, seed: SyntheticFourFold(n_splits, seed),
    )
    assert len(inner) == len(train) and len(seen) == len(train)
    assert not np.array_equal(inner, frozen_outer_values[train])

    bad = synthetic_router(); bad[0]["year"] = "2018"
    expect_block(validate_router, bad)
    bad = synthetic_router(); bad[0]["route_DI"] = "0"
    expect_block(validate_router, bad)
    assert HOST_KEYS == ("sha256", "host_values", "host_offsets", "real_host_count")

    print("SOURCE_NEEDGATE_SELFTEST_PASS")
    print("source_rows = 799")
    print("year = 2017")
    print("route_DI = 200")
    print("future_rows_loaded = 0")
    print("ego_consumed = false")


if __name__ == "__main__":
    main()
