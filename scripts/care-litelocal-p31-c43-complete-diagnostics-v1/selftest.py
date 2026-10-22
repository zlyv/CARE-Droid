#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from common import align_fold_global24, metric_counts


def main():
    label = np.asarray([0, 1, 1, 0])
    base = np.asarray([.1, .2, .8, .9])
    final = np.asarray([.1, .8, .8, .2])
    result = metric_counts(label, base, final, np.asarray([0, 1, 0, 1]))
    if result != {"routed": 2, "fixed": 2, "broken": 0, "net": 2}:
        raise RuntimeError("metric-count selftest failed: " + repr(result))
    canonical = [f"global_{index}" for index in range(24)] + ["role_node__a", "role_edge__a"]
    fold_names = [f"global_{index}" for index in range(23)] + [
        "role_node__a", "role_edge__a", "role_node__package_installation"]
    raw = np.arange(2 * len(fold_names), dtype=float).reshape(2, -1)
    train, validation, audit = align_fold_global24(raw, raw[:1], fold_names, canonical)
    if train.shape != (2, 24) or validation.shape != (1, 24):
        raise RuntimeError("Global24 alignment shape selftest failed")
    if train[:, 23].any() or audit["missing_canonical_global"] != ["global_23"]:
        raise RuntimeError("Global24 missing-column zero-fill selftest failed")
    if audit["ignored_fold_only_role"] != ["role_node__package_installation"]:
        raise RuntimeError("Global24 extra-role isolation selftest failed")
    tree = ast.parse((Path(__file__).with_name("run_within_year.py")).read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "original_compact_matrix_named"]
    if len(calls) != 1 or len(calls[0].args) != 4:
        raise RuntimeError("original_compact_matrix_named must be called with its four-argument API")
    print("P31_C43_COMPLETE_DIAGNOSTICS_SELFTEST_PASS")
    print("protocols=frozen_2017_to_future,independent_same_year_5fold")
    print("stage1=P31_MEAN_COMMITTEE router=P31_DI25 stage2=C43_RELATION_QUERY")
    print("fold_only_role_columns=ignored_with_explicit_audit global24_schema=frozen_by_name")
    print("ego_consumed=false")


if __name__ == "__main__":
    main()
