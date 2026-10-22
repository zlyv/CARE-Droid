#!/usr/bin/env python3
"""Fast contract checks for the within-year matrix package."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np


def test_compact_contract(here: Path) -> None:
    sys.path.insert(0, str(here))
    import source_frozen_compact_contract as contract

    canonical = [f"feature_{index}" for index in range(76)]
    fold_names = canonical + [
        "role_node__package_installation",
        "role_edge__package_installation",
    ]

    class Original:
        @staticmethod
        def original_compact_matrix_named(train, test, by_sha, schema):
            train_values = np.arange(len(train) * 78, dtype=float).reshape(len(train), 78)
            test_values = np.arange(len(test) * 78, dtype=float).reshape(len(test), 78)
            return train_values, test_values, list(fold_names)

    class Audit:
        allratio = Original()

    class V4:
        audit = Audit()

    contract.install_source_frozen_compact_contract(V4)
    train, test, names = V4.audit.restore_raw_compact76(
        [{"sha256": "a"}], [{"sha256": "b"}], {}, {}, canonical
    )
    assert train.shape == (1, 76) and test.shape == (1, 76)
    assert names == canonical
    audit = contract.compact_contract_audit()
    assert audit["later_year_extra_features_enter_model"] is False
    assert audit["later_year_extra_features_ignored"] == sorted(fold_names[-2:])


def main() -> None:
    here = Path(__file__).resolve().parent
    required = [
        "build_within_year_router_inputs.py", "run_within_year_arm.py",
        "summarize_within_year_matrix.py", "ensure_future_compact_features.py",
        "source_frozen_compact_contract.py",
        "configs/semantic_mapping_v1.yaml", "configs/sensitive_endpoint_priority_v1.yaml",
    ]
    for name in required:
        path = here / name
        assert path.is_file(), name
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    router_source = (here / "build_within_year_router_inputs.py").read_text(encoding="utf-8")
    arm_source = (here / "run_within_year_arm.py").read_text(encoding="utf-8")
    summary_source = (here / "summarize_within_year_matrix.py").read_text(encoding="utf-8")
    assert "StratifiedKFold" in router_source
    assert "n_splits=args.folds" in router_source
    assert "target_rows, target_rows" not in router_source
    assert "install_source_frozen_compact_contract" in router_source
    assert "inner_oof_base" in arm_source
    assert "target_rows, target_rows" not in arm_source
    assert "fold_global24" in arm_source
    assert "install_source_frozen_compact_contract" in arm_source
    assert "max-delta=2.25" in arm_source
    prepare_source = (here / "prepare_frozen_inputs.py").read_text(encoding="utf-8")
    assert '"within_year_label_file"' in prepare_source
    assert '"within_year_labels_used_for_training_and_routing": True' in prepare_source
    assert '"future_labels_unsealed_only_by_final_summarizer": False' in prepare_source
    assert 'ROUTE_COLUMNS = {"A25": "route_A25", "DI25": "route_DI"}' in summary_source
    assert "router.replace('25', '')" not in summary_source
    for count in (793, 399, 397, 400):
        assert max(1, min(count, int(round(0.25 * count)))) in {198, 100, 99}
    test_compact_contract(here)
    print("WITHIN_YEAR_MATRIX_SELFTEST_PASS")


if __name__ == "__main__":
    main()
