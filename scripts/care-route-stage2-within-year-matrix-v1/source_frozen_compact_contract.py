#!/usr/bin/env python3
"""Install the source-frozen Compact76 contract for same-year experiments.

The feature dictionary is fixed by the 799-APK 2017 source cohort.  A later
year may expose additional semantic-role columns, but those columns are not
allowed to silently expand the model input.  They are explicitly audited and
ignored.  Missing source columns are represented by zero, exactly as in the
existing fold-safe source implementation.

Only the column alignment is patched here.  Ratio means and standard
deviations continue to be fitted by ``compact_matrix_variant`` on the current
outer/inner training fold, so validation rows do not influence preprocessing.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


_UNEXPECTED = Counter()
_INSTALL_MARKER = "_care_source_frozen_compact76_installed"


def install_source_frozen_compact_contract(v4) -> None:
    """Patch the exact audit module used by ``v4`` once per process."""
    audit = v4.audit
    if getattr(audit, _INSTALL_MARKER, False):
        return

    def restore_source_frozen_compact76(
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
        canonical_original_names,
    ):
        fold_train, fold_test, fold_names = (
            audit.allratio.original_compact_matrix_named(
                train_rows, test_rows, compact_by_sha, compact_schema
            )
        )
        fold_train = np.asarray(fold_train, dtype=np.float64)
        fold_test = np.asarray(fold_test, dtype=np.float64)
        fold_names = list(fold_names)
        canonical = list(canonical_original_names)

        if len(canonical) != 76 or len(set(canonical)) != 76:
            raise RuntimeError(
                "source-frozen Compact dictionary must contain 76 unique fields"
            )
        if len(set(fold_names)) != len(fold_names):
            raise RuntimeError("duplicate names in fold Compact feature matrix")
        if fold_train.shape != (len(train_rows), len(fold_names)):
            raise RuntimeError("fold-train Compact matrix/name alignment failed")
        if fold_test.shape != (len(test_rows), len(fold_names)):
            raise RuntimeError("fold-validation Compact matrix/name alignment failed")

        canonical_set = set(canonical)
        unexpected = sorted(set(fold_names) - canonical_set)
        for name in unexpected:
            _UNEXPECTED[name] += 1

        fold_index = {name: index for index, name in enumerate(fold_names)}
        restored_train = np.zeros((len(train_rows), 76), dtype=np.float64)
        restored_test = np.zeros((len(test_rows), 76), dtype=np.float64)
        for target_index, name in enumerate(canonical):
            source_index = fold_index.get(name)
            if source_index is None:
                continue
            restored_train[:, target_index] = fold_train[:, source_index]
            restored_test[:, target_index] = fold_test[:, source_index]
        return restored_train, restored_test, canonical

    audit.restore_raw_compact76 = restore_source_frozen_compact76
    setattr(audit, _INSTALL_MARKER, True)


def compact_contract_audit() -> dict:
    return {
        "feature_dictionary_source": "fixed 2017 Compact76",
        "canonical_dimension": 76,
        "later_year_extra_features_enter_model": False,
        "later_year_extra_features_ignored": sorted(_UNEXPECTED),
        "extra_feature_fold_occurrences": dict(sorted(_UNEXPECTED.items())),
        "ratio_preprocessing": "fit on the current training fold only",
        "validation_rows_used_for_preprocessing_fit": False,
    }

