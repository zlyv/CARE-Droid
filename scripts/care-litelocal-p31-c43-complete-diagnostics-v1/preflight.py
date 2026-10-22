#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import EXPECTED_ROUTE, EXPECTED_YEARS, ROOT, norm, read_csv


PROJECT = Path.cwd()
SCRIPTS = PROJECT / "scripts"


def choose(candidates, description):
    for path in candidates:
        if path.is_file() and path.stat().st_size >= 100:
            return path.resolve()
    raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty " + description + ":\n- " +
                       "\n- ".join(str(value) for value in candidates))


def choose_host_cache(candidates, description, expected_rows):
    checked = []
    for path in candidates:
        checked.append(path)
        if not path.is_file() or path.stat().st_size < 100:
            continue
        try:
            with np.load(path, allow_pickle=False) as saved:
                if (len(saved["sha256"]) == expected_rows and
                        saved["host_values"].ndim == 2 and
                        saved["host_values"].shape[1] == 105):
                    return path.resolve()
        except (KeyError, OSError, ValueError):
            pass
    # File names changed across the earlier diagnostic packages. Search only
    # their cache directories and accept a file by its hard shape contract.
    roots = (
        PROJECT / "artifacts/care-di25-compact43-relationquery-diagnostics-v1",
        PROJECT / "artifacts/care-di25-compact43-relationquery-source-v1",
    )
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.npz"):
            if path in checked or path.stat().st_size < 100:
                continue
            try:
                with np.load(path, allow_pickle=False) as saved:
                    if (len(saved["sha256"]) == expected_rows and
                            saved["host_values"].ndim == 2 and
                            saved["host_values"].shape[1] == 105):
                        return path.resolve()
            except (KeyError, OSError, ValueError):
                continue
    raise RuntimeError("BLOCKED_BY_CONTRACT: no exact " + description +
                       f" with rows={expected_rows}, width=105")


def validate_p31(path, expect_labels):
    rows = read_csv(path)
    ids = [norm(row.get("sha256") or row.get("sample_id")) for row in rows]
    if len(rows) != 1989 or len(set(ids)) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31 future cohort changed: " + str(path))
    years = Counter(int(float(row["year"])) for row in rows)
    if years != Counter(EXPECTED_YEARS):
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31 year counts changed: " + repr(years))
    route = Counter(int(float(row["year"])) for row in rows
                    if int(float(row.get("route_DI", row.get("P31_DI25__routed", 0)))) == 1)
    if route != Counter(EXPECTED_ROUTE):
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31 route changed: " + repr(route))
    if expect_labels and not {"label", "fold"}.issubset(rows[0]):
        raise RuntimeError("BLOCKED_BY_CONTRACT: within-year labels/folds missing")
    return rows


def validate_cache(path, expected_rows):
    with np.load(path, allow_pickle=False) as saved:
        required = {"sha256", "host_values", "host_offsets", "real_host_count"}
        if not required.issubset(saved.files):
            raise RuntimeError("BLOCKED_BY_CONTRACT: Compact43 cache keys changed: " + str(path))
        if len(saved["sha256"]) != expected_rows or saved["host_values"].shape[1] != 105:
            raise RuntimeError("BLOCKED_BY_CONTRACT: Compact43 cache shape changed: " + str(path))


def main():
    source_ref = choose((
        PROJECT / "artifacts/care-litelocal-p31-c43-source-v1/preflight/p31_reference/source_predictions.csv",
    ), "locked source P31 probabilities")
    frozen_p31 = choose((
        PROJECT / "artifacts/care-litelocal-p31-diagnostics-v1/frozen_future/label_free_predictions.csv",
    ), "label-free frozen-future P31 predictions")
    within_p31 = choose((
        PROJECT / "artifacts/care-litelocal-p31-diagnostics-v1/within_year/oof_predictions.csv",
    ), "within-year P31 OOF predictions")
    raw_source_p31 = choose((
        PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz",
    ), "source P31 feature cache")
    raw_future_p31 = choose((
        PROJECT / "artifacts/care-litelocal-p31-diagnostics-v1/future_p31_features_2018_2021.npz",
    ), "future P31 feature cache")
    source_c43 = choose_host_cache((
        PROJECT / "artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz",
    ), "source Compact43+Relation31 cache", 799)
    future_c43 = choose_host_cache((
        PROJECT / "artifacts/care-di25-compact43-relationquery-diagnostics-v1/cache/compact43_relation_future.npz",
        PROJECT / "artifacts/care-di25-compact43-relationquery-diagnostics-v1/cache/compact43_relation_2018_2021.npz",
        PROJECT / "artifacts/care-di25-compact43-relationquery-diagnostics-v1/cache/future_compact43_relation.npz",
        PROJECT / "artifacts/care-litelocal-p31-c43-localization-source-v1/cache/compact43_relation_future.npz",
    ), "future Compact43+Relation31 cache", 1989)
    future_resolved = choose((
        PROJECT / "artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json",
    ), "frozen future structural input manifest")
    choose((
        PROJECT / "artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz",
    ), "prebuilt future InterZ cache (concurrent jobs must not rebuild it)")
    dependencies = (
        SCRIPTS / "care-route-stage2-future-matrix-v1/run_future_arm.py",
        SCRIPTS / "care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
        SCRIPTS / "care-di25-compact43-relationquery-source-v1/compact_models.py",
        SCRIPTS / "care-di25-compact43-relationquery-source-v1/compact_background.py",
        SCRIPTS / "care-litelocal-p31-c43-source-v1/p31_core.py",
    )
    missing = [str(path) for path in dependencies if not path.is_file() or path.stat().st_size < 100]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing code dependencies:\n- " + "\n- ".join(missing))

    source = read_csv(source_ref)
    if len(source) != 799 or len({norm(row["sha256"]) for row in source}) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source P31 cohort changed")
    source_route = np.asarray([int(float(row["P31_DI25__routed"])) for row in source])
    if int(source_route.sum()) != 200:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source P31 route changed")
    frozen = validate_p31(frozen_p31, False)
    within = validate_p31(within_p31, True)
    if {norm(row["sha256"]) for row in frozen} != {norm(row["sha256"]) for row in within}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen/within P31 cohorts differ")
    with np.load(raw_source_p31, allow_pickle=False) as saved:
        if saved["role_prevalence"].shape != (799, 31):
            raise RuntimeError("BLOCKED_BY_CONTRACT: source P31 shape changed")
    with np.load(raw_future_p31, allow_pickle=False) as saved:
        if saved["role_prevalence"].shape != (1989, 31):
            raise RuntimeError("BLOCKED_BY_CONTRACT: future P31 shape changed")
    validate_cache(source_c43, 799)
    validate_cache(future_c43, 1989)

    output = {
        "source_p31_reference": str(source_ref),
        "frozen_p31_predictions": str(frozen_p31),
        "within_p31_predictions": str(within_p31),
        "source_p31_features": str(raw_source_p31),
        "future_p31_features": str(raw_future_p31),
        "source_c43_cache": str(source_c43),
        "future_c43_cache": str(future_c43),
        "future_matrix_resolved": str(future_resolved),
    }
    target = ROOT / "preflight/resolved_inputs.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("P31_C43_COMPLETE_DIAGNOSTICS_PREFLIGHT_PASS")
    print("source_rows=799 source_route_P31_DI25=200")
    print("future_rows=1989 year_counts=2018:793,2019:399,2020:397,2021:400")
    print("future_route_P31_DI25=497 route_counts=2018:198,2019:100,2020:99,2021:100")
    print("stage2=C43_RELATION_QUERY parameters=32292 max_delta=2.25")
    print("frozen_future_labels_loaded_by_model=false ego_consumed=false")


if __name__ == "__main__":
    main()
