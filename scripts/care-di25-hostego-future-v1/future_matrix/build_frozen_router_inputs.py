#!/usr/bin/env python3
"""Rebuild frozen Global24, A25 and DI25 inputs from 2017 only.

This removes the operational dependency on old committee-result directories.
Future labels are absent from every object opened by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import balanced_accuracy_score


HERE = Path(__file__).resolve().parent
for path in (HERE, HERE / "vendor"):
    sys.path.insert(0, str(path))

import run_care_rolegraph_conditioned_dualstream_v4 as v4
from committee_uncertainty_core import (
    MEMBER_NAMES, fit_calibrated_committee, groupwise_top_fraction,
)
from source_label_contract import install_source_label_contract


def norm(value) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def global_matrices(train_rows, test_rows, by_sha, schema, original_names):
    x_train, x_test, names = v4.audit.compact_matrix_variant(
        v4.VARIANT, train_rows, test_rows, by_sha, schema, original_names)
    index, _ = v4.evidence.split_view_indices(names)
    if len(index) != 24:
        raise RuntimeError("Global24 dimension changed")
    return (
        np.asarray(x_train[:, index], dtype=np.float64),
        np.asarray(x_test[:, index], dtype=np.float64),
        [names[i] for i in index],
    )


def decision_instability(base: np.ndarray, members: np.ndarray) -> np.ndarray:
    mean = np.mean(np.asarray(members, dtype=np.float64), axis=1)
    return np.where(np.asarray(base) >= 0.5, 1.0 - mean, mean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved", type=Path, default=Path(
        "artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json"))
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    install_source_label_contract(v4)
    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    load_args = SimpleNamespace(
        features=Path(resolved["features"]), schema=Path(resolved["schema"]),
        p72_oof=Path(resolved["source_manifest"]),
    )
    manifest, source_rows, _, compact_by_sha, schema, original_names = (
        v4.audit.load_inputs(load_args)
    )
    labels = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in manifest], dtype=np.int64)
    fold_sizes = sorted(int(np.sum(folds == fold)) for fold in np.unique(folds))
    if len(labels) != 799 or fold_sizes != [159, 160, 160, 160, 160]:
        raise RuntimeError("fixed 2017 fold contract changed")

    future = read_csv(Path(resolved["future_cohort_label_free"]))
    future_ids = [norm(row["sha256"]) for row in future]
    years = np.asarray([int(row["year"]) for row in future], dtype=np.int64)
    target_rows = []
    for value in future_ids:
        row = compact_by_sha[value]
        leaked = {key: item for key, item in row.items()
                  if "label" in key.casefold() and str(item).strip()}
        if leaked:
            raise RuntimeError("future label visible while rebuilding routers: " + value)
        target_rows.append(row)

    source_base = np.full(799, np.nan, dtype=np.float64)
    source_member = np.full((799, len(MEMBER_NAMES)), np.nan, dtype=np.float64)
    names_reference = None
    audits = []
    for fold in sorted(np.unique(folds)):
        train_index = np.flatnonzero(folds != fold)
        validation_index = np.flatnonzero(folds == fold)
        train_rows = [source_rows[i] for i in train_index]
        validation_rows = [source_rows[i] for i in validation_index]
        x_train, x_validation, names = global_matrices(
            train_rows, validation_rows, compact_by_sha, schema, original_names)
        if names_reference is None:
            names_reference = names
        elif names != names_reference:
            raise RuntimeError("Global24 names/order changed across folds")
        source_base[validation_index] = v4.audit.p72.fit_predict(
            x_train, labels[train_index], x_validation, seed=int(fold) - 1)[0]
        fitted = fit_calibrated_committee(
            x_train, labels[train_index], x_validation,
            inner_folds=args.inner_folds,
            split_seed=args.seed + 100000 * int(fold),
            model_seed=args.seed + 10000 * int(fold), jobs=args.jobs,
        )
        source_member[validation_index] = fitted.target_probability
        audits.extend({"scope": "source_outer_fold", "fold": int(fold), **row}
                      for row in fitted.audit)
        print(f"router rebuild source fold {int(fold)} done", flush=True)
    if not np.isfinite(source_base).all() or not np.isfinite(source_member).all():
        raise RuntimeError("incomplete source router reconstruction")
    source_ba = float(balanced_accuracy_score(labels, source_base >= 0.5))
    if abs(source_ba - 0.8196741854636592) > 1e-9:
        raise RuntimeError(f"Global24 source reproduction changed: BA={source_ba:.12f}")

    x_source, x_target, names = global_matrices(
        source_rows, target_rows, compact_by_sha, schema, original_names)
    if names != names_reference:
        raise RuntimeError("future Global24 names/order changed")
    future_base = v4.audit.p72.fit_predict(
        x_source, labels, x_target, seed=0)[0]
    fitted = fit_calibrated_committee(
        x_source, labels, x_target,
        inner_folds=args.inner_folds, split_seed=args.seed + 800000,
        model_seed=args.seed + 900000, jobs=args.jobs,
    )
    future_member = fitted.target_probability
    audits.extend({"scope": "future_deployment_fit", **row} for row in fitted.audit)

    source_a = 1.0 - 2.0 * np.abs(source_base - 0.5)
    future_a = 1.0 - 2.0 * np.abs(future_base - 0.5)
    source_di = decision_instability(source_base, source_member)
    future_di = decision_instability(future_base, future_member)
    source_route_a = groupwise_top_fraction(source_a, folds, 0.25)
    source_route_di = groupwise_top_fraction(source_di, folds, 0.25)
    future_route_a = groupwise_top_fraction(future_a, years, 0.25)
    future_route_di = groupwise_top_fraction(future_di, years, 0.25)

    source_output = Path(resolved["source_scores"])
    future_output = Path(resolved["future_router_inputs_label_free"])
    source_rows_out = []
    for i, row in enumerate(manifest):
        source_rows_out.append({
            "sha256": norm(row["sha256"]), "label": int(labels[i]),
            "fold": int(folds[i]), "p_stage1": float(source_base[i]),
            **{f"p_{name.lower()}": float(source_member[i, j])
               for j, name in enumerate(MEMBER_NAMES)},
            "U_A25": float(source_a[i]), "U_DI": float(source_di[i]),
            "route_A25": int(source_route_a[i]), "route_DI": int(source_route_di[i]),
        })
    future_rows_out = []
    for i, value in enumerate(future_ids):
        future_rows_out.append({
            "sha256": value, "year": int(years[i]),
            "p_stage1": float(future_base[i]),
            **{f"p_{name.lower()}": float(future_member[i, j])
               for j, name in enumerate(MEMBER_NAMES)},
            "U_A25": float(future_a[i]), "U_DI": float(future_di[i]),
            "route_A25": int(future_route_a[i]), "route_DI": int(future_route_di[i]),
        })
    write_csv(source_output, source_rows_out)
    write_csv(future_output, future_rows_out)
    write_csv(args.resolved.parent / "committee_calibration_audit.csv", audits)

    resolved.update({
        "future_router_inputs_sha256": sha256(future_output),
        "source_scores_sha256": sha256(source_output),
        "global24_source_BA_reproduced": source_ba,
        "routes": {
            "A25": {str(y): int(np.sum(future_route_a & (years == y))) for y in sorted(set(years))},
            "DI25": {str(y): int(np.sum(future_route_di & (years == y))) for y in sorted(set(years))},
        },
        "router_models_trained_on_2017_only": True,
        "future_labels_loaded_by_router_builder": False,
    })
    args.resolved.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    print("FROZEN_A25_DI_ROUTER_REBUILD_PASS", flush=True)


if __name__ == "__main__":
    main()
