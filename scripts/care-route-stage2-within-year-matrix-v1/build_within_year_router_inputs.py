#!/usr/bin/env python3
"""Build strict within-year OOF Global24, A25 and DI25 routes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve().parent
for path in (HERE, HERE / "vendor"):
    sys.path.insert(0, str(path))

import run_care_rolegraph_conditioned_dualstream_v4 as v4
from committee_uncertainty_core import MEMBER_NAMES, fit_calibrated_committee, groupwise_top_fraction
from source_label_contract import install_source_label_contract
from source_frozen_compact_contract import (
    compact_contract_audit,
    install_source_frozen_compact_contract,
)


EXPECTED_YEARS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
FEATURE_CONTRACT_VERSION = "source-frozen-compact76-foldsafe-v1"


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


def label_of(row: dict) -> int:
    value = str(row.get("label") or row.get("metadata_label") or "").strip().casefold()
    if value in {"0", "0.0", "benign", "goodware"}:
        return 0
    if value in {"1", "1.0", "malware", "malicious"}:
        return 1
    raise RuntimeError("unknown within-year label: " + repr(value))


def decision_instability(base: np.ndarray, members: np.ndarray) -> np.ndarray:
    mean = np.mean(np.asarray(members, dtype=np.float64), axis=1)
    return np.where(np.asarray(base) >= 0.5, 1.0 - mean, mean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.folds != 5:
        raise SystemExit("frozen same-year protocol requires five outer folds")

    install_source_label_contract(v4)
    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    load_args = SimpleNamespace(
        features=Path(resolved["features"]), schema=Path(resolved["schema"]),
        p72_oof=Path(resolved["source_manifest"]),
    )
    _, _, _, compact_by_sha, schema, original_names = v4.audit.load_inputs(load_args)
    install_source_frozen_compact_contract(v4)
    if v4.VARIANT != "allratio76_ratio52":
        raise RuntimeError("unexpected frozen Compact variant: " + str(v4.VARIANT))
    lineage_path = resolved.get("within_year_label_file")
    if not lineage_path:
        raise RuntimeError("resolved inputs do not declare within_year_label_file")
    lineage = read_csv(Path(lineage_path))
    ids = [norm(row.get("sha256") or row.get("sample_id")) for row in lineage]
    years = np.asarray([int(row.get("year") or row.get("metadata_year")) for row in lineage], dtype=np.int64)
    labels = np.asarray([label_of(row) for row in lineage], dtype=np.int64)
    if len(ids) != 1989 or len(set(ids)) != 1989:
        raise RuntimeError("within-year cohort must contain 1989 unique APKs")
    if dict(Counter(years.tolist())) != EXPECTED_YEARS:
        raise RuntimeError("within-year annual counts changed")
    target_rows = [compact_by_sha[value] for value in ids]

    base = np.full(1989, np.nan, dtype=np.float64)
    members = np.full((1989, len(MEMBER_NAMES)), np.nan, dtype=np.float64)
    folds = np.zeros(1989, dtype=np.int64)
    audits = []
    for year in sorted(EXPECTED_YEARS):
        year_index = np.flatnonzero(years == year)
        splitter = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed + year)
        for fold, (train_local, validation_local) in enumerate(
            splitter.split(np.zeros(len(year_index)), labels[year_index]), start=1
        ):
            train_index = year_index[train_local]
            validation_index = year_index[validation_local]
            folds[validation_index] = fold
            x_train, x_validation, names = v4.audit.compact_matrix_variant(
                v4.VARIANT,
                [target_rows[int(i)] for i in train_index],
                [target_rows[int(i)] for i in validation_index],
                compact_by_sha,
                schema,
                original_names,
            )
            global_index, _ = v4.evidence.split_view_indices(names)
            x_train = np.asarray(x_train[:, global_index], dtype=np.float64)
            x_validation = np.asarray(
                x_validation[:, global_index], dtype=np.float64
            )
            if x_train.shape[1] != 24 or x_validation.shape[1] != 24:
                raise RuntimeError("within-year fold Global24 contract failed")
            base[validation_index] = v4.audit.p72.fit_predict(
                x_train, labels[train_index], x_validation,
                seed=args.seed + year * 10 + fold,
            )[0]
            fitted = fit_calibrated_committee(
                x_train, labels[train_index], x_validation,
                inner_folds=args.inner_folds,
                split_seed=args.seed + year * 100 + fold,
                model_seed=args.seed + year * 1000 + fold,
                jobs=args.jobs,
            )
            members[validation_index] = fitted.target_probability
            audits.extend({"year": year, "outer_fold": fold, **item} for item in fitted.audit)
            print(f"within-year router {year} fold {fold}/5 done", flush=True)
    if not np.isfinite(base).all() or not np.isfinite(members).all() or np.any(folds == 0):
        raise RuntimeError("incomplete within-year OOF router predictions")

    ua = 1.0 - 2.0 * np.abs(base - 0.5)
    udi = decision_instability(base, members)
    route_a = groupwise_top_fraction(ua, years, 0.25)
    route_di = groupwise_top_fraction(udi, years, 0.25)
    rows = []
    for i, value in enumerate(ids):
        rows.append({
            "sha256": value, "year": int(years[i]), "label": int(labels[i]),
            "fold": int(folds[i]), "p_stage1": float(base[i]),
            **{f"p_{name.lower()}": float(members[i, j]) for j, name in enumerate(MEMBER_NAMES)},
            "U_A25": float(ua[i]), "U_DI": float(udi[i]),
            "route_A25": int(route_a[i]), "route_DI": int(route_di[i]),
        })
    write_csv(args.output, rows)
    write_csv(args.output.with_name("committee_calibration_audit.csv"), audits)
    report = {
        "protocol": "strict independent within-year five-fold OOF",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "future_years_used_for_training": True,
        "temporal_generalization_result": False,
        "budget": 0.25,
        "global24_oof_BA": float(balanced_accuracy_score(labels, base >= 0.5)),
        "annual_global24_oof_BA": {
            str(year): float(balanced_accuracy_score(labels[years == year], base[years == year] >= 0.5))
            for year in sorted(EXPECTED_YEARS)
        },
        "routes": {
            name: {str(year): int(np.sum(route & (years == year))) for year in sorted(EXPECTED_YEARS)}
            for name, route in (("A25", route_a), ("DI25", route_di))
        },
    }
    args.output.with_name("router_protocol.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_name("compact76_contract_audit.json").write_text(
        json.dumps(compact_contract_audit(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("WITHIN_YEAR_ROUTER_OOF_PASS", flush=True)


if __name__ == "__main__":
    main()
