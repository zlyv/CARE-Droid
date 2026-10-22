#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/care-litelocal-p31-c43-complete-diagnostics-v1")
EXPECTED_YEARS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
EXPECTED_ROUTE = {2018: 198, 2019: 100, 2020: 99, 2021: 100}
METHOD = "P31Mean__DI25__C43RelationQuery__Bounded"


def norm(value) -> str:
    return str(value or "").strip().casefold()


def read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty CSV: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolved() -> dict:
    return json.loads((ROOT / "preflight/resolved_inputs.json").read_text(encoding="utf-8"))


def aligned_rows(target_rows: list[dict], path: Path) -> list[dict]:
    rows = read_csv(path)
    lookup = {norm(row.get("sha256") or row.get("sample_id")): row for row in rows}
    if len(lookup) != len(rows):
        raise RuntimeError("duplicate SHA in " + str(path))
    output = []
    for target in target_rows:
        sha = norm(target.get("sha256") or target.get("sample_id"))
        if sha not in lookup:
            raise RuntimeError(f"{path} is missing SHA {sha}")
        output.append(lookup[sha])
    return output


def load_bags(path: Path, target_rows: list[dict], expected_width: int = 105):
    with np.load(path, allow_pickle=False) as saved:
        ids = [norm(value) for value in saved["sha256"].astype(str)]
        values = np.asarray(saved["host_values"], dtype=np.float32)
        offsets = np.asarray(saved["host_offsets"], dtype=np.int64)
        real = np.asarray(saved["real_host_count"], dtype=np.int64)
    lookup = {value: index for index, value in enumerate(ids)}
    if len(lookup) != len(ids) or len(offsets) != len(ids) + 1:
        raise RuntimeError("invalid Host cache index: " + str(path))
    bags, counts = [], []
    for row in target_rows:
        sha = norm(row["sha256"])
        if sha not in lookup:
            raise RuntimeError(f"{path} is missing SHA {sha}")
        index = lookup[sha]
        bag = values[offsets[index]:offsets[index + 1]]
        if bag.ndim != 2 or bag.shape[0] < 1 or bag.shape[1] != expected_width:
            raise RuntimeError(f"invalid {expected_width}-D Host bag for {sha}: {bag.shape}")
        bags.append(bag)
        counts.append(int(real[index]))
    return bags, np.asarray(counts, dtype=np.int64)


def subset_bags(bags: list[np.ndarray], index: np.ndarray) -> list[np.ndarray]:
    return [bags[int(value)] for value in index]


def align_fold_global24(raw_train, raw_validation, fold_names, canonical_names):
    """Align only the fixed Global24 view and ignore fold-only role columns.

    Some annual folds expose a role whose source-2017 Compact76 column was
    absent (for example package_installation).  C43 consumes only Global24 as
    correction context, so rejecting the fold because of an extra *role*
    column is incorrect.  This routine freezes Global24 by name, zero-fills a
    missing canonical global column, and refuses any unexpected non-role
    column.  It never copies an extra role column into Global24.
    """
    raw_train = np.asarray(raw_train, dtype=np.float64)
    raw_validation = np.asarray(raw_validation, dtype=np.float64)
    fold_names = list(fold_names); canonical_names = list(canonical_names)
    is_role = lambda name: name.startswith("role_node__") or name.startswith("role_edge__")
    canonical_global = [name for name in canonical_names if not is_role(name)]
    fold_global = [name for name in fold_names if not is_role(name)]
    if len(canonical_global) != 24 or len(set(canonical_global)) != 24:
        raise RuntimeError("BLOCKED_BY_CONTRACT: canonical Global24 names changed")
    unexpected_global = sorted(set(fold_global) - set(canonical_global))
    if unexpected_global:
        raise RuntimeError("BLOCKED_BY_CONTRACT: unexpected global columns: " + repr(unexpected_global))
    if raw_train.shape[1] != len(fold_names) or raw_validation.shape[1] != len(fold_names):
        raise RuntimeError("BLOCKED_BY_CONTRACT: fold Compact matrix/name width mismatch")
    lookup = {name: index for index, name in enumerate(fold_names)}
    if len(lookup) != len(fold_names):
        raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate fold Compact names")
    train = np.zeros((len(raw_train), 24), dtype=np.float32)
    validation = np.zeros((len(raw_validation), 24), dtype=np.float32)
    for target, name in enumerate(canonical_global):
        source = lookup.get(name)
        if source is not None:
            train[:, target] = raw_train[:, source]
            validation[:, target] = raw_validation[:, source]
    audit = {
        "fold_compact_width": len(fold_names),
        "canonical_global_width": 24,
        "missing_canonical_global": sorted(set(canonical_global) - set(fold_global)),
        "ignored_fold_only_role": sorted(
            name for name in set(fold_names) - set(canonical_names) if is_role(name)),
        "unexpected_global": unexpected_global,
    }
    return train, validation, audit


def install_paths() -> None:
    project = Path.cwd()
    scripts = project / "scripts"
    paths = (
        scripts / "care-litelocal-p31-c43-complete-diagnostics-v1",
        scripts / "care-litelocal-p31-c43-source-v1",
        scripts / "care-di25-compact43-relationquery-source-v1",
        scripts / "care-di25-graphglobal-path-source-v1",
        scripts / "care-route-stage2-future-matrix-v1",
        scripts / "care-route-stage2-future-matrix-v1/vendor",
        scripts / "care-rolegraph-conditioned-v4-script",
        scripts / "care-rgcondattn-dedup-bounded-source-v1",
        scripts / "care-malben-host-v2",
        scripts,
    )
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def install_c43(v4):
    from compact_background import make_fit
    from compact_models import make_compact_class

    v4.AUGMENTED_REGION_DIM = 78
    v4.RoleGraphConditionedEvidence = make_compact_class(v4, "C43_RELATION_QUERY")
    v4.v3.fit_scaled_dualproto_bags = make_fit(v4.v3)


def metric_counts(label, base_probability, final_probability, route):
    label = np.asarray(label, int)
    base_probability = np.asarray(base_probability, float)
    final_probability = np.asarray(final_probability, float)
    route = np.asarray(route, bool)
    base_correct = (base_probability >= .5) == label
    final_correct = (final_probability >= .5) == label
    return {
        "routed": int(route.sum()),
        "fixed": int(np.sum(~base_correct & final_correct)),
        "broken": int(np.sum(base_correct & ~final_correct)),
        "net": int(np.sum(final_correct) - np.sum(base_correct)),
    }
