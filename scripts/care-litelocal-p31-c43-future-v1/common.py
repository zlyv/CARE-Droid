#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT = Path('/home/boyuanjie/projects/tasa-droid')
SCRIPTS = PROJECT / 'scripts'
OUT = PROJECT / 'artifacts/care-litelocal-p31-c43-future-v1'
SOURCE_P31 = PROJECT / 'artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz'
SOURCE_ROUTER = PROJECT / 'artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv'
SOURCE_REFERENCE = PROJECT / 'artifacts/care-litelocal-p31-c43-source-v1/preflight/p31_reference/source_predictions.csv'
FUTURE_LABELS = PROJECT / 'artifacts/care-di25-host-m0-future-replay-v1/replay_predictions_with_labels.csv'
FUTURE_INTER = PROJECT / 'artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz'

LABEL_FREE_COHORT_CANDIDATES = (
    PROJECT / 'artifacts/care-di25-host-m0-future-replay-v1/preflight/future_cohort_label_free.csv',
    PROJECT / 'artifacts/care-route-stage2-future-matrix-v1/preflight/future_cohort_label_free.csv',
    PROJECT / 'artifacts/care-future-local-assets-v2/future_lineage_2018_2021.csv',
)
SEALED_FEATURE_CANDIDATES = (
    PROJECT / 'artifacts/care-di25-host-m0-future-replay-v1/preflight/compact_features_2017_train_2018_2021_label_sealed.csv',
    PROJECT / 'artifacts/care-route-stage2-future-matrix-v1/preflight/compact_features_2017_train_2018_2021_label_sealed.csv',
    PROJECT / 'artifacts/care-route-stage2-within-year-matrix-v1/preflight/compact_features_2017_train_2018_2021_label_sealed.csv',
)


def norm(value):
    return str(value or '').strip().casefold()


def read_csv(path: Path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: refusing to write empty csv: {path}')
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def file_sha256(path: Path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def first_existing(paths: Iterable[Path], label: str) -> Path:
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise RuntimeError('BLOCKED_BY_CONTRACT: no ' + label + ' found. Checked:\n- ' + '\n- '.join(map(str, paths)))


def infer_year(row):
    for key in ('year', 'metadata_year'):
        value = str(row.get(key, '')).strip()
        if value:
            return int(float(value))
    raise RuntimeError(f'BLOCKED_BY_CONTRACT: row has no year field: keys={list(row)[:12]}')


def load_future_cohort():
    path = first_existing(LABEL_FREE_COHORT_CANDIDATES, 'label-free future cohort')
    rows = read_csv(path)
    out = []
    for row in rows:
        sid = norm(row.get('sha256'))
        if not sid:
            continue
        year = infer_year(row)
        if year in (2018, 2019, 2020, 2021):
            out.append((sid, year))
    if len(out) != 1989 or len({x[0] for x in out}) != 1989:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: future cohort must be 1989 unique rows, got n={len(out)} unique={len({x[0] for x in out})} from {path}')
    counts = {year: sum(y == year for _, y in out) for year in (2018, 2019, 2020, 2021)}
    expected = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
    if counts != expected:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: future year counts changed: {counts}')
    return path, out


def discover_future_p31_cache(explicit: str | None = None) -> Path:
    _, cohort = load_future_cohort()
    future_ids = {sid for sid, _ in cohort}
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = sorted((PROJECT / 'artifacts').glob('**/*.npz'))
    matches = []
    seen_semantic = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                if 'role_prevalence' not in z or 'sha256' not in z:
                    continue
                p = np.asarray(z['role_prevalence'])
                ids = [norm(x) for x in z['sha256']]
                if p.ndim != 2 or p.shape[1] != 31 or len(ids) != p.shape[0]:
                    continue
                coverage = len(future_ids.intersection(ids))
                seen_semantic.append((coverage, len(ids), str(path)))
                if coverage == len(future_ids):
                    bonus = int('future' in str(path).casefold()) + int('semantic' in str(path).casefold())
                    matches.append((bonus, -len(ids), path))
        except Exception:
            continue
    if not matches:
        detail = '\n'.join(f'  coverage={c}/1989 rows={n} {p}' for c, n, p in sorted(seen_semantic, reverse=True)[:20])
        raise RuntimeError(
            'BLOCKED_BY_CONTRACT: no NPZ containing role_prevalence[*,31] + sha256 covers all 1989 future APKs.\n'
            'This experiment will NOT substitute role_node counts for P31.\n'
            'Best semantic NPZ candidates:\n' + (detail or '  <none>') +
            '\nIf you already have the correct cache, rerun with FUTURE_P31_CACHE=/path/to/file.npz.'
        )
    matches.sort(reverse=True, key=lambda x: (x[0], x[1], str(x[2])))
    best = matches[0][2]
    if explicit and best.resolve() != Path(explicit).resolve():
        raise RuntimeError('BLOCKED_BY_CONTRACT: explicit future P31 cache failed coverage contract')
    return best


def discover_future_runner() -> Path:
    exact = [
        SCRIPTS / 'run_care_rolegraph_conditioned_v4_future.py',
        SCRIPTS / 'care-rolegraph-conditioned-v4-future-script/run_care_rolegraph_conditioned_v4_future.py',
    ]
    for p in exact:
        if p.is_file():
            return p
    found = sorted(SCRIPTS.glob('**/run_care_rolegraph_conditioned_v4_future.py'))
    if len(found) == 1:
        return found[0]
    raise RuntimeError(f'BLOCKED_BY_CONTRACT: expected one generic v4 future runner, found={found}')


def discover_c43_future_region_cache(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise RuntimeError(f'BLOCKED_BY_CONTRACT: C43 future region cache does not exist: {p}')
        return p
    globs = []
    for pattern in ('**/*compact43*relation*future*.npz', '**/*compact43*relation*2017*2021*.npz', '**/*compact43*relation*.npz'):
        globs.extend((PROJECT / 'artifacts').glob(pattern))
    unique = []
    seen = set()
    source_only = PROJECT / 'artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz'
    for p in globs:
        if p in seen or not p.is_file() or p.resolve() == source_only.resolve():
            continue
        seen.add(p); unique.append(p)
    # Prefer names that explicitly say future or 2017_2021.
    preferred = [p for p in unique if 'future' in str(p).casefold() or '2017_2021' in str(p).casefold() or '2018_2021' in str(p).casefold()]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        raise RuntimeError('BLOCKED_BY_CONTRACT: multiple C43 future region-cache candidates; set C43_FUTURE_REGION_CACHE explicitly:\n- ' + '\n- '.join(map(str, preferred)))
    if len(unique) == 1:
        return unique[0]
    raise RuntimeError('BLOCKED_BY_CONTRACT: cannot uniquely identify a future C43 relation-query region cache. Set C43_FUTURE_REGION_CACHE=/exact/path.npz. Candidates:\n- ' + ('\n- '.join(map(str, unique)) if unique else '<none>'))


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
