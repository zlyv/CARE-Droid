#!/usr/bin/env python3
"""Shared paths and fail-closed I/O for the extraction-only snapshot."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT / 'scripts/care-az-random15k-available-extraction-v1'
OUT = ROOT / 'outputs/care_az_random15k_available_extraction_v1'
PRIOR = ROOT / 'outputs/care_expanded2017_benign_aware_v2'
PYTHON = Path('/home/boyuanjie/envs/tif/bin/python')
COMPACT = ROOT / 'scripts/care-di25-host-gatedgraph-source-v1/run_compact_feature_extraction_ag.py'
ENDPOINT = ROOT / 'scripts/care-di25-host-gatedgraph-source-v1/run_endpoint_centered_tensor_cache_ag.py'
RESOLVED = ROOT / 'artifacts/care-litelocal-semantic-source-v1/resolved_inputs.json'
PRIORITY = ROOT / 'configs/protocol/sensitive_endpoint_priority_v1.yaml'
EDGE = ROOT / 'configs/protocol/semantic_edge_mapping_v1.yaml'


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    fields = list(rows[0]) if rows else ['status']
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')


def digest(path):
    path = Path(path); h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            h.update(chunk)
    return {'path': str(path.resolve()), 'size': path.stat().st_size, 'sha256': h.hexdigest()}


def compact_ids(path):
    if not Path(path).is_file(): return set()
    return {str(r.get('sha256', '')).strip().lower() for r in read_csv(path)
            if r.get('status') == 'OK' and r.get('sha256')}


def endpoint_ids(path):
    result = set(); path = Path(path)
    if not path.is_file(): return result
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip(): continue
                try: row = json.loads(line)
                except json.JSONDecodeError: continue
                if row.get('status') == 'OK' and row.get('sha256'):
                    result.add(str(row['sha256']).strip().lower())
    except (OSError, EOFError):
        pass
    return result

