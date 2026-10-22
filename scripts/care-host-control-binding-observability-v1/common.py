#!/usr/bin/env python3
from __future__ import annotations
import csv, gzip, hashlib, json, os
from pathlib import Path
import numpy as np

NAME = 'care-host-control-binding-observability-v1'
HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('HCB_PROJECT', '/home/boyuanjie/projects/tasa-droid')).resolve()
OUTPUT = PROJECT / 'outputs' / NAME
PARENT = Path(os.environ.get('HCB_PARENT', str(PROJECT / 'outputs/care-local-control-constraint-incremental-source-v1'))).resolve()

def norm(v):
    return (v.decode() if isinstance(v, bytes) else str(v)).strip().casefold()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def load_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def safe(path):
    path = Path(path).resolve()
    root = (PROJECT / 'outputs').resolve()
    if OUTPUT.resolve() != root / NAME or path == root:
        raise RuntimeError('unsafe experiment output root')
    if path != OUTPUT.resolve() and OUTPUT.resolve() not in path.parents:
        raise RuntimeError('write outside independent output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def dump(path, obj):
    p = safe(path)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)

def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    p = safe(path)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def asset(path, purpose):
    p = Path(path).resolve()
    s = p.stat()
    return dict(absolute_path=str(p), purpose=purpose, size_bytes=s.st_size, mtime_ns=s.st_mtime_ns, sha256=sha256(p))

def first(paths, description):
    for raw in paths:
        if raw and Path(raw).is_file():
            return Path(raw).resolve()
    raise RuntimeError('BLOCKED: missing ' + description + '; checked=' + repr([str(p) for p in paths if p]))

def protocol():
    return read_json(HERE / 'protocol.json')
