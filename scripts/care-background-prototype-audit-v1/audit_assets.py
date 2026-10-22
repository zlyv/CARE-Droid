#!/usr/bin/env python3
"""Read-only preflight and named-asset SHA256 guard; writes only this new audit output."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT / 'scripts/care-background-prototype-audit-v1'
OUT = ROOT / 'outputs/care_background_prototype_audit_v1'
TAGS = ('C0R_2017','T2_C1','T3_C1','T4_C1')
SEEDS = (900000,900001,900002)
MODEL = ROOT / 'outputs/care_multiyear_development_v1'


def sha_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    s = path.stat()
    return {'path': str(path.resolve()), 'sha256': h.hexdigest(),
            'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def read_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    p = Path(path)
    if OUT not in p.resolve().parents: raise RuntimeError('write outside isolated output')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def write_json(path, obj):
    p = Path(path)
    if OUT not in p.resolve().parents: raise RuntimeError('write outside isolated output')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.write('\n')


def protected():
    paths = [MODEL/'data.npz', MODEL/'sample_manifest.csv',
             ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py',
             ROOT/'scripts/run_care_joint_dualprototype_ablation.py',
             ROOT/'scripts/care-multiyear-development-v1/train_stage2.py',
             ROOT/'outputs/care_crossyear_layerwise_diagnosis_v2/per_apk_error_contract.csv',
             ROOT/'outputs/care-2020-fn-tn-dex-review-v2/matched_pairs_restricted.csv']
    paths += [MODEL/'models/stage2'/tag/(str(seed)+'.pkl')
              for tag in TAGS for seed in SEEDS]
    lineage = ROOT/'../tasa-droid-storage/care-source-local-assets-v2/source_lineage_2017.csv'
    if not lineage.is_file():
        lineage = Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/source_lineage_2017.csv')
    old = read_csv(lineage)
    sample = [old[i]['absolute_path'] for i in (0,len(old)//2,len(old)-1)]
    paths += list(map(Path,sample))
    return paths


def verify_manifest():
    before = json.loads((OUT/'baseline_asset_manifest_before.json').read_text())['assets']
    changed = []
    for row in before:
        try: now = sha_file(row['path'])
        except OSError as exc:
            changed.append({'path': row['path'], 'error': str(exc)})
            continue
        if any(now[k] != row[k] for k in ('sha256','size','mtime_ns')):
            changed.append({'path': row['path'], 'before': row, 'after': now})
    return {'status': 'PASS' if not changed else 'FAIL', 'asset_count': len(before),
            'changed': changed,
            'scope': 'named data, source functions, 12 checkpoints, frozen predictions/pairs and 3 sampled original APKs; not whole repository or all APKs'}


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--audit-only', action='store_true')
    g.add_argument('--verify', action='store_true')
    g.add_argument('--verify-readonly', action='store_true')
    args = p.parse_args()
    if args.audit_only:
        OUT.mkdir(parents=True, exist_ok=False)
        (OUT/'tmp').mkdir()
        paths = protected()
        before = [sha_file(x) for x in paths]
        write_json(OUT/'baseline_asset_manifest_before.json', {'assets': before})
        write_json(OUT/'preflight.json', {'status':'PASS','tags':TAGS,'seeds':SEEDS,
            'asset_count':len(before),'read_only':True,'new_model_training':False,
            'original_apk_samples_hashed':3})
        print('PROTOTYPE_PREFLIGHT_PASS assets='+str(len(before)), flush=True)
    elif args.verify_readonly:
        result = verify_manifest()
        print('PROTOTYPE_INTEGRITY_'+result['status'], flush=True)
        if result['status'] != 'PASS': raise RuntimeError('protected asset changed')
    else:
        result = verify_manifest()
        write_json(OUT/'baseline_integrity_check.json', result)
        print('PROTOTYPE_INTEGRITY_'+result['status'], flush=True)
        if result['status'] != 'PASS': raise RuntimeError('protected asset changed')


if __name__ == '__main__': main()
