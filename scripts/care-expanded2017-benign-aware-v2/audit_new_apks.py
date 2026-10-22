#!/usr/bin/env python3
"""Audit downloaded APK lineage and form an isolated 2017 union; writes only EXPANDED_OUTPUT.

The earlier read-only audit content-verifies all APKs. This program verifies its
manifest and metadata hashes, retains its per-APK status, and never edits inputs.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path('/home/boyuanjie/projects/tasa-droid')
OUT = Path(os.environ.get('EXPANDED_OUTPUT', PROJECT / 'outputs/care_expanded2017_benign_aware_v2'))
PRIOR = PROJECT / 'outputs/care_expanded2017_crossyear_v1'
SOURCE = PROJECT / 'outputs/care_multiyear_development_v1/sample_manifest.csv'
LINEAGE = Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/source_lineage_2017.csv')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ['status'])
    w.writeheader()
    w.writerows(rows)
    expected = buf.getvalue()
    if path.exists():
        if path.read_text(encoding='utf-8') == expected:
            return
        raise RuntimeError(f'refusing to overwrite nonidentical {path}')
    with path.open('x', encoding='utf-8', newline='') as f:
        f.write(expected)


def write_json(path, obj):
    path = Path(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')


def file_record(path):
    path = Path(path)
    s = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return {'path': str(path), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'sha256': h.hexdigest()}


def main():
    OUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    required = [PRIOR / 'dataset_manifest.csv', PRIOR / 'dataset_audit.json',
                PRIOR / 'label_provenance.json', PRIOR / 'baseline_integrity_check.json',
                SOURCE, LINEAGE]
    before = [file_record(p) for p in required]
    if json.loads((PRIOR / 'baseline_integrity_check.json').read_text())['status'] != 'PASS':
        raise RuntimeError('prior APK audit did not pass integrity')
    prior = json.loads((PRIOR / 'dataset_audit.json').read_text())
    if prior['downloaded_apk_count'] != 3989 or prior['status'] != 'AUDIT_COMPLETE':
        raise RuntimeError('unexpected prior APK audit scope')
    newer = read_csv(PRIOR / 'dataset_manifest.csv')
    old_info = {r['sha256'].lower(): r for r in read_csv(SOURCE) if r['year'] == '2017'}
    old_paths = {r['sha256'].lower(): r for r in read_csv(LINEAGE)}
    if len(old_info) != 799 or len(old_paths) != 799:
        raise RuntimeError('old 2017 lineage is not 799 unique samples')
    manifest = []
    for sid, r in old_info.items():
        path = old_paths[sid].get('absolute_path', '')
        if not Path(path).is_file():
            raise RuntimeError('missing old 2017 APK: ' + sid)
        label = r['label']
        manifest.append({'sha256': sid, 'year': '2017', 'source_dataset': 'OLD_2017',
                         'apk_path': path, 'file_size': Path(path).stat().st_size,
                         'raw_label': label, 'normalized_label': label,
                         'label_provenance': 'existing frozen old-2017 source lineage',
                         'label_confidence_or_quality': 'historical-protocol',
                         'p31_status': 'PREEXISTING', 'c43_status': 'PREEXISTING',
                         'duplicate_status': 'UNIQUE_OLD', 'eligible_for_train': 1,
                         'eligible_for_evaluation': 0, 'reason': ''})
    counts = Counter()
    overlaps = []
    for r in newer:
        sid, year, label = r['sha256'].lower(), int(r['year']), r['label']
        reasons = set(filter(None, r.get('exclusion_reasons', '').split('|')))
        duplicate = 'OLD_2017_DUPLICATE' if year == 2017 and sid in old_info else 'UNIQUE_NEW'
        if duplicate != 'UNIQUE_NEW':
            if label != old_info[sid]['label']:
                reasons.add('LABEL_CONFLICT_WITH_OLD')
            overlaps.append({'sha256': sid, 'year': year, 'type': duplicate,
                             'new_label': label, 'old_label': old_info[sid]['label']})
        if r.get('content_sha256', '').lower() != sid or r.get('dex_error'):
            reasons.add('INVALID_APK_OR_DEX')
        severe = {'LABEL_CONFLICT_WITH_OLD', 'CONTENT_SHA_MISMATCH', 'DEX_UNREADABLE',
                  'INVALID_APK_OR_DEX', 'NEW_TRAIN_TEST_SHA_OVERLAP',
                  'HISTORICAL_LABEL_CONFLICT'}
        train = year == 2017 and duplicate == 'UNIQUE_NEW' and label in ('0', '1') and not reasons.intersection(severe)
        test = year >= 2018 and label in ('0', '1') and not reasons.intersection(severe)
        counts[f'{year}_{label}'] += 1
        manifest.append({'sha256': sid, 'year': year, 'source_dataset': 'AZ_2017_2022_Random15K',
                         'apk_path': r['absolute_path'], 'file_size': r['apk_bytes'],
                         'raw_label': label, 'normalized_label': label,
                         'label_provenance': 'AndroZoo VT count: 0 benign, >=4 malware; dex_date year',
                         'label_confidence_or_quality': 'AV-consensus-proxy-not-behavior-ground-truth',
                         'p31_status': 'PENDING', 'c43_status': 'PENDING' if train else 'ROUTE_DEPENDENT',
                         'duplicate_status': duplicate, 'eligible_for_train': int(train),
                         'eligible_for_evaluation': int(test), 'reason': '|'.join(sorted(reasons))})
    union = [r for r in manifest if int(r['eligible_for_train'])]
    by_sha = defaultdict(list)
    for r in union: by_sha[r['sha256']].append(r)
    if any(len(v) != 1 for v in by_sha.values()):
        raise RuntimeError('union train SHA overlap')
    by_label = Counter(r['normalized_label'] for r in union)
    if not {'0', '1'}.issubset(by_label):
        raise RuntimeError('union has no legal binary training pool')
    write_csv(OUT / 'dataset_manifest.csv', manifest)
    write_csv(OUT / 'union_2017_manifest.csv', union)
    write_csv(OUT / 'overlap_audit.csv', overlaps)
    write_csv(OUT / 'dataset_summary.csv',
              [{'year': y, 'label': l, 'count': counts[f'{y}_{l}']} for y in range(2017, 2023) for l in ('0', '1')])
    write_csv(OUT / 'class_distribution.csv',
              [{'scope': scope, 'label': l, 'n': sum(r['normalized_label'] == l for r in rows)}
               for scope, rows in [('OLD_2017', manifest[:799]), ('UNION_2017', union)] for l in ('0', '1')])
    write_json(OUT / 'label_audit.json', {'label_rule': 'VT detection count 0/≥4',
                'label_ground_truth': False, 'source_counts': dict(counts),
                'old_new_sha_overlaps': len(overlaps), 'union_label_counts': dict(by_label),
                'union_n': len(union), 'historical_test_not_blind': True})
    write_json(OUT / 'dataset_audit.json', {'status': 'PASS', 'prior_content_audit': str(PRIOR),
                'new_downloaded': len(newer), 'union_n': len(union), 'union_classes': dict(by_label),
                'prior_exact_dex_duplicate_groups': prior['exact_dex_duplicate_groups']})
    after = [file_record(p) for p in required]
    changed = [a['path'] for a, b in zip(before, after) if a != b]
    write_json(OUT / 'baseline_integrity_check.json',
               {'status': 'PASS' if not changed else 'FAIL', 'asset_count': len(required),
                'changed': changed, 'before': before, 'after': after,
                'scope': 'listed reused audit, lineage and manifest files; prior audit content-hashed all 3989 APKs'})
    if changed: raise RuntimeError('reused asset changed during audit')
    print('EXPANDED_AUDIT_PASS', dict(by_label), 'new=', len(newer), flush=True)


if __name__ == '__main__': main()
