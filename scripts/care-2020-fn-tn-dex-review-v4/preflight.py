#!/usr/bin/env python3
"""Organizer-only preflight. Never copy identities or labels into the blind area."""
import csv
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path('/home/boyuanjie/projects/tasa-droid')
SOURCE = ROOT / 'outputs/care_crossyear_layerwise_diagnosis_v2/per_apk_error_contract.csv'
OLD = ROOT / 'outputs/care-2020-fn-tn-dex-review-v2'
OUT = ROOT / 'outputs/care-2020-fn-tn-dex-review-v4'
EXPECTED = {('1', 'UNFIXED_FN'): 24, ('1', 'FIXED_FN'): 2,
            ('0', 'STABLE_CORRECT'): 48, ('0', 'BROKEN_TN'): 3}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def exclusive_json(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')

def main():
    if OUT.exists():
        raise RuntimeError(f'FAIL_CLOSED: output already exists: {OUT}')
    packets = sorted((OLD / 'blind_packets').glob('M[0-9][0-9].json'))
    if [p.stem for p in packets] != [f'M{i:02d}' for i in range(1, 9)]:
        raise RuntimeError('FAIL_CLOSED: blind packet IDs are not M01..M08')
    form = OLD / 'blind_review_form.csv'
    with SOURCE.open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['tag'] == 'T3_C1' and r['year'] == '2020'
                and r['routed'] == '1' and r['stage1_prediction'] == '0']
    from collections import Counter
    counts = Counter((r['label'], r['category']) for r in rows)
    unique = len({r['sha256'] for r in rows})
    actual = {f'{k[1]}_label{k[0]}': v for k, v in sorted(counts.items())}
    matched = len(rows) == unique == 77 and dict(counts) == EXPECTED
    with form.open(newline='', encoding='utf-8') as f:
        form_rows = list(csv.DictReader(f))
        headers = list(form_rows[0]) if form_rows else []
    if [r['case_id'] for r in form_rows] != [p.stem for p in packets]:
        raise RuntimeError('FAIL_CLOSED: blank form IDs differ from packet IDs')
    if any(any(v.strip() for k, v in r.items() if k != 'case_id') for r in form_rows):
        raise RuntimeError('FAIL_CLOSED: source form is not blank')
    sources = [SOURCE, form, *packets]
    before = {str(p): digest(p) for p in sources}
    OUT.mkdir(mode=0o700)
    organizer = OUT / 'organizer_only'
    organizer.mkdir(mode=0o700)
    blind = OUT / 'blind_workspace'
    blind.mkdir(mode=0o700)
    (blind / 'blind_packets').mkdir(mode=0o700)
    for p in packets:
        shutil.copyfile(p, blind / 'blind_packets' / p.name)
    shutil.copyfile(form, blind / 'blind_review_working.csv')
    shutil.copyfile(pathlib.Path(__file__).with_name('blind_audit.py'), blind / 'blind_audit.py')
    shutil.copyfile(pathlib.Path(__file__).with_name('freeze_review.py'), blind / 'freeze_review.py')
    exclusive_json(organizer / 'cohort_preflight_restricted.json', {
        'status': 'PASS' if matched else 'COUNT_MISMATCH', 'total': len(rows),
        'unique_sha_count': unique, 'actual_counts': actual,
        'expected_counts': {f'{k[1]}_label{k[0]}': v for k, v in EXPECTED.items()},
        'note': 'Organizer-only; never mount this path in the blind sandbox.'})
    exclusive_json(organizer / 'input_hashes_before.json', before)
    exclusive_json(OUT / 'preflight_status.json', {
        'status': 'PASS' if matched else 'COUNT_MISMATCH',
        'cohort_size': len(rows), 'unique_sha_count': unique,
        'blank_form_headers': headers, 'blind_packets': len(packets),
        'blind_boundary': 'bubblewrap blind process may see /review only, not project or organizer_only'})
    if not matched:
        raise RuntimeError('FAIL_CLOSED: 77-case composition mismatch; see organizer-only record')
    print('PREFLIGHT_PASS cohort=77 packets=8 unique_sha=77')

if __name__ == '__main__':
    main()
