#!/usr/bin/env python3
"""Read-only AndroZoo/new-vs-historical audit for CARE-Droid expanded 2017 study.

This phase does not extract model features, fit models, or mutate source assets.
All results are written beneath the new experiment output directory only.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import zipfile

ROOT = pathlib.Path('/home/boyuanjie/projects/tasa-droid')
DATA = pathlib.Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/datasets/android/AndroZoo/AZ_2017_2022_Random15K')
SOURCE = pathlib.Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/datasets/android/AndroZoo/final_longitudinal_2015_2024/manifests')
OUT = ROOT / 'outputs/care_expanded2017_crossyear_v1'
OLD = ROOT / 'outputs/care_multiyear_development_v1/sample_manifest.csv'
HISTORICAL = {
    'OLD_DEV_2017_2021': OLD,
    'MalEval2026': ROOT / 'outputs/care-maleval2026-external-v1/dataset_manifest_label_sealed.csv',
    'CICMalDroid2020': ROOT / 'outputs/care-cicmaldroid2020-direct-route-budget-v3/dataset_manifest_label_sealed.csv',
}
SHA = re.compile(r'^[a-f0-9]{64}$')
YEARS = range(2017, 2023)

def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def write_json(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write('\n')

def dex_fingerprint(path):
    parts = []
    try:
        with zipfile.ZipFile(path) as z:
            for n in sorted(x for x in z.namelist() if re.fullmatch(r'classes\d*\.dex', x)):
                with z.open(n) as f:
                    h = hashlib.sha256()
                    for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                        h.update(block)
                    parts.append((n, h.hexdigest()))
    except (zipfile.BadZipFile, OSError, RuntimeError, ValueError) as exc:
        return '', 0, type(exc).__name__ + ': ' + str(exc)[:200]
    if not parts:
        return '', 0, 'NO_CLASSES_DEX'
    serial = json.dumps(parts, separators=(',', ':')).encode()
    return hashlib.sha256(serial).hexdigest(), len(parts), ''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit-only', action='store_true', required=True)
    args = ap.parse_args()
    if OUT.exists():
        raise SystemExit('FAIL_CLOSED: output directory already exists; choose fresh version')
    selected = DATA / 'manifests/selected_all.csv'
    sampling = DATA / 'manifests/sampling_audit.json'
    source_manifest = SOURCE / 'manifest_primary.csv'
    source_summary = SOURCE / 'sampling_summary.json'
    required = [selected, sampling, source_manifest, source_summary, *HISTORICAL.values()]
    if any(not p.is_file() for p in required):
        raise SystemExit('FAIL_CLOSED: missing required bounded manifest: ' + repr([str(p) for p in required if not p.is_file()]))
    if any(not (DATA / f'apks/year{year}').is_dir() for year in YEARS):
        raise SystemExit('FAIL_CLOSED: expected year directory missing')
    OUT.mkdir(mode=0o700)
    before = {str(p): hash_file(p) for p in required}
    write_json(OUT / 'asset_hashes_before.json', before)
    selected_rows = read_csv(selected)
    by_sha = collections.defaultdict(list)
    for r in selected_rows:
        by_sha[r['sha256'].strip().lower()].append(r)
    historical = {}
    hist_labels = collections.defaultdict(set)
    for scope, path in HISTORICAL.items():
        rows = read_csv(path)
        historical[scope] = {r['sha256'].strip().lower() for r in rows if SHA.fullmatch(r.get('sha256', '').strip().lower())}
        for r in rows:
            s = r.get('sha256', '').strip().lower()
            if SHA.fullmatch(s) and r.get('label', '').strip() in ('0', '1'):
                hist_labels[s].add(r['label'].strip())
    sampled = json.loads(sampling.read_text())
    source_protocol = json.loads(source_summary.read_text())
    if hash_file(source_manifest) != sampled.get('source_manifest_sha256'):
        raise RuntimeError('FAIL_CLOSED: selected source manifest hash does not match sampling audit')
    manifest = []
    dedup = []
    counts = collections.Counter()
    reasons = collections.Counter()
    found_paths = set()
    start = time.monotonic()
    for year in YEARS:
        base = DATA / f'apks/year{year}'
        paths = sorted(base.glob('*/*.apk'))
        print(f'AUDIT_YEAR_START year={year} files={len(paths)}', flush=True)
        for j, p in enumerate(paths, 1):
            stat = p.stat()
            expected = p.stem.lower()
            matches = by_sha.get(expected, [])
            m = matches[0] if len(matches) == 1 else {}
            label = m.get('label', '')
            folder = p.parent.name.lower()
            selected_year = m.get('year', '')
            dex_date = m.get('dex_date', '')
            size_ok = stat.st_size > 0 and str(stat.st_size) == m.get('apk_size', '')
            sha_actual = hash_file(p) if stat.st_size else ''
            sha_ok = sha_actual == expected and SHA.fullmatch(expected) is not None
            dex_hash, dex_count, dex_error = dex_fingerprint(p) if sha_ok else ('', 0, 'APK_SHA_FAILED')
            old_scopes = [scope for scope, values in historical.items() if expected in values]
            conflict = bool(hist_labels.get(expected) and label not in hist_labels[expected])
            row_reasons = []
            if len(matches) != 1: row_reasons.append('SELECTION_AMBIGUOUS_OR_MISSING')
            if not size_ok: row_reasons.append('SIZE_MISMATCH_OR_EMPTY')
            if not sha_ok: row_reasons.append('CONTENT_SHA_MISMATCH')
            if selected_year != str(year): row_reasons.append('YEAR_MANIFEST_MISMATCH')
            if not dex_date.startswith(f'{year}-'): row_reasons.append('DEX_DATE_YEAR_MISMATCH')
            if label not in ('0', '1') or folder != {'0': 'benign', '1': 'malware'}.get(label):
                row_reasons.append('LABEL_FOLDER_OR_MANIFEST_MISMATCH')
            if conflict: row_reasons.append('HISTORICAL_LABEL_CONFLICT')
            if dex_error: row_reasons.append('DEX_UNREADABLE')
            if year >= 2018 and 'OLD_DEV_2017_2021' in old_scopes:
                row_reasons.append('HISTORICAL_DEV_OVERLAP')
            if year >= 2018 and any(x in old_scopes for x in ('MalEval2026', 'CICMalDroid2020')):
                row_reasons.append('HISTORICAL_EXTERNAL_OVERLAP')
            for reason in row_reasons: reasons[reason] += 1
            counts[(year, label or 'UNKNOWN')] += 1
            found_paths.add(expected)
            d = {
                'scope': f'NEW_{year}', 'sha256': expected, 'absolute_path': str(p),
                'year': year, 'label': label, 'label_name': m.get('label_name', ''),
                'vt_detection': m.get('vt_detection', ''), 'vt_scan_date': m.get('vt_scan_date', ''),
                'dex_date': dex_date, 'pkg_name': m.get('pkg_name', ''),
                'apk_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'content_sha256': sha_actual, 'dex_fingerprint': dex_hash,
                'dex_count': dex_count, 'dex_error': dex_error,
                'historical_scopes': '|'.join(old_scopes), 'exclusion_reasons': '|'.join(row_reasons),
            }
            manifest.append(d)
            if j % 50 == 0 or j == len(paths):
                print(f'AUDIT_PROGRESS year={year} {j}/{len(paths)} seconds={time.monotonic()-start:.0f}', flush=True)
    new_by_sha = collections.defaultdict(list)
    dex_by_hash = collections.defaultdict(list)
    for r in manifest:
        new_by_sha[r['sha256']].append(r)
        if r['dex_fingerprint']:
            dex_by_hash[r['dex_fingerprint']].append(r)
    for r in manifest:
        reasons_here = set(filter(None, r['exclusion_reasons'].split('|')))
        sha = r['sha256']
        if len(new_by_sha[sha]) > 1: reasons_here.add('NEW_DATA_SHA_DUPLICATE')
        if r['year'] >= 2018 and any(x['year'] == 2017 for x in new_by_sha[sha]):
            reasons_here.add('NEW_TRAIN_TEST_SHA_OVERLAP')
        if r['year'] == 2017 and sha in historical['OLD_DEV_2017_2021']:
            reasons_here.add('OLD_NEW_2017_SHA_OVERLAP')
        if r['dex_fingerprint'] and len(dex_by_hash[r['dex_fingerprint']]) > 1:
            r['dex_exact_duplicate_count'] = len(dex_by_hash[r['dex_fingerprint']])
        else:
            r['dex_exact_duplicate_count'] = 1 if r['dex_fingerprint'] else 0
        r['exclusion_reasons'] = '|'.join(sorted(reasons_here))
        r['all_eligible_new'] = int(r['year'] in (2018, 2019, 2020) and not reasons_here.intersection({
            'SELECTION_AMBIGUOUS_OR_MISSING', 'SIZE_MISMATCH_OR_EMPTY', 'CONTENT_SHA_MISMATCH',
            'YEAR_MANIFEST_MISMATCH', 'DEX_DATE_YEAR_MISMATCH', 'LABEL_FOLDER_OR_MANIFEST_MISMATCH',
            'HISTORICAL_LABEL_CONFLICT', 'DEX_UNREADABLE', 'NEW_DATA_SHA_DUPLICATE',
            'NEW_TRAIN_TEST_SHA_OVERLAP'}))
        r['novel_test'] = int(r['all_eligible_new'] and not reasons_here.intersection({
            'HISTORICAL_DEV_OVERLAP', 'HISTORICAL_EXTERNAL_OVERLAP'}))
        dedup.append({'scope': r['scope'], 'sha256': sha, 'year': r['year'],
                      'label': r['label'], 'historical_scopes': r['historical_scopes'],
                      'dex_exact_duplicate_count': r['dex_exact_duplicate_count'],
                      'exclusion_reasons': r['exclusion_reasons'],
                      'all_eligible_new': r['all_eligible_new'], 'novel_test': r['novel_test']})
    fields = list(manifest[0]) if manifest else []
    write_csv(OUT / 'dataset_manifest.csv', manifest, fields)
    write_csv(OUT / 'deduplication_audit.csv', dedup, list(dedup[0]) if dedup else [])
    scope_sets = {'NEW_' + str(y): {r['sha256'] for r in manifest if r['year'] == y} for y in YEARS}
    scope_sets.update(historical)
    matrix = [{'scope_a': a, 'scope_b': b, 'sha_overlap': len(av & bv)}
              for a, av in scope_sets.items() for b, bv in scope_sets.items()]
    write_csv(OUT / 'dataset_overlap_matrix.csv', matrix, ['scope_a', 'scope_b', 'sha_overlap'])
    provenance = {
        'metadata_source': str(source_manifest),
        'metadata_sha256': before[str(source_manifest)],
        'selected_manifest': str(selected), 'selected_manifest_sha256': before[str(selected)],
        'year_field': 'dex_date (year field checked against it)',
        'label_rule': source_protocol.get('protocol', {}).get('labels'),
        'malware_threshold': source_protocol.get('malware_threshold'),
        'label_quality_boundary': 'AndroZoo VirusTotal detection-count proxy, not manual behavior ground truth; scan timing and AV consensus may differ by sample.',
        'historical_overlap_scope': list(HISTORICAL),
        'historical_overlap_limitation': 'Only listed historical manifests; cannot claim all prior experiments are covered.'}
    write_json(OUT / 'label_provenance.json', provenance)
    summary = {
        'status': 'AUDIT_COMPLETE', 'data_root': str(DATA), 'downloaded_apk_count': len(manifest),
        'counts_by_year_label': {f'{y}_{lab}': n for (y, lab), n in sorted(counts.items())},
        'reason_counts': dict(sorted(reasons.items())),
        'old_2017_count': sum(1 for r in read_csv(OLD) if r['year'] == '2017'),
        'new_2017_count': sum(r['year'] == 2017 for r in manifest),
        'union_2017_sha_count': len(scope_sets['NEW_2017'] | {r['sha256'].lower() for r in read_csv(OLD) if r['year'] == '2017'}),
        'novel_test_by_year': {str(y): sum(r['novel_test'] for r in manifest if r['year'] == y) for y in (2018, 2019, 2020)},
        'all_eligible_new_by_year': {str(y): sum(r['all_eligible_new'] for r in manifest if r['year'] == y) for y in (2018, 2019, 2020)},
        'exact_dex_duplicate_groups': sum(len(v) > 1 for v in dex_by_hash.values()),
        'near_duplicate_code_not_established': True,
        'runtime_seconds': round(time.monotonic() - start, 2),
    }
    write_json(OUT / 'dataset_audit.json', summary)
    after = {str(p): hash_file(p) for p in required}
    changed = [p for p in before if before[p] != after[p]]
    write_json(OUT / 'baseline_integrity_check.json', {
        'status': 'PASS' if not changed else 'FAIL', 'asset_count': len(required),
        'changed': changed, 'scope': 'source/selected manifests and three named historical sample manifests; APK content SHA verified individually but pre/post APK metadata was not resampled after audit'})
    hard_fail = bool(changed or any(r['exclusion_reasons'] for r in manifest if r['year'] == 2017))
    if summary['new_2017_count'] != 1910 or summary['old_2017_count'] != 799:
        hard_fail = True
    decision = {'status': 'AUDIT_GATE_FAILED' if hard_fail else 'AUDIT_COMPLETE_AWAITING_SMOKE_TEST',
                'phase': 'PHASE_0', 'training_started': False, 'feature_extraction_started': False,
                'reason': 'Dataset audit complete; no training or full extraction is permitted before compatibility self-test and source-only protocol validation.',
                'integrity': 'PASS' if not changed else 'FAIL'}
    write_json(OUT / 'final_decision.json', decision)
    (OUT / 'final_report.md').write_text(
        '# CARE-Droid expanded 2017 cross-year study — Phase 0\n\n'
        f"Current status: **{decision['status']}**. Audited {len(manifest)} downloaded APKs by content SHA256 "
        'and exact classes*.dex fingerprints. See dataset_audit.json, deduplication_audit.csv, '
        'dataset_overlap_matrix.csv, label_provenance.json and baseline_integrity_check.json.\n\n'
        'No model, threshold, router, prototype or expert was fitted. Historical overlap only covers '
        'the named manifests; near-duplicate code beyond exact DEX identity has not been established. '
        '2018–2021 were used in prior development and will not be described as untouched blind tests. '
        '2021/2022 each contain only two downloaded APKs and cannot support yearly BA.\n', encoding='utf-8')
    print('EXPANDED2017_PHASE0_STATUS=' + decision['status'], flush=True)

if __name__ == '__main__':
    main()
