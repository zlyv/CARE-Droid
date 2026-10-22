#!/usr/bin/env python3
"""2017--2021 full-cohort Host-local CFG Witness audit.

This package freezes the existing Witness definition, scans all 2,788 APKs,
and reports both the complete cohort and the deployed routed subset. It fits no
model and never modifies an APK or a formal CARE-Droid asset.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT / 'scripts/care-host-cfg-rolegraph-witness-all-years-v2'
PARENT = ROOT / 'scripts/care-local-control-constraint-feasibility-v2'
OUT = ROOT / 'artifacts/care_cfg_witness_all_years_v2'
ENDPOINT = ROOT / 'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz'
SOURCE_LINEAGE = ROOT / 'artifacts/care-source-recovery-stage2only-v1/source_lineage_2017.csv'
FUTURE_LINEAGE = ROOT / 'artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_lineage_2018_2021_exact.csv'
SOURCE_PRED = ROOT / 'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv'
FUTURE_PRED = ROOT / 'outputs/care-p31-classifier-family-stage2-v1/frozen_future_stage2_predictions.csv'
SOURCE_INTER = ROOT / 'artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz'
FUTURE_INTER = ROOT / 'artifacts/care-droid-formal-diagnostics-v1/cache/future_interz.npz'
MAPPING = ROOT / 'scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml'
HOSTREL = ROOT / 'scripts/care-di25-compact43-relationquery-source-v1/hostrel_features.py'
EXPECTED_BY_YEAR = {2017: 799, 2018: 793, 2019: 399, 2020: 397, 2021: 400}
EXPECTED_ROUTED = {2017: 200, 2018: 198, 2019: 100, 2020: 99, 2021: 100}


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    def encode_nonstandard(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f'Object of type {type(item).__name__} is not JSON serializable')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               default=encode_nonstandard) + '\n', encoding='utf-8')
    temp.replace(path)


def atomic_csv(path, values):
    values = list(values); fields = list(dict.fromkeys(k for r in values for k in r)) or ['status']
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(values)
    temp.replace(path)


def digest(path):
    path = Path(path); h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''): h.update(chunk)
    return {'path': str(path.resolve()), 'size': path.stat().st_size, 'sha256': h.hexdigest()}


def load_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle: return json.load(handle)


def load_predictions():
    selected = []
    for path, default_year in ((SOURCE_PRED, 2017), (FUTURE_PRED, None)):
        for row in rows(path):
            if row.get('arm') != 'SVM': continue
            selected.append({'sha256': row['sha256'].lower(), 'year': int(row.get('year') or default_year),
                'label': int(float(row['label'])), 'routed': int(float(row['routed'])),
                'stage1_prob': float(row['stage1_prob']), 'expert_prob': float(row['expert_prob']),
                'stage2_prob': float(row['stage2_prob'])})
    if len(selected) != 2788 or len({r['sha256'] for r in selected}) != 2788:
        raise RuntimeError('prediction membership is not 2,788 unique APKs')
    if Counter(r['year'] for r in selected) != Counter(EXPECTED_BY_YEAR):
        raise RuntimeError('year membership changed')
    routed = Counter(r['year'] for r in selected if r['routed'])
    if routed != Counter(EXPECTED_ROUTED): raise RuntimeError('frozen routed membership changed')
    return selected


def load_paths(wanted):
    found = {}
    for path in (SOURCE_LINEAGE, FUTURE_LINEAGE):
        for row in rows(path):
            sid = row['sha256'].lower(); apk = Path(row['absolute_path'])
            if sid in wanted and sid not in found and apk.is_file(): found[sid] = str(apk.resolve())
    if set(found) != wanted:
        raise RuntimeError(f'APK path coverage {len(found)}/{len(wanted)} missing={sorted(wanted-set(found))[:10]}')
    return found


def endpoint_records(wanted):
    result = {}
    with gzip.open(ENDPOINT, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip(): continue
            row = json.loads(line); sid = str(row.get('sha256', '')).lower()
            if sid in wanted and row.get('status') == 'OK':
                if sid in result: raise RuntimeError('duplicate endpoint ' + sid)
                result[sid] = {'payload': row['payload'], 'summary': row.get('summary', {})}
    if set(result) != wanted: raise RuntimeError(f'endpoint coverage {len(result)}/{len(wanted)}')
    return result


def frozen_view(record):
    payload = record['payload']; ids = payload['node_ids']; roles = payload['node_role_ids']
    base = payload['node_base_features']; edges = payload['edge_index']
    if not (len(ids) == len(roles) == len(base)): raise RuntimeError('endpoint schema mismatch')
    pairs = {}
    for a, b in zip(*edges):
        a, b = int(a), int(b)
        if base[b][4] > .5 and roles[b]:
            pairs[str(int(ids[a])) + ':' + str(int(ids[b]))] = list(map(int, roles[b]))
    return {'pairs': pairs, 'host_ids': sorted(set(k.split(':')[0] for k in pairs)),
            'summary': record.get('summary', {})}


def preflight():
    if OUT.exists(): raise RuntimeError('artifact directory exists; refusing overwrite')
    OUT.mkdir(parents=True); (OUT / 'cache').mkdir(); (OUT / 'status').mkdir(); (OUT / 'logs').mkdir(); (OUT / 'frozen').mkdir()
    prediction = load_predictions(); wanted = {r['sha256'] for r in prediction}
    paths = load_paths(wanted); endpoints = endpoint_records(wanted)
    required = [HERE/'pipeline.py', HERE/'scan_one.py', HERE/'witness_core.py',
        PARENT/'control_core.py', PARENT/'dex_adapter.py', PARENT/'common.py', ENDPOINT,
        SOURCE_LINEAGE, FUTURE_LINEAGE, SOURCE_PRED, FUTURE_PRED, SOURCE_INTER,
        FUTURE_INTER, MAPPING, HOSTREL]
    atomic_json(OUT / 'asset_manifest_before.json', [digest(p) for p in required])
    atomic_csv(OUT / 'selection_label_free.csv',
        [{'sha256': r['sha256'], 'year': r['year'], 'routed': r['routed']} for r in prediction])
    atomic_json(OUT / 'apk_paths.json', paths)
    for n, sid in enumerate(sorted(wanted), 1):
        atomic_json(OUT / 'frozen' / f'{sid}.json', frozen_view(endpoints[sid]))
        if n % 250 == 0: print(f'CFGW_FROZEN {n}/2788', flush=True)
    atomic_json(OUT / 'preflight.json', {'status': 'PASS', 'APKs': 2788,
        'by_year': EXPECTED_BY_YEAR, 'routed_by_year': EXPECTED_ROUTED,
        'labels_used_for_scan': False, 'model_training_performed': False,
        'family_labels_available': False,
        'family_concentration_claim': 'not testable without a reliable family contract'})
    print('CFGW_ALL_PREFLIGHT_PASS apks=2788 routed=697 labels_for_scan=false', flush=True)


def worker(row, paths):
    sid = row['sha256']; output = OUT/'cache'/f'{sid}.json.gz'; status = OUT/'status'/f'{sid}.json'
    cmd = [sys.executable, str(HERE/'scan_one.py'), '--apk', paths[sid], '--sha', sid,
        '--year', str(row['year']), '--mapping', str(MAPPING), '--frozen', str(OUT/'frozen'/f'{sid}.json'),
        '--output', str(output), '--status', str(status)]
    started = time.monotonic()
    with (OUT/'logs'/f'{sid}.log').open('a', encoding='utf-8') as log:
        try: process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        except subprocess.TimeoutExpired:
            atomic_json(status, {'sha256': sid, 'status': 'TIMEOUT', 'wall_seconds': time.monotonic()-started})
            return json.loads(status.read_text())
    if status.is_file(): return json.loads(status.read_text())
    return {'sha256': sid, 'status': 'ERROR', 'reason': f'worker exit {process.returncode}'}


def scan_all(workers):
    selection = rows(OUT/'selection_label_free.csv'); paths = json.loads((OUT/'apk_paths.json').read_text())
    smoke = sorted(selection, key=lambda r:r['sha256'])[:5]
    for n, row in enumerate(smoke, 1):
        result = worker(row, paths)
        if result.get('status') != 'OK': raise RuntimeError('smoke failed ' + repr(result))
        print(f'CFGW_ALL_SMOKE {n}/5', flush=True)
    def completed(row):
        status_path = OUT/'status'/f"{row['sha256']}.json"
        cache_path = OUT/'cache'/f"{row['sha256']}.json.gz"
        if not status_path.is_file() or not cache_path.is_file():
            return False
        try:
            return json.loads(status_path.read_text()).get('status') == 'OK'
        except (OSError, ValueError):
            return False
    pending = [r for r in selection if not completed(r)]
    done = len(selection)-len(pending); started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(worker, r, paths): r for r in pending}
        for job in concurrent.futures.as_completed(jobs):
            result = job.result(); done += 1
            if done % 10 == 0 or done == len(selection): print(f'CFGW_ALL_SCAN {done}/2788 status={result.get("status")}', flush=True)
    statuses = [json.loads((OUT/'status'/f"{r['sha256']}.json").read_text()) for r in selection]
    atomic_csv(OUT/'scan_status.csv', statuses)
    atomic_json(OUT/'runtime.json', {'wall_seconds': time.monotonic()-started, 'workers': workers,
        'gpu_used': False, 'models_trained': 0, 'APKs_scanned': len(selection)})


def auc(y, score):
    y = np.asarray(y, int); score = np.asarray(score, float)
    if len(np.unique(y)) < 2: return None
    pos, neg = score[y == 1], score[y == 0]
    return float(sum((p > neg).sum() + .5*(p == neg).sum() for p in pos) / (len(pos)*len(neg)))


def load_inter():
    result = {}
    for path in (SOURCE_INTER, FUTURE_INTER):
        with np.load(path, allow_pickle=True) as data:
            for sid, vector in zip(data['sha256'].astype(str), data['inter_z_clipped']):
                result[sid.lower()] = np.asarray(vector).reshape(31, 31)
    if len(result) != 2788: raise RuntimeError('InterZ membership changed')
    return result


def apk_summary(scan, year, routed, inter):
    hosts, events, pairs = scan['hosts'], scan['events'], scan['pairs']
    relation = Counter(p['relation'] for p in pairs); witness = set()
    for pair in pairs:
        a, b = int(pair['role_a']), int(pair['role_b'])
        if pair['relation'] in ('A_TO_B','BIDIRECTIONAL'): witness.add((a,b))
        if pair['relation'] in ('B_TO_A','BIDIRECTIONAL'): witness.add((b,a))
    unique = sum(abs(float(inter[a,b])) < 1e-12 for a,b in witness)
    return {'sha256': scan['sha256'], 'year': year, 'routed': routed,
        'scan_status': scan['status'], 'frozen_hosts': scan['frozen_hosts'],
        'recovered_hosts': scan['recovered_hosts'], 'parsed_hosts': sum(h['cfg_status']=='PARSED' for h in hosts),
        'hosts_with_two_calls': sum(h['aligned_sensitive_calls']>=2 for h in hosts),
        'hosts_with_two_roles': sum(h['distinct_roles']>=2 for h in hosts),
        'hosts_with_cycle': sum(bool(h['has_reachable_cycle']) for h in hosts),
        'aligned_sensitive_calls': len(events), 'ordered_witness_role_pairs': len(witness),
        'a_to_b_pairs': relation['A_TO_B'] + relation['B_TO_A'],
        'bidirectional_pairs': relation['BIDIRECTIONAL'], 'neither_pairs': relation['NEITHER'],
        'sensitive_loop_calls': sum(bool(e['in_loop']) for e in events),
        'witness_pairs_interz_zero': unique,
        'witness_unique_fraction_vs_interz': unique/max(len(witness),1),
        'witness_apk': int(bool(witness)), 'pair_cap_methods': scan['pair_cap_methods']}


def verify_assets():
    before = json.loads((OUT/'asset_manifest_before.json').read_text()); changed = []
    for item in before:
        try: now = digest(item['path'])
        except OSError as exc: changed.append({'path': item['path'], 'error': repr(exc)}); continue
        if now != item: changed.append({'before': item, 'after': now})
    return {'status': 'PASS' if not changed else 'FAIL', 'checked': len(before), 'changed': changed,
        'scope': 'named code, endpoint cache, lineages, predictions and InterZ assets; APKs opened read-only'}


def summarize():
    prediction = load_predictions(); inter = load_inter(); apk_rows = []; representative = []
    for pred in prediction:
        path = OUT/'cache'/f"{pred['sha256']}.json.gz"
        if not path.is_file(): continue
        scan = load_gz(path); row = apk_summary(scan, pred['year'], pred['routed'], inter[pred['sha256']])
        final_prob = pred['stage2_prob']; label = pred['label']; final_label = int(final_prob >= .5)
        row.update(label=label, stage1_prob=pred['stage1_prob'], expert_prob=pred['expert_prob'],
            stage2_prob=final_prob, prediction_group=('TP' if label and final_label else 'FN' if label else 'FP' if final_label else 'TN'))
        apk_rows.append(row)
        if len(representative) < 40 and row['witness_pairs_interz_zero']:
            representative.append({'sha256': pred['sha256'], 'year': pred['year'], 'routed': pred['routed'],
                'witness_pairs_interz_zero': row['witness_pairs_interz_zero'],
                'ordered_witness_role_pairs': row['ordered_witness_role_pairs']})
    atomic_csv(OUT/'cfg_witness_apk.csv', apk_rows); atomic_csv(OUT/'representative_examples.csv', representative)
    if len(apk_rows) < .95 * 2788: raise RuntimeError(f'scan coverage too low {len(apk_rows)}/2788')
    features = ('hosts_with_two_calls','hosts_with_two_roles','ordered_witness_role_pairs','a_to_b_pairs',
        'bidirectional_pairs','neither_pairs','sensitive_loop_calls','witness_pairs_interz_zero',
        'witness_unique_fraction_vs_interz','frozen_hosts','aligned_sensitive_calls')
    control = {'frozen_hosts','aligned_sensitive_calls'}; direction = []
    for scope, scoped in (('ALL', apk_rows), ('ROUTED', [r for r in apk_rows if r['routed']])):
        for year in (2017,2018,2019,2020,2021,'POOLED'):
            year_rows = scoped if year == 'POOLED' else [r for r in scoped if r['year']==year]
            definitions = [('LABEL_MALWARE_vs_BENIGN', lambda r:r['label']==1, lambda r:True),
                ('FN_vs_TN', lambda r:r['prediction_group']=='FN', lambda r:r['prediction_group'] in ('FN','TN')),
                ('TP_vs_FP', lambda r:r['prediction_group']=='TP', lambda r:r['prediction_group'] in ('TP','FP'))]
            for name, positive, include in definitions:
                sample = [r for r in year_rows if include(r)]; y = [int(positive(r)) for r in sample]
                for feature in features:
                    direction.append({'scope': scope, 'year': year, 'direction': name, 'feature': feature,
                        'n': len(sample), 'positive_n': sum(y), 'AUC': auc(y, [r[feature] for r in sample]),
                        'is_size_control': int(feature in control)})
    atomic_csv(OUT/'witness_direction_auc.csv', direction)
    coverage = []
    for scope, scoped in (('ALL', apk_rows), ('ROUTED', [r for r in apk_rows if r['routed']])):
        for year in (2017,2018,2019,2020,2021,'POOLED'):
            sample = scoped if year == 'POOLED' else [r for r in scoped if r['year']==year]
            total_witness = sum(r['ordered_witness_role_pairs'] for r in sample); unique = sum(r['witness_pairs_interz_zero'] for r in sample)
            coverage.append({'scope': scope, 'year': year, 'APKs': len(sample),
                'witness_APKs': sum(r['witness_apk'] for r in sample),
                'witness_APK_rate': sum(r['witness_apk'] for r in sample)/max(len(sample),1),
                'ordered_witness_role_pairs': total_witness, 'witness_pairs_absent_from_InterZ': unique,
                'unique_fraction_vs_InterZ': unique/max(total_witness,1),
                'frozen_hosts': sum(r['frozen_hosts'] for r in sample),
                'parsed_host_rate': sum(r['parsed_hosts'] for r in sample)/max(sum(r['recovered_hosts'] for r in sample),1)})
    atomic_csv(OUT/'coverage_by_year_scope.csv', coverage)
    candidates = []
    for name in ('LABEL_MALWARE_vs_BENIGN','FN_vs_TN','TP_vs_FP'):
        pooled = [r for r in direction if r['scope']=='ROUTED' and r['year']=='POOLED' and r['direction']==name]
        base = max(abs(float(r['AUC'])-.5) for r in pooled if r['is_size_control'] and r['AUC'] is not None)
        for item in pooled:
            if item['is_size_control'] or item['AUC'] is None or abs(float(item['AUC'])-.5) < base+.03: continue
            sign = np.sign(float(item['AUC'])-.5)
            years = [r for r in direction if r['scope']=='ROUTED' and r['year']!='POOLED'
                and r['direction']==name and r['feature']==item['feature'] and r['AUC'] is not None]
            aligned = sum(np.sign(float(r['AUC'])-.5)==sign for r in years)
            if aligned >= 3: candidates.append({**item, 'aligned_years': aligned, 'control_abs_AUC': base})
    integrity = verify_assets(); atomic_json(OUT/'baseline_integrity_check.json', integrity)
    if integrity['status'] != 'PASS': raise RuntimeError('protected asset changed')
    decision = {'status': 'SUPPORTED_FOR_FIXED_ABLATION' if candidates else 'NO_STABLE_CROSSYEAR_CLASS_DIRECTION',
        'APKs_scanned': len(apk_rows), 'routed_APKs_scanned': sum(r['routed'] for r in apk_rows),
        'candidate_witness_features': candidates, 'class_direction_beyond_size_control': bool(candidates),
        'family_concentration_tested': False,
        'family_boundary': 'No reliable family labels were available; cross-year consistency tests temporal breadth, not family independence.',
        'model_training_performed': False, 'formal_model_changed': False,
        'semantic_boundary': 'Static local CFG reachability is possible path structure, not runtime execution or malicious intent.',
        'next_action': ('One fixed-capacity ablation may be specified; no Witness-rule tuning.' if candidates else
            'Stop CFG Witness model integration; larger temporal coverage did not establish independent direction.')}
    atomic_json(OUT/'final_decision.json', decision)
    report = ['# CARE-Droid 2017—2021全量Host局部CFG Witness审计','',f"最终状态：**{decision['status']}**。",'',
        f"- 成功汇总APK：{len(apk_rows)}/2788；路由子集：{sum(r['routed'] for r in apk_rows)}/697。",
        '- 全量与路由子集分别统计，不把Host、Role对或年份当作独立APK。',
        '- Witness定义与2017 Pilot完全相同；未训练模型、未搜索Witness规则。',
        '- 跨年份可以检验时间稳定性，但没有可靠家族标签，不能据此证明或否定家族集中假设。','',
        '## 年份/口径覆盖','']
    for row in coverage:
        if row['year'] != 'POOLED':
            report.append(f"- {row['scope']} {row['year']}：APK={row['APKs']}，Witness率={row['witness_APK_rate']:.3f}，InterZ未覆盖比例={row['unique_fraction_vs_InterZ']:.3f}。")
    report += ['', '## 判定','']
    if candidates:
        for item in candidates: report.append(f"- {item['direction']} / {item['feature']}：pooled AUC={item['AUC']:.4f}，同方向年份={item['aligned_years']}/5。")
    else: report.append('- 没有Witness特征在路由总体中超过规模/调用数对照0.03，并保持至少3/5年份同方向。')
    report += ['', '因此本审计只回答跨年份现象是否稳定；不能把结构重合归因于家族集中，也不能把Witness解释为恶意语义。','']
    (OUT/'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print('CFGW_ALL_SUMMARY_DONE status=' + decision['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('preflight','scan','summarize'))
    parser.add_argument('--workers', type=int, default=4); args = parser.parse_args()
    if args.phase == 'preflight': preflight()
    elif args.phase == 'scan': scan_all(args.workers)
    else: summarize()


if __name__ == '__main__': main()
