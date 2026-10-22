#!/usr/bin/env python3
"""Frozen Internal-background self-reference audit.

Role in experiment: reconstruct the exact formal Internal column, perform a
single measurement intervention (leave the scored Host out of its own selected
cluster centroid), compare it with matched same-cluster peer removal, and replay
the already-trained M0 model. No APK is read and no model is fitted. Writes are
restricted to the new experiment output directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT / 'scripts/care-internal-self-reference-audit-v1'
OUT = ROOT / 'outputs/care_internal_self_reference_audit_v1'
BFA = ROOT / 'scripts/care-background-functional-attribution-v1'
HAIG = ROOT / 'scripts/care-p31rf-c43-haig-host-ablation-source-v1'
sys.path[:0] = [str(BFA), str(HAIG)]

SEEDS = (900000, 900001, 900002)
SPLITS = ('S1', 'S2', 'S3', 'S4', 'S5', 'T1', 'T2', 'T3', 'T4')
YEAR = {'S1': 2017, 'S2': 2017, 'S3': 2017, 'S4': 2017, 'S5': 2017,
        'T1': 2018, 'T2': 2019, 'T3': 2020, 'T4': 2021}
REPEATS = 10


def safe_scale(values, fallback=1.0):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float(fallback)
    # Exact frozen implementation uses median positive value; keep compatible.
    positive = finite[finite > 1e-8]
    if len(positive):
        return max(float(np.median(positive)), 1e-6)
    return max(float(fallback), 1e-6)


def pairwise_rms(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).mean(2), 0.0))


def selected_clusters(assignment, n, coverage):
    counts = np.bincount(assignment)
    ordered = np.argsort(-counts)
    target = max(1, int(math.ceil(float(coverage) * n)))
    selected, covered = [], 0
    for cluster in ordered:
        if counts[cluster] <= 0:
            continue
        selected.append(int(cluster))
        covered += int(counts[cluster])
        if covered >= target:
            break
    return selected


def self_exclusion(x, assignment, coverage, repeats, random_seed):
    """Return original, leave-self-out, and matched peer-removal Internal values.

    The selected dominant cluster set and the original bag scale remain frozen.
    This isolates target membership in its reference centroid. Hosts outside the
    selected reference are unchanged. Peer controls remove a different Host from
    the same selected cluster and never use labels.
    """
    x = np.asarray(x, np.float32)
    assignment = np.asarray(assignment, int)
    n = len(x)
    selected = selected_clusters(assignment, n, coverage)
    sums = {c: x[assignment == c].sum(0, dtype=np.float64) for c in selected}
    counts = {c: int(np.sum(assignment == c)) for c in selected}
    centers = np.asarray([sums[c] / counts[c] for c in selected], np.float32)
    distance_matrix = pairwise_rms(x, centers)
    raw_original = distance_matrix.min(1)
    scale = safe_scale(raw_original)
    original = np.log1p(raw_original / scale).astype(np.float32)
    loo = original.copy()
    random_values = np.tile(original[:, None], (1, repeats)).astype(np.float32)
    evaluable = np.ones(n, dtype=bool)
    self_in_reference = np.isin(assignment, np.asarray(selected, int))
    rng = np.random.RandomState(int(random_seed) & 0x7fffffff)

    for position, cluster in enumerate(selected):
        idx = np.flatnonzero(assignment == cluster)
        count = len(idx)
        if len(selected) > 1:
            other_raw = np.min(np.delete(distance_matrix[idx], position, axis=1), axis=1)
        else:
            other_raw = np.full(count, np.inf, dtype=np.float32)
        if count == 1:
            loo_raw = other_raw
            evaluable[idx] = np.isfinite(loo_raw)
            if evaluable[idx][0]:
                loo[idx] = np.log1p(loo_raw / scale)
            continue

        own_loo_center = (sums[cluster][None, :] - x[idx]) / float(count - 1)
        own_loo_raw = np.sqrt(np.maximum(((x[idx] - own_loo_center) ** 2).mean(1), 0.0))
        loo_raw = np.minimum(own_loo_raw, other_raw)
        loo[idx] = np.log1p(loo_raw / scale).astype(np.float32)

        # Each repeat is a derangement-like cyclic peer selection within cluster.
        for repeat in range(repeats):
            shift = int(rng.randint(1, count))
            donor = idx[(np.arange(count) + shift) % count]
            peer_center = (sums[cluster][None, :] - x[donor]) / float(count - 1)
            peer_raw = np.sqrt(np.maximum(((x[idx] - peer_center) ** 2).mean(1), 0.0))
            random_raw = np.minimum(peer_raw, other_raw)
            random_values[idx, repeat] = np.log1p(random_raw / scale).astype(np.float32)

    # Independent exact reconstruction of the frozen formula for fail-closed replay.
    replay = np.log1p(raw_original / safe_scale(raw_original)).astype(np.float32)
    return {'original_internal': original, 'loo_internal': loo,
            'random_internal': random_values, 'evaluable': evaluable,
            'self_in_reference': self_in_reference,
            'selected_clusters': len(selected), 'scale': float(scale),
            'exact_replay_max_abs': float(np.max(np.abs(replay - original)))}


def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            h.update(chunk)
    return {'path': str(path.resolve()), 'sha256': h.hexdigest(), 'size': path.stat().st_size}


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader(); writer.writerows(rows)


def write_text(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(value)


def source_assets():
    from bfa_core import asset_paths
    return list(asset_paths()) + [
        BFA / 'bfa_core.py', HAIG / 'haig_background.py',
        ROOT / 'scripts/run_care_joint_dualprototype_ablation.py']


def status_of(y, base, expert, routed):
    if not routed:
        return 'UNROUTED'
    stage1_ok = int(base >= .5) == int(y)
    expert_ok = int(expert >= .5) == int(y)
    if not stage1_ok and expert_ok: return 'FIXED'
    if not stage1_ok and not expert_ok: return 'UNFIXED'
    if stage1_ok and not expert_ok: return 'BROKEN'
    return 'KEPT'


def balanced_accuracy(y, p):
    y = np.asarray(y, int); pred = np.asarray(p) >= .5
    tpr = np.mean(pred[y == 1]) if np.any(y == 1) else float('nan')
    tnr = np.mean(~pred[y == 0]) if np.any(y == 0) else float('nan')
    return float((tpr + tnr) / 2)


def hard_assignment(pack, bag, raw):
    import haig_background as hb
    state = pack['old']['background']['state']
    sp = pack['old']['background']['sp']
    x = np.asarray(bag[:, :107], np.float32)
    role = np.maximum(np.asarray(raw[:, 43:74], np.float32), 0)
    denom = role.sum(1, keepdims=True)
    role = np.divide(role, denom, out=np.zeros_like(role), where=denom > 0)
    d = pairwise_rms(x, state['shared_centers'])
    assignment = hb.hard(d / np.maximum(state['shared_scale'][None, :], 1e-6), role, sp)
    return x, assignment


def changed_bag(bag, internal):
    out = np.asarray(bag, np.float32).copy()
    out[:, 108] = np.asarray(internal, np.float32)
    out[:, 110] = (1 - np.clip(out[:, 109], 0, 1)) * np.sqrt(
        np.maximum(out[:, 107], 0) * np.maximum(out[:, 108], 0))
    if not np.isfinite(out).all():
        raise RuntimeError('nonfinite intervention')
    return out


def predict(pack, bags):
    pred, _ = pack['v4'].predict_model(pack['model'], pack['global_val'],
        np.full(len(bags), .5), pack['graph_val'], bags, pack['dev'])
    return np.asarray(pred, float)


def phase_preflight():
    if OUT.exists():
        raise RuntimeError('output directory already exists; no overwrite permitted')
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'logs').mkdir()
    assets = [digest(path) for path in source_assets()]
    write_json(OUT / 'asset_manifest_before.json', assets)
    write_json(OUT / 'experiment_contract.json', {
        'status': 'FROZEN', 'source_only_model_training': False,
        'model_training_performed': False, 'APK_files_opened': 0,
        'years': [2017, 2018, 2019, 2020, 2021], 'new_APKs_used': False,
        'intervention': 'leave scored Host out of its selected APK-local centroid; keep selected clusters and original scale fixed',
        'control': 'remove a different Host from the same selected cluster',
        'control_repeats': REPEATS, 'threshold_changed': False,
        'router_changed': False, 'future_years_are_development': True})
    print('ISR_PREFLIGHT_PASS assets=' + str(len(assets)), flush=True)


def verify_assets():
    before = json.loads((OUT / 'asset_manifest_before.json').read_text(encoding='utf-8'))
    changed = []
    for item in before:
        try: now = digest(item['path'])
        except OSError as exc:
            changed.append({'path': item['path'], 'error': str(exc)}); continue
        if now != item: changed.append({'path': item['path'], 'before': item, 'after': now})
    return {'status': 'PASS' if not changed else 'FAIL', 'checked': len(before),
            'changed': changed, 'scope': 'named frozen code/caches/checkpoints only; no APK bytes'}


def run(device):
    from bfa_core import formal_split, load_data_contract
    data, _ = load_data_contract()
    per_seed, host_rows, replay_rows, coverage_rows = [], [], [], []
    started = time.perf_counter()
    for split in SPLITS:
        for seed in SEEDS:
            pack = formal_split(data, split, seed, device=device, train_data=False)
            loo_bags, random_bags = [], []
            val = pack['val']; y = np.asarray(data['y'][val], int)
            max_replay = 0.; self_hosts = eval_hosts = total_hosts = 0
            for local, (bag, raw) in enumerate(zip(pack['bags_val'], pack['raw_val'])):
                x, assignment = hard_assignment(pack, bag, raw)
                sid = str(data['ids'][val[local]])
                stable = int(hashlib.sha256((split + str(seed) + sid).encode()).hexdigest()[:8], 16)
                effect = self_exclusion(x, assignment, pack['args'].internal_background_coverage,
                                        REPEATS, stable)
                frozen_diff = float(np.max(np.abs(effect['original_internal'] - bag[:, 108])))
                max_replay = max(max_replay, frozen_diff)
                if frozen_diff > 2e-5:
                    raise RuntimeError(f'Internal replay mismatch {split}/{seed}/{sid}: {frozen_diff}')
                random_mean = effect['random_internal'].mean(1)
                loo_bags.append(changed_bag(bag, effect['loo_internal']))
                random_bags.append(changed_bag(bag, random_mean))
                total_hosts += len(bag); self_hosts += int(effect['self_in_reference'].sum())
                eval_hosts += int((effect['evaluable'] & effect['self_in_reference']).sum())
                mask = effect['self_in_reference'] & effect['evaluable']
                loo_delta = effect['loo_internal'] - effect['original_internal']
                random_delta = random_mean - effect['original_internal']
                if np.any(mask):
                    values = loo_delta[mask]; controls = random_delta[mask]
                    summary = {'mean_loo_internal_delta': float(values.mean()),
                               'median_loo_internal_delta': float(np.median(values)),
                               'max_loo_internal_delta': float(values.max()),
                               'top3_mean_loo_internal_delta': float(np.mean(np.sort(values)[-min(3,len(values)):])),
                               'mean_peer_control_delta': float(controls.mean()),
                               'mean_excess_self_delta': float((values-controls).mean())}
                else:
                    summary = {key: None for key in ('mean_loo_internal_delta','median_loo_internal_delta',
                        'max_loo_internal_delta','top3_mean_loo_internal_delta','mean_peer_control_delta','mean_excess_self_delta')}
                per_seed.append({'split': split, 'year': YEAR[split], 'seed': seed,
                    'sha256': sid, 'label': int(y[local]), 'routed': int(pack['route'][local]),
                    'stage1_probability': float(pack['base'][local]),
                    'original_expert_probability': float(pack['old']['probability'][local]),
                    'hosts': len(bag), 'self_reference_hosts': int(effect['self_in_reference'].sum()),
                    'evaluable_self_reference_hosts': int(mask.sum()), **summary})
                if pack['route'][local]:
                    for j in np.flatnonzero(mask):
                        host_rows.append({'split': split, 'year': YEAR[split], 'seed': seed,
                            'sha256': sid, 'host_index': int(j),
                            'original_internal': float(effect['original_internal'][j]),
                            'loo_internal': float(effect['loo_internal'][j]),
                            'peer_control_internal': float(random_mean[j]),
                            'loo_delta': float(loo_delta[j]),
                            'peer_delta': float(random_delta[j]),
                            'excess_self_delta': float(loo_delta[j]-random_delta[j])})
            loo_prob = predict(pack, loo_bags)
            random_prob = predict(pack, random_bags)
            original_prob = np.asarray(pack['old']['probability'], float)
            for local in range(len(val)):
                row = per_seed[-len(val)+local]
                row.update({'loo_expert_probability': float(loo_prob[local]),
                            'peer_control_expert_probability': float(random_prob[local]),
                            'loo_probability_delta': float(loo_prob[local]-original_prob[local]),
                            'peer_control_probability_delta': float(random_prob[local]-original_prob[local]),
                            'excess_self_probability_delta': float(loo_prob[local]-random_prob[local]),
                            'original_status': status_of(y[local], pack['base'][local], original_prob[local], pack['route'][local])})
            replay_rows.append({'split': split, 'year': YEAR[split], 'seed': seed,
                'M0_probability_max_abs_difference': pack['replay_diff'],
                'Internal_column_max_abs_difference': max_replay, 'status': 'PASS'})
            coverage_rows.append({'split': split, 'year': YEAR[split], 'seed': seed,
                'APKs': len(val), 'hosts': total_hosts, 'self_reference_hosts': self_hosts,
                'evaluable_self_reference_hosts': eval_hosts,
                'self_reference_host_rate': self_hosts/max(total_hosts,1),
                'evaluable_rate_within_self_reference': eval_hosts/max(self_hosts,1)})
            print(f'ISR_SPLIT_SEED_DONE split={split} seed={seed} apks={len(val)} hosts={total_hosts}', flush=True)
    write_csv(OUT / 'per_apk_seed_effects.csv', per_seed)
    write_csv(OUT / 'routed_per_host_effects.csv', host_rows)
    write_csv(OUT / 'replay_validation.csv', replay_rows)
    write_csv(OUT / 'coverage_by_split_seed.csv', coverage_rows)
    aggregate(data, per_seed)
    write_json(OUT / 'runtime_cost.json', {'audit_seconds': time.perf_counter()-started,
        'device': device, 'models_replayed': len(SPLITS)*len(SEEDS), 'models_trained': 0,
        'APK_files_opened': 0, 'random_control_repeats': REPEATS})


def aggregate(data, rows):
    grouped = defaultdict(list)
    for row in rows: grouped[(row['split'], row['sha256'])].append(row)
    ensemble = []
    for (split, sid), values in grouped.items():
        if len(values) != 3: raise RuntimeError('missing seed for ' + split + '/' + sid)
        first = values[0]
        item = {k:first[k] for k in ('split','year','sha256','label','routed','stage1_probability','hosts','original_status')}
        numeric = ('original_expert_probability','loo_expert_probability','peer_control_expert_probability',
                   'loo_probability_delta','peer_control_probability_delta','excess_self_probability_delta',
                   'mean_loo_internal_delta','median_loo_internal_delta','max_loo_internal_delta',
                   'top3_mean_loo_internal_delta','mean_peer_control_delta','mean_excess_self_delta')
        for key in numeric:
            vals = [v[key] for v in values if v[key] is not None]
            item[key] = float(np.mean(vals)) if vals else None
        base = item['stage1_probability']; y = item['label']; routed = bool(item['routed'])
        item['original_status'] = status_of(y, base, item['original_expert_probability'], routed)
        item['loo_status'] = status_of(y, base, item['loo_expert_probability'], routed)
        item['peer_control_status'] = status_of(y, base, item['peer_control_expert_probability'], routed)
        ensemble.append(item)
    write_csv(OUT / 'per_apk_ensemble_effects.csv', ensemble)

    result_rows, association = [], []
    for split_name, year in [('SOURCE_2017_OOF',2017),('T1',2018),('T2',2019),('T3',2020),('T4',2021)]:
        subset = [r for r in ensemble if r['year']==year]
        y = np.asarray([r['label'] for r in subset], int)
        base = np.asarray([r['stage1_probability'] for r in subset], float)
        route = np.asarray([r['routed'] for r in subset], bool)
        for arm, field in [('ORIGINAL','original_expert_probability'),('LEAVE_SELF_OUT','loo_expert_probability'),('PEER_REMOVAL_CONTROL','peer_control_expert_probability')]:
            expert = np.asarray([r[field] for r in subset], float)
            final = np.where(route, expert, base)
            pred = final >= .5
            stage1 = base >= .5
            fixed = int(np.sum((stage1 != y) & (pred == y)))
            broken = int(np.sum((stage1 == y) & (pred != y)))
            result_rows.append({'split':split_name,'year':year,'arm':arm,'n':len(y),
                'routed':int(route.sum()),'BA':balanced_accuracy(y,final),
                'accuracy':float(np.mean(pred==y)),'TN':int(np.sum((y==0)&(~pred))),
                'FP':int(np.sum((y==0)&pred)),'FN':int(np.sum((y==1)&(~pred))),
                'TP':int(np.sum((y==1)&pred)),'fixed':fixed,'broken':broken,'net':fixed-broken})
        for status in ('FIXED','UNFIXED','BROKEN','KEPT'):
            group = [r for r in subset if r['original_status']==status]
            vals = np.asarray([r['excess_self_probability_delta'] for r in group], float)
            intern = np.asarray([r['mean_excess_self_delta'] for r in group if r['mean_excess_self_delta'] is not None], float)
            association.append({'split':split_name,'year':year,'group':status,'n':len(group),
                'mean_excess_self_probability_delta':float(vals.mean()) if len(vals) else None,
                'median_excess_self_probability_delta':float(np.median(vals)) if len(vals) else None,
                'mean_excess_self_internal_delta':float(intern.mean()) if len(intern) else None})
    write_csv(OUT / 'frozen_model_intervention_results.csv', result_rows)
    write_csv(OUT / 'error_association.csv', association)

    # Conservative gate: observable above peer deletion and linked to errors in several periods.
    routed = [r for r in ensemble if r['routed']]
    positive_excess = [r for r in routed if r['mean_excess_self_delta'] is not None and r['mean_excess_self_delta'] > 0]
    observable = len(positive_excess) / max(len(routed),1) >= .20
    year_links = 0
    for year in (2017,2018,2019,2020,2021):
        err = [r['mean_excess_self_delta'] for r in routed if r['year']==year and r['original_status'] in ('UNFIXED','BROKEN') and r['mean_excess_self_delta'] is not None]
        ok = [r['mean_excess_self_delta'] for r in routed if r['year']==year and r['original_status'] in ('FIXED','KEPT') and r['mean_excess_self_delta'] is not None]
        if err and ok and np.median(err) > np.median(ok): year_links += 1
    association_supported = observable and year_links >= 3
    status = ('INTERNAL_SELF_REFERENCE_ERROR_ASSOCIATION_CANDIDATE' if association_supported else
              'INTERNAL_SELF_REFERENCE_EFFECT_OBSERVABLE' if observable else
              'NO_VERIFIED_INTERNAL_SELF_REFERENCE_BOTTLENECK')
    integrity = verify_assets()
    if integrity['status'] != 'PASS': raise RuntimeError('frozen assets changed during audit')
    write_json(OUT / 'baseline_integrity_check.json', integrity)
    decision = {'status':status,'self_reference_effect_observable':observable,
        'error_association_supported':association_supported,'years_error_effect_higher':year_links,
        'routed_APKs':len(routed),'routed_positive_excess_fraction':len(positive_excess)/max(len(routed),1),
        'formal_integrity':integrity['status'],'model_training_performed':False,
        'new_APKs_used':False,'future_years_are_development':True,
        'formal_model_changed':False,
        'permit_separate_controlled_training_experiment':association_supported,
        'claim_boundary':'Estimator intervention is not an APK/program counterfactual and cannot prove malicious intent or causality.',
        'next_action':('Specify a separate frozen source-first estimator experiment; do not alter formal model yet.' if association_supported
                       else 'Stop this Internal self-reference branch; retain formal three-background model.')}
    write_json(OUT / 'final_decision.json', decision)
    table = ['| 年份 | 方案 | BA | ACC | TN | FP | FN | TP | Fixed | Broken | Net |','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result_rows:
        table.append(f"| {r['year']} | {r['arm']} | {r['BA']:.4f} | {r['accuracy']:.4f} | {r['TN']} | {r['FP']} | {r['FN']} | {r['TP']} | {r['fixed']} | {r['broken']} | {r['net']} |")
    report = f'''# CARE-Droid Internal背景自引用审计 v1

最终状态：**{status}**。

本实验读取冻结2017—2021缓存与27个M0 checkpoint，不读取APK、不训练模型、不改变Stage-I、Router、阈值或正式三背景。2018—2021均为已使用的开发年份。

## 干预定义

对每个Host，仅在其属于Internal主导参考簇时，把该Host从自身簇质心中排除；主导簇集合与原APK尺度保持冻结。负对照从同一簇移除另一个Host。该操作检验估计器自引用，不是语义保持APK变换，也不是程序因果反事实。

## 冻结模型重放

{chr(10).join(table)}

## 判定

- 路由APK数：{len(routed)}。
- 相对同簇随机移除，Internal自排除效应为正的路由APK比例：{len(positive_excess)/max(len(routed),1):.4f}。
- 错误组中位效应高于正确组的年份：{year_links}/5。
- 只有同时超过同簇移除对照并在多个时期与错误稳定相关，才允许另立受控训练实验；本包本身不授权修改正式模型。
- 即使LEAVE_SELF_OUT的BA变化，也不能证明原Internal“导致”某个误判，因为这是对已训练模型输入的测量干预。

冻结资产完整性：**{integrity['status']}**；核验{integrity['checked']}项命名资产，原始APK打开数为0。
'''
    write_text(OUT / 'REPORT.md', report)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args=parser.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    phase_preflight()
    run(args.device)
    print('CARE_INTERNAL_SELF_REFERENCE_AUDIT_DONE', flush=True)


if __name__ == '__main__':
    main()

