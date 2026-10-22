#!/usr/bin/env python3
"""Frozen BFA model-input attribution; reads cached Host tensors/checkpoints, writes only new BFA results, never executes or opens APKs."""
from __future__ import annotations

import csv
import math

import numpy as np

from bfa_core import (OUT, ROOT, SEEDS, anchors, formal_split, immutable_csv,
                      immutable_json, load_data_contract, recompute_residual,
                      verify_assets)

BRANCHES = ('External', 'Shared_group', 'Internal')


def neutral_bag(original, baseline, branch):
    b = original.copy()
    if branch == 'External':
        b[:, 107] = baseline[0]
    elif branch == 'Shared_group':
        b[:, 108] = baseline[1]
        b[:, 109] = baseline[2]
    elif branch == 'Internal':
        b[:, 108] = baseline[1]
    else:
        raise ValueError(branch)
    return recompute_residual(b)


def torch_bag(torch, original, dev):
    raw = torch.as_tensor(original, dtype=torch.float32, device=dev)
    primitives = raw[:, 107:110].detach().clone().requires_grad_(True)
    residual = (1 - torch.clamp(primitives[:, 2], 0, 1)) * torch.sqrt(
        torch.clamp(primitives[:, 0] * primitives[:, 1], min=1e-12))
    bag = torch.cat([raw[:, :107], primitives, residual[:, None], raw[:, 111:]], dim=1)
    return bag, primitives


def one_logit(model, torch, dev, global_t, graph_t, bag, i):
    graph = {k: value[i] for k, value in graph_t.items()}
    base_logit = torch.zeros((), dtype=torch.float32, device=dev)
    return model(global_t[i], base_logit, graph, bag)[0]


def classify_group(y, base, p):
    first = int(base >= .5)
    final = int(p >= .5)
    if first == 0 and y == 1:
        return 'FIXED_FN' if final == y else 'UNFIXED_FN'
    if first == 0 and y == 0:
        return 'BROKEN_TN' if final != y else 'KEPT_TN'
    if first == 1 and y == 0:
        return 'FIXED_FP' if final == y else 'UNFIXED_FP'
    return 'BROKEN_TP' if final != y else 'KEPT_TP'


def run(device, resume, full=False):
    if verify_assets()['status'] != 'PASS':
        raise RuntimeError('frozen assets changed before attribution')
    path = OUT / ('background_logit_effects.csv' if full else 'pilot_2020_logit_effects.csv')
    if path.exists():
        if resume:
            print('BFA_ATTRIBUTION_RESUME_COMPLETE', flush=True)
            return
        raise RuntimeError('attribution result exists without --resume')
    data, _ = load_data_contract()
    rows = []
    replay = []
    validity = []
    splits = ('T2', 'T3', 'T4') if full else ('T3',)
    for split in splits:
        for seed in SEEDS:
            pack = formal_split(data, split, seed, device)
            v4 = pack['v4']
            torch = v4.torch
            dev = pack['dev']
            model = pack['model']
            baseline = anchors(pack['bags_train'])
            graph_t = v4.graph_tensors(pack['graph_val'], dev)
            global_t = torch.as_tensor(pack['global_val'], dtype=torch.float32, device=dev)
            ensemble = np.mean([__import__('bfa_core').load_m0(split, s)['probability'] for s in SEEDS], axis=0)
            replay.append({'split': split, 'seed': seed,
                           'M0_full_replay_max_difference': pack['replay_diff'],
                           'routed': int(pack['route'].sum())})
            route_idx = np.flatnonzero(pack['route'])
            if not full:
                with (ROOT / 'outputs/care-2020-fn-tn-dex-review-v2/matched_pairs_restricted.csv').open(
                        encoding='utf-8-sig', newline='') as f:
                    pilot_pairs = list(csv.DictReader(f))[:8]
                wanted = {s for pair in pilot_pairs for s in (pair['fn_sha256'], pair['tn_sha256'])}
                route_idx = np.asarray([i for i in route_idx
                                        if str(data['ids'][pack['val'][i]]) in wanted], int)
                if len(route_idx) != 16:
                    raise RuntimeError('8-pair pilot SHA alignment mismatch')
            for i in route_idx:
                i = int(i)
                original = pack['bags_val'][i]
                bag_t, primitive = torch_bag(torch, original, dev)
                logit = one_logit(model, torch, dev, global_t, graph_t, bag_t, i)
                p = float(torch.sigmoid(logit).detach().cpu())
                target = float(pack['old']['probability'][i])
                if abs(p - target) > 2e-6:
                    raise RuntimeError(f'gradient-path replay failed {split}/{seed}/{i}: {p-target}')
                gradient = torch.autograd.grad(logit, primitive, retain_graph=False)[0].detach().cpu().numpy()
                if not np.isfinite(gradient).all():
                    raise RuntimeError('nonfinite input gradient')
                model_logit = float(logit.detach().cpu())
                sid = str(data['ids'][pack['val'][i]])
                y = int(data['y'][pack['val'][i]])
                group = classify_group(y, float(pack['base'][i]), float(ensemble[i]))
                for branch in BRANCHES:
                    variant = neutral_bag(original, baseline, branch)
                    if float(np.max(np.abs(variant[:, 110] - recompute_residual(variant)[:, 110]))) > 1e-6:
                        raise RuntimeError('inconsistent intervention residual')
                    with torch.no_grad():
                        changed = one_logit(model, torch, dev, global_t, graph_t,
                            torch.as_tensor(variant, dtype=torch.float32, device=dev), i)
                    changed_logit = float(changed.detach().cpu())
                    if branch == 'External':
                        local_grad = gradient[:, 0] * (original[:, 107] - baseline[0])
                    elif branch == 'Internal':
                        local_grad = gradient[:, 1] * (original[:, 108] - baseline[1])
                    else:
                        local_grad = (gradient[:, 1] * (original[:, 108] - baseline[1]) +
                                      gradient[:, 2] * (original[:, 109] - baseline[2]))
                    neutral_p = float(torch.sigmoid(changed).detach().cpu())
                    rows.append({'split': split, 'year': int(data['years'][pack['val'][i]]),
                                 'seed': seed, 'sha256': sid, 'label': y,
                                 'stage1_probability': float(pack['base'][i]),
                                 'M0_ensemble_probability': float(ensemble[i]),
                                 'M0_seed_probability': target, 'group': group,
                                 'branch': branch, 'original_logit': model_logit,
                                 'neutral_logit': changed_logit,
                                 'original_minus_neutral_logit': model_logit - changed_logit,
                                 'neutral_minus_original_logit': changed_logit - model_logit,
                                 'neutral_probability': neutral_p,
                                 'gradient_times_input_local': float(np.sum(local_grad)),
                                 'host_count': len(original),
                                 'anchor_external': float(baseline[0]),
                                 'anchor_internal': float(baseline[1]),
                                 'anchor_commonness': float(baseline[2]),
                                 'intervention_type': 'class-blind train-only equal-APK mean',
                                 'residual_recomputed': True,
                                 'program_counterfactual': False})
                if (len(rows) // 3) % 50 == 0:
                    print('BFA_ATTR_PROGRESS', split, seed, len(rows) // 3, flush=True)
            validity.append({'split': split, 'seed': seed, 'routed': len(route_idx),
                             'all_original_probabilities_replayed': True,
                             'all_derived_residuals_recomputed': True,
                             'input_distribution_check': 'train-only mean within each scalar training range',
                             'program_level_counterfactual': False})
            print('BFA_ATTR_DONE', split, seed, 'routed=' + str(len(route_idx)), flush=True)
    immutable_csv(path, rows)
    if not full:
        immutable_json(OUT / 'pilot_2020_validation.json', {
            'status': 'PASS', 'routed_seed_rows': len(rows) // 3,
            'max_full_probability_difference': max(r['M0_full_replay_max_difference'] for r in replay),
            'all_seeds': len(replay) == 3,
            'pilot_boundary': 'All T3 routed APKs were checked; no intervention chosen from labels'})
        print('BFA_ATTRIBUTION_PILOT_PASS', len(rows), flush=True)
        return
    immutable_csv(OUT / 'background_gradient_attribution.csv', [
        {k: r[k] for k in ('split', 'year', 'seed', 'sha256', 'label', 'group', 'branch',
                          'gradient_times_input_local', 'host_count')} for r in rows])
    immutable_csv(OUT / 'background_intervention_validity.csv', validity)
    immutable_json(OUT / 'baseline_replay.json', {'status': 'PASS', 'fold_seed_replay': replay,
                   'max_full_probability_difference': max(r['M0_full_replay_max_difference'] for r in replay),
                   'attribution_is_model_input_sensitivity_not_causal_program_evidence': True})
    _pair_and_seed(rows)
    print('BFA_ATTRIBUTION_DONE', len(rows), flush=True)


def _pair_and_seed(rows):
    by_key = {(r['split'], r['seed'], r['sha256'], r['branch']): r for r in rows}
    pairs = []
    path = ROOT / 'outputs/care-2020-fn-tn-dex-review-v2/matched_pairs_restricted.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        source = list(csv.DictReader(f))
    if len(source) != 22:
        raise RuntimeError('frozen 2020 matched pair count changed')
    for pair_id, pair in enumerate(source, 1):
        for seed in SEEDS:
            for branch in BRANCHES:
                a = by_key.get(('T3', seed, pair['fn_sha256'], branch))
                b = by_key.get(('T3', seed, pair['tn_sha256'], branch))
                if a is None or b is None:
                    raise RuntimeError('frozen FN/TN pair missing from M0 route')
                pairs.append({'pair_id': pair_id, 'seed': seed, 'branch': branch,
                              'fn_sha256': pair['fn_sha256'], 'tn_sha256': pair['tn_sha256'],
                              'FN_effect': a['original_minus_neutral_logit'],
                              'TN_effect': b['original_minus_neutral_logit'],
                              'FN_minus_TN_effect': a['original_minus_neutral_logit'] - b['original_minus_neutral_logit'],
                              'match_stage1_probability_difference': pair['stage1_probability_difference'],
                              'match_host_count_ratio': pair['host_count_ratio']})
    immutable_csv(OUT / 'matched_fn_tn_background_effects.csv', pairs)
    # Same predictions and fixed route; do not rematch using attribution outcomes.
    summaries = []
    for split in ('T2', 'T3', 'T4'):
        for branch in BRANCHES:
            sample = [r for r in rows if r['split'] == split and r['branch'] == branch]
            if not sample:
                continue
            for group in sorted(set(r['group'] for r in sample)):
                subset = [r for r in sample if r['group'] == group]
                summaries.append({'split': split, 'branch': branch, 'group': group,
                                  'APKs': len(set(r['sha256'] for r in subset)),
                                  'seed_observations': len(subset),
                                  'mean_effect': float(np.mean([r['original_minus_neutral_logit'] for r in subset])),
                                  'negative_effect_APKs_any_seed': len(set(r['sha256'] for r in subset if r['original_minus_neutral_logit'] < 0))})
    immutable_csv(OUT / 'background_effect_group_summary.csv', summaries)
    signs = []
    apks = sorted(set((r['split'], r['sha256'], r['branch']) for r in rows))
    for split, sid, branch in apks:
        subset = [by_key[(split, seed, sid, branch)] for seed in SEEDS]
        values = [r['original_minus_neutral_logit'] for r in subset]
        signs.append({'split': split, 'sha256': sid, 'branch': branch,
                      'group': subset[0]['group'],
                      'seeds': len(values), 'all_negative': all(x < 0 for x in values),
                      'all_positive': all(x > 0 for x in values),
                      'same_nonzero_sign': all(x < 0 for x in values) or all(x > 0 for x in values),
                      'mean_effect': float(np.mean(values))})
    immutable_csv(OUT / 'seed_direction_consistency.csv', signs)
    # Pre-registered 2019 FP/TP descriptive table; no outcome-conditioned rematching.
    immutable_csv(OUT / 'matched_fp_tp_background_effects.csv', [
        {k: r[k] for k in ('split', 'seed', 'sha256', 'label', 'group', 'branch',
                          'original_minus_neutral_logit', 'host_count')} for r in rows
        if r['split'] == 'T2' and r['group'] in ('UNFIXED_FP', 'KEPT_TP', 'FIXED_FP', 'BROKEN_TP')])
