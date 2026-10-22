#!/usr/bin/env python3
"""BFA-V1 read-only source adapter; reads frozen caches, writes only the new BFA output directory, never opens APKs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT / 'scripts/care-background-functional-attribution-v1'
OUT = ROOT / 'outputs/care_background_functional_attribution_v1'
SCBE = ROOT / 'outputs/care_scbe_v1'
sys.path[:0] = [str(HERE), str(ROOT / 'scripts/care-scbe-v1'),
                str(ROOT / 'scripts/care-multiyear-development-v1')]

SEEDS = (900000, 900001, 900002)
SPLITS = ('S1', 'S2', 'S3', 'S4', 'S5', 'T1', 'T2', 'T3', 'T4')
ARMS = ('B0_ORIGINAL', 'B1_EXTERNAL_NEUTRAL', 'B2_SHARED_NEUTRAL',
        'B3_INTERNAL_NEUTRAL', 'B4_EXTERNAL_ROLE_SHUFFLE')
COLS = {'external': 107, 'internal': 108, 'commonness': 109, 'residual': 110}


def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return {'path': str(path.resolve()), 'sha256': h.hexdigest(), 'size': path.stat().st_size}


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def immutable_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(value)


def immutable_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def asset_paths():
    old = ROOT / 'outputs/care_multiyear_development_v1'
    paths = [ROOT / 'scripts/care-scbe-v1/train.py',
             ROOT / 'scripts/care-scbe-v1/preflight.py',
             ROOT / 'scripts/care-multiyear-development-v1/train_stage2.py',
             ROOT / 'scripts/care-multiyear-development-v1/myd_common.py',
             ROOT / 'scripts/care-multiyear-development-v1/metrics.py',
             ROOT / 'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py',
             ROOT / 'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_models.py',
             ROOT / 'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py',
             old / 'data.npz', old / 'feature_rows.pkl', old / 'sample_manifest.csv',
             ROOT / 'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv',
             ROOT / 'outputs/care-2020-fn-tn-dex-review-v2/matched_pairs_restricted.csv',
             SCBE / 'final_decision.json', SCBE / 'baseline_replay.json']
    paths += [old / 'predictions' / split / 'router.npz' for split in SPLITS[5:]]
    paths += [SCBE / 'models' / split / str(seed) / 'M0_REPRODUCED.pkl'
              for split in SPLITS for seed in SEEDS]
    return paths


def preflight(resume=False):
    from preflight import verify as scbe_verify
    if scbe_verify()['status'] != 'PASS':
        raise RuntimeError('SCBE frozen input integrity failed')
    manifest = OUT / 'asset_manifest_before.json'
    if manifest.exists():
        if not resume:
            raise RuntimeError('BFA output exists; use --resume, never overwrite')
        return verify_assets()
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'tmp').mkdir()
    items = [digest(p) for p in asset_paths()]
    immutable_json(manifest, items)
    result = {'status': 'PASS', 'checked': len(items), 'new_APKs_used': False,
              'APK_files_opened': 0, 'source_rows': 799,
              'development_years': [2017, 2018, 2019, 2020, 2021],
              'scope': 'named frozen code, cached features, routes, pairs, and 27 M0 checkpoints; no APK bytes'}
    immutable_json(OUT / 'asset_integrity_report.json', result)
    return result


def verify_assets():
    prior = json.loads((OUT / 'asset_manifest_before.json').read_text(encoding='utf-8'))
    changed = []
    for item in prior:
        try:
            current = digest(item['path'])
        except OSError as e:
            changed.append({'path': item['path'], 'error': str(e)})
            continue
        if current != item:
            changed.append({'path': item['path'], 'before': item, 'after': current})
    return {'status': 'PASS' if not changed else 'FAIL', 'checked': len(prior),
            'changed': changed,
            'scope': 'actual named frozen code/data/routes/M0 checkpoints; no APK files opened; not a whole-server audit'}


def load_data_contract():
    from audit_data import load_data
    from train import load_source_contract
    data = load_data()
    source = load_source_contract(data)
    years = {int(y): int(np.sum(data['years'] == y)) for y in np.unique(data['years'])}
    if years != {2017: 799, 2018: 793, 2019: 399, 2020: 397, 2021: 400}:
        raise RuntimeError('historic dataset inventory changed: ' + repr(years))
    if len(set(map(str, data['ids']))) != 2788:
        raise RuntimeError('duplicate or missing SHA in historical data')
    return data, source


def m0_path(split, seed):
    return SCBE / 'models' / split / str(seed) / 'M0_REPRODUCED.pkl'


def load_m0(split, seed):
    with m0_path(split, seed).open('rb') as f:
        obj = pickle.load(f)
    if obj['split'] != split or obj['seed'] != seed or obj['arm'] != 'M0_REPRODUCED':
        raise RuntimeError('M0 checkpoint identity mismatch')
    return obj


def formal_split(data, split, seed, device='cpu', train_data=True):
    """Reconstruct the exact frozen M0 tensors and fail if all validation probabilities do not replay."""
    from myd_common import runtime
    from train import global24, split_contract, load_source_contract
    from train_stage2 import raw_bags, transform_saved
    v4, _, args = runtime()
    args.device = device
    args.keep_weight = args.fix_weight = 1.
    args.keep_consistency = 0.
    tr, va, base, route, score = split_contract(data, split, load_source_contract(data))
    old = load_m0(split, seed)
    if old['ids'] != list(map(str, data['ids'][va])):
        raise RuntimeError('M0 validation SHA order differs')
    gx, gz = global24(v4, data, tr, va)
    raw_tr = raw_bags(data, tr) if train_data else None
    raw_va = raw_bags(data, va)
    bags_tr = transform_saved(v4, raw_tr, old['background'], args) if train_data else None
    bags_va = transform_saved(v4, raw_va, old['background'], args)
    gs = old['graph_scaler']
    graph_tr = v4.transform_role_graphs(data['s7'][tr], gs) if train_data else None
    graph_va = v4.transform_role_graphs(data['s7'][va], gs)
    torch = v4.torch
    dev = v4.resolve_device(device)
    model = v4.RoleGraphConditionedEvidence(24, 'RGCondAttn', args).to(dev)
    model.load_state_dict(old['state_dict'])
    model.eval()
    pred, _ = v4.predict_model(model, gz, np.full(len(va), .5), graph_va, bags_va, dev)
    diff = float(np.max(np.abs(np.asarray(pred) - np.asarray(old['probability']))))
    if diff > 1e-6:
        raise RuntimeError(f'BFA M0 full validation replay mismatch {split}/{seed}: {diff}')
    return dict(v4=v4, args=args, dev=dev, model=model, old=old, train=tr, val=va,
                base=base, route=route, router_score=score, global_train=gx,
                global_val=gz, raw_train=raw_tr, raw_val=raw_va,
                bags_train=bags_tr, bags_val=bags_va,
                graph_train=graph_tr, graph_val=graph_va, replay_diff=diff)


def anchors(train_bags):
    """Class-blind equal-APK scalar means; preserves train-only and avoids large APK dominance."""
    if not train_bags:
        raise RuntimeError('missing training bags')
    means = np.asarray([np.mean(b[:, 107:110], axis=0) for b in train_bags], np.float64)
    anchor = np.mean(means, axis=0)
    if not np.isfinite(anchor).all() or np.any(anchor < 0):
        raise RuntimeError('invalid neutral background anchor')
    return anchor.astype(np.float32)


def recompute_residual(bag):
    out = np.asarray(bag, np.float32).copy()
    out[:, 110] = (1 - np.clip(out[:, 109], 0, 1)) * np.sqrt(
        np.maximum(out[:, 107], 0) * np.maximum(out[:, 108], 0))
    if not np.isfinite(out).all():
        raise RuntimeError('nonfinite background intervention')
    return out


def role_ids(raw_bags):
    ids = []
    for raw in raw_bags:
        r = np.maximum(np.asarray(raw[:, 43:74], np.float32), 0)
        out = np.argmax(r, axis=1).astype(np.int16)
        out[r.sum(axis=1) == 0] = -1
        ids.append(out)
    return ids


def transform_arm(train_bags, val_bags, raw_train, raw_val, arm, seed):
    """Neutralize one legal primitive and always recompute the derived Residual."""
    if arm == 'B0_ORIGINAL':
        return train_bags, val_bags, {'type': 'FROZEN_ORIGINAL', 'downstream_recomputed': []}
    neutral = anchors(train_bags)
    source = {'B1_EXTERNAL_NEUTRAL': 107, 'B2_SHARED_NEUTRAL': 109,
              'B3_INTERNAL_NEUTRAL': 108}.get(arm)
    if source is not None:
        val = float(neutral[source - 107])
        tr = []
        va = []
        for bags, target in ((train_bags, tr), (val_bags, va)):
            for bag in bags:
                copy = bag.copy()
                copy[:, source] = val
                if arm == 'B2_SHARED_NEUTRAL':
                    # Internal is itself built using a Shared assignment.
                    # Remove both Shared-derived paths rather than leaving
                    # a stale Internal path while claiming Shared is gone.
                    copy[:, 108] = float(neutral[1])
                target.append(recompute_residual(copy))
        return tr, va, {'type': 'INFORMATION_NEUTRALIZATION',
                        'changed_primitive': source, 'neutral_value': val,
                        'also_neutralized_internal_for_shared_group': arm == 'B2_SHARED_NEUTRAL',
                        'downstream_recomputed': ['Residual'],
                        'neutral_estimator': 'train-only equal-APK mean'}
    if arm != 'B4_EXTERNAL_ROLE_SHUFFLE':
        raise ValueError(arm)
    rng = np.random.RandomState(seed + 202601)
    tr_roles, va_roles = role_ids(raw_train), role_ids(raw_val)
    pool = {}
    for r in np.unique(np.concatenate(tr_roles)):
        pool[int(r)] = np.concatenate([b[roles == r, 107] for b, roles in zip(train_bags, tr_roles)
                                       if np.any(roles == r)])
    all_values = np.concatenate([b[:, 107] for b in train_bags])
    def one(bags, roles_list):
        out = []
        for bag, roles in zip(bags, roles_list):
            copy = bag.copy()
            for j, r in enumerate(roles):
                choices = pool.get(int(r), all_values)
                copy[j, 107] = choices[rng.randint(len(choices))]
            out.append(recompute_residual(copy))
        return out
    return one(train_bags, tr_roles), one(val_bags, va_roles), {
        'type': 'ROLE_MATCHED_EXTERNAL_SHUFFLE_CONTROL',
        'changed_primitive': 107, 'downstream_recomputed': ['Residual'],
        'donors': 'training-window only, class-blind, with-replacement role-matched Host values',
        'seed': seed + 202601}


def audit_bags(original, changed, arm):
    if len(original) != len(changed):
        raise RuntimeError('bag count changed')
    altered = 0
    for a, b in zip(original, changed):
        if a.shape != b.shape or a.shape[1] != 142 or not np.isfinite(b).all():
            raise RuntimeError('Host shape/finite invariant failed')
        if not np.array_equal(a[:, :107], b[:, :107]) or not np.array_equal(a[:, 111:], b[:, 111:]):
            raise RuntimeError('base Compact/Relation changed')
        if arm == 'B1_EXTERNAL_NEUTRAL' or arm == 'B4_EXTERNAL_ROLE_SHUFFLE':
            if not np.array_equal(a[:, 108:110], b[:, 108:110]):
                raise RuntimeError('other background primitive changed')
        elif arm == 'B2_SHARED_NEUTRAL':
            if not np.array_equal(a[:, 107], b[:, 107]):
                raise RuntimeError('other background primitive changed')
        elif arm == 'B3_INTERNAL_NEUTRAL':
            if not np.array_equal(a[:, 107], b[:, 107]) or not np.array_equal(a[:, 109], b[:, 109]):
                raise RuntimeError('other background primitive changed')
        expected = recompute_residual(b)
        if float(np.max(np.abs(expected[:, 110] - b[:, 110]))) > 1e-6:
            raise RuntimeError('Residual formula violated')
        altered += int(not np.array_equal(a, b))
    return {'arm': arm, 'bags': len(original), 'changed_bags': altered,
            'all_shapes_legal': True, 'all_residuals_recomputed': True}
