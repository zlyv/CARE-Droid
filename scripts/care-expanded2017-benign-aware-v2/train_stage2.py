#!/usr/bin/env python3
"""Run original Direct-C43 with old, union and class-balanced 2017 training.

Existing CARE-Droid training functions are called without editing them. Their
write root is redirected to EXPANDED_OUTPUT in this process. Future test labels
remain masked (-1) until the separate evaluation step.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from audit_new_apks import OUT, PROJECT, write_csv

MYD = PROJECT / 'scripts/care-multiyear-development-v1'
OLD = PROJECT / 'outputs/care_multiyear_development_v1'


def original_stage2_module():
    """Load the existing implementation without importing this same-named file."""
    from myd_common import module
    return module('_expanded_original_myd_stage2', MYD / 'train_stage2.py')


def class_balanced_epochs(v4, model, optimizer, g, base, graph, bags, y, weights,
                          correct, epochs, args, seed, years):
    torch = v4.torch; device = g.device
    bl = torch.as_tensor(v4.probability_to_logit(base), dtype=torch.float32, device=device)
    b = torch.as_tensor(base, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    wt = torch.as_tensor(weights, dtype=torch.float32, device=device)
    ct = torch.as_tensor(correct, dtype=torch.bool, device=device)
    rng = np.random.RandomState(seed+811)
    neg = np.flatnonzero(np.asarray(y) == 0); pos = np.flatnonzero(np.asarray(y) == 1)
    if not len(neg) or not len(pos): raise RuntimeError('C3 requires both classes')
    draws = []; final = float('nan')
    for epoch in range(int(epochs)):
        model.train()
        n = len(y); order = np.concatenate([rng.choice(neg, n//2, replace=True),
                                             rng.choice(pos, n-n//2, replace=True)])
        rng.shuffle(order)
        draws.append({'epoch': epoch+1, 'total': n, 'unique_apks': int(len(set(order))),
                      'benign_draws': int(np.sum(np.asarray(y)[order] == 0)),
                      'malware_draws': int(np.sum(np.asarray(y)[order] == 1))})
        total = 0.0
        for start in range(0, n, args.batch_size):
            ix = order[start:start+args.batch_size]; optimizer.zero_grad(set_to_none=True)
            logits = v4.model_forward_batch(model, g, bl, graph, bags, ix)
            it = torch.as_tensor(ix, dtype=torch.long, device=device)
            loss = v4.batch_loss(logits, yt[it], wt[it], b[it], ct[it], args)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); total += float(loss.detach().cpu())*len(ix)
        final = total/n
    return final, draws


def old_frozen_predict(data, test, device, original):
    from myd_common import runtime
    v4, _, args = runtime(); args.device = device
    dev = v4.resolve_device(device)
    rows = data['feature_rows']['rows']; old = list(rows[:799]); targets = [rows[i] for i in test]
    _, gx, names = v4.audit.compact_matrix_variant(v4.VARIANT, old, targets,
                data['feature_rows']['compact_by_sha'], data['feature_rows']['schema'],
                data['feature_rows']['canonical_names'])
    gi, _ = v4.evidence.split_view_indices(names)
    global_target = np.asarray(gx[:,gi], np.float32)
    raw = original.raw_bags(data, test)
    result = []
    for seed in original.SEEDS:
        path = OLD / 'models/stage2/C0R_2017' / (str(seed)+'.pkl')
        with path.open('rb') as f: checkpoint = pickle.load(f)
        prepared = original.transform_saved(v4, raw, checkpoint['background'], args)
        graphs = v4.transform_role_graphs(data['s7'][test], checkpoint['graph_scaler'])
        model = v4.RoleGraphConditionedEvidence(24, 'RGCondAttn', args).to(dev)
        model.load_state_dict(checkpoint['state_dict'])
        p, _ = v4.predict_model(model, global_target, np.full(len(test), .5),
                                graphs, prepared, dev)
        result.append(p)
        print('C0_OLD_FROZEN_SEED_DONE', seed, flush=True)
    return np.mean(result, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-stage2', action='store_true', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    args_cli = parser.parse_args()
    if str(MYD) not in sys.path: sys.path.insert(0, str(MYD))
    import myd_common
    original = original_stage2_module()
    original.OUTPUT = OUT
    myd_common.OUTPUT = OUT
    with np.load(OUT/'stage2_data.npz', allow_pickle=False) as a:
        data = {k:a[k] for k in a.files}
    with (OUT/'stage2_feature_rows.pkl').open('rb') as f:
        data['feature_rows'] = pickle.load(f)
    data['signature'] = hashlib.sha256(('expanded2017-v2:'+','.join(data['ids'])).encode()).hexdigest()
    old = np.arange(799); train = np.flatnonzero(data['years']==2017)
    test = np.flatnonzero(data['years']>2017)
    if len(old)!=799 or len(train)<=799 or not len(test): raise RuntimeError('Stage-II pool malformed')
    if not np.all(data['y'][test] == -1): raise RuntimeError('test labels entered Stage-II data')
    frozen = old_frozen_predict(data, test, args_cli.device, original)
    _, _ = original.fit(data, old, test, 'C1_OLD_RETRAIN', args_cli.device)
    union, epochs = original.fit(data, train, test, 'C2_UNION', args_cli.device)
    original.train_balanced = class_balanced_epochs
    balanced, _ = original.fit(data, train, test, 'C3_UNION_BALANCED',
                               args_cli.device, epoch_reference=epochs)
    with np.load(OUT/'predictions/C1_OLD_RETRAIN/expert.npz', allow_pickle=False) as a:
        old_retrain = np.asarray(a['probability'], float)
    entries = []
    for i, index in enumerate(test):
        entries.append({'sha256': str(data['ids'][index]), 'year': int(data['years'][index]),
            'C0_OLD_FROZEN': float(frozen[i]), 'C1_OLD_RETRAIN': float(old_retrain[i]),
            'C2_UNION': float(union[i]), 'C3_UNION_BALANCED': float(balanced[i])})
    write_csv(OUT/'stage2_expert_predictions_label_free.csv', entries)
    (OUT/'stage2_training_audit.json').write_text(json.dumps({
        'status': 'PASS', 'old_train_n': 799, 'union_train_n': len(train),
        'routed_test_complete': len(test), 'future_labels_loaded': 0,
        'C2': 'original Direct-C43, original 32+24+internal background, equal BCE',
        'C3': 'same network/background; fixed epoch count from C2; class-balanced APK draw with replacement',
        'training_outputs_within': str(OUT), 'formal_source_modified': False}, indent=2)+'\n')
    print('EXPANDED_STAGE2_TRAIN_DONE', len(test), flush=True)


if __name__ == '__main__': main()
