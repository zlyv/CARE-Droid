#!/usr/bin/env python3
"""BFA-V2 audit, fixed B13 retraining and evaluation. Reads frozen caches/checkpoints, never APKs; writes only outputs/care_background_complementarity_v2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT / 'scripts/care-background-complementarity-v2'
V1_CODE = ROOT / 'scripts/care-background-functional-attribution-v1'
V1 = ROOT / 'outputs/care_background_functional_attribution_v1'
OUT = ROOT / 'outputs/care_background_complementarity_v2'
sys.path.insert(0, str(V1_CODE))
from bfa_core import (SEEDS, SPLITS, asset_paths, anchors, digest, formal_split,
                      load_data_contract, load_m0, recompute_residual, role_ids,
                      transform_arm)
from metrics import cascade, paired_intervals
from train import split_contract, train_one

B0 = 'B0_ORIGINAL'
B1 = 'B1_EXTERNAL_NEUTRAL'
B3 = 'B3_INTERNAL_NEUTRAL'
B4 = 'B4_EXTERNAL_ROLE_SHUFFLE'
B13 = 'B13_EXTERNAL_INTERNAL_NEUTRAL'
ARMS = (B0, B1, B3, B4)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def put_json(name, value, resume=False):
    path = OUT / name
    if path.exists():
        if resume:
            return
        raise RuntimeError('refusing to overwrite ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def put_text(name, value, resume=False):
    path = OUT / name
    if path.exists():
        if resume:
            return
        raise RuntimeError('refusing to overwrite ' + str(path))
    with path.open('x', encoding='utf-8') as f:
        f.write(value)


def put_csv(name, rows, resume=False):
    path = OUT / name
    if path.exists():
        if resume:
            return
        raise RuntimeError('refusing to overwrite ' + str(path))
    rows = list(rows)
    cols = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, cols)
        writer.writeheader()
        writer.writerows(rows)


def immutable_assets():
    paths = list(asset_paths())
    paths += [V1_CODE / name for name in ('bfa_core.py', 'bfa_run.py', 'bfa_evaluate.py')]
    paths += [V1 / name for name in ('final_decision.json', 'per_apk_predictions.csv',
              'background_ablation_source2017.csv', 'background_ablation_rolling.csv',
              'fixed_broken_analysis.csv', 'conditional_auc_analysis.csv', 'baseline_replay_training.json')]
    for split in SPLITS:
        for seed in SEEDS:
            paths += [V1 / 'models' / split / str(seed) / (arm + '.pkl')
                      for arm in (B1, B3, B4)]
    return list(dict.fromkeys(paths))


def verify_assets():
    manifest = json.loads((OUT / 'asset_manifest_before.json').read_text())
    changed = []
    for item in manifest:
        try:
            now = digest(item['path'])
            if now != item:
                changed.append({'path': item['path'], 'before': item, 'after': now})
        except OSError as exc:
            changed.append({'path': item['path'], 'error': str(exc)})
    return {'status': 'PASS' if not changed else 'FAIL', 'checked': len(manifest),
            'changed': changed, 'scope': 'named reused frozen code, caches, routes, V1 predictions and checkpoints only; no APK files opened'}


def preflight(resume):
    if OUT.exists():
        if not resume:
            raise RuntimeError('V2 output already exists; use --resume only for this same run')
        if not (OUT / 'asset_manifest_before.json').is_file():
            raise RuntimeError('existing V2 output lacks integrity manifest')
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        (OUT / 'tmp').mkdir()
        manifest = [digest(p) for p in immutable_assets()]
        put_json('asset_manifest_before.json', manifest)
    check = verify_assets()
    if check['status'] != 'PASS':
        raise RuntimeError('frozen historical asset changed')
    put_json('asset_integrity_report.json', check, resume)
    return check


def checkpoint(split, seed, arm):
    if arm == B0:
        return load_m0(split, seed)
    path = V1 / 'models' / split / str(seed) / (arm + '.pkl')
    with path.open('rb') as f:
        obj = pickle.load(f)
    if (obj['arm'], obj['split'], obj['seed']) != (arm, split, seed):
        raise RuntimeError('V1 checkpoint identity mismatch ' + str(path))
    return obj


def aligned_v1():
    rows = read_csv(V1 / 'per_apk_predictions.csv')
    by_key = defaultdict(dict)
    for row in rows:
        key = (row['split'], int(row['year']), row['sha256'])
        if row['arm'] in by_key[key]:
            raise RuntimeError('duplicate V1 split/year/SHA/arm ' + str(key))
        by_key[key][row['arm']] = row
    if len(by_key) != 2788 or len(rows) != 2788 * 5:
        raise RuntimeError('V1 coverage mismatch: ' + str((len(by_key), len(rows))))
    data, source = load_data_contract()
    expected = set()
    for split in SPLITS:
        _, va, base, route, score = split_contract(data, split, source)
        ids = [str(x) for x in data['ids'][va]]
        for i, sid in enumerate(ids):
            key = (split, int(data['years'][va[i]]), sid)
            expected.add(key)
            group = by_key.get(key)
            if group is None or set(group) != set(('B0_ORIGINAL', 'B1_EXTERNAL_NEUTRAL',
                                                    'B2_SHARED_NEUTRAL', 'B3_INTERNAL_NEUTRAL', B4)):
                raise RuntimeError('V1 arms or APK missing ' + str(key))
            for arm, row in group.items():
                if int(row['label']) != int(data['y'][va[i]]) or int(row['routed']) != int(route[i]):
                    raise RuntimeError('label/route mismatch ' + str((key, arm)))
                if abs(float(row['stage1_probability']) - float(base[i])) > 1e-12:
                    raise RuntimeError('Stage-I probability mismatch ' + str((key, arm)))
                if np.isfinite(score[i]) and abs(float(row['router_score']) - float(score[i])) > 1e-12:
                    raise RuntimeError('router score mismatch ' + str((key, arm)))
                if not route[i] and abs(float(row['final_probability']) - float(base[i])) > 1e-12:
                    raise RuntimeError('unrouted decision changed ' + str((key, arm)))
                expected_final = float(row['expert_probability']) if route[i] else float(base[i])
                if abs(float(row['final_probability']) - expected_final) > 1e-12:
                    raise RuntimeError('expert/final mismatch ' + str((key, arm)))
        for arm in ARMS:
            probs = []
            for seed in SEEDS:
                obj = checkpoint(split, seed, arm)
                if obj['ids'] != ids or int(obj['selected_epochs']) != int(load_m0(split, seed)['selected_epochs']):
                    raise RuntimeError('V1 checkpoint SHA/epoch mismatch ' + str((split, seed, arm)))
                probs.append(np.asarray(obj['probability'], float))
            ensemble = np.mean(probs, axis=0)
            for i, sid in enumerate(ids):
                row = by_key[(split, int(data['years'][va[i]]), sid)][arm]
                if abs(float(row['expert_probability']) - float(ensemble[i])) > 1e-10:
                    raise RuntimeError('V1 CSV/checkpoint version mismatch ' + str((split, sid, arm)))
    if set(by_key) != expected:
        raise RuntimeError('V1 has extra or missing keys')
    return by_key


def logit(p):
    p = min(max(float(p), 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def analyze_existing(resume):
    if verify_assets()['status'] != 'PASS':
        raise RuntimeError('asset integrity failed')
    by_key = aligned_v1()
    put_json('bfa_v1_alignment_audit.json', {
        'status': 'PASS', 'unique_split_year_sha': len(by_key), 'arms': list(ARMS),
        'all_labels_stage1_routes_and_router_scores_equal': True,
        'all_V1_ensemble_probabilities_match_three_checkpoints': True,
        'unrouted_final_decisions_unchanged': True, 'threshold': 0.5,
        'source_is_five_disjoint_outer_folds': True, 'APK_files_opened': 0}, resume)
    # Preserve seed-level probability directions; seeds are stability checks,
    # never additional APK-level observations.
    data,source=load_data_contract()
    seed_lookup={}
    for split in SPLITS:
        _,va,_,_,_=split_contract(data,split,source)
        ids=[str(x) for x in data['ids'][va]]
        for arm in (B0,B1,B3):
            for seed in SEEDS:
                p=np.asarray(checkpoint(split,seed,arm)['probability'],float)
                for sid,value in zip(ids,p):seed_lookup[(split,sid,arm,seed)]=float(value)
    repair, damage, transitions, probability, seed_probability = [], [], [], [], []
    sets = defaultdict(lambda: defaultdict(set))
    for key, group in by_key.items():
        split, year, sid = key
        ref = group[B0]
        if int(ref['routed']) != 1:
            continue
        label = int(ref['label'])
        correctness = {arm: int(group[arm]['final_correct']) for arm in ARMS}
        kind = ('FN' if label else 'FP') if not correctness[B0] else ('TP' if label else 'TN')
        b1p = float(group[B1]['final_probability'])
        b3p = float(group[B3]['final_probability'])
        b0p = float(ref['final_probability'])
        delta1, delta3 = b1p - b0p, b3p - b0p
        category = (('REPAIR' if not correctness[B0] else 'DAMAGE') + '_' +
                    ('BOTH' if correctness[B1] == correctness[B3] != correctness[B0] else
                     'B1_ONLY' if correctness[B1] != correctness[B0] else
                     'B3_ONLY' if correctness[B3] != correctness[B0] else 'NEITHER'))
        item = {'split': split, 'year': year, 'sha256': sid, 'label': label,
                'B0_kind': kind, 'B0_correct': correctness[B0],
                'B1_correct': correctness[B1], 'B3_correct': correctness[B3],
                'B4_correct': correctness[B4], 'category': category,
                'B0_probability': b0p, 'B1_probability': b1p, 'B3_probability': b3p,
                'B1_minus_B0_probability': delta1, 'B3_minus_B0_probability': delta3,
                'B1_minus_B0_logit': logit(b1p)-logit(b0p),
                'B3_minus_B0_logit': logit(b3p)-logit(b0p),
                'opposite_delta_direction': int(delta1 * delta3 < 0),
                'B0_distance_to_0p5': abs(b0p - .5)}
        (damage if correctness[B0] else repair).append(item)
        probability.append(item)
        seed_row={'split':split,'year':year,'sha256':sid,'label':label,'B0_kind':kind}
        b1_sign=[];b3_sign=[]
        for seed in SEEDS:
            p0=seed_lookup[(split,sid,B0,seed)];p1=seed_lookup[(split,sid,B1,seed)];p3=seed_lookup[(split,sid,B3,seed)]
            seed_row[f'B0_seed_{seed}']=p0;seed_row[f'B1_seed_{seed}']=p1;seed_row[f'B3_seed_{seed}']=p3
            seed_row[f'B1_minus_B0_seed_{seed}']=p1-p0;seed_row[f'B3_minus_B0_seed_{seed}']=p3-p0
            b1_sign.append(np.sign(p1-p0));b3_sign.append(np.sign(p3-p0))
        seed_row['B1_all_seed_direction_same']=int(len(set(b1_sign))==1)
        seed_row['B3_all_seed_direction_same']=int(len(set(b3_sign))==1)
        seed_row['B1_B3_all_six_deltas_same_direction']=int(len(set(b1_sign+b3_sign))==1)
        seed_probability.append(seed_row)
        for arm in (B1, B3):
            if correctness[arm] != correctness[B0]:
                sets[(split, kind)][arm].add(sid)
        transitions.append({'split': split, 'year': year, 'B0_kind': kind,
                            'B0_correct': correctness[B0], 'B1_correct': correctness[B1],
                            'B3_correct': correctness[B3], 'count': 1})
    summary = []
    for split in SPLITS:
        for kind in ('FN','FP','TN','TP'):
            a = sets[(split, kind)][B1]
            b = sets[(split, kind)][B3]
            summary.append({'split': split, 'year': 2017 if split.startswith('S') else 2017+int(split[1:]),
                            'B0_kind': kind, 'B1_only': len(a-b), 'B3_only': len(b-a),
                            'both': len(a&b), 'union': len(a|b), 'B1_total': len(a), 'B3_total': len(b)})
    grouped = defaultdict(int)
    for row in transitions:
        grouped[(row['year'], row['B0_correct'], row['B1_correct'], row['B3_correct'], row['B0_kind'])] += 1
    matrix = [{'year': y, 'B0_correct': c0, 'B1_correct': c1, 'B3_correct': c3,
               'B0_kind': kind, 'count': n} for (y,c0,c1,c3,kind),n in sorted(grouped.items())]
    put_csv('per_apk_repair_overlap.csv', repair, resume)
    put_csv('per_apk_damage_overlap.csv', damage, resume)
    put_csv('repair_set_intersections.csv', [r for r in summary if r['B0_kind'] in ('FN','FP')], resume)
    put_csv('damage_set_intersections.csv', [r for r in summary if r['B0_kind'] in ('TN','TP')], resume)
    put_csv('prediction_transition_matrix.csv', matrix, resume)
    put_csv('b1_b3_probability_directions.csv', probability, resume)
    put_csv('b1_b3_seed_probability_directions.csv', seed_probability, resume)
    lines = ['# BFA-V1 B1/B3 逐APK互补性（训练B13前冻结）', '',
             '所有集合限制在相同冻结路由内；B0、B1、B3、B4的三种子集成与V1 checkpoint逐APK一致。',
             '不同重新训练模型的概率差不是独立背景因果效应。', '',
             '|窗口|原始错误种类|B1独有|B3独有|共同|并集|', '|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['B0_kind'] in ('FN','FP'):
            lines.append(f"|{r['split']}|{r['B0_kind']}|{r['B1_only']}|{r['B3_only']}|{r['both']}|{r['union']}|")
    oracle = {}
    for year in (2017,2018,2019,2020,2021):
        rows = [v for k,v in by_key.items() if k[1] == year]
        oracle[str(year)] = {'n': len(rows), 'nondeployment_oracle_errors': sum(
            not (int(v[B1]['final_correct']) or int(v[B3]['final_correct'])) for v in rows),
            'uses_true_labels': True}
    lines += ['', 'B1/B3按真实标签二选一的Oracle仅是非部署上界，不能用于路由或门控。',
              f'Oracle错误数：{oracle}。']
    put_text('b1_b3_complementarity_summary.md', '\n'.join(lines)+'\n', resume)
    put_json('nondeployment_oracle.json', oracle, resume)
    print('BFA_V2_EXISTING_ALIGNMENT_PASS 2788 APKs', flush=True)


def b13_transform(train_bags, val_bags):
    neutral = anchors(train_bags)
    def one(bags):
        out = []
        for bag in bags:
            changed = bag.copy()
            changed[:, 107] = neutral[0]
            changed[:, 108] = neutral[1]
            out.append(recompute_residual(changed))
        return out
    return one(train_bags), one(val_bags), neutral


def validate_b13(original, changed, neutral):
    count = 0
    for a,b in zip(original, changed):
        if a.shape != b.shape or a.shape[1] != 142 or not np.isfinite(b).all():
            raise RuntimeError('B13 bag shape/finite invalid')
        if not (np.array_equal(a[:, :107], b[:, :107]) and
                np.array_equal(a[:, 109], b[:, 109]) and
                np.array_equal(a[:, 111:], b[:, 111:])):
            raise RuntimeError('B13 changed non-target Host/Role input')
        if not (np.all(b[:,107] == neutral[0]) and np.all(b[:,108] == neutral[1])):
            raise RuntimeError('B13 neutral anchor mismatch')
        if np.max(np.abs(recompute_residual(b)[:,110] - b[:,110])) > 1e-6:
            raise RuntimeError('B13 stale residual')
        count += int(not np.array_equal(a,b))
    return {'n_bags': len(original), 'changed_bags': count, 'width': 142, 'residual_consistent': True}


def smoke(device, resume):
    data,_ = load_data_contract()
    pack = formal_split(data, 'S1', SEEDS[0], device)
    tr,va,neutral = b13_transform(pack['bags_train'], pack['bags_val'])
    b13check = validate_b13(pack['bags_val'], va, neutral)
    control = {}
    for ex, internal, arm in ((False,False,B0),(True,False,B1),(False,True,B3)):
        oldtr,oldva,_ = transform_arm(pack['bags_train'],pack['bags_val'],pack['raw_train'],pack['raw_val'],arm,SEEDS[0])
        def combo(bags):
            if not ex and not internal: return bags
            output=[]
            for bag in bags:
                b=bag.copy()
                if ex: b[:,107]=neutral[0]
                if internal: b[:,108]=neutral[1]
                output.append(recompute_residual(b))
            return output
        newtr,newva=combo(pack['bags_train']),combo(pack['bags_val'])
        differences = [float(np.max(np.abs(x-y))) for x,y in zip(oldtr+oldva,newtr+newva)]
        if max(differences)>0:
            raise RuntimeError('B0/B1/B3 V1 feature replay mismatch '+arm)
        control[arm]={'max_abs_difference':max(differences),'bags':len(differences)}
    report={'status':'PASS','split':'S1','seed':SEEDS[0], 'anchor':neutral.tolist(),
            'reference_modes':control,'B13':b13check,
            'Commonness_unchanged':True,'Host_order_Role_graph_unchanged':True,
            'B13_Residual_formula':'(1-clip(Commonness,0,1))*sqrt(max(External,0)*max(Internal,0))'}
    put_json('b13_feature_validation.json',report,resume)
    put_json('b13_dependency_consistency.json',{'status':'PASS','B0_B1_B3_exact':True,
        'B13_joint_neutralization_from_same_train_only_anchors':True,
        'Shared_Commonness_preserved':True,'Residual_recomputed':True,
        'no_new_APKs_or_Host_extractor':True},resume)
    # One-epoch full-fold interface test. RoleGraph batches are structured
    # dictionaries rather than indexable lists, so keep their exact frozen
    # alignment instead of inventing a slicing adapter for a smoke test.
    y=np.asarray(data['y'][pack['train']],int)
    pred,_,cost=train_one(pack['v4'],pack['args'],pack['dev'],SEEDS[0],1,
        pack['global_train'],pack['global_val'],
        pack['graph_train'],pack['graph_val'],tr,va,y)
    if pred.shape != (len(pack['val']),) or not np.isfinite(pred).all():
        raise RuntimeError('B13 one-epoch interface test failed')
    put_json('b13_training_selftest.json',{'status':'PASS','train_n':len(pack['train']),
        'test_n':len(pack['val']),
        'epochs':1,'parameter_n':cost['parameters'],'smoke_only':True},resume)
    primitive_args={k:v for k,v in vars(pack['args']).items()
                    if isinstance(v,(str,int,float,bool)) or v is None}
    put_json('b13_training_protocol.json',{
        'status':'FROZEN_BEFORE_FORMAL_B13_RESULTS','architecture':'RGCondAttn Direct-C43',
        'host_input_width':142,'global_input_width':24,'objective':'equal-weight BCE',
        'optimizer':'Adam','seeds':list(SEEDS),'threshold':.5,'router':'frozen MeanOpposition Top25%',
        'epochs':'reuse corresponding SCBE M0 selected_epochs without retuning',
        'runtime_arguments':primitive_args,'new_APKs_used':False},resume)
    print('BFA_V2_SMOKE_PASS',flush=True)


def train_phase(kind, device, resume):
    if verify_assets()['status']!='PASS': raise RuntimeError('asset integrity failed')
    if not (OUT/'b13_training_selftest.json').is_file(): raise RuntimeError('run --smoke-test first')
    data,_=load_data_contract()
    for split in (SPLITS[:5] if kind=='source' else SPLITS[5:]):
        for seed in SEEDS:
            path=OUT/'models'/split/str(seed)/(B13+'.pkl')
            if path.exists():
                if not resume: raise RuntimeError('existing checkpoint '+str(path))
                with path.open('rb') as f: old=pickle.load(f)
                expected=[str(x) for x in data['ids'][split_contract(data,split,load_data_contract()[1])[1]]]
                if old['arm']!=B13 or old['split']!=split or old['seed']!=seed or old['ids']!=expected:
                    raise RuntimeError('resume checkpoint mismatch '+str(path))
                print('BFA_V2_RESUME',split,seed,flush=True);continue
            pack=formal_split(data,split,seed,device)
            ref=pack['old']
            tr,va,neutral=b13_transform(pack['bags_train'],pack['bags_val'])
            validity=validate_b13(pack['bags_val'],va,neutral)
            if pack['dev'].type=='cuda': pack['v4'].torch.cuda.reset_peak_memory_stats(pack['dev'])
            start=time.perf_counter()
            pred,state,cost=train_one(pack['v4'],pack['args'],pack['dev'],seed,
                int(ref['selected_epochs']),pack['global_train'],pack['global_val'],
                pack['graph_train'],pack['graph_val'],tr,va,
                np.asarray(data['y'][pack['train']],int))
            obj={'arm':B13,'split':split,'seed':seed,
                 'ids':list(map(str,data['ids'][pack['val']])),
                 'train_ids':list(map(str,data['ids'][pack['train']])),
                 'train_n':len(pack['train']),'test_n':len(pack['val']),
                 'probability':pred,'state_dict':state,
                 'selected_epochs':int(ref['selected_epochs']),
                 'parameter_n':int(cost['parameters']),'cost':cost,
                 'feature_protocol':{'type':'joint information neutralization',
                    'neutral_anchor':neutral.tolist(),'estimator':'train-only equal-APK mean',
                    'changed_primitives':[107,108],'recomputed':[110]},
                 'intervention_validity':validity,'new_APKs_used':False,
                 'Stage1_and_Router_changed':False,
                 'elapsed_seconds':time.perf_counter()-start}
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
            print('BFA_V2_B13_DONE',split,seed,'epochs='+str(ref['selected_epochs']),flush=True)
    print('BFA_V2_TRAIN_'+kind.upper()+'_DONE',flush=True)


def eval_v2(resume):
    if verify_assets()['status']!='PASS': raise RuntimeError('asset integrity failed')
    by_key=aligned_v1()
    data,source=load_data_contract()
    per=[];metrics=[];conditional=[];fixed=[];runtime=[];paired=[];seeds=[]
    all_source=[]
    for split in SPLITS:
        tr,va,base,route,score=split_contract(data,split,source)
        ids=[str(x) for x in data['ids'][va]]
        y=np.asarray(data['y'][va],int)
        predictions=[]
        for seed in SEEDS:
            path=OUT/'models'/split/str(seed)/(B13+'.pkl')
            if not path.is_file():raise RuntimeError('missing B13 checkpoint '+str(path))
            with path.open('rb') as f:obj=pickle.load(f)
            if (obj['arm'],obj['split'],obj['seed'],obj['ids'])!=(B13,split,seed,ids):
                raise RuntimeError('B13 checkpoint identity mismatch '+str(path))
            predictions.append(np.asarray(obj['probability'],float))
            seeds.append({'split':split,'year':int(data['years'][va[0]]),'seed':seed,
                          **cascade(y,base,predictions[-1],route)})
            runtime.append({'split':split,'seed':seed,'arm':B13,
                            'selected_epochs':obj['selected_epochs'],
                            'train_n':obj['train_n'],'test_n':obj['test_n'],
                            **obj['cost'],'elapsed_seconds':obj['elapsed_seconds']})
        p=np.mean(predictions,axis=0)
        m=cascade(y,base,p,route)
        metrics.append({'split':split,'year':int(data['years'][va[0]]),'arm':B13,'seed':'ENSEMBLE',**m})
        conditional.append({'split':split,'year':int(data['years'][va[0]]),'arm':B13,
                            **{k:m[k] for k in ('Routed_AUC','FN_vs_TN_AUC','TP_vs_FP_AUC',
                                                 'FN_n','TN_n','TP_n','FP_n')}})
        fixed.append({'split':split,'arm':B13,**{k:m[k] for k in ('Fixed','Broken','Net','Fixed_FN',
                            'Fixed_FP','Broken_TN','Broken_TP','routed_errors','unrouted_errors')}})
        for i,sid in enumerate(ids):
            key=(split,int(data['years'][va[i]]),sid)
            old=by_key[key][B0]
            if abs(float(old['stage1_probability'])-float(base[i]))>1e-12 or int(old['routed'])!=int(route[i]):
                raise RuntimeError('V1/V2 frozen upstream mismatch '+str(key))
            final=float(p[i] if route[i] else base[i])
            per.append({'split':split,'year':int(data['years'][va[i]]),'sha256':sid,
                        'label':int(y[i]),'stage1_probability':float(base[i]),
                        'router_score':float(score[i]) if np.isfinite(score[i]) else '',
                        'routed':int(route[i]),'B13_expert_probability':float(p[i]),
                        'B13_final_probability':final,'B13_correct':int((final>=.5)==int(y[i])),
                        **{f'B13_seed_{seed}_probability':float(predictions[j][i]) for j,seed in enumerate(SEEDS)}})
            if split.startswith('S'):all_source.append((key,int(y[i]),float(base[i]),bool(route[i]),float(p[i])))
        for oldarm in (B0,B1,B3):
            oldp=np.asarray([float(by_key[(split,int(data['years'][va[i]]),sid)][oldarm]['expert_probability'])
                             for i,sid in enumerate(ids)])
            cand_correct=(np.where(route,p,base)>=.5)==y
            ref_correct=(np.where(route,oldp,base)>=.5)==y
            cand_only=int(np.sum(cand_correct&~ref_correct));ref_only=int(np.sum(~cand_correct&ref_correct))
            from scipy.stats import binomtest
            mc=float(binomtest(cand_only,cand_only+ref_only,.5).pvalue) if cand_only+ref_only else 1.0
            for stat in paired_intervals(y,base,route,p,oldp,repeats=500):
                paired.append({'split':split,'candidate':B13,'reference':oldarm,
                               'candidate_only_correct':cand_only,'reference_only_correct':ref_only,
                               'mcnemar_exact_p':mc,**stat})
    if len(all_source)!=799 or len({key for key,*_ in all_source})!=799:
        raise RuntimeError('source OOF incomplete')
    sy=np.asarray([x[1] for x in all_source]);sb=np.asarray([x[2] for x in all_source]);
    sr=np.asarray([x[3] for x in all_source]);sp=np.asarray([x[4] for x in all_source])
    m=cascade(sy,sb,sp,sr)
    metrics.append({'split':'SOURCE_2017_OOF','year':2017,'arm':B13,'seed':'ENSEMBLE',**m})
    conditional.append({'split':'SOURCE_2017_OOF','year':2017,'arm':B13,
        **{k:m[k] for k in ('Routed_AUC','FN_vs_TN_AUC','TP_vs_FP_AUC','FN_n','TN_n','TP_n','FP_n')}})
    fixed.append({'split':'SOURCE_2017_OOF','arm':B13,
        **{k:m[k] for k in ('Fixed','Broken','Net','Fixed_FN','Fixed_FP','Broken_TN','Broken_TP','routed_errors','unrouted_errors')}})
    for oldarm in (B0,B1,B3):
        op=np.asarray([float(by_key[key][oldarm]['expert_probability']) for key,*_ in all_source])
        cand_correct=(np.where(sr,sp,sb)>=.5)==sy
        ref_correct=(np.where(sr,op,sb)>=.5)==sy
        cand_only=int(np.sum(cand_correct&~ref_correct));ref_only=int(np.sum(~cand_correct&ref_correct))
        from scipy.stats import binomtest
        mc=float(binomtest(cand_only,cand_only+ref_only,.5).pvalue) if cand_only+ref_only else 1.0
        for stat in paired_intervals(sy,sb,sr,sp,op,repeats=500):
            paired.append({'split':'SOURCE_2017_OOF','candidate':B13,'reference':oldarm,
                           'candidate_only_correct':cand_only,'reference_only_correct':ref_only,
                           'mcnemar_exact_p':mc,**stat})
    # Holm adjustment is applied over every emitted split/reference comparison,
    # not over bootstrap metric rows or random seeds.
    keys=list(dict.fromkeys((r['split'],r['reference']) for r in paired))
    raw={(s,a):next(r['mcnemar_exact_p'] for r in paired if r['split']==s and r['reference']==a) for s,a in keys}
    order=sorted(keys,key=lambda k:raw[k]);adjusted={};running=0.0;m=len(order)
    for rank,key in enumerate(order):
        value=min(1.0,(m-rank)*raw[key]);running=max(running,value);adjusted[key]=running
    for r in paired:r['mcnemar_holm_p']=adjusted[(r['split'],r['reference'])]
    put_csv('b13_training_metrics.csv',seeds,resume)
    put_csv('b13_source2017_results.csv',[r for r in metrics if r['split'].startswith('S')],resume)
    put_csv('b13_rolling_results.csv',[r for r in metrics if r['split'].startswith('T')],resume)
    put_csv('b13_per_apk_predictions.csv',per,resume)
    put_csv('b13_fixed_broken_analysis.csv',fixed,resume)
    put_csv('b13_conditional_auc.csv',conditional,resume)
    put_csv('paired_statistical_analysis.csv',paired,resume)
    put_csv('runtime_cost.csv',runtime,resume)
    print('BFA_V2_EVALUATION_DONE',flush=True)


def combined(resume):
    by_key=aligned_v1()
    new={(r['split'],int(r['year']),r['sha256']):r for r in read_csv(OUT/'b13_per_apk_predictions.csv')}
    if set(by_key)!=set(new):raise RuntimeError('V1/B13 APK alignment mismatch')
    repair=[];damage=[];retention=[];fresh=[];nonadd=[]
    totals=defaultdict(lambda:Counter())
    for key,group in by_key.items():
        v=new[key]
        if int(v['routed'])!=1:continue
        c0=int(group[B0]['final_correct']);c1=int(group[B1]['final_correct']);
        c3=int(group[B3]['final_correct']);c13=int(v['B13_correct'])
        kind=('FN' if int(v['label']) else 'FP') if not c0 else ('TP' if int(v['label']) else 'TN')
        t=totals[(v['year'],kind)]
        t['n']+=1
        if not c0:
            if c1 and not c3:t['B1_unique']+=1;t['B1_retained']+=int(c13)
            if c3 and not c1:t['B3_unique']+=1;t['B3_retained']+=int(c13)
            if c1 and c3:t['joint_repair']+=1;t['joint_retained']+=int(c13)
            if c13 and not (c1 or c3):t['B13_new_repair']+=1
        else:
            if not c13 and c1 and c3:t['B13_new_damage']+=1
        x={'year':v['year'],'split':v['split'],'sha256':v['sha256'],'label':v['label'],
           'B0_kind':kind,'B0_correct':c0,'B1_correct':c1,'B3_correct':c3,'B13_correct':c13,
           'B0_probability':group[B0]['final_probability'],
           'B1_probability':group[B1]['final_probability'],
           'B3_probability':group[B3]['final_probability'],
           'B13_probability':v['B13_final_probability']}
        seed_correct=[]
        for seed in SEEDS:
            value=float(v[f'B13_seed_{seed}_probability'])
            x[f'B13_seed_{seed}_probability']=value
            x[f'B13_seed_{seed}_correct']=int((value>=.5)==int(v['label']))
            seed_correct.append(x[f'B13_seed_{seed}_correct'])
        x['B13_seed_correct_count']=sum(seed_correct)
        x['B13_all_seeds_same_correctness']=int(len(set(seed_correct))==1)
        if not c0:repair.append(x)
        else:damage.append(x)
        z=logit(v['B13_final_probability'])-logit(group[B1]['final_probability'])-\
          logit(group[B3]['final_probability'])+logit(group[B0]['final_probability'])
        nonadd.append({'year':v['year'],'split':v['split'],'sha256':v['sha256'],
                       'RETRAINED_MODEL_NONADDITIVITY':z,'causal_interaction':False})
    for (year,kind),c in sorted(totals.items()):
        retention.append({'year':year,'B0_kind':kind,'n':c['n'],
                          **{k:c[k] for k in ('B1_unique','B1_retained','B3_unique','B3_retained',
                                              'joint_repair','joint_retained','B13_new_repair')}})
        fresh.append({'year':year,'B0_kind':kind,'B13_new_damage':c['B13_new_damage'],
                      'B13_new_repair':c['B13_new_repair']})
    put_csv('combined_repair_retention.csv',retention,resume)
    put_csv('combined_new_damage.csv',fresh,resume)
    put_csv('combined_repair_details.csv',repair,resume)
    put_csv('combined_damage_details.csv',damage,resume)
    put_csv('2020_b0_routed_fn_tracking.csv',[r for r in repair if int(r['year'])==2020 and r['B0_kind']=='FN'],resume)
    put_csv('retrained_model_nonadditivity.csv',nonadd,resume)
    print('BFA_V2_COMPLEMENTARITY_DONE',flush=True)


def distribution(device,resume):
    """Training-referenced joint/full-input/Role/Host-size/activation audit; no test-label use."""
    data,_=load_data_contract()
    rows=[]; role_acc=defaultdict(lambda:{'hosts':0,'sum':np.zeros(4),'delta':np.zeros(4)})
    for split in SPLITS:
        # Bank is seed-specific; compare each seed with its own training-window reference.
        for seed in SEEDS:
            pack=formal_split(data,split,seed,device)
            transformed_train,transformed_val,neutral=b13_transform(pack['bags_train'],pack['bags_val'])
            original_train=pack['bags_train'];original_val=pack['bags_val']
            raw_train,raw_val=pack['raw_train'],pack['raw_val']
            original=np.asarray([np.mean(b[:,107:111],axis=0) for b in original_train],float)
            mu=np.mean(original,axis=0)
            cov=np.cov(original,rowvar=False)+np.eye(4)*1e-4
            inverse=np.linalg.inv(cov)
            host_train=np.concatenate(original_train,axis=0).astype(np.float64)
            feature_mean=np.mean(host_train,axis=0)
            feature_std=np.std(host_train,axis=0)+1e-6
            q01,q99=np.quantile(host_train,[.01,.99],axis=0)
            host_quartiles=np.quantile([len(b) for b in original_train],[.25,.5,.75])
            def dist(x):
                d=x-mu
                return float(np.sqrt(max(d@inverse@d,0)))
            reference=np.asarray([dist(x) for x in original])
            threshold=float(np.quantile(reference,.99))
            # Load B13 solely for first-layer activation auditing after training.
            b13_path=OUT/'models'/split/str(seed)/(B13+'.pkl')
            if not b13_path.is_file():raise RuntimeError('distribution audit missing '+str(b13_path))
            with b13_path.open('rb') as f:b13_obj=pickle.load(f)
            b13_model=pack['v4'].RoleGraphConditionedEvidence(24,'RGCondAttn',pack['args']).to(pack['dev'])
            b13_model.load_state_dict(b13_obj['state_dict']);b13_model.eval();pack['model'].eval()
            torch=pack['v4'].torch
            for group,oldbags,newbags,rawbags in (('train',original_train,transformed_train,raw_train),
                                           ('validation',original_val,transformed_val,raw_val)):
                ids=pack['train'] if group=='train' else pack['val']
                roles=role_ids(rawbags)
                for idx,a,b,rids in zip(ids,oldbags,newbags,roles):
                    x=np.mean(a[:,107:111],axis=0);z=np.mean(b[:,107:111],axis=0)
                    host_n=len(a)
                    stratum=('Q1' if host_n<=host_quartiles[0] else 'Q2' if host_n<=host_quartiles[1]
                             else 'Q3' if host_n<=host_quartiles[2] else 'Q4')
                    with torch.no_grad():
                        ta=torch.as_tensor(a[:,:111],dtype=torch.float32,device=pack['dev'])
                        tb=torch.as_tensor(b[:,:111],dtype=torch.float32,device=pack['dev'])
                        b0a=pack['model'].region_encoder[0](ta).detach().cpu().numpy()
                        b0b=pack['model'].region_encoder[0](tb).detach().cpu().numpy()
                        b13b=b13_model.region_encoder[0](tb).detach().cpu().numpy()
                    for arm,vec,bag,own,frozen_b0 in ((B0,x,a,b0a,b0a),(B13,z,b,b13b,b0b)):
                        out_of_range=np.mean((bag<q01)|(bag>q99))
                        rows.append({'split':split,'seed':seed,'group':group,'year':int(data['years'][idx]),
                                     'sha256':str(data['ids'][idx]),'arm':arm,'hosts':len(a),
                                     'host_count_stratum':stratum,
                                     'external_mean':float(vec[0]),'internal_mean':float(vec[1]),
                                     'commonness_mean':float(vec[2]),'residual_mean':float(vec[3]),
                                     'train_99pct_joint_distance':threshold,
                                     'joint_mahalanobis':dist(vec),
                                     'outside_train_99pct_joint':int(dist(vec)>threshold),
                                     'full142_standardized_rms_shift_from_original':float(np.sqrt(np.mean(((bag-a)/feature_std)**2))),
                                     'host_feature_fraction_outside_train_1_99pct':float(out_of_range),
                                     'own_model_first_linear_mean':float(np.mean(own)),
                                     'own_model_first_linear_std':float(np.std(own)),
                                     'own_model_first_linear_relu_zero_fraction':float(np.mean(own<=0)),
                                     'frozen_B0_first_linear_mean':float(np.mean(frozen_b0)),
                                     'frozen_B0_first_linear_std':float(np.std(frozen_b0)),
                                     'frozen_B0_first_linear_relu_zero_fraction':float(np.mean(frozen_b0<=0)),
                                     'reference':'original training-window APK-level 4D mean/cov + 1e-4I'})
                    for role in np.unique(rids):
                        mask=rids==role
                        key=(split,seed,group,int(role))
                        role_acc[key]['hosts']+=int(mask.sum())
                        role_acc[key]['sum']+=np.sum(b[mask,107:111],axis=0)
                        role_acc[key]['delta']+=np.sum(b[mask,107:111]-a[mask,107:111],axis=0)
            del b13_model
            if pack['dev'].type=='cuda':torch.cuda.empty_cache()
    put_csv('b13_distribution_shift.csv',rows,resume)
    role_rows=[]
    for (split,seed,group,role),x in sorted(role_acc.items()):
        n=x['hosts'];mean=x['sum']/max(n,1);delta=x['delta']/max(n,1)
        role_rows.append({'split':split,'seed':seed,'group':group,'role_id':role,'hosts':n,
                          'B13_external_mean':mean[0],'B13_internal_mean':mean[1],
                          'B13_commonness_mean':mean[2],'B13_residual_mean':mean[3],
                          'external_mean_delta':delta[0],'internal_mean_delta':delta[1],
                          'commonness_mean_delta':delta[2],'residual_mean_delta':delta[3]})
    put_csv('b13_distribution_by_role.csv',role_rows,resume)
    print('BFA_V2_DISTRIBUTION_DONE',flush=True)


def collect(resume):
    required=('bfa_v1_alignment_audit.json','b13_feature_validation.json',
              'b13_source2017_results.csv','b13_rolling_results.csv',
              'combined_repair_retention.csv','b13_distribution_shift.csv')
    if any(not (OUT/p).is_file() for p in required):raise RuntimeError('V2 incomplete; missing required phases')
    check=verify_assets()
    if check['status']!='PASS':raise RuntimeError('historical asset integrity changed')
    v1=read_csv(V1/'background_ablation_source2017.csv')+read_csv(V1/'background_ablation_rolling.csv')
    new=read_csv(OUT/'b13_source2017_results.csv')+read_csv(OUT/'b13_rolling_results.csv')
    def get(split,arm):
        pool=new if arm==B13 else v1
        return next(r for r in pool if r['split']==split and r['arm']==arm and r['seed']=='ENSEMBLE')
    tab=[]
    for split in ('SOURCE_2017_OOF','T1','T2','T3','T4'):
        for arm in (B0,B1,B3,B4,B13):
            r=get(split,arm)
            tab.append({'split':split,'year':r['year'],'arm':arm,
                        **{k:r[k] for k in ('BA','AUC','FN','FP','Fixed','Broken','Net',
                                              'FN_vs_TN_AUC','TP_vs_FP_AUC')}})
    put_csv('b13_comparison_table.csv',tab,resume)
    gains={}
    for arm in (B0,B1,B3):
        diffs=[float(get(t,B13)['BA'])-float(get(t,arm)['BA']) for t in ('T2','T3','T4')]
        net=sum(int(get(t,B13)['Net'])-int(get(t,arm)['Net']) for t in ('T2','T3','T4'))
        gains[arm]={'T2_T4_mean_BA_difference':float(np.mean(diffs)),
                    'T2_T4_Net_difference':net,'positive_years':sum(d>0 for d in diffs),
                    'T2_T4_FN_difference':sum(int(get(t,B13)['FN'])-int(get(t,arm)['FN']) for t in ('T2','T3','T4')),
                    'T2_T4_FP_difference':sum(int(get(t,B13)['FP'])-int(get(t,arm)['FP']) for t in ('T2','T3','T4')),
                    'source_BA_difference':float(get('SOURCE_2017_OOF',B13)['BA'])-float(get('SOURCE_2017_OOF',arm)['BA']),
                    'T1_BA_difference':float(get('T1',B13)['BA'])-float(get('T1',arm)['BA'])}
    shift=read_csv(OUT/'b13_distribution_shift.csv')
    shift_summary={}
    for arm in (B0,B13):
        rs=[r for r in shift if r['group']=='validation' and r['arm']==arm]
        shift_summary[arm]=sum(int(r['outside_train_99pct_joint']) for r in rs)/len(rs)
    abnormal=shift_summary[B13]>shift_summary[B0]+.1
    key_auc={'2019_TP_vs_FP':float(get('T2',B13)['TP_vs_FP_AUC'])-float(get('T2',B0)['TP_vs_FP_AUC']),
             '2020_FN_vs_TN':float(get('T3',B13)['FN_vs_TN_AUC'])-float(get('T3',B0)['FN_vs_TN_AUC']),
             '2021_FN_vs_TN':float(get('T4',B13)['FN_vs_TN_AUC'])-float(get('T4',B0)['FN_vs_TN_AUC'])}
    difficult_not_materially_worse=all(v>=-.05 for v in key_auc.values())
    baseline_effective=(gains[B0]['T2_T4_mean_BA_difference']>0 and gains[B0]['T2_T4_Net_difference']>0
                        and gains[B0]['positive_years']>=2 and gains[B0]['T2_T4_FN_difference']<=0
                        and gains[B0]['source_BA_difference']>=-.01 and gains[B0]['T1_BA_difference']>=-.01)
    incremental=all(gains[x]['T2_T4_mean_BA_difference']>0 and
                    gains[x]['T2_T4_Net_difference']>=0 and gains[x]['positive_years']>=2
                    for x in (B1,B3))
    complementary=baseline_effective and incremental and difficult_not_materially_worse and not abnormal
    harmful=(gains[B1]['T2_T4_mean_BA_difference']<0 and gains[B3]['T2_T4_mean_BA_difference']<0 and
             gains[B1]['T2_T4_Net_difference']<0 and gains[B3]['T2_T4_Net_difference']<0)
    statuses=[]
    if complementary:statuses.append('B13_COMPLEMENTARY_GAIN_SUPPORTED_ON_DEVELOPMENT')
    elif baseline_effective:statuses.append('B13_GAIN_OVER_BASELINE_ONLY')
    else:statuses.append('B13_NO_ADDITIVE_GAIN')
    if harmful:statuses.append('B13_HARMFUL_INTERACTION_CANDIDATE')
    if abnormal:statuses.append('B13_DISTRIBUTION_SHIFT_CONFOUNDED')
    if not complementary:statuses.append('NO_VERIFIED_BACKGROUND_COMPLEMENTARITY')
    status=statuses[0]
    decision={'status':status,'statuses':statuses,'comparisons':gains,
              'key_conditional_AUC_differences_vs_B0':key_auc,
              'difficult_APK_AUC_not_materially_worse_5pp_rule':difficult_not_materially_worse,
              'baseline_effective_rule':baseline_effective,'incremental_over_B1_and_B3_rule':incremental,
              'validation_joint_outside_99pct':shift_summary,
              'formal_integrity':check['status'],'new_APKs_used':False,
              'threshold':.5,'future_years_are_development':True,
              'formal_mainline_changed':False,
              'next_action':'仅开发集诊断；不自动修改正式三背景或引入新模块。'}
    put_json('asset_integrity_final.json',check,resume)
    put_json('final_decision.json',decision,resume)
    ret=read_csv(OUT/'combined_repair_retention.csv')
    dmg=read_csv(OUT/'combined_new_damage.csv')
    repair_rows=read_csv(OUT/'per_apk_repair_overlap.csv')
    damage_rows=read_csv(OUT/'per_apk_damage_overlap.csv')
    yearly=[]
    for year in (2017,2018,2019,2020,2021):
        for kind in ('FN','FP'):
            rr=[r for r in repair_rows if int(r['year'])==year and r['B0_kind']==kind]
            yearly.append({'year':year,'B0_kind':kind,'B0_errors':len(rr),
                'B1_only_repair':sum(int(r['B1_correct']) and not int(r['B3_correct']) for r in rr),
                'B3_only_repair':sum(int(r['B3_correct']) and not int(r['B1_correct']) for r in rr),
                'common_repair':sum(int(r['B1_correct']) and int(r['B3_correct']) for r in rr),
                'neither_repair':sum(not int(r['B1_correct']) and not int(r['B3_correct']) for r in rr)})
        dd=[r for r in damage_rows if int(r['year'])==year]
        yearly.append({'year':year,'B0_kind':'B0_CORRECT', 'B0_errors':0,
            'B1_only_damage':sum(not int(r['B1_correct']) and int(r['B3_correct']) for r in dd),
            'B3_only_damage':sum(not int(r['B3_correct']) and int(r['B1_correct']) for r in dd),
            'common_damage':sum(not int(r['B1_correct']) and not int(r['B3_correct']) for r in dd),
            'neither_damage':sum(int(r['B1_correct']) and int(r['B3_correct']) for r in dd)})
    put_csv('repair_damage_intersections_by_year.csv',yearly,resume)
    rr_all=[r for r in yearly if r['B0_kind'] in ('FN','FP')]
    b1_unique=sum(int(r['B1_only_repair']) for r in rr_all);b3_unique=sum(int(r['B3_only_repair']) for r in rr_all)
    common=sum(int(r['common_repair']) for r in rr_all)
    probability=read_csv(OUT/'b1_b3_probability_directions.csv')
    dev_prob=[r for r in probability if int(r['year']) in (2019,2020,2021)]
    d1=np.asarray([float(r['B1_minus_B0_probability']) for r in dev_prob]);d3=np.asarray([float(r['B3_minus_B0_probability']) for r in dev_prob])
    delta_corr=float(np.corrcoef(d1,d3)[0,1]) if len(d1)>1 and np.std(d1)>0 and np.std(d3)>0 else float('nan')
    opposite_fraction=float(np.mean(d1*d3<0)) if len(d1) else float('nan')
    align=json.loads((OUT/'bfa_v1_alignment_audit.json').read_text())
    report=['# CARE-Droid BFA-V2 双背景互补性验证', '',f'状态：**{"; ".join(statuses)}**。',
            '本轮仅用冻结2017–2021开发缓存；B13重新训练，其他臂复用V1。未读取新增APK，未改变Stage-I、Router、阈值或正式模型。',
            '', '## 五年对照', '',
            '|窗口|模型|BA|AUC|FN|FP|Fixed|Broken|Net|FN/TN AUC|TP/FP AUC|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in tab:
        report.append(f"|{r['split']}|{r['arm']}|{float(r['BA']):.4f}|{float(r['AUC']):.4f}|{r['FN']}|{r['FP']}|{r['Fixed']}|{r['Broken']}|{r['Net']}|{float(r['FN_vs_TN_AUC']):.4f}|{float(r['TP_vs_FP_AUC']):.4f}|")
    report+=['','## B13相对关键对照（T2–T4）','']
    for arm,x in gains.items():report.append(f"- 对{arm}：平均BA差{x['T2_T4_mean_BA_difference']:+.5f}；Net差{x['T2_T4_Net_difference']:+d}；正向年份{x['positive_years']}/3。")
    report+=['',f"关键条件AUC相对B0：2019 TP/FP {key_auc['2019_TP_vs_FP']:+.4f}；2020 FN/TN {key_auc['2020_FN_vs_TN']:+.4f}；2021 FN/TN {key_auc['2021_FN_vs_TN']:+.4f}。",
             '预注册实现判据：T2–T4需优于B0且Net增加、至少2年同向、FN不增、source/T1 BA退化不超过1pp；还需同时超过B1/B3、关键条件AUC不恶化超过5pp，并通过联合分布检查。']
    report+=['',f'原始/联合中性化验证集APK级四维联合响应超出训练期99%支持范围比例：{shift_summary[B0]:.3f}/{shift_summary[B13]:.3f}。',
             '该诊断使用各折/种子的原始训练窗口APK级四维均值和协方差（正则1e-4I）；不能单独证明真实程序输入可实现性。',
             '', '## 十五项判断',
             f'1. B1/B3的修复集合并非相同：五年合计B1独有{b1_unique}、B3独有{b3_unique}、共同{common}；逐年FN/FP见`repair_damage_intersections_by_year.csv`。',
             f'2. B1和B3共同修复{common}个B0错误APK。',
             f'3. B1独有修复{b1_unique}个；4. B3独有修复{b3_unique}个。',
             '5. B1正确/B3错误及相反样本的SHA、标签和概率见`per_apk_repair_overlap.csv`及`per_apk_damage_overlap.csv`，没有以净FN/FP代替逐APK状态。',
             f'6. T2–T4路由APK上两种概率变化Pearson相关为{delta_corr:+.4f}，反向变化比例为{opposite_fraction:.3f}；重训练模型差值不是独立因果贡献。',
             '7. B0/B1/B3的四种开关特征构造精确重放，B13只联合使用同一训练窗口锚点并重算Residual；验证状态PASS。',
             f"8. 2017 source中B13相对B0的BA差为{gains[B0]['source_BA_difference']:+.5f}。",
             '9. 2018–2021逐年BA、AUC、FN、FP、Fixed/Broken/Net均在上表完整报告，未删除不利年份。',
             f"10. T2–T4平均BA：B13相对B0/B1/B3分别为{gains[B0]['T2_T4_mean_BA_difference']:+.5f}/{gains[B1]['T2_T4_mean_BA_difference']:+.5f}/{gains[B3]['T2_T4_mean_BA_difference']:+.5f}。",
             f"11. 相对B0的T2–T4 FN/FP总差为{gains[B0]['T2_T4_FN_difference']:+d}/{gains[B0]['T2_T4_FP_difference']:+d}；是否同时减少由这两个实际方向决定。",
             f"12. 相对B0的T2–T4 Net差为{gains[B0]['T2_T4_Net_difference']:+d}；各年Fixed/Broken见上表和`combined_*`文件。",
             f"13. 关键条件AUC差：2019 TP/FP {key_auc['2019_TP_vs_FP']:+.4f}，2020 FN/TN {key_auc['2020_FN_vs_TN']:+.4f}，2021 FN/TN {key_auc['2021_FN_vs_TN']:+.4f}。",
             f'14. B13验证集联合响应超出训练99%支持范围的比例为{shift_summary[B13]:.3f}（B0={shift_summary[B0]:.3f}）；另报告142维尺度、首层激活、Role和Host数量分层。',
             ('15. 本轮只形成开发集联合消融候选，正式三背景保持不变，下一步需独立数据验证。' if complementary else
              '15. 没有证明两种收益可可靠组合；保留正式三背景，不应把B13或单背景消融替换为主线。'),
             '', '## 可复核性与边界',
             f"V1逐APK/三种子校验：{align['status']}。冻结旧资产SHA256校验：{check['status']}（{check['checked']}项）。",
             'B13同时使用B1/B3训练期等APK类盲锚点，保留Commonness并重算Residual。`RETRAINED_MODEL_NONADDITIVITY`不是单模型Shapley或程序行为因果交互。',
             '三种子不能当作三个独立APK；2018–2021为开发评估，不是独立盲测。',
             '本实验不支持凭结果继续搜索锚点、阈值、路由或新Loss。']
    put_text('final_report.md','\n'.join(report)+'\n',resume)
    put_text('reproduce_commands.md','# BFA-V2 复现顺序\n\n旧输出不可覆盖；仅在新版本目录中运行。\n\n'
        '`bash scripts/care-background-complementarity-v2/run.sh --audit-only`\n'
        '`bash scripts/care-background-complementarity-v2/run.sh --analyze-existing`\n'
        '`bash scripts/care-background-complementarity-v2/run.sh --smoke-test --device cuda`\n'
        '`bash scripts/care-background-complementarity-v2/run.sh --train-source --device cuda --resume`\n'
        '`bash scripts/care-background-complementarity-v2/run.sh --train-rolling --device cuda --resume`\n'
        '`bash scripts/care-background-complementarity-v2/run.sh --evaluate --resume`\n'
        '`bash scripts/care-background-complementarity-v2/run.sh --analyze-complementarity --resume`\n'
        '`bash scripts/care-background-complementarity-v2/run.sh --distribution-shift --device cuda --resume`\n'
        '`bash scripts/care-background-complementarity-v2/run.sh --collect --resume`\n',resume)
    print('BFA_V2_FINAL',status,flush=True)


def main():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    for name in ('audit-only','analyze-existing','smoke-test','train-source','train-rolling',
                 'evaluate','analyze-complementarity','distribution-shift','collect'):
        g.add_argument('--'+name,action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if a.workers!=1:raise RuntimeError('fixed single-worker protocol; do not contend with other experiments')
    if a.device=='cuda':
        import torch
        t=torch.zeros(1,device='cuda');del t
    if a.audit_only:return preflight(a.resume)
    if not (OUT/'asset_manifest_before.json').is_file() or verify_assets()['status']!='PASS':
        raise RuntimeError('run --audit-only first; assets must remain unchanged')
    if a.analyze_existing:return analyze_existing(a.resume)
    if not (OUT/'bfa_v1_alignment_audit.json').is_file():raise RuntimeError('run --analyze-existing first')
    if a.smoke_test:return smoke(a.device,a.resume)
    if a.train_source:return train_phase('source',a.device,a.resume)
    if a.train_rolling:return train_phase('rolling',a.device,a.resume)
    if a.evaluate:return eval_v2(a.resume)
    if a.analyze_complementarity:return combined(a.resume)
    if a.distribution_shift:return distribution(a.device,a.resume)
    if a.collect:return collect(a.resume)


if __name__=='__main__':main()
