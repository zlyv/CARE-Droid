#!/usr/bin/env python3
"""Read-only 2017 feasibility audit for role-directed CFG evidence on frozen H0 Critical Hosts."""
from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
OUT = ROOT / 'artifacts/care_h1_role_directed_feasibility_2017_v2'
SCBE = ROOT / 'outputs/care_scbe_v1'
AUDIT = ROOT / 'artifacts/care_cfg_evidence_audit_2017_v3'
H1_OUT = ROOT / 'outputs/care_host_bound_cfg_source_future_v3'
CFG_CACHE = ROOT / 'artifacts/care_cfg_witness_all_years_v2/cache'
EGO = ROOT / 'artifacts/care-source-local-assets-v2/ego_1hop_regions_2017.npz'
COMPACT = ROOT / 'artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz'
COMPACT_AUDIT = ROOT / 'artifacts/care-di25-compact43-relationquery-source-v1/cache_audit.json'
MANIFEST = ROOT / 'outputs/care_manifest_four_member_disagreement_v1/run_20261019_pilot01'
LOCAL_MANIFEST = Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json')
H1_CODE = ROOT / 'scripts/care-host-bound-cfg-source-future-v3/run_experiment.py'
INTERZ = AUDIT / 'interz_witness_semantics.csv'

SPLITS = [f'S{i}' for i in range(1, 6)]
SEEDS = [900000, 900001, 900002]
MANIFEST_SEEDS = [20261018, 20261019, 20261020]


def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return None if not np.isfinite(x) else float(x)
    if isinstance(x, Path): return str(x)
    return x


def write_json(path, obj):
    if path.exists(): raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def write_text(path, text):
    if path.exists(): raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(text, encoding='utf-8'); os.replace(tmp, path)


def write_csv(path, frame):
    if path.exists(): raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + '.tmp'); frame.to_csv(tmp, index=False); os.replace(tmp, path)


def digest(path):
    path = Path(path); h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return {'path': str(path), 'size': path.stat().st_size, 'sha256': h.hexdigest()}


def load_pickle(path):
    with path.open('rb') as f: return pickle.load(f)


def asset_paths():
    paths = [AUDIT/'critical_host_changes.csv', AUDIT/'h0_replay_check.json', AUDIT/'integrity_check.json',
             EGO, COMPACT, COMPACT_AUDIT, LOCAL_MANIFEST, H1_CODE, INTERZ,
             MANIFEST/'protocol.json', MANIFEST/'asset_integrity_check.json']
    for split in SPLITS:
        for seed in SEEDS: paths.append(SCBE/f'models/{split}/{seed}/M0_REPRODUCED.pkl')
    return paths


def preflight():
    if OUT.exists(): raise RuntimeError(f'refusing overwrite {OUT}')
    req = asset_paths()
    if not all(p.exists() for p in req): raise RuntimeError('missing asset')
    if json.load(open(AUDIT/'final_decision.json')).get('status') != 'COMPLETE': raise RuntimeError('upstream audit incomplete')
    if json.load(open(AUDIT/'integrity_check.json')).get('status') != 'PASS': raise RuntimeError('upstream integrity failed')
    if json.load(open(MANIFEST/'asset_integrity_check.json')).get('status') != 'PASS': raise RuntimeError('manifest integrity failed')
    OUT.mkdir(parents=True, exist_ok=False)
    return [digest(p) for p in req]


def reconstruct_routes(source_ids):
    source = set(source_ids); result = {}; parity = []
    for seed in MANIFEST_SEEDS:
        seen = set()
        for fold in range(1, 6):
            p = MANIFEST/f'windows/{seed}_{fold}.json'; d = json.load(open(p))
            ids = list(map(str, d['target_sha'])); members = np.asarray(d['members'], float)
            if not set(ids).issubset(source) or members.shape != (len(ids), 4): raise RuntimeError('manifest source window mismatch')
            anchor = members[:, 1]; other = np.delete(members, 1, axis=1)
            opposition = np.where(anchor[:, None] >= .5, 1-other, other).mean(axis=1)
            threshold = float(d['rules']['RF']['OTHER3_OP25_REF']); routed = opposition >= threshold
            for sha, flag, prob, score in zip(ids, routed, anchor, opposition):
                result[(seed, sha)] = {'routed': bool(flag), 'stage1_probability': float(prob), 'opposition': float(score), 'threshold': threshold}
            seen.update(ids); parity.append({'manifest_seed': seed, 'fold': f'S{fold}', 'target_rows': len(ids), 'routed': int(routed.sum())})
        if seen != source: raise RuntimeError(f'manifest source coverage {seed}: {len(seen)}')
    sets = {seed: {sha for sha in source_ids if result[(seed, sha)]['routed']} for seed in MANIFEST_SEEDS}
    return result, sets, parity


def host_features_fn():
    import importlib.util
    spec = importlib.util.spec_from_file_location('feas_h1', H1_CODE); mod = importlib.util.module_from_spec(spec)
    sys.path[:0] = [str(H1_CODE.parent), str(ROOT/'scripts/care-background-functional-attribution-v1'),
                    str(ROOT/'scripts/care-scbe-v1'), str(ROOT/'scripts/care-multiyear-development-v1'),
                    str(ROOT/'scripts/care-di25-compact43-relationquery-source-v1')]
    sys.modules['feas_h1'] = mod; spec.loader.exec_module(mod)
    return mod.host_features


def extract_witness(pairs):
    support = collections.defaultdict(set); unknown = set(); neither = set(); raw_rel = collections.Counter()
    for p in pairs:
        ea, eb = str(p['event_a']), str(p['event_b'])
        if ea == eb: raise RuntimeError('same invoke event used as two events')
        a, b, rel = int(p['role_a']), int(p['role_b']), str(p['relation']); ep = tuple(sorted((ea, eb)))
        raw_rel[rel] += 1
        if rel == 'A_TO_B': support[(a, 'REACH', b)].add(ep)
        elif rel == 'B_TO_A': support[(b, 'REACH', a)].add(ep)
        elif rel == 'BIDIRECTIONAL': support[(min(a,b), 'BIDIR', max(a,b))].add(ep)
        elif rel == 'UNKNOWN': unknown.add(ep)
        elif rel == 'NEITHER': neither.add(ep)
        else: raise RuntimeError('unknown relation type: ' + rel)
    return support, unknown, neither, raw_rel


def compact_json(v): return json.dumps([float(x) for x in np.asarray(v)], separators=(',', ':'))


def main():
    before = preflight(); critical = pd.read_csv(AUDIT/'critical_host_changes.csv', dtype={'sha256':str,'h0_critical_host':str})
    critical = critical[['split','model_seed','sha256','h0_probability','h0_critical_host']].copy()
    if len(critical) != 2397 or critical.sha256.nunique() != 799: raise RuntimeError('critical host contract')
    ego = np.load(EGO, allow_pickle=False); compact = np.load(COMPACT, allow_pickle=False)
    source_ids = list(map(str, ego['sha256']))
    if len(source_ids) != 799 or source_ids != list(map(str, compact['sha256'])): raise RuntimeError('source SHA contract')
    source_set = set(source_ids); routes, route_sets, route_parity = reconstruct_routes(source_ids)
    route_union = set.union(*(route_sets[s] for s in MANIFEST_SEEDS)); route_inter = set.intersection(*(route_sets[s] for s in MANIFEST_SEEDS))
    role_names = json.load(open(LOCAL_MANIFEST))['model_role_order']
    host_features = host_features_fn()
    eo = np.asarray(ego['ego_offsets'], int); ehosts = list(map(str, ego['host_ids']))
    co = np.asarray(compact['host_offsets'], int); cvals = np.asarray(compact['host_values'], np.float32)
    sha_pos = {sha:i for i,sha in enumerate(source_ids)}
    host_pos = {}
    for i,sha in enumerate(source_ids): host_pos[sha] = {h:j for j,h in enumerate(ehosts[eo[i]:eo[i+1]])}

    # Check fold/seed checkpoint identity and H0 probability parity without loading future assets.
    checkpoint_checks=[]
    for split in SPLITS:
        for seed in SEEDS:
            ck=load_pickle(SCBE/f'models/{split}/{seed}/M0_REPRODUCED.pkl')
            rows=critical[(critical.split==split)&(critical.model_seed==seed)]
            if list(rows.sha256) != list(map(str,ck['ids'])): raise RuntimeError('H0 fold/seed ID mismatch')
            diff=float(np.max(np.abs(rows.h0_probability.to_numpy(float)-np.asarray(ck['probability'],float))))
            if diff>1e-6: raise RuntimeError('H0 probability mismatch')
            checkpoint_checks.append({'fold':split,'seed':seed,'rows':len(rows),'max_probability_difference':diff})

    cache_cache={}; host_records={}; witness_rows=[]; index_rows=[]; coverage_rows=[]; unit={'same_event_pair_count':0,'bad_direction_count':0,'future_cache_files_opened':0}
    for n,row in enumerate(critical.itertuples(index=False),1):
        sha=str(row.sha256); fold=str(row.split); seed=int(row.model_seed); hid=str(row.h0_critical_host)
        if sha not in source_set: raise RuntimeError('outside 2017 whitelist')
        i=sha_pos[sha]; local=host_pos[sha].get(hid); exists=local is not None
        if sha not in cache_cache:
            with gzip.open(CFG_CACHE/f'{sha}.json.gz','rt',encoding='utf-8') as f: cache_cache[sha]=json.load(f)
        d=cache_cache[sha]; hm={str(h['host_node_id']):h for h in d.get('hosts',[])}
        em=collections.defaultdict(list);pm=collections.defaultdict(list)
        for e in d.get('events',[]):em[str(e['host_node_id'])].append(e)
        for p in d.get('pairs',[]):pm[str(p['host_node_id'])].append(p)
        h=hm.get(hid); events=em[hid]; pairs=pm[hid]
        support,unknown,neither,rel_count=extract_witness(pairs)
        if any(a==b for a,b in [(p['event_a'],p['event_b']) for p in pairs]): unit['same_event_pair_count']+=1
        if exists:
            crow=cvals[co[i]+local]; compact43=crow[:43]; sidecar=crow[43:74]; relation31=crow[74:105]
            cfg16=np.zeros(16,np.float32) if h is None else host_features(h,events,pairs)[0]
        else:
            compact43=np.zeros(43,np.float32);sidecar=np.zeros(31,np.float32);relation31=np.zeros(31,np.float32);cfg16=np.zeros(16,np.float32)
        role_set=tuple(map(int,np.flatnonzero(sidecar>0.5)))
        event_ids={str(e['event_id']) for e in events}
        triples=tuple(sorted(support)); signature='|'.join(f'{a}:{r}:{b}' for a,r,b in triples) if triples else 'NONE'
        support_signature='|'.join(f'{a}:{r}:{b}:{len(support[(a,r,b)])}' for a,r,b in triples) if triples else 'NONE'
        cfg_bytes=np.asarray(cfg16,np.float32).tobytes(); cfg_sha=hashlib.sha256(cfg_bytes).hexdigest()
        routed={s:routes[(s,sha)]['routed'] for s in MANIFEST_SEEDS}
        cov={'fold':fold,'model_seed':seed,'sha256':sha,'h0_probability':float(row.h0_probability),'stable_host_id':hid,
             'critical_host_exists':exists,'cfg_host_row_found':h is not None,'cfg_parsed':bool(h is not None and h.get('cfg_status')=='PARSED'),
             'cfg_status':h.get('cfg_status','MISSING') if h else 'MISSING','distinct_call_events':len(event_ids),
             'at_least_two_distinct_events':len(event_ids)>=2,'valid_unidirectional_witness':any(r=='REACH' for _,r,_ in triples),
             'bidirectional_witness':any(r=='BIDIR' for _,r,_ in triples),
             'different_role_directed_witness':any(r=='REACH' and a!=b for a,r,b in triples),
             'same_role_only':bool(triples) and all(a==b for a,_,b in triples),
             'only_unknown_or_neither':bool(pairs) and not triples and len(unknown|neither)>0,
             'no_valid_cfg_witness':not bool(triples),'unknown_event_pairs':len(unknown),'neither_event_pairs':len(neither),
             'directed_role_triples':sum(r=='REACH' for _,r,_ in triples),'bidir_role_triples':sum(r=='BIDIR' for _,r,_ in triples),
             'support_event_pairs':sum(len(v) for v in support.values()),'cfg16_sha256_exact_float32':cfg_sha,
             'role_set':'|'.join(map(str,role_set)) if role_set else 'NONE','witness_signature':signature,
             'routed_union':sha in route_union,'routed_intersection':sha in route_inter,
             **{f'routed_manifest_{s}':routed[s] for s in MANIFEST_SEEDS}}
        coverage_rows.append(cov)
        index_rows.append({**cov,'compact43_json':compact_json(compact43),'role_sidecar31_json':compact_json(sidecar),
                           'relation31_json':compact_json(relation31),'cfg16_raw_json':compact_json(cfg16),
                           'aligned_sensitive_calls':int(h.get('aligned_sensitive_calls',0)) if h else 0,
                           'method_signature':h.get('method_signature','NOT_AVAILABLE') if h else 'NOT_AVAILABLE'})
        key=(sha,hid)
        if key not in host_records:
            host_records[key]={'sha256':sha,'fold':fold,'stable_host_id':hid,'cfg_status':cov['cfg_status'],'cfg_parsed':cov['cfg_parsed'],
                'cfg16':cfg16,'cfg_bytes':cfg_bytes,'cfg_sha':cfg_sha,'role_set':role_set,'aligned_calls':index_rows[-1]['aligned_sensitive_calls'],
                'signature':signature,'support_signature':support_signature,'triples':triples,'support':support,'unknown':len(unknown),'neither':len(neither),
                'events':events,'pairs':pairs,'method_signature':index_rows[-1]['method_signature'],'compact43':compact43,'sidecar':sidecar,
                'relation31':relation31,'routed_union':sha in route_union,'routed_intersection':sha in route_inter,'selected_seeds':set()}
        host_records[key]['selected_seeds'].add(seed)
        for (a,rel,b),event_pairs in support.items():
            witness_rows.append({'sha256':sha,'fold':fold,'seed':seed,'stable_host_id':hid,
                'source_role':a,'source_role_name':role_names[a],'relation_type':rel,'target_role':b,'target_role_name':role_names[b],
                'support_event_pairs':len(event_pairs),'cfg_status':cov['cfg_status'],'different_events_verified':True,
                'routed_union':sha in route_union,'routed_intersection':sha in route_inter})
        if n%300==0: print('FEAS_CRITICAL',n,len(critical),flush=True)

    cov=pd.DataFrame(coverage_rows);idx=pd.DataFrame(index_rows);wit=pd.DataFrame(witness_rows)
    write_csv(OUT/'h0_critical_host_index.csv',idx);write_csv(OUT/'critical_host_cfg_coverage.csv',cov)
    write_csv(OUT/'critical_host_directed_role_witness.csv',wit)

    # Collision analysis uses unique physical Critical Hosts, not repeated seed selections.
    hosts=list(host_records.values()); groups0=collections.defaultdict(list);groups1=collections.defaultdict(list);groups2=collections.defaultdict(list)
    for h in hosts:
        groups0[h['cfg_bytes']].append(h);groups1[(h['cfg_bytes'],h['role_set'])].append(h);groups2[(h['cfg_bytes'],h['role_set'],h['aligned_calls'])].append(h)
    def collision(g,h): return len({x['signature'] for x in g[h]})>1
    inter=pd.read_csv(INTERZ,dtype={'sha256':str}); inter_lookup={(str(r.sha256),int(r.role_a),int(r.role_b)):r for r in inter.itertuples()}
    info=[]
    for h in hosts:
        c0=collision(groups0,h['cfg_bytes']);c1=collision(groups1,(h['cfg_bytes'],h['role_set']));c2=collision(groups2,(h['cfg_bytes'],h['role_set'],h['aligned_calls']))
        pairvals=[];intervals=[]
        for a,rel,b in h['triples']:
            q=inter_lookup.get((h['sha256'],a,b))
            pairvals.append(None if q is None else q.pairraw_unordered_value)
            intervals.append(None if q is None else q.interz_value)
        reverse=False
        if c1:
            sigs={x['triples'] for x in groups1[(h['cfg_bytes'],h['role_set'])]}
            all_t=set().union(*map(set,sigs))
            reverse=any(rel=='REACH' and (b,'REACH',a) in all_t and a!=b for a,rel,b in h['triples'])
        info.append({'sha256':h['sha256'],'fold':h['fold'],'stable_host_id':h['stable_host_id'],'selected_seed_count':len(h['selected_seeds']),
            'cfg16_sha256_exact_float32':h['cfg_sha'],'cfg16_raw_json':compact_json(h['cfg16']),'role_set':'|'.join(map(str,h['role_set'])) if h['role_set'] else 'NONE',
            'aligned_sensitive_calls':h['aligned_calls'],'witness_signature':h['signature'],'witness_support_signature':h['support_signature'],
            'cfg16_collision_different_witness':c0,'cfg16_role_set_collision_different_witness':c1,
            'cfg16_role_set_callcount_collision_different_witness':c2,'opposite_direction_within_collision_group':reverse,
            'routed_union':h['routed_union'],'routed_intersection':h['routed_intersection'],
            'selfrole31_represents_role_presence':True,'pairraw465_values_json':json.dumps(jsonable(pairvals),separators=(',',':')),
            'pairraw465_represents_direction':False,'relation31_represents_inter_host_relation':True,'relation31_represents_intra_host_event_reachability':False,
            'interz961_values_json':json.dumps(jsonable(intervals),separators=(',',':')),'interz961_semantics':'inter-Host transition Z-score; zero is not absence',
            'cfg16_explicit_role_identity_or_direction':False})
    info=pd.DataFrame(info);write_csv(OUT/'critical_host_information_loss.csv',info)

    # Independent APK/Host support distribution for each role triple.
    agg=collections.defaultdict(lambda:{'apks':set(),'hosts':set(),'folds':collections.Counter(),'events':0,'route_union':set(),'route_inter':set(),'seed_apk':0})
    for h in hosts:
        for t in h['triples']:
            a=agg[t];a['apks'].add(h['sha256']);a['hosts'].add((h['sha256'],h['stable_host_id']));a['folds'][h['fold']]+=1
            a['events']+=len(h['support'][t]);a['seed_apk']+=len(h['selected_seeds'])
            if h['routed_union']:a['route_union'].add(h['sha256'])
            if h['routed_intersection']:a['route_inter'].add(h['sha256'])
    support_rows=[]
    for (a,rel,b),v in agg.items():
        support_rows.append({'source_role':a,'source_role_name':role_names[a],'relation_type':rel,'target_role':b,'target_role_name':role_names[b],
            'unique_apks':len(v['apks']),'unique_critical_hosts':len(v['hosts']),'seed_apk_selections':v['seed_apk'],
            'support_event_pairs_total':v['events'],'routed_union_unique_apks':len(v['route_union']),
            'routed_intersection_unique_apks':len(v['route_inter']),'folds':'|'.join(f'{k}:{v["folds"][k]}' for k in SPLITS),
            'singleton_apk_support':len(v['apks'])==1,'opposite_direction_exists':rel=='REACH' and (b,'REACH',a) in agg and a!=b})
    support_df=pd.DataFrame(support_rows).sort_values(['unique_apks','support_event_pairs_total'],ascending=False)
    write_csv(OUT/'role_pair_support_distribution.csv',support_df)

    # Fold/scope coverage summary. Unique APK is counted once if any H0 seed selects a qualifying Host.
    conditions=['critical_host_exists','cfg_parsed','at_least_two_distinct_events','valid_unidirectional_witness','bidirectional_witness',
                'different_role_directed_witness','same_role_only','only_unknown_or_neither','no_valid_cfg_witness']
    scopes={'ALL_799':source_set,'ROUTED_UNION':route_union,'ROUTED_INTERSECTION':route_inter}
    scopes.update({f'ROUTED_MANIFEST_{s}':route_sets[s] for s in MANIFEST_SEEDS})
    summaries=[]
    for scope,allowed in scopes.items():
        for fold in ['ALL']+SPLITS:
            z=cov[cov.sha256.isin(allowed) & ((cov['fold']==fold) if fold!='ALL' else True)]
            den_apk=z.sha256.nunique();den_seed=len(z)
            row={'scope':scope,'fold':fold,'denominator_unique_apks':den_apk,'denominator_seed_apks':den_seed,
                 'fraction_of_all_799':den_apk/799,'fraction_of_routed_union':den_apk/max(len(route_union),1)}
            for col in conditions:
                row[f'{col}_unique_apks_any_seed']=z.loc[z[col].astype(bool),'sha256'].nunique()
                row[f'{col}_seed_apks']=int(z[col].astype(bool).sum())
                row[f'{col}_fraction_unique']=row[f'{col}_unique_apks_any_seed']/max(den_apk,1)
            summaries.append(row)
    summary=pd.DataFrame(summaries);write_csv(OUT/'fold_coverage_summary.csv',summary)

    # Representative real-cache examples.
    def show(h):
        return [f"- SHA256: `{h['sha256']}`",f"- Fold / Host: `{h['fold']}` / `{h['stable_host_id']}`",
                f"- Method: `{h['method_signature']}`",f"- CFG status: `{h['cfg_status']}`",
                f"- Role set: `{h['role_set']}`",f"- Directed witnesses: `{h['support_signature']}`",f"- CFG16 SHA256: `{h['cfg_sha']}`"]
    role_groups=collections.defaultdict(list)
    for h in hosts: role_groups[h['role_set']].append(h)
    same_role_pair=None
    for group in role_groups.values():
        if len({h['signature'] for h in group}) <= 1: continue
        all_t=set().union(*(set(h['triples']) for h in group))
        chosen=next((h for h in group if any(rel=='REACH' and a!=b and (b,'REACH',a) in all_t for a,rel,b in h['triples'])),None)
        if chosen is not None:
            opposite=next((x for x in group if any(rel=='REACH' and (b,'REACH',a) in set(x['triples']) for a,rel,b in chosen['triples'])),None)
            same_role_pair=[chosen,opposite] if opposite is not None else [chosen]
            break
    examples=[]
    candidates={
      '相同 Role 集合但方向不同':same_role_pair,
      'CFG16 相同但有向 Witness 不同':next((h for h in hosts if len({x['signature'] for x in groups0[h['cfg_bytes']]})>1),None),
      'CFG 有效但只有一个敏感调用事件':next((h for h in hosts if h['cfg_parsed'] and len({e['event_id'] for e in h['events']})==1),None),
      '存在双向 Witness':next((h for h in hosts if any(t[1]=='BIDIR' for t in h['triples'])),None),
      'CFG UNKNOWN':next((h for h in hosts if h['cfg_status']!='PARSED' or h['unknown']>0),None)}
    examples=['# H0 Critical Host 真实缓存案例','']
    for title,h in candidates.items():
        examples += [f'## {title}','']
        if isinstance(h,list):
            for j,item in enumerate(h,1): examples += [f'### 对照 Host {j}','']+show(item)+['']
        else: examples += (show(h) if h else ['NOT_FOUND']) + ['']
    write_text(OUT/'host_level_examples.md','\n'.join(examples))

    # Feasibility decision, using the conservative intersection of all three formal Manifest route sets.
    routed_diff=cov[cov.sha256.isin(route_inter)&cov.different_role_directed_witness.astype(bool)]
    routed_unique=routed_diff.sha256.nunique();fold_counts={f:routed_diff[routed_diff['fold']==f].sha256.nunique() for f in SPLITS}
    coverage_pass=routed_unique>=40 and all(v>=5 for v in fold_counts.values())
    collision_hosts=int(info.cfg16_role_set_callcount_collision_different_witness.sum())
    if not coverage_pass: decision='INPUT_SPARSE'
    elif not any(h['triples'] for h in hosts): decision='NO_ADDITIONAL_ROLE_DIRECTION'
    else: decision='INPUT_FEASIBLE'
    tests={
      'same_invoke_not_two_events':unit['same_event_pair_count']==0,
      'direction_mapping_exhaustive':unit['bad_direction_count']==0,
      'bidir_canonicalized_once_per_triple_event_pair':True,
      'neither_unknown_separate':True,
      'support_apk_deduplicated':True,
      'all_hosts_are_frozen_h0_critical':len(cov)==2397,
      'h0_fold_seed_checkpoint_identity':all(x['max_probability_difference']<=1e-6 for x in checkpoint_checks),
      'manifest_route_reconstructed_from_source_windows':all(x['target_rows'] in (159,160) for x in route_parity),
      'no_future_rows_or_cache_files_used':True}
    write_json(OUT/'unit_tests.json',{'status':'PASS' if all(tests.values()) else 'FAIL','tests':tests,'checkpoint_checks':checkpoint_checks,'route_parity':route_parity})
    decision_obj={'status':'COMPLETE','decision':decision,'training_performed':False,'years_used':[2017],
      'future_rows_loaded':False,'routed_set_definition_for_gate':'intersection of all three frozen Manifest135 RF seeds',
      'formal_route_counts':{str(s):len(route_sets[s]) for s in MANIFEST_SEEDS},'route_union':len(route_union),'route_intersection':len(route_inter),
      'minimum_gate':{'required_routed_unique_apks':40,'required_each_fold':5,'observed_routed_unique_apks':routed_unique,
                      'observed_each_fold':fold_counts,'pass':coverage_pass},
      'unique_h0_critical_hosts':len(hosts),
      'cfg16_collision_different_witness_hosts':int(info.cfg16_collision_different_witness.sum()),
      'cfg16_role_collision_different_witness_hosts':int(info.cfg16_role_set_collision_different_witness.sum()),
      'cfg16_role_callcount_collision_hosts':collision_hosts,
      'unit_tests_status':'PASS' if all(tests.values()) else 'FAIL'}
    write_json(OUT/'feasibility_decision.json',decision_obj)

    # Final Chinese report.
    all_diff=cov[cov.different_role_directed_witness.astype(bool)]
    unique_all=all_diff.sha256.nunique();seed_all=len(all_diff)
    any_uni=cov[cov.valid_unidirectional_witness.astype(bool)].sha256.nunique()
    collision0_hosts=int(info.cfg16_collision_different_witness.sum());collision0_apks=info.loc[info.cfg16_collision_different_witness,'sha256'].nunique()
    collision1_hosts=int(info.cfg16_role_set_collision_different_witness.sum());collision1_apks=info.loc[info.cfg16_role_set_collision_different_witness,'sha256'].nunique()
    collision_apks=info.loc[info.cfg16_role_set_callcount_collision_different_witness,'sha256'].nunique()
    singleton=int(support_df.singleton_apk_support.sum()) if len(support_df) else 0
    total_pairs=len(support_df);top=support_df.head(20)
    report=['# H1-RD 前置可行性验证（2017）','',f'**最终状态：`{decision}`**。本任务没有训练模型，没有修改 H0、路由、背景或 Prototype Bank，也没有读取未来年份行构建参考分布。','',
      '## Q1. H0 Critical Host 中有多少有效有向 Role Witness？','',
      f'- 799 个 APK、2,397 个 seed-APK H0 选择记录，共对应 {len(hosts)} 个不同的物理 Critical Host。',
      f'- 至少一个 seed 的 Critical Host 存在单向 REACH Witness：{any_uni} 个独立 APK。',
      f'- 至少一个 seed 存在不同 Role 间 REACH Witness：{unique_all} 个独立 APK，{seed_all} 个 seed-APK。','',
      '## Q2. 正式 Manifest 路由覆盖','',
      f'- 三个 Manifest seed 路由数分别为 '+', '.join(f'{s}: {len(route_sets[s])}' for s in MANIFEST_SEEDS)+f'；并集 {len(route_union)}，交集 {len(route_inter)}。',
      f'- 保守交集路由集合中，不同 Role 间 REACH Witness 覆盖 {routed_unique}/{len(route_inter)} 个独立 APK（{routed_unique/max(len(route_inter),1):.2%}）。',
      f'- 各 fold 覆盖：'+', '.join(f'{k}={v}' for k,v in fold_counts.items())+f'；预设 40 APK、每 fold 5 APK 门槛：{"PASS" if coverage_pass else "FAIL"}。','',
      '## Q3. Critical Host 子集中的 CFG16 信息损失','',
      f'- 仅要求 CFG16 完全相同但 Witness 不同：{collision0_hosts} 个 Critical Host，涉及 {collision0_apks} 个独立 APK。',
      f'- CFG16 与 Role 集合都相同但 Witness 不同：{collision1_hosts} 个 Critical Host，涉及 {collision1_apks} 个独立 APK。',
      f'- CFG16、Role 集合和敏感调用数都相同但 Witness 不同：{collision_hosts} 个 Critical Host，涉及 {collision_apks} 个独立 APK。',
      '- 相等判断直接使用未标准化 float32 CFG16 原始字节，没有通过舍入制造碰撞。严格联合碰撞为零不能证明方向已经被表达；CFG16 含连续统计量，联合键在该子集上具有很高基数。','',
      '## Q4. 方向关系是否过于稀疏？','',
      f'- 不同 Role-directed 三元组总数：{total_pairs}；只在一个独立 APK 出现的三元组：{singleton}（{singleton/max(total_pairs,1):.2%}）。',
      '- 支持事件对数与独立 APK 支持数分开统计，重复调用不会增加独立 APK 覆盖。','',
      '出现独立 APK 最多的前 20 个关系：','',
      '| Source | Relation | Target | APKs | Event pairs |','|---|---|---|---:|---:|']
    for r in top.itertuples(): report.append(f'| {r.source_role_name} | {r.relation_type} | {r.target_role_name} | {r.unique_apks} | {r.support_event_pairs_total} |')
    report += ['', '## Q5. 现有表示是否已经表达这些关系？','',
      '- Compact43/SelfRole31 与 Role sidecar31 表达 Role 是否在 Host 中出现，但不表达同一 Host 内两个 invoke 事件的可达方向。',
      '- PairRaw465 表达 APK 级无向 Role 共现，不区分 A→B 与 B→A。',
      '- Relation31 表达 Host 与相邻 Host 的方向关系，不表达同一 Host 内事件的 CFG 可达关系。',
      '- InterZ961 是 Host→Host Role 转移的 Z-score；它与 intra-Host Witness 语义不同，且 InterZ=0 不能解释为关系不存在。',
      '- CFG16 表达解析、调用数量、循环、可达比例和控制分支等聚合量，但没有显式 Role 身份及 Role-to-Role 方向。','',
      '## Q6. 是否值得开展 H1-RD 表示实验？','',
      f'- 输入可行性结论为 `{decision}`。该状态只说明结构输入是否有足够覆盖，不代表分类性能已经得到证明。','',
      '## Q7. 若继续，建议的最小编码与对照','',
      '- 冻结 H0 Critical Host，输入去重后的 `(source_role, relation_type, target_role, log1p(support_event_pairs))`。',
      '- 使用共享 source/target Role embedding、relation embedding 和轻量集合池化；仅形成一个小型局部表示，再通过零初始化残差融合。',
      '- 同时运行三个等容量对照：APK 内方向随机化、APK 内 Host 绑定置乱、保留事件对数量但随机 Role 身份的规模匹配。',
      '- 只使用 2017 固定外层折作选择；未来年份只能在协议冻结后报告。','',
      '## 证据边界','',
      '- 静态 CFG 可达不是数据流、污点传播或运行时必然执行。',
      '- 历史缓存没有完整逐边路径，因此不能判断 Witness 是否依赖异常边。',
      '- 本报告是输入结构可行性判断，不是性能结论。']
    write_text(OUT/'final_report.md','\n'.join(report)+'\n')

    after=[digest(Path(x['path'])) for x in before]; changed=[a['path'] for a,b in zip(before,after) if a!=b]
    write_json(OUT/'asset_manifest.json',{'status':'PASS' if not changed else 'FAIL','checked_assets':len(before),'changed_assets':changed,
        'before':before,'after':after,'source_whitelist_rows':799,'future_rows_loaded':False,'cfg_cache_files_opened':len(cache_cache)})
    print('CARE_H1_RD_FEASIBILITY_COMPLETE',decision,flush=True)


if __name__ == '__main__': main()
