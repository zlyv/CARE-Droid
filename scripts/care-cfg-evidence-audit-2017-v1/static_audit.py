from __future__ import annotations

import collections
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import *


def load_contract():
    add_paths()
    from bfa_core import load_data_contract
    data, contract = load_data_contract()
    ids = list(map(str, data['ids']))
    years = np.asarray(data['years'])
    source = np.where(years == 2017)[0]
    if len(source) != 799 or set(years[source]) != {2017}:
        raise RuntimeError('2017 source contract mismatch')
    return data, contract, ids, source


def asset_and_protocol(data, contract, ids, source):
    required = [H1_OUT, CFG_ABL, CFG_FIXED, MANIFEST_CFG, SCBE, CFG_ART, CFG_FEATURES,
                EGO_SOURCE, ENDPOINT_CACHE, LOCAL_ASSET_MANIFEST]
    rows = []
    for p in required:
        rows.append({'path': str(p), 'exists': p.exists(), 'kind': 'directory' if p.is_dir() else 'file',
                     'size': p.stat().st_size if p.is_file() else None,
                     'sha256': sha256(p) if p.is_file() else None})
    h1_protocol = read_json(H1_OUT / 'protocol.json')
    integrity = read_json(H1_OUT / 'baseline_integrity_check.json')
    report = [
        '# CARE-Droid CFG evidence audit: asset inventory', '',
        f'- Audit scope: **2017 only**, {len(source)} APKs.',
        '- Future-year labels loaded by this audit: **No**.',
        '- This audit reuses frozen assets and cached predictions; it does not modify checkpoints, prototypes, APKs, or formal source.',
        f'- Prior H1 integrity check: **{integrity.get("status", "UNKNOWN")}**.', '',
        '| Asset | Exists | Kind | SHA256 (files) |', '|---|---:|---|---|'
    ]
    for r in rows:
        report.append(f'| `{r["path"]}` | {r["exists"]} | {r["kind"]} | {r["sha256"] or "—"} |')
    report += ['', '## Evidence boundaries', '',
               '- CFG cache keeps Host/event/pair summaries but does not keep the full basic-block graph or concrete witness path edge list.',
               '- Therefore exact path reconstruction and exceptional-edge dependence are marked `NOT_AVAILABLE`.',
               '- Historical timing is used only where the producer saved measured wall-clock fields; absent per-stage timings are marked `NOT_MEASURED`.']
    write_text_new(OUT / 'asset_audit.md', '\n'.join(report) + '\n')
    source_hash = hashlib.sha256('\n'.join(ids[i] for i in source).encode()).hexdigest()
    proto = {
        'status': 'PASS' if all(r['exists'] for r in rows) and integrity.get('status') == 'PASS' else 'FAIL',
        'audit_scope_years': [2017], 'source_rows': len(source), 'source_sha256_order_hash': source_hash,
        'labels_loaded': [2017], 'future_labels_loaded': False,
        'future_predictions_used_for_decisions': False,
        'training_protocol': h1_protocol.get('training_protocol', h1_protocol),
        'stage1_required': 'Manifest135 RF / OTHER3_OP25_REF',
        'stage2_base': 'frozen Direct-C43 / SCBE M0',
        'h1_integrity_status': integrity.get('status'),
        'assets': rows,
    }
    write_json_new(OUT / 'source_protocol_audit.json', proto)


def load_cfg_source(source_ids):
    cfg = {}
    for j, sha in enumerate(source_ids):
        cfg[sha] = read_gzip_json(CFG_CACHE / f'{sha}.json.gz')
        if (j + 1) % 100 == 0:
            print(f'[static] cfg caches {j+1}/{len(source_ids)}', flush=True)
    return cfg


def alignment_and_coverage(data, ids, source, cfg):
    ego = np.load(EGO_SOURCE, allow_pickle=False)
    feat = np.load(CFG_FEATURES, allow_pickle=False)
    ego_ids = list(map(str, ego['sha256']))
    feat_ids = list(map(str, feat['sha256']))
    if ego_ids != [ids[i] for i in source] or feat_ids[:len(source)] != ego_ids:
        raise RuntimeError('source order mismatch')
    eo = ego['ego_offsets']; fo = feat['host_offsets']; host_ids = list(map(str, ego['host_ids']))
    align_rows, cov_rows, all_real = [], [], []
    dimension_values = []
    for pos, sha in enumerate(ego_ids):
        l0,r0 = int(eo[pos]),int(eo[pos+1]); l1,r1=int(fo[pos]),int(fo[pos+1])
        cache_hosts = cfg[sha]['hosts']
        cache_host_ids = [str(h['host_node_id']) for h in cache_hosts]
        expected = host_ids[l0:r0]
        ordered_match = expected == cache_host_ids
        count_match = (r0-l0)==(r1-l1)==len(cache_hosts)
        real = np.asarray(feat['real'][l1:r1], dtype=np.float32)
        all_real.append(real); dimension_values.append(real)
        parsed = sum(h['cfg_status']=='PARSED' for h in cache_hosts)
        unknown = len(cache_hosts)-parsed
        one = sum(int(h['aligned_sensitive_calls']) == 1 for h in cache_hosts)
        two = sum(int(h['aligned_sensitive_calls']) >= 2 for h in cache_hosts)
        directed = sum((int(h['a_to_b_pairs'])+int(h['bidirectional_pairs'])) > 0 for h in cache_hosts)
        bidi = sum(int(h['bidirectional_pairs']) > 0 for h in cache_hosts)
        zero = int(np.sum(np.linalg.norm(real, axis=1) <= 1e-12)) if len(real) else 0
        pairs = cfg[sha]['pairs']; events=cfg[sha]['events']
        directed_pairs=sum(p['relation'] in ('A_TO_B','B_TO_A','BIDIRECTIONAL') for p in pairs)
        align_rows.append({'sha256':sha,'apk_position':pos,'expected_host_rows':r0-l0,
                           'cfg_feature_host_rows':r1-l1,'cache_host_rows':len(cache_hosts),
                           'count_match':count_match,'host_identity_order_match':ordered_match,
                           'first_mismatch': next((k for k,(a,b) in enumerate(zip(expected,cache_host_ids)) if a!=b), None),
                           'cfg_offset_left':l1,'cfg_offset_right':r1})
        cov_rows.append({'sha256':sha,'label':int(data['y'][source[pos]]),'fold':source_fold_map()[sha],
                         'host_count':len(cache_hosts),'parsed_host_count':parsed,'unknown_host_count':unknown,
                         'one_sensitive_call_hosts':one,'two_or_more_sensitive_event_hosts':two,
                         'directed_witness_hosts':directed,'bidirectional_witness_hosts':bidi,
                         'cfg16_all_zero_hosts':zero,'event_count':len(events),'pair_count':len(pairs),
                         'directed_pair_count':directed_pairs,'scan_wall_seconds':cfg[sha].get('wall_seconds'),
                         'host_instruction_sum':sum(int(h['instruction_count']) for h in cache_hosts)})
    write_csv_new(OUT/'host_alignment_audit.csv', pd.DataFrame(align_rows))
    cov = pd.DataFrame(cov_rows)
    witness_path = CFG_ABL/'per_apk_predictions.csv'
    if witness_path.exists():
        w = pd.read_csv(witness_path)
        cols=[c for c in w.columns if 'witness' in c.lower()]
        if cols:
            uu=w.drop_duplicates('sha256').set_index('sha256')
            for c in cols: cov[c]=cov.sha256.map(uu[c])
    write_csv_new(OUT/'cfg_coverage.csv', cov)
    mat=np.concatenate(dimension_values,axis=0)
    summary={
        'apk_count':len(cov),'host_count':int(cov.host_count.sum()),
        'parsed_host_fraction':float(cov.parsed_host_count.sum()/max(1,cov.host_count.sum())),
        'unknown_host_fraction':float(cov.unknown_host_count.sum()/max(1,cov.host_count.sum())),
        'one_sensitive_call_host_fraction':float(cov.one_sensitive_call_hosts.sum()/max(1,cov.host_count.sum())),
        'two_or_more_event_host_fraction':float(cov.two_or_more_sensitive_event_hosts.sum()/max(1,cov.host_count.sum())),
        'directed_witness_host_fraction':float(cov.directed_witness_hosts.sum()/max(1,cov.host_count.sum())),
        'bidirectional_witness_host_fraction':float(cov.bidirectional_witness_hosts.sum()/max(1,cov.host_count.sum())),
        'cfg16_zero_variance_dimensions':np.where(np.std(mat,axis=0)<=1e-12)[0].tolist(),
        'correlations':{
            'directed_pairs_vs_host_count':pearson_spearman(cov.directed_pair_count,cov.host_count),
            'directed_pairs_vs_instruction_sum':pearson_spearman(cov.directed_pair_count,cov.host_instruction_sum),
        }}
    write_json_new(OUT/'cfg_coverage_summary.json',summary)
    return ego, feat, cov


def role_semantics(ego, feat, cfg):
    eo=ego['ego_offsets']; fo=feat['host_offsets']; shas=list(map(str,ego['sha256']))
    rows=[]; signatures=collections.defaultdict(list); path_rows=[]
    for i,sha in enumerate(shas):
        l,r=int(fo[i]),int(fo[i+1]); real=np.asarray(feat['real'][l:r])
        events={e['event_id']:e for e in cfg[sha]['events']}
        host_pairs=collections.defaultdict(list)
        for p in cfg[sha]['pairs']: host_pairs[str(p['host_node_id'])].append(p)
        for j,h in enumerate(cfg[sha]['hosts']):
            hid=str(h['host_node_id']); pairs=host_pairs[hid]
            dirs=sorted({f"{int(p['role_a'])}>{int(p['role_b'])}:{p['relation']}" for p in pairs})
            pattern='|'.join(dirs) if dirs else 'NONE'
            sig=tuple(np.round(real[j],6).tolist())
            signatures[sig].append((sha,hid,tuple(h['role_ids']),pattern,int(h['aligned_sensitive_calls'])))
            rows.append({'sha256':sha,'host_node_id':hid,'method_signature':h['method_signature'],
                         'role_set':'|'.join(map(str,h['role_ids'])),'aligned_sensitive_calls':h['aligned_sensitive_calls'],
                         'cfg_status':h['cfg_status'],'cfg16_signature_sha256':hashlib.sha256(np.asarray(sig,dtype=np.float32).tobytes()).hexdigest(),
                         'directed_role_witness_pattern':pattern,'pair_count':len(pairs),
                         'distinct_event_pairs':len({(p['event_a'],p['event_b']) for p in pairs}),
                         'same_invocation_role_pair_count':sum(p['event_a']==p['event_b'] for p in pairs)})
            for p in pairs[:3]:
                ea=events.get(p['event_a'],{}); eb=events.get(p['event_b'],{})
                path_rows.append({'sha256':sha,'host_node_id':hid,'method_signature':h['method_signature'],
                    'role_a':p['role_a'],'role_b':p['role_b'],'relation':p['relation'],
                    'offset_a':p['offset_a'],'offset_b':p['offset_b'],'reference_a':ea.get('reference'),
                    'reference_b':eb.get('reference'),'same_basic_block':p['same_basic_block'],
                    'a_in_loop':p['a_in_loop'],'b_in_loop':p['b_in_loop'],
                    'shared_verified_control_branch':p['shared_verified_control_branch'],
                    'different_verified_control_sets':p['different_verified_control_sets'],
                    'concrete_cfg_path_edges':'NOT_AVAILABLE','exception_edge_dependency':'NOT_AVAILABLE',
                    'reason':'historical cache stores reachability result and event metadata, not full CFG/path edge sequence'})
    collision_size={k:len({x[3] for x in v}) for k,v in signatures.items()}
    for row in rows:
        # signature hash lookup is deliberately kept as an auditable row-level field; aggregate collisions below.
        pass
    sem=pd.DataFrame(rows)
    sig_to_patterns=sem.groupby('cfg16_signature_sha256').directed_role_witness_pattern.nunique()
    sem['same_cfg16_multiple_witness_patterns']=sem.cfg16_signature_sha256.map(sig_to_patterns).gt(1)
    write_csv_new(OUT/'cfg_role_semantics.csv',sem)
    write_csv_new(OUT/'cfg_path_reliability.csv',pd.DataFrame(path_rows))
    write_json_new(OUT/'cfg_role_semantics_summary.json',{
        'host_rows':len(sem),'same_invocation_pair_rows':int(sem.same_invocation_role_pair_count.sum()),
        'cfg16_exact_signature_groups':int(sig_to_patterns.size),
        'signature_groups_with_multiple_witness_patterns':int((sig_to_patterns>1).sum()),
        'hosts_in_ambiguous_signature_groups':int(sem.same_cfg16_multiple_witness_patterns.sum()),
        'path_edges_available':False,'exception_edge_dependency_available':False})


def load_endpoint_source(source_ids):
    wanted=set(source_ids); out={}
    with gzip.open(ENDPOINT_CACHE,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); sha=str(r['sha256'])
            if sha in wanted:
                out[sha]=r['payload']
                if len(out)==len(wanted): break
    if len(out)!=len(wanted): raise RuntimeError(f'endpoint source missing {len(wanted)-len(out)}')
    return out


def interz_semantics(data, ids, source, cfg):
    hostrel=load_module('audit_hostrel',ROOT/'scripts/care-di25-compact43-relationquery-source-v1/hostrel_features.py')
    mapping=read_json(LOCAL_ASSET_MANIFEST)['payload_to_model_role']
    source_ids=[ids[i] for i in source]; payloads=load_endpoint_source(source_ids)
    rows=[]
    for pos,(idx,sha) in enumerate(zip(source,source_ids)):
        payload=payloads[sha]; records=hostrel.reconstruct_host_relations(payload,mapping)
        host_nodes={int(r['host_node']):tuple(map(int,np.flatnonzero(r['endpoint_counts']>0))) for r in records}
        edge_index=payload.get('edge_index') or [[],[]]
        raw=np.zeros((31,31),dtype=np.int64)
        for a,b in zip(edge_index[0],edge_index[1]):
            a=int(a);b=int(b)
            if a in host_nodes and b in host_nodes and a!=b:
                for ra in host_nodes[a]:
                    for rb in host_nodes[b]: raw[ra,rb]+=1
        s7=np.asarray(data['s7'][idx]); pair=s7[120:585]; inter=s7[585:].reshape(31,31)
        witnessed=collections.Counter()
        for p in cfg[sha]['pairs']:
            a,b=int(p['role_a']),int(p['role_b']); rel=p['relation']
            if rel in ('A_TO_B','BIDIRECTIONAL'): witnessed[(a,b)]+=1
            if rel in ('B_TO_A','BIDIRECTIONAL'): witnessed[(b,a)]+=1
        for (a,b),count in witnessed.items():
            if a==b: pairraw=None
            else:
                x,y=sorted((a,b)); pair_idx=x*30 - x*(x-1)//2 + (y-x-1)
                pairraw=float(pair[pair_idx])
            rows.append({'sha256':sha,'fold':source_fold_map()[sha],'role_a':a,'role_b':b,
                         'within_host_directed_witness_count':count,'inter_host_raw_directed_count':int(raw[a,b]),
                         'interz_value':float(inter[a,b]),'interz_exact_zero':bool(inter[a,b]==0),
                         'interz_near_zero_1e6':bool(abs(float(inter[a,b]))<=1e-6),
                         'pairraw_unordered_value':pairraw,
                         'semantics':'CFG witness is intra-Host event reachability; InterZ is inter-Host call-edge role transition z-score'})
    frame=pd.DataFrame(rows)
    write_csv_new(OUT/'interz_witness_semantics.csv',frame)
    write_json_new(OUT/'interz_witness_semantics_summary.json',{
        'witness_role_direction_rows':len(frame),
        'interz_exact_zero_fraction':float(frame.interz_exact_zero.mean()) if len(frame) else None,
        'interz_near_zero_fraction':float(frame.interz_near_zero_1e6.mean()) if len(frame) else None,
        'inter_raw_nonzero_fraction':float((frame.inter_host_raw_directed_count>0).mean()) if len(frame) else None,
        'pairraw_nonzero_fraction':float((frame.pairraw_unordered_value.fillna(0)>0).mean()) if len(frame) else None})


def runtime_audit(cov):
    timings={}
    timings['cfg_scan_per_apk_seconds']=qstats(cov.scan_wall_seconds.dropna().values)
    tc=pd.read_csv(H1_OUT/'training_cost.csv')
    timings['historical_h1_training_cost_rows']=tc.to_dict('records')
    timings['manifest135_extraction_per_apk_seconds']='NOT_MEASURED'
    timings['host_rolegraph_extraction_per_apk_seconds']='NOT_MEASURED'
    timings['witness3_incremental_extraction_per_apk_seconds']='NOT_SEPARABLE_FROM_CFG_SCAN'
    timings['h0_inference_per_apk_distribution']='NOT_MEASURED'
    timings['h1_inference_per_apk_distribution']='NOT_MEASURED'
    timings['explanation']='CFG cache wall_seconds is a measured full APK CFG scan time. Historical model files save validation-batch inference totals, not per-APK latency distributions.'
    write_json_new(OUT/'runtime_cost_audit.json',timings)


def main():
    if OUT.exists():
        raise RuntimeError(f'refusing to overwrite existing audit directory: {OUT}')
    OUT.mkdir(parents=True,exist_ok=False)
    data,contract,ids,source=load_contract()
    asset_and_protocol(data,contract,ids,source)
    source_ids=[ids[i] for i in source]
    cfg=load_cfg_source(source_ids)
    ego,feat,cov=alignment_and_coverage(data,ids,source,cfg)
    role_semantics(ego,feat,cfg)
    interz_semantics(data,ids,source,cfg)
    runtime_audit(cov)
    print('[static] complete',flush=True)


if __name__=='__main__': main()
