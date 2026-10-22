#!/usr/bin/env python3
"""Orchestrate preflight, label-free CFG scan, and source-only OOF diagnosis.

Writes only artifacts/care_cfg_witness_pilot_2017. Labels are isolated to the
final summarize phase and never affect route membership or witness extraction.
"""
from __future__ import annotations
import argparse, concurrent.futures, csv, gzip, hashlib, json, math, os, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT / 'scripts/care-host-cfg-rolegraph-witness-pilot-v1'
PARENT = ROOT / 'scripts/care-local-control-constraint-feasibility-v2'
PARENT_OUT = ROOT / 'outputs/care-local-control-constraint-feasibility-v2'
OUT = ROOT / 'artifacts/care_cfg_witness_pilot_2017'
PRED = ROOT / 'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv'
INTER = ROOT / 'artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz'
MAPPING = ROOT / 'scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml'
SCHEMA = 'care-cfg-rolegraph-witness-v1.0'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); tmp.replace(path)


def atomic_csv(path, rows):
    rows=list(rows); fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def load_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def preflight():
    if OUT.exists(): raise RuntimeError('artifact directory exists; refusing overwrite')
    OUT.mkdir(parents=True); (OUT/'cache').mkdir(); (OUT/'status').mkdir(); (OUT/'logs').mkdir()
    parent_decision=json.loads((PARENT_OUT/'final_decision.json').read_text(encoding='utf-8'))
    parent_integrity=json.loads((PARENT_OUT/'baseline_integrity_check.json').read_text(encoding='utf-8'))
    if parent_decision.get('status')!='LCC_CONTROL_OBSERVABILITY_SUPPORTED' or parent_integrity.get('status')!='PASS':
        raise RuntimeError('validated LCC-v2 parent unavailable')
    selected=rows(PARENT_OUT/'selection_label_free.csv'); paths=json.loads((PARENT_OUT/'apk_paths.json').read_text(encoding='utf-8'))
    if len(selected)!=200 or len({r['sha256'] for r in selected})!=200 or any(int(r['routed'])!=1 for r in selected):
        raise RuntimeError('frozen route membership changed')
    if Counter(int(r['fold']) for r in selected)!=Counter({1:40,2:40,3:40,4:40,5:40}):
        raise RuntimeError('fold route contract changed')
    required=[HERE/'scan_one.py',HERE/'witness_core.py',PARENT/'control_core.py',PARENT/'dex_adapter.py',
              PARENT/'common.py',PARENT_OUT/'selection_label_free.csv',PARENT_OUT/'apk_paths.json',
              PARENT_OUT/'baseline_integrity_check.json',PRED,INTER,MAPPING]
    required += [PARENT_OUT/'cache/frozen'/f"{r['sha256']}.json" for r in selected]
    if any(not p.is_file() for p in required): raise RuntimeError('required frozen asset missing')
    manifest=[{'path':str(p.resolve()),'sha256':sha256(p),'size':p.stat().st_size} for p in required]
    atomic_json(OUT/'asset_manifest_before.json',manifest)
    atomic_csv(OUT/'selection_label_free.csv',selected)
    atomic_json(OUT/'apk_paths.json',{sid:paths[sid] for sid in sorted(paths) if sid in {r['sha256'] for r in selected}})
    inspection={
      'status':'PASS','source_rows':799,'selected_routed_APKs':200,'fold_counts':dict(Counter(int(r['fold']) for r in selected)),
      'requested_route_name':'DI25','actual_frozen_route':'2017 SVM MeanOpposition Top25%',
      'protocol_difference':'Task permits current formal frozen configuration; membership reused exactly, no threshold recomputation.',
      'APK_parser':'Androguard APK/DEX via validated LCC-v2 dex_adapter.decode_method',
      'role_mapping':str(MAPPING),'host_contract':str(PARENT_OUT/'cache/frozen/<sha>.json'),
      'host_identity':'canonical DEX method signature -> first 8 SHA256 bytes as unsigned integer',
      'CFG_engine':str(PARENT/'control_core.py')+':build_cfg; instruction exceptional CFG, try/catch edges, strong reachability',
      'instruction_call_positions':'DEX invoke offset bytes retained by dex_adapter.decode_method',
      'basic_blocks':'derived conservatively from normal instruction CFG; reachability itself remains instruction-level',
      'existing_role_features':{'SelfRole31':'frozen Host endpoint Role31 sets','RolePair465':str(INTER)+':pair_raw',
        'InterZ961':str(INTER)+':inter_z_clipped','RoleGraph':'frozen endpoint/Host mapping and formal RoleGraph code'},
      'OOF_predictions':str(PRED),'labels_used_before_summarize':False,'future_rows_loaded':0,
      'parent_observability':{'frozen_host_method_recovery':0.998269197484567,'aligned_event_cfg_parse_rate':0.9894093204736246},
      'obstacle':None,'full_scan_estimate':'about 8-20 minutes with four workers based on parent measured scan cost'}
    atomic_json(OUT/'project_inspection.json',inspection)
    md=['# 项目检查与最小实现方案','',f"- 冻结路由：{inspection['actual_frozen_route']}（任务写DI25，但当前可复用正式资产并非DI25命名）。",
        '- Host与Role：直接读取LCC-v2从正式endpoint cache重建并校验的冻结Host/endpoint Role31映射。',
        '- CFG：复用已验证的DEX指令级异常CFG，不重新定义Host；同块顺序由指令边和偏移共同处理。',
        '- 新信息：只产生同一Host内敏感调用位置之间的可达/双向/不可达、循环与严格控制分支集合关系。',
        '- 已有信息对照：SelfRole31、PairRaw465、InterZ961；不把Host内Witness冒充Host→Host边。',
        '- 标签隔离：扫描完成后才读取2017 OOF标签/预测；不加载2018—2021。',
        '- 预计成本：按父实验实测，4 worker约8—20分钟；不使用GPU。','']
    (OUT/'project_inspection.md').write_text('\n'.join(md),encoding='utf-8')
    print('CFGW_PREFLIGHT_PASS selected=200 future=0 labels=false',flush=True)


def worker(row, paths):
    sid=row['sha256']; output=OUT/'cache'/f'{sid}.json.gz'; status=OUT/'status'/f'{sid}.json'
    cmd=[sys.executable,str(HERE/'scan_one.py'),'--apk',paths[sid],'--sha',sid,'--mapping',str(MAPPING),
         '--frozen',str(PARENT_OUT/'cache/frozen'/f'{sid}.json'),'--output',str(output),'--status',str(status)]
    log=OUT/'logs'/f'{sid}.log'; started=time.monotonic()
    with log.open('a',encoding='utf-8') as f:
        try: proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
        except subprocess.TimeoutExpired:
            atomic_json(status,{'sha256':sid,'status':'TIMEOUT','wall_seconds':time.monotonic()-started}); return json.loads(status.read_text())
    if status.is_file(): return json.loads(status.read_text())
    return {'sha256':sid,'status':'ERROR','reason':'worker exit '+str(proc.returncode),'wall_seconds':time.monotonic()-started}


def scan_all(workers):
    selected=rows(OUT/'selection_label_free.csv'); paths=json.loads((OUT/'apk_paths.json').read_text())
    # Fixed SHA smoke subset, independent of labels.
    smoke=sorted(selected,key=lambda r:r['sha256'])[:5]
    for i,row in enumerate(smoke,1):
        result=worker(row,paths)
        if result.get('status')!='OK': raise RuntimeError('smoke failed '+repr(result))
        record=load_gz(OUT/'cache'/f"{row['sha256']}.json.gz")
        if record.get('schema')!=SCHEMA or record.get('labels_used') or record.get('future_rows_loaded')!=0:
            raise RuntimeError('smoke contract failed')
        print(f'CFGW_SMOKE {i}/5 status=OK',flush=True)
    atomic_json(OUT/'smoke_test.json',{'status':'PASS','sha_order':[r['sha256'] for r in smoke],
        'selection':'first five lexicographic frozen route SHA; label-free','future_rows_loaded':0})
    pending=[r for r in selected if not (OUT/'status'/f"{r['sha256']}.json").is_file()]
    statuses=[]; started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(worker,r,paths):r for r in pending}
        for job in concurrent.futures.as_completed(jobs):
            statuses.append(job.result()); print(f'CFGW_SCAN {len(statuses)+len(smoke)}/200 status={statuses[-1].get("status")}',flush=True)
    statuses=[json.loads((OUT/'status'/f"{r['sha256']}.json").read_text()) for r in selected]
    atomic_csv(OUT/'scan_status.csv',statuses)
    atomic_json(OUT/'scan_runtime.json',{'wall_seconds':time.monotonic()-started,'workers':workers,
        'sum_worker_seconds':sum(float(r.get('wall_seconds',0)) for r in statuses),'gpu_used':False})
    print('CFGW_SCAN_DONE',flush=True)


def auc(y,score):
    y=np.asarray(y,int); score=np.asarray(score,float)
    if len(np.unique(y))<2:return None
    pos=score[y==1]; neg=score[y==0]
    return float((sum((p>neg).sum()+.5*(p==neg).sum() for p in pos))/(len(pos)*len(neg)))


def spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    def rank(x):
        order=np.argsort(x,kind='mergesort'); r=np.empty(len(x),float); r[order]=np.arange(len(x));
        for v in np.unique(x):
            idx=np.flatnonzero(x==v); r[idx]=r[idx].mean()
        return r
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(rank(a),rank(b))[0,1])


def verify_assets():
    before=json.loads((OUT/'asset_manifest_before.json').read_text()); changed=[]
    for item in before:
        p=Path(item['path'])
        try: now={'path':str(p.resolve()),'sha256':sha256(p),'size':p.stat().st_size}
        except OSError as exc: changed.append({'path':str(p),'error':repr(exc)}); continue
        if now!=item: changed.append({'before':item,'after':now})
    return {'status':'PASS' if not changed else 'FAIL','checked':len(before),'changed':changed,
            'scope':'named source-only code, caches, predictions and mappings; APK bytes read-only, not installed/executed'}


def summarize():
    selected=rows(OUT/'selection_label_free.csv'); fold={r['sha256']:int(r['fold']) for r in selected}
    scans=[]
    for row in selected:
        p=OUT/'cache'/f"{row['sha256']}.json.gz"
        if p.is_file(): scans.append(load_gz(p))
    apk_rows=[]; host_rows=[]; pair_rows=[]; event_rows=[]; failure_reasons=Counter()
    for scan in scans:
        sid=scan['sha256']; hosts=scan['hosts']; events=scan['events']; pairs=scan['pairs']
        for h in hosts: host_rows.append({'fold':fold[sid],**h})
        for e in events: event_rows.append({'fold':fold[sid],**e})
        for p in pairs: pair_rows.append({'fold':fold[sid],**p})
        for f in scan['failures']: failure_reasons[f['reason']]+=1
        rel=Counter(p['relation'] for p in pairs); roles={int(r) for e in events for r in e['role_ids']}
        local_role_pairs={(int(p['role_a']),int(p['role_b'])) for p in pairs if p['relation'] in ('A_TO_B','B_TO_A','BIDIRECTIONAL')}
        apk_rows.append({'sha256':sid,'fold':fold[sid],'scan_status':scan['status'],'dex_count':scan['dex_count'],
          'frozen_hosts':scan['frozen_hosts'],'recovered_hosts':scan['recovered_hosts'],'parsed_hosts':sum(h['cfg_status']=='PARSED' for h in hosts),
          'hosts_with_two_calls':sum(h['aligned_sensitive_calls']>=2 for h in hosts),'hosts_with_two_roles':sum(h['distinct_roles']>=2 for h in hosts),
          'hosts_with_cycle':sum(bool(h['has_reachable_cycle']) for h in hosts),'aligned_sensitive_calls':len(events),
          'distinct_roles':len(roles),'event_role_pairs':len(pairs),'ordered_witness_role_pairs':len(local_role_pairs),
          'a_to_b_pairs':rel['A_TO_B']+rel['B_TO_A'],'bidirectional_pairs':rel['BIDIRECTIONAL'],
          'neither_pairs':rel['NEITHER'],'unknown_pairs':rel['UNKNOWN'],'sensitive_loop_calls':sum(bool(e['in_loop']) for e in events),
          'pair_cap_methods':scan['pair_cap_methods']})
    if len(scans)!=200: blocked=True
    else: blocked=False
    z=np.load(INTER,allow_pickle=True); zi={str(s):i for i,s in enumerate(z['sha256'].astype(str))}
    by_sid_pairs=defaultdict(set)
    for p in pair_rows:
        if p['relation'] in ('A_TO_B','B_TO_A','BIDIRECTIONAL'):
            by_sid_pairs[p['sha256']].add((int(p['role_a']),int(p['role_b'])))
            if p['relation']=='B_TO_A': by_sid_pairs[p['sha256']].add((int(p['role_b']),int(p['role_a'])))
    for row in apk_rows:
        i=zi[row['sha256']]; inter=np.asarray(z['inter_z_clipped'][i]).reshape(31,31); pair=np.asarray(z['pair_raw'][i])
        witness=by_sid_pairs[row['sha256']]
        row['existing_pairraw_nonzero']=int(np.count_nonzero(pair))
        row['existing_interz_nonzero']=int(np.count_nonzero(inter))
        row['witness_pairs_interz_zero']=sum(abs(float(inter[a,b]))<1e-12 for a,b in witness)
        row['witness_pairs_interz_nonzero']=sum(abs(float(inter[a,b]))>=1e-12 for a,b in witness)
        row['witness_unique_fraction_vs_interz']=row['witness_pairs_interz_zero']/max(len(witness),1)
        row['witness_apk']=int(bool(witness))
        row['multi_call_host_apk']=int(row['hosts_with_two_calls']>0)
    atomic_csv(OUT/'cfg_witness_apk.csv',apk_rows); atomic_csv(OUT/'cfg_witness_host.csv',host_rows)
    atomic_csv(OUT/'cfg_witness_events.csv',event_rows); atomic_csv(OUT/'cfg_witness_pairs.csv',pair_rows)

    # Same Role set, different CFG witness signature: label-free novelty audit.
    role_groups=defaultdict(list)
    for h in host_rows:
        key=tuple(h['role_ids'])
        if key: role_groups[key].append(h)
    variable_groups=0; variable_hosts=0; examples=[]
    for key,group in sorted(role_groups.items()):
        signatures={(g['a_to_b_pairs']>0,g['bidirectional_pairs']>0,g['neither_pairs']>0,g['sensitive_calls_in_loop']>0) for g in group}
        if len(group)>=2 and len(signatures)>=2:
            variable_groups+=1; variable_hosts+=len(group)
            if len(examples)<20:
                for g in sorted(group,key=lambda x:(x['sha256'],x['host_node_id']))[:2]:
                    examples.append({'role_ids':list(key),'sha256':g['sha256'],'host_node_id':g['host_node_id'],
                        'method_signature':g['method_signature'],'aligned_sensitive_calls':g['aligned_sensitive_calls'],
                        'a_to_b_pairs':g['a_to_b_pairs'],'bidirectional_pairs':g['bidirectional_pairs'],
                        'neither_pairs':g['neither_pairs'],'sensitive_calls_in_loop':g['sensitive_calls_in_loop']})
    atomic_csv(OUT/'representative_witness_examples.csv',examples)

    # Labels/probabilities enter only now.
    pred=[r for r in rows(PRED) if r.get('arm')=='SVM']
    by_pred={r['sha256']:r for r in pred}
    if len(by_pred)!=799: raise RuntimeError('SVM OOF prediction contract changed')
    if {r['sha256'] for r in selected}!={sid for sid,r in by_pred.items() if int(float(r['routed']))==1}:
        raise RuntimeError('OOF route differs from label-free frozen selection')
    features=('hosts_with_two_calls','hosts_with_two_roles','ordered_witness_role_pairs','a_to_b_pairs','bidirectional_pairs',
              'neither_pairs','sensitive_loop_calls','witness_pairs_interz_zero','witness_unique_fraction_vs_interz',
              'frozen_hosts','aligned_sensitive_calls')
    error_rows=[]
    for row in apk_rows:
        p=by_pred[row['sha256']]; y=int(float(p['label'])); prob=float(p['stage2_prob']); pred_label=int(prob>=.5)
        group=('TP' if y and pred_label else 'FN' if y else 'FP' if pred_label else 'TN')
        row.update(label=y,stage2_probability=prob,prediction_group=group)
        error_rows.append({k:row.get(k) for k in ('sha256','fold','label','stage2_probability','prediction_group')+features})
    atomic_csv(OUT/'oof_error_analysis.csv',error_rows)
    group_summary=[]
    for group in ('TP','FN','FP','TN'):
        subset=[r for r in error_rows if r['prediction_group']==group]
        for feature in features:
            vals=np.asarray([float(r[feature]) for r in subset],float)
            group_summary.append({'group':group,'feature':feature,'n':len(vals),'mean':float(vals.mean()) if len(vals) else None,
                'median':float(np.median(vals)) if len(vals) else None,'q25':float(np.quantile(vals,.25)) if len(vals) else None,
                'q75':float(np.quantile(vals,.75)) if len(vals) else None})
    atomic_csv(OUT/'oof_group_summary.csv',group_summary)
    direction=[]
    controls=('frozen_hosts','aligned_sensitive_calls')
    for name,positive,negative in (('FN_vs_TN','FN','TN'),('TP_vs_FP','TP','FP')):
        sample=[r for r in error_rows if r['prediction_group'] in (positive,negative)]
        y=[int(r['prediction_group']==positive) for r in sample]
        for feature in features:
            score=[float(r[feature]) for r in sample]; overall=auc(y,score); folds=[]
            for f in range(1,6):
                sub=[r for r in sample if int(r['fold'])==f]
                value=auc([int(r['prediction_group']==positive) for r in sub],[float(r[feature]) for r in sub])
                if value is not None: folds.append(value)
            direction.append({'direction':name,'feature':feature,'n':len(sample),'AUC':overall,
                'folds_evaluable':len(folds),'folds_AUC_above_0_5':sum(v>.5 for v in folds),
                'fold_AUC_mean':float(np.mean(folds)) if folds else None,'is_size_control':feature in controls})
    atomic_csv(OUT/'witness_direction_auc.csv',direction)
    correlations=[]
    for feature in features[:-2]:
        for control in controls:
            correlations.append({'feature':feature,'control':control,
                'spearman':spearman([r[feature] for r in apk_rows],[r[control] for r in apk_rows]),'n':len(apk_rows)})
    atomic_csv(OUT/'size_frequency_correlations.csv',correlations)

    successful=sum(s['status']=='OK' for s in scans); total_hosts=sum(r['frozen_hosts'] for r in apk_rows)
    recovered=sum(r['recovered_hosts'] for r in apk_rows); parsed=sum(r['parsed_hosts'] for r in apk_rows)
    witness_apks=sum(r['witness_apk'] for r in apk_rows); multicall=sum(r['multi_call_host_apk'] for r in apk_rows)
    witness_total=sum(r['ordered_witness_role_pairs'] for r in apk_rows); unique=sum(r['witness_pairs_interz_zero'] for r in apk_rows)
    fold_witness={f:sum(r['witness_apk'] for r in apk_rows if int(r['fold'])==f) for f in range(1,6)}
    coverage={'selected_APKs':200,'successful_scans':successful,'scan_success_rate':successful/200,
      'frozen_hosts':total_hosts,'recovered_hosts':recovered,'host_recovery_rate':recovered/max(total_hosts,1),
      'parsed_hosts':parsed,'parsed_host_rate':parsed/max(recovered,1),'APKs_with_multi_call_host':multicall,
      'multi_call_host_APK_rate':multicall/200,'APKs_with_ordered_witness':witness_apks,'witness_APK_rate':witness_apks/200,
      'ordered_witness_role_pairs':witness_total,'witness_pairs_absent_from_InterZ':unique,
      'witness_unique_fraction_vs_InterZ':unique/max(witness_total,1),'fold_witness_APKs':fold_witness,
      'pair_cap_methods':sum(r['pair_cap_methods'] for r in apk_rows),'failure_reasons':dict(failure_reasons),
      'denominator_note':'APK, Host, event and role-pair denominators are reported separately; no Host is treated as an independent APK.'}
    atomic_json(OUT/'coverage_report.json',coverage)
    novelty={'same_role_signature_groups':len(role_groups),'variable_CFG_groups':variable_groups,
      'hosts_in_variable_CFG_groups':variable_hosts,'representative_examples':len(examples),
      'interpretation':'Variation within the same Role set and InterZ-zero local ordered pairs are structural novelty candidates, not malicious semantics.'}
    atomic_json(OUT/'redundancy_report.json',novelty)

    control_best=max((r['AUC'] or .5) for r in direction if r['is_size_control'])
    candidates=[r for r in direction if not r['is_size_control'] and r['AUC'] is not None and
                max(r['AUC'],1-r['AUC'])>=max(control_best,1-control_best)+.03 and r['folds_AUC_above_0_5']>=3]
    gates={'scan_success_rate>=0.95':coverage['scan_success_rate']>=.95,
      'host_recovery_rate>=0.95':coverage['host_recovery_rate']>=.95,
      'parsed_host_rate>=0.90':coverage['parsed_host_rate']>=.90,
      'witness_APK_rate>=0.15':coverage['witness_APK_rate']>=.15,
      'at_least_4_folds_have_witness':sum(v>0 for v in fold_witness.values())>=4,
      'structural_novelty':coverage['witness_unique_fraction_vs_InterZ']>=.05 or variable_groups>=5,
      'class_direction_beyond_size_control':bool(candidates),'formal_integrity=PASS':False}
    integrity=verify_assets(); gates['formal_integrity=PASS']=integrity['status']=='PASS'
    if integrity['status']!='PASS': raise RuntimeError('protected source asset changed')
    if blocked or coverage['scan_success_rate']<.95: status='BLOCKED'
    elif all(gates.values()): status='SUPPORTED_FOR_NEXT_ABLATION'
    elif coverage['witness_APK_rate']<.15 or not gates['structural_novelty']: status='NOT_SUPPORTED'
    else: status='INCONCLUSIVE'
    decision={'status':status,'criteria':gates,'candidate_witness_features':candidates,'future_rows_loaded':0,
      'labels_used_only_after_witness_freeze':True,'model_training_performed':False,'formal_model_changed':False,
      'requested_DI25_but_actual_route':'2017 SVM MeanOpposition Top25%',
      'semantic_boundary':'Static local CFG reachability is possible path structure, not runtime execution, data flow, leakage or malicious intent.',
      'next_action':('Specify one fixed source-only end-to-end ablation; do not tune Witness rules.' if status=='SUPPORTED_FOR_NEXT_ABLATION' else
                     'Do not integrate CFG Witness into Direct-C43; inspect coverage/novelty report only.')}
    atomic_json(OUT/'baseline_integrity_check.json',integrity); atomic_json(OUT/'final_decision.json',decision)
    atomic_json(OUT/'run_manifest.json',{'schema':SCHEMA,'python':sys.executable,'workers':int(os.environ.get('CFGW_WORKERS','4')),
      'source_APKs':799,'routed_APKs':200,'future_rows_loaded':0,'new_APKs_used':False,'GPU_used':False,
      'parent_engine':str(PARENT),'route_source':str(PARENT_OUT/'selection_label_free.csv'),'asset_manifest_sha256':sha256(OUT/'asset_manifest_before.json')})
    report=['# CARE-Droid Host局部CFG与RoleGraph Witness Pilot','',f'最终状态：**{status}**。','',
      '本实验只使用2017源域冻结SVM MeanOpposition Top25%路由200个APK；任务中的DI25名称与当前可复用正式资产不一致，未重新计算路由。未加载2018—2021，未训练模型。','',
      '## 覆盖率','',f"- 扫描成功：{successful}/200；冻结Host恢复：{recovered}/{total_hosts}；CFG解析Host：{parsed}/{recovered}。",
      f"- 含多敏感调用Host的APK：{multicall}/200；获得有向局部Witness的APK：{witness_apks}/200。",
      f"- 局部有向Role Witness总数：{witness_total}；对应InterZ单元为零：{unique}（{unique/max(witness_total,1):.3f}）。",
      f"- 相同Role集合却出现不同CFG组织的组：{variable_groups}；涉及Host：{variable_hosts}。",'',
      '## 与现有RoleGraph的边界','',
      'InterZ961描述APK级跨Host有向角色关系；本Pilot保存的是同一Host方法内部敏感调用位置的指令CFG可达关系。两者只在APK/角色对粒度做覆盖对照，不将局部Witness伪装成Host→Host边。',
      '', '## 源域后验诊断','']
    if candidates:
        for c in candidates: report.append(f"- {c['direction']} 的 `{c['feature']}`：AUC={c['AUC']:.4f}，同方向fold={c['folds_AUC_above_0_5']}/{c['folds_evaluable']}；仅为开发源域候选。")
    else: report.append('- 没有Witness特征同时超过规模/调用数对照预设边界并保持至少3折同方向。')
    report += ['', '## 结论边界','',
      '- 可达表示静态CFG中存在路径，不代表运行时执行；循环不代表恶意；无路径也不等于语义互斥。',
      '- 标签只在Witness冻结后用于TP/FN/FP/TN后验统计，不能把相关性解释为恶意行为真值。',
      '- 本状态最多授权下一轮单次固定消融，不证明性能提升，也不授权CFG-GNN、未来调参或修改正式三背景。','']
    (OUT/'summary.md').write_text('\n'.join(report),encoding='utf-8')
    print('CFGW_SUMMARY_DONE status='+status,flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('preflight','scan','summarize')); p.add_argument('--workers',type=int,default=4); a=p.parse_args()
    if a.phase=='preflight':preflight()
    elif a.phase=='scan':scan_all(a.workers)
    else:summarize()


if __name__=='__main__':main()

