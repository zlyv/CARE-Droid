from __future__ import annotations

import collections
import copy
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from common import *


ARMS = ['H0_FROZEN_DIRECT_C43','G1_APK_GLOBAL_WITNESS3','W2_SIZE3_CONTROL','W3_SIZE_CONDITIONAL_SHUFFLE',
        'H1_HOST_BOUND_CFG','H2_HOST_SIZE_CONTROL','H3_HOST_BINDING_SHUFFLE','H4_HOST_TOPOLOGY_BINDING_SHUFFLE']


def entropy_norm(a):
    a=np.asarray(a,dtype=float)
    return float(-np.sum(a*np.log(np.clip(a,1e-12,None)))/math.log(max(len(a),2)))


def group_stats(frame, value, extra):
    rows=[]
    groupings=[('all',[]),('label',['label']),('correctness',['h1_correct']),('top3',['h1_top3']),
               ('critical',['h1_critical']),('label_x_correctness',['label','h1_correct'])]
    for scope,cols in groupings:
        for keys,g in frame.groupby(cols,dropna=False) if cols else [((),frame)]:
            if not isinstance(keys,tuple): keys=(keys,)
            r={'scope':scope,'value':value,**extra}
            r.update({c:k for c,k in zip(cols,keys)}); r.update(qstats(g[value].values)); rows.append(r)
    return rows


def load_all():
    add_paths()
    from bfa_core import load_data_contract, formal_split
    data,_=load_data_contract()
    ids=list(map(str,data['ids'])); years=np.asarray(data['years']); source=np.where(years==2017)[0]
    if len(source)!=799: raise RuntimeError('source n')
    h1=load_module('audit_h1',ROOT/'scripts/care-host-bound-cfg-source-future-v3/run_experiment.py')
    real,control=h1.load_feature_cache(data)
    ego=np.load(EGO_SOURCE,allow_pickle=False)
    return data,ids,source,formal_split,h1,real,control,ego


def model_probability_maps(ids, data):
    maps={a:{s:{} for s in SEEDS} for a in ARMS}
    for split in SPLITS:
        for seed in SEEDS:
            old=load_pickle(SCBE/f'models/{split}/{seed}/M0_REPRODUCED.pkl')
            for sha,p in zip(old['ids'],old['probability']): maps['H0_FROZEN_DIRECT_C43'][seed][str(sha)]=float(p)
            for arm in ['H1_HOST_BOUND_CFG','H2_HOST_SIZE_CONTROL','H3_HOST_BINDING_SHUFFLE','H4_HOST_TOPOLOGY_BINDING_SHUFFLE']:
                ck=load_pickle(H1_OUT/f'models/{split}/{seed}/{arm}.pkl')
                for sha,p in zip(ck['ids'],ck['probability']): maps[arm][seed][str(sha)]=float(p)
            for src_arm,dst_arm in [('W1_WITNESS3','G1_APK_GLOBAL_WITNESS3'),('W2_SIZE3_CONTROL','W2_SIZE3_CONTROL'),('W3_SIZE_CONDITIONAL_SHUFFLE','W3_SIZE_CONDITIONAL_SHUFFLE')]:
                ck=load_pickle(CFG_ABL/f'models/{split}/{seed}/{src_arm}.pkl')
                for sha,p in zip(ck['ids'],ck['probability']): maps[dst_arm][seed][str(sha)]=float(p)
    return maps


def traces_and_replay(data, ids, source, formal_split, h1, real, control, ego):
    torch=None; replay=[]; gate_rows=[]; residual_rows=[]; critical=[]; sensitivity=[]
    eo=ego['ego_offsets']; host_ids=list(map(str,ego['host_ids'])); role_offsets=ego['host_role_offsets']; role_ids=ego['host_role_ids']
    source_pos={int(idx):p for p,idx in enumerate(source)}
    param_audit=None
    for split in SPLITS:
        for seed in SEEDS:
            pack=formal_split(data,split,seed,'cuda')
            v4,args,dev=pack['v4'],pack['args'],pack['dev']; torch=v4.torch
            ck=load_pickle(H1_OUT/f'models/{split}/{seed}/H1_HOST_BOUND_CFG.pkl')
            cfg,_=h1.prepare_cfg_bags(data,pack['val'],'H1_HOST_BOUND_CFG',seed,real,control,ck['feature_scaler'])
            bags=h1.append_cfg(pack['bags_val'],cfg)
            h0=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev)
            h0.load_state_dict(pack['old']['state_dict'],strict=True); h0.eval()
            h1m=h1.host_bound_model(v4,args).to(dev); h1.initialize_from_m0(h1m,pack['old'])
            combined=dict(pack['old']['state_dict']);combined.update(ck['adapter_state']);h1m.load_state_dict(combined,strict=True);h1m.eval()
            disabled=h1.host_bound_model(v4,args).to(dev);h1.initialize_from_m0(disabled,pack['old']);disabled.load_state_dict(combined,strict=True)
            with torch.no_grad():
                disabled.cfg_adapter[2].weight.zero_();disabled.cfg_adapter[2].bias.zero_()
            disabled.eval()
            if param_audit is None:
                h0sd=h0.state_dict();h1sd=h1m.state_dict(); base_names=set(h0sd); common=set(h0sd)&set(h1sd)
                param_audit={
                    'base_parameter_or_buffer_key_count':len(base_names),
                    'h1_key_count':len(h1sd),'base_keys_missing_in_h1':sorted(base_names-set(h1sd)),
                    'base_state_max_abs_diff':max(float(torch.max(torch.abs(h0sd[k]-h1sd[k])).cpu()) for k in common),
                    'cfg_only_keys':sorted(set(h1sd)-set(h0sd)),
                    'batchnorm_module_count_h0':sum('BatchNorm' in type(m).__name__ for m in h0.modules()),
                    'batchnorm_module_count_h1':sum('BatchNorm' in type(m).__name__ for m in h1m.modules()),
                    'h0_eval':not h0.training,'h1_eval':not h1m.training,'disabled_eval':not disabled.training,
                    'base_backbone_frozen_claim':bool(ck.get('base_backbone_frozen')),
                    'backgrounds_changed':ck.get('backgrounds_changed'),'prototype_bank_changed':ck.get('prototype_bank_changed')}
            gt=torch.as_tensor(pack['global_val'],dtype=torch.float32,device=dev)
            graphs=v4.graph_tensors(pack['graph_val'],dev)
            max_prob=max_att=0.; crit_mismatch=0; old_replay=0.; label_mismatch=0
            for i,idx in enumerate(pack['val']):
                idx=int(idx); sha=ids[idx]; graph={k:x[i] for k,x in graphs.items()}
                base_region=torch.as_tensor(pack['bags_val'][i],dtype=torch.float32,device=dev)
                full_region=torch.as_tensor(bags[i],dtype=torch.float32,device=dev)
                zero=torch.zeros((),dtype=torch.float32,device=dev)
                with torch.no_grad():
                    l0,_,t0=h0.forward_with_trace(gt[i],zero,graph,base_region)
                    ld,_,td=disabled.forward_with_trace(gt[i],zero,graph,full_region)
                    l1,_,t1=h1m.forward_with_trace(gt[i],zero,graph,full_region)
                    p0=float(torch.sigmoid(l0).cpu());p_disabled=float(torch.sigmoid(ld).cpu());p1=float(torch.sigmoid(l1).cpu())
                    a0=t0['attention'].detach().cpu().numpy();ad=td['attention'].detach().cpu().numpy();a1=t1['attention'].detach().cpu().numpy()
                    c0=int(np.argmax(a0));cd=int(np.argmax(ad));c1=int(np.argmax(a1))
                    core=full_region[:,:111];cfgx=full_region[:,142:]
                    host=h1m.region_encoder(core); emb=h1m.cfg_adapter(cfgx)
                    gamma=torch.sigmoid(h1m.cfg_gate(torch.cat([host,emb],1))).reshape(-1)
                    res=gamma[:,None]*emb; rn=torch.linalg.vector_norm(res,dim=1)
                    gam=gamma.cpu().numpy(); rnorm=rn.cpu().numpy()
                max_prob=max(max_prob,abs(p0-p_disabled)); max_att=max(max_att,float(np.max(np.abs(a0-ad))))
                crit_mismatch+=c0!=cd; old_replay=max(old_replay,abs(p0-float(pack['old']['probability'][i])))
                label=int(data['y'][idx]); pred0=p0>=.5;pred1=p1>=.5
                pos=source_pos[idx]; left,right=int(eo[pos]),int(eo[pos+1]); one_ids=host_ids[left:right]
                if len(one_ids)==0 and len(a0)==1:
                    one_ids=['__EMPTY_HOST_PLACEHOLDER__']
                if len(one_ids)!=len(a0): raise RuntimeError(f'trace host alignment {sha} ego={len(one_ids)} model={len(a0)}')
                def roles(j):
                    if one_ids[j]=='__EMPTY_HOST_PLACEHOLDER__': return 'NONE'
                    q=left+j; return '|'.join(map(str,role_ids[int(role_offsets[q]):int(role_offsets[q+1])]))
                rank=np.empty(len(a1),int);rank[np.argsort(-a1)]=np.arange(1,len(a1)+1)
                score0=t0['attention'].detach().cpu().numpy();score1=t1['attention'].detach().cpu().numpy()
                for j in range(len(a1)):
                    base={'split':split,'model_seed':seed,'sha256':sha,'label':label,'host_node_id':one_ids[j],
                          'host_roles':roles(j),'host_rank_h1':int(rank[j]),'h1_top3':bool(rank[j]<=3),
                          'h1_critical':bool(j==c1),'h0_critical':bool(j==c0),'h1_correct':bool(pred1==label),
                          'attention_h0':float(a0[j]),'attention_h1':float(a1[j]),
                          'attention_change':float(a1[j]-a0[j]),'critical_changed':bool(c0!=c1)}
                    gate_rows.append({**base,'gamma':float(gam[j])})
                    residual_rows.append({**base,'residual_norm':float(rnorm[j])})
                top0=np.argsort(-a0)[:3];top1=np.argsort(-a1)[:3]
                critical.append({'split':split,'model_seed':seed,'sha256':sha,'label':label,
                    'h0_probability':p0,'h1_probability':p1,'probability_change':p1-p0,
                    'h0_correct':bool(pred0==label),'h1_correct':bool(pred1==label),
                    'fixed':bool(pred0!=label and pred1==label),'broken':bool(pred0==label and pred1!=label),
                    'critical_changed':bool(c0!=c1),'h0_critical_host':one_ids[c0],'h1_critical_host':one_ids[c1],
                    'h0_critical_roles':roles(c0),'h1_critical_roles':roles(c1),
                    'h0_top3_hosts':'|'.join(one_ids[j] for j in top0),'h1_top3_hosts':'|'.join(one_ids[j] for j in top1),
                    'h0_attention_entropy':entropy_norm(a0),'h1_attention_entropy':entropy_norm(a1),
                    'h0_max_attention':float(a0[c0]),'h1_max_attention':float(a1[c1]),
                    'h1_critical_gamma':float(gam[c1]),'h1_critical_residual_norm':float(rnorm[c1])})

                # Exact local sensitivity for this frozen H1 forward.
                h1m.zero_grad(set_to_none=True)
                grad_region=full_region.detach().clone().requires_grad_(True)
                captured={}
                def hook(_m,_inp,out):
                    out.retain_grad(); captured['embedding']=out
                handle=h1m.cfg_adapter.register_forward_hook(hook)
                lg,_,tr=h1m.forward_with_trace(gt[i],zero,graph,grad_region)
                lg.backward(); handle.remove()
                gx=grad_region.grad[:,142:].detach().cpu().numpy()
                eg=captured['embedding'].grad.detach().cpu().numpy()
                embv=captured['embedding'].detach().cpu().numpy()
                contrib=np.sum(eg*(gam[:,None]*embv),axis=1)
                sensitivity.append({'split':split,'model_seed':seed,'sha256':sha,'label':label,
                    'cfg_input_gradient_l1':float(np.abs(gx).sum()),
                    'critical_cfg_input_gradient_l1':float(np.abs(gx[c1]).sum()),
                    'gated_embedding_first_order_sensitivity_sum':float(np.abs(contrib).sum()),
                    'critical_gated_embedding_first_order_sensitivity':float(contrib[c1]),
                    **{f'cfg_dim_{d}_abs_gradient_sum':float(np.abs(gx[:,d]).sum()) for d in range(16)}})
            replay.append({'split':split,'model_seed':seed,'n':len(pack['val']),
                'h0_vs_h1_disabled_max_probability_diff':max_prob,
                'h0_vs_h1_disabled_max_attention_diff':max_att,
                'h0_vs_h1_disabled_critical_mismatch_count':crit_mismatch,
                'h0_forward_vs_saved_probability_max_diff':old_replay,
                'validation_ids_match_checkpoint':list(map(str,data['ids'][pack['val']]))==list(map(str,ck['ids']))})
            print('[model]',split,seed,'replay',max_prob,'trace done',flush=True)
            del h0,h1m,disabled,gt,graphs
            if dev.type=='cuda': torch.cuda.empty_cache()
    replay_status=all(r['h0_vs_h1_disabled_max_probability_diff']<=1e-6 and r['h0_vs_h1_disabled_max_attention_diff']<=1e-6 and r['h0_vs_h1_disabled_critical_mismatch_count']==0 and r['validation_ids_match_checkpoint'] for r in replay)
    write_json_new(OUT/'h0_replay_check.json',{'status':'PASS' if replay_status else 'FAIL','tolerance':1e-6,'parameter_and_buffer_audit':param_audit,'fold_seed_checks':replay})
    gate=pd.DataFrame(gate_rows);res=pd.DataFrame(residual_rows);crit=pd.DataFrame(critical);sens=pd.DataFrame(sensitivity)
    gs=[];rs=[]
    for (sp,se),g in gate.groupby(['split','model_seed']): gs+=group_stats(g,'gamma',{'split':sp,'model_seed':se})
    for (sp,se),g in res.groupby(['split','model_seed']): rs+=group_stats(g,'residual_norm',{'split':sp,'model_seed':se})
    gs += group_stats(gate,'gamma',{'split':'ALL','model_seed':'ALL'})
    rs += group_stats(res,'residual_norm',{'split':'ALL','model_seed':'ALL'})
    write_csv_new(OUT/'gate_statistics.csv',pd.DataFrame(gs))
    write_csv_new(OUT/'residual_norm_statistics.csv',pd.DataFrame(rs))
    write_csv_new(OUT/'critical_host_changes.csv',crit)
    write_csv_new(OUT/'cfg_sensitivity_statistics.csv',sens)
    write_json_new(OUT/'gate_residual_correlations.json',{
        'gamma_vs_attention_h1':pearson_spearman(gate.gamma,gate.attention_h1),
        'gamma_vs_attention_change':pearson_spearman(gate.gamma,gate.attention_change),
        'residual_norm_vs_attention_h1':pearson_spearman(res.residual_norm,res.attention_h1),
        'residual_norm_vs_attention_change':pearson_spearman(res.residual_norm,res.attention_change),
        'critical_change_fraction':float(crit.critical_changed.mean()),
        'fixed_count':int(crit.fixed.sum()),'broken_count':int(crit.broken.sum())})


def paired_results(data, ids, source, maps):
    ymap={ids[i]:int(data['y'][i]) for i in source}; folds=source_fold_map(); source_ids=[ids[i] for i in source]
    stage=pd.read_csv(H1_OUT/'per_apk_predictions.csv.gz')
    stage=stage[(stage.year==2017)&(stage.arm=='H0_FROZEN_DIRECT_C43')]
    routing={}
    for ms,g in stage.groupby('manifest_seed'):
        routing[int(ms)]={str(r.sha256):(float(r.stage1_probability),bool(r.routed)) for r in g.itertuples()}
    rows=[]; fixed=[]
    for arm in ARMS:
        for seed in SEEDS:
            for split in SPLITS+['ALL']:
                ss=[s for s in source_ids if split=='ALL' or folds[s]==split]
                y=np.array([ymap[s] for s in ss]); p=np.array([maps[arm][seed][s] for s in ss])
                rows.append({'level':'stage2_all_apk','arm':arm,'reference':None,'split':split,'model_seed':seed,'manifest_seed':None,**binary_metrics(y,p)})
                for ms,route in routing.items():
                    final=np.array([maps[arm][seed][s] if route[s][1] else route[s][0] for s in ss])
                    rows.append({'level':'manifest_routed','arm':arm,'reference':None,'split':split,'model_seed':seed,'manifest_seed':ms,**binary_metrics(y,final)})
    comparisons=[('H1_HOST_BOUND_CFG','H0_FROZEN_DIRECT_C43'),('H1_HOST_BOUND_CFG','G1_APK_GLOBAL_WITNESS3'),
                 ('H1_HOST_BOUND_CFG','H2_HOST_SIZE_CONTROL'),('H1_HOST_BOUND_CFG','H3_HOST_BINDING_SHUFFLE'),
                 ('H1_HOST_BOUND_CFG','H4_HOST_TOPOLOGY_BINDING_SHUFFLE')]
    foldarr=np.array([folds[s] for s in source_ids]); y=np.array([ymap[s] for s in source_ids])
    for arm,ref in comparisons:
        for seed in SEEDS:
            pa=np.array([maps[arm][seed][s] for s in source_ids]);pb=np.array([maps[ref][seed][s] for s in source_ids])
            ci=stratified_paired_bootstrap(y,pa,pb,foldarr,n_boot=5000,seed=20261020+seed)
            rows.append({'level':'paired_bootstrap_stage2','arm':arm,'reference':ref,'split':'ALL','model_seed':seed,'manifest_seed':None,**ci})
            for s,yy,a,b in zip(source_ids,y,pa,pb):
                ca=a>=.5;cb=b>=.5
                if ca!=cb:
                    fixed.append({'level':'stage2_all_apk','arm':arm,'reference':ref,'model_seed':seed,'manifest_seed':None,
                                  'sha256':s,'fold':folds[s],'label':int(yy),'candidate_probability':a,'reference_probability':b,
                                  'fixed':bool(cb!=yy and ca==yy),'broken':bool(cb==yy and ca!=yy)})
            for ms,route in routing.items():
                fa=np.array([maps[arm][seed][s] if route[s][1] else route[s][0] for s in source_ids])
                fb=np.array([maps[ref][seed][s] if route[s][1] else route[s][0] for s in source_ids])
                ci=stratified_paired_bootstrap(y,fa,fb,foldarr,n_boot=5000,seed=20262020+seed+ms)
                rows.append({'level':'paired_bootstrap_manifest_routed','arm':arm,'reference':ref,'split':'ALL','model_seed':seed,'manifest_seed':ms,**ci})
                for s,yy,a,b in zip(source_ids,y,fa,fb):
                    ca=a>=.5;cb=b>=.5
                    if ca!=cb:
                        fixed.append({'level':'manifest_routed','arm':arm,'reference':ref,'model_seed':seed,'manifest_seed':ms,
                                      'sha256':s,'fold':folds[s],'label':int(yy),'candidate_probability':a,'reference_probability':b,
                                      'fixed':bool(cb!=yy and ca==yy),'broken':bool(cb==yy and ca!=yy)})
    write_csv_new(OUT/'paired_ablation_results.csv',pd.DataFrame(rows))
    write_csv_new(OUT/'fixed_broken_analysis.csv',pd.DataFrame(fixed))


def main():
    if not OUT.exists(): raise RuntimeError('run static_audit.py first')
    for name in ['h0_replay_check.json','gate_statistics.csv','paired_ablation_results.csv']:
        if (OUT/name).exists(): raise RuntimeError(f'refusing overwrite {name}')
    data,ids,source,formal_split,h1,real,control,ego=load_all()
    traces_and_replay(data,ids,source,formal_split,h1,real,control,ego)
    maps=model_probability_maps(ids,data)
    paired_results(data,ids,source,maps)
    print('[model] complete',flush=True)


if __name__=='__main__': main()
