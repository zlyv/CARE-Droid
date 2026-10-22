"""Read saved Direct-C43 models, verify numerical replay, and trace same frozen Host identities; never fit."""
from __future__ import annotations
import argparse,gzip,json
from collections import defaultdict
import numpy as np
from audit_common import *

def endpoints(mc,wanted,data,roles):
    path=PROJECT/'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz';payloads={}
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            d=json.loads(line);sid=str(d.get('sha256','')).casefold()
            if sid in wanted and d.get('status')=='OK':
                if sid in payloads and payloads[sid]!=d['payload']:raise RuntimeError('different duplicate endpoint payload '+sid)
                payloads[sid]=d['payload']
    mapping_path=Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json');manifest=read_json(mapping_path)
    if manifest['model_role_order']!=roles:raise RuntimeError('frozen model Role order differs')
    mapping=manifest['payload_to_model_role'];module=mc.module('_meea_hostrel',PROJECT/'scripts/care-di25-compact43-relationquery-source-v1/hostrel_features.py')
    position={s:i for i,s in enumerate(data['ids'])};out={};audit=[]
    for sid in sorted(wanted):
        payload=payloads[sid];r=module.reconstruct_host_relations(payload,mapping);i=position[sid];off=data['host_offsets'];b=data['host_values'][off[i]:off[i+1]];n=int(data['real_host_count'][i])
        if len(r)!=n:raise RuntimeError('Host count alignment failed '+sid)
        if n:
            p=np.stack([a['presence'] for a in r]);q=np.stack([a['endpoint_counts'] for a in r]);rel=np.stack([a['directed_neighbor_role'] for a in r])
            difference=max(float(np.max(np.abs(b[:,:31]-p))),float(np.max(np.abs(b[:,43:74]-q))),float(np.max(np.abs(b[:,74:105]-rel))))
            if difference>1e-6:raise RuntimeError('frozen Host order/Role/Relation differs '+sid+' '+str(difference))
        else:difference=0.
        slots=[];target_nodes=set(np.flatnonzero(np.asarray(payload['node_base_features'])[:,4]>.5)) if payload['node_base_features'] else set()
        for k,a in enumerate(r):
            node=a['host_node'];targets=sorted({int(t) for s,t in zip(*payload['edge_index']) if int(s)==node and int(t) in target_nodes})
            slots.append(dict(slot=k,host_node_id=str(a['host_id']),role_indices=np.flatnonzero(a['presence']).tolist(),endpoint_ids=[str(payload['node_ids'][t]) for t in targets],endpoint_roles={str(payload['node_ids'][t]):[roles[mapping[int(v)]] for v in payload['node_role_ids'][t]] for t in targets}))
        if not n:slots=[dict(slot=0,host_node_id='EMPTY_PLACEHOLDER',role_indices=[],endpoint_ids=[],endpoint_roles={})]
        out[sid]=slots;audit.append(dict(sha256=sid,status='EXACT',real_hosts=n,stored_slots=len(b),max_feature_difference=difference))
    write_csv(OUTPUT/'host_alignment.csv',audit);return out

def run(mc,data,members,descriptor,roles,device='cuda'):
    v4,_,_=mc.runtime();torch=v4.torch
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but PyTorch driver runtime is unavailable; use an explicit CPU audit version')
    dev=v4.resolve_device(device)
    if dev.type=='cuda' and torch.cuda.mem_get_info()[0]<1024**3:raise RuntimeError('insufficient free GPU memory; use --device cpu')
    from train_stage2 import raw_bags,transform_saved
    position={s:i for i,s in enumerate(data['ids'])};wanted={r['sha256'] for r in members};host=endpoints(mc,wanted,data,roles)
    traces={s:[] for s in wanted};checks=[];coverage=[]
    tags=['C0R_2017']+[w+'_'+c for w in ['T2','T3','T4'] for c in ['C1','C2']]
    for tag in tags:
        selected=[r for r in members if tag=='C0R_2017' or tag.startswith(r['window']+'_')]
        if not selected:continue
        ix=np.asarray([position[r['sha256']] for r in selected]);bags=raw_bags(data,ix)
        for seed in SEEDS:
            path=BASE/'models/stage2'/tag/(str(seed)+'.pkl');obj=load_checkpoint(path,mc,data);meta=obj['metadata'];args=argparse.Namespace(**meta['args'])
            train=np.asarray([position[s] for s in meta['train_sha']],int)
            if set(meta['train_sha'])&{r['sha256'] for r in selected} or np.any(data['years'][train]>=min(data['years'][ix])):raise RuntimeError('model train/validation scope invalid '+tag)
            vp={s:j for j,s in enumerate(meta['validation_sha'])};vi=np.asarray([vp[r['sha256']] for r in selected]);gv=np.asarray(obj['global_validation'][vi],np.float32)
            augmented=transform_saved(v4,bags,obj['background'],args);graphs=v4.transform_role_graphs(data['s7'][ix],obj['graph_scaler']);gt=v4.graph_tensors(graphs,dev)
            v4.seed_everything(seed);model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev);model.load_state_dict(obj['state_dict']);model.eval();prob=[]
            original_select=model.select_local_evidence;capture={}
            def tracking(g,r):
                cache,gg=original_select(g,r);capture['cache']=cache;return cache,gg
            model.select_local_evidence=tracking
            with torch.no_grad():
                for j,r in enumerate(selected):
                    sid=r['sha256'];one_graph={k:v[j] for k,v in gt.items()};region=torch.as_tensor(augmented[j],dtype=torch.float32,device=dev)
                    logit,diag,trace=model.forward_with_trace(torch.as_tensor(gv[j],device=dev),torch.zeros((),device=dev),one_graph,region)
                    p=float(torch.sigmoid(logit).cpu());prob.append(p);cache=capture['cache'];att=cache.attention.detach().cpu().numpy();score=cache.score.detach().cpu().numpy()
                    if len(att)!=len(host[sid]) or not np.isfinite(att).all() or abs(att.sum()-1)>1e-5:raise RuntimeError('attention/Host alignment failed '+sid)
                    rank=np.lexsort((np.arange(len(att)),-att));top=rank[:5];critical=int(np.argmax(score));background=np.asarray(augmented[j][:,107:111],float)
                    row=dict(tag=tag,seed=seed,probability=p,logit=float(logit.cpu()),critical_slot=critical,critical_host_node_id=host[sid][critical]['host_node_id'],critical_roles=[roles[k] for k in host[sid][critical]['role_indices']],attention=att.tolist(),host_score=score.tolist(),top_hosts=[dict(**host[sid][int(k)],attention=float(att[k]),local_score=float(score[k]),background_values=background[k].tolist()) for k in top],within_model_background_mean=background.mean(0).tolist(),global_input_sha256=hashlib.sha256(gv[j].tobytes()).hexdigest(),diagnostic=diag.detach().cpu().tolist(),seed_label_correct=bool((p>=.5)==r['label']))
                    traces[sid].append(row)
            diff=float(np.max(np.abs(np.asarray(prob)-obj['probability'][vi])))
            checks.append(dict(tag=tag,seed=seed,rows=len(selected),max_probability_difference=diff,tolerance=1e-6,status='PASS' if diff<=1e-6 else 'FAIL'))
            if diff>1e-6:dump(OUTPUT/'replay_validation.json',dict(status='FAIL',details=checks));raise RuntimeError('saved model replay failed '+tag+' '+str(diff))
            # Descriptive raw-space coverage only; no same-label Host truth inferred.
            distance=np.sqrt(np.mean((descriptor[ix,None,:]-descriptor[train][None,:,:])**2,axis=2))
            for j,r in enumerate(selected):
                order=np.lexsort((data['ids'][train],distance[j]))[:10];nearest=train[order]
                coverage.append(dict(sha256=r['sha256'],tag=tag,seed=seed,training_n=len(train),nearest_train_distance=float(distance[j,order[0]]),nearest_10_same_label=sum(data['y'][nearest]==r['label']).item(),neighbor_sha='|'.join(data['ids'][nearest]),neighbor_labels='|'.join(map(str,data['y'][nearest].tolist())),neighbor_years='|'.join(map(str,data['years'][nearest].tolist())),interpretation='posthoc fixed-raw-space coverage; sensitive to training N; not a cause or class-semantic proof'))
            del model,gt,augmented,obj
            if dev.type=='cuda':torch.cuda.empty_cache()
            print('MEEA_REPLAY',tag,seed,'rows',len(selected),'difference',diff,flush=True)
    compact=[];selected_slots={}
    for r in members:
        sid=r['sha256'];records=traces[sid];blocks={}
        for tag in sorted({a['tag'] for a in records}):
            rows=[a for a in records if a['tag']==tag];att=np.mean([a['attention'] for a in rows],axis=0);rank=np.lexsort((np.arange(len(att)),-att));p=np.mean([a['probability'] for a in rows]);old_key='C0' if tag=='C0R_2017' else tag[-2:]
            expected=r[old_key+'_probability']
            if abs(p-expected)>1e-6:raise RuntimeError('ensemble differs from frozen per-APK results '+sid+' '+tag)
            blocks[old_key]=dict(tag=tag,mean_probability=float(p),seed_probabilities=[a['probability'] for a in rows],correct_seed_count=sum(a['seed_label_correct'] for a in rows),critical_host_ids=[a['critical_host_node_id'] for a in rows],mean_attention_top3=[dict(**host[sid][int(k)],mean_attention=float(att[k])) for k in rank[:3]])
        slots=sorted({a['critical_slot'] for a in records}|{h['slot'] for b in blocks.values() for h in b['mean_attention_top3']})
        selected_slots[sid]=[host[sid][k] for k in slots]
        dump(OUTPUT/'private/traces'/(sid+'.json'),dict(sha256=sid,member_id=r['member_id'],models=blocks,full_seed_traces=records,selected_code_slots=slots,hidden_vectors_compared_across_models=False,attention_not_semantic_or_causal_proof=True,background_scales_not_cross_model_comparable=True))
        compact.append(dict(sha256=sid,member_id=r['member_id'],year=r['year'],category=r['category'],target=r['target'],label=r['label'],C0_correct_seeds=blocks['C0']['correct_seed_count'],C1_correct_seeds=blocks['C1']['correct_seed_count'],C2_correct_seeds=blocks['C2']['correct_seed_count'],C0_probability=blocks['C0']['mean_probability'],C1_probability=blocks['C1']['mean_probability'],C2_probability=blocks['C2']['mean_probability'],C0_C1_mean_critical_same=blocks['C0']['mean_attention_top3'][0]['host_node_id']==blocks['C1']['mean_attention_top3'][0]['host_node_id'],code_host_slots=len(slots),cause='UNKNOWN_REQUIRES_VERIFIED_PROGRAM_FACTS'))
    write_csv(OUTPUT/'private/model_behavior.csv',compact);write_csv(OUTPUT/'private/training_coverage.csv',coverage);dump(OUTPUT/'private/selected_code_hosts.json',selected_slots);dump(OUTPUT/'replay_validation.json',dict(status='PASS',details=checks,model_training_performed=False,preprocessing_refit=False,tolerance=1e-6,device=str(dev)))
    return selected_slots,compact
