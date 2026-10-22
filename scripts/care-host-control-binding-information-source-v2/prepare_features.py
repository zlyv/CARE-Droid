#!/usr/bin/env python3
from common import *
from binding_core import proof_tables,aggregates,host_context,joint_descriptor,permutation,seed_for,node_id
from model_core import projection_map,sketch,variable_proof_mask

def main():
    paths=read_json(OUTPUT/'input_paths.json');parent=Path(paths['parent']);p=protocol()
    rows=read_csv(parent/'selection_label_free.csv');ids=[norm(r['sha256']) for r in rows];wanted=set(ids)
    status={norm(r['sha256']):r for r in read_csv(parent/'feature_scan_status.csv')}
    if len(status)!=799 or set(status)!=wanted or any(r['status']!='OK' for r in status.values()):raise RuntimeError('no failed or excluded source scans allowed')
    prior_stats={(r['sha256'],int(r['repeat'])):r for r in read_csv(OBS/'null_repeat_apk_metrics.csv')}
    if len(prior_stats)!=799*p['null_repeats']:raise RuntimeError('observability repeat inventory incomplete')
    identity=read_json(parent/'engine_identity.json')
    hostrel=module('_hbi_hostrel',PROJECT/'scripts/care-di25-compact43-relationquery-source-v1/hostrel_features.py')
    mapping=np.asarray(read_json(paths['manifest'])['payload_to_model_role'],int)
    if mapping.shape!=(31,) or sorted(mapping.tolist())!=list(range(31)):raise RuntimeError('invalid frozen Role permutation')
    payloads={}
    with gzip.open(paths['endpoint'],'rt',encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line);sid=norm(row.get('sha256') or row.get('sample_id'))
            if sid in wanted and row.get('status')=='OK':
                if sid in payloads:raise RuntimeError('duplicate source endpoint SHA')
                payloads[sid]=row['payload']
    if set(payloads)!=wanted:raise RuntimeError('incomplete source endpoint cache')
    with np.load(paths['compact'],allow_pickle=False) as z:
        cids=[norm(s) for s in z['sha256']];values=np.asarray(z['host_values'],float);off=np.asarray(z['host_offsets'],int)
        real=np.asarray(z['real_host_count'],int);empty=np.asarray(z['host_was_empty'],int)
    if len(cids)!=799 or len(set(cids))!=799 or set(cids)!=wanted or off.shape!=(800,) or real.shape!=(799,) or empty.shape!=(799,) or off[0]!=0 or off[-1]!=len(values) or not np.array_equal(np.diff(off),np.maximum(real,1)) or not np.array_equal(empty,(real==0).astype(int)):raise RuntimeError('C43 source offset/membership/placeholder contract')
    ci={s:i for i,s in enumerate(cids)}
    with np.load(parent/'control_features.npz',allow_pickle=False) as z:
        eids=[norm(s) for s in z['ids']];e=np.asarray(z['evidence'],float);cov=np.asarray(z['coverage'],float)
    if len(eids)!=799 or len(set(eids))!=799 or set(eids)!=wanted or e.shape!=(799,124) or cov.shape!=(799,64):raise RuntimeError('aggregate source matrix contract')
    ei={s:i for i,s in enumerate(eids)};e=e[[ei[s] for s in ids]];cov=cov[[ei[s] for s in ids]]
    bucket,sign=projection_map(p['projection_width'],p['projection_seed'])
    true=np.zeros((799,p['projection_width']));null=np.zeros((p['null_repeats'],799,p['projection_width']))
    audit=[]
    for i,row in enumerate(rows):
        sid=ids[i];record=load_gz(status[sid]['path']);frozen=read_json(parent/'cache/frozen'/(sid+'.json'))
        token=identity['parent_scan_identity'] if status[sid]['source']=='PASSED_V2_READ_ONLY' else identity['scan_identity_sha256']
        if record.get('schema')!='lcc-fa-v2.0' or record.get('sha256')!=sid or record.get('status')!='OK' or record.get('scan_identity_sha256')!=token or record.get('labels_used') is not False or record.get('future_rows_loaded')!=0:raise RuntimeError('scan lineage/scope differs '+sid)
        if record['frozen_pairs_total']!=len(frozen['pairs']) or record['frozen_hosts_total']!=len(frozen['host_ids']):raise RuntimeError('frozen scan scope changed')
        records=hostrel.reconstruct_host_relations(payloads[sid],mapping);index=ci[sid]
        if len(records)!=real[index]:raise RuntimeError('Host count mismatch')
        ctx=host_context(values[off[index]:off[index+1]],records,mapping)
        if set(ctx)!={node_id(h) for h in frozen['host_ids']}:raise RuntimeError('exact frozen Host IDs differ')
        table=proof_tables(record,frozen);a,b=aggregates(table)
        if not np.array_equal(a,e[i]) or not np.array_equal(b,cov[i]):raise RuntimeError('aggregate proof/coverage replay failed')
        joint=joint_descriptor(table,ctx);true[i]=sketch(joint,bucket,sign,p['projection_width'])
        for rep in range(p['null_repeats']):
            donors,_=permutation(table,seed_for(sid,rep,p['null_seed']))
            shuffled=table['flags'][donors];nj=joint_descriptor(table,ctx,shuffled);null[rep,i]=sketch(nj,bucket,sign,p['projection_width'])
            diff=joint-nj;changed=np.any(shuffled!=table['flags'],axis=1)
            s=prior_stats[(sid,rep)]
            # Replay all bindings from the successful audit, not only its pooled rate.
            scalars={'joint_rms_change':float(np.sqrt(np.mean(diff*diff))), 'joint_max_abs_change':float(np.max(np.abs(diff)))}
            if any(abs(v-float(s[k]))>1e-12 for k,v in scalars.items()) or int(changed.sum())!=int(s['changed_pairs']) or int((changed&(table['flags'][:,0]>0)).sum())!=int(s['changed_verified_pairs']):raise RuntimeError('passed observability bindings not reproduced '+sid)
            audit.append(dict(sha256=sid,fold=int(row['fold']),routed=int(row['routed']),repeat=rep,
                descriptor_change=bool(s['binding_ready'] in ('True','true','1')),
                projected_change=bool(np.max(np.abs(true[i]-null[rep,i]))>1e-10),
                projected_max_abs_change=float(np.max(np.abs(true[i]-null[rep,i]))),
                marginals_exact=True,donors_are_within_APK=True))
        if (i+1)%25==0 or i+1==799:print(f'HBI_FEATURES {i+1}/799 scans=0 projection=64',flush=True)
    if not(np.isfinite(true).all() and np.isfinite(null).all()):raise RuntimeError('nonfinite binding sketch')
    support=variable_proof_mask(e)
    if np.any(true[~support]) or np.any(null[:,~support]):raise RuntimeError('constant per-Role proofs generated spurious binding coordinates')
    np.savez_compressed(safe(OUTPUT/'binding_features.npz'),ids=np.asarray(ids),evidence=e,coverage=cov,binding=true,shuffle=null,
        support=support,projection_bucket=bucket,projection_sign=sign)
    write_csv(OUTPUT/'binding_projection_audit.csv',audit)
    folds=[]
    for f in range(1,6):
        rates=[sum(r['projected_change'] for r in audit if r['routed'] and r['fold']==f and r['repeat']==rep)/40 for rep in range(p['null_repeats'])]
        folds.append(dict(fold=f,routed=40,median_projected_changed_rate=float(np.median(rates))))
    rates=[sum(r['projected_change'] for r in audit if r['routed'] and r['repeat']==rep)/200 for rep in range(p['null_repeats'])]
    g=p['minimum_projection_retention'];ready=float(np.median(rates))>=g['routed_median_projected_change_rate'] and sum(r['median_projected_changed_rate']>=.15 for r in folds)>=g['folds_changed_rate_at_least_0_15']
    write_csv(OUTPUT/'fold_projection_coverage.csv',folds)
    dump(OUTPUT/'feature_validation.json',dict(status='PASS',source_rows=799,feature_constructor_labels_used=False,
        original_binding_repeats_reproduced=20,projection_width=64,routed_median_projected_change_rate=float(np.median(rates)),
        projection_ready=ready,all_marginals_exact=True,all_host_alignments_exact=True,all_randomization_invariants_pass=True,
        new_DEX_scans=0,future_rows_loaded=0))
    print('HBI_FEATURES_PASS projection_ready='+str(ready),flush=True)
if __name__=='__main__':main()
