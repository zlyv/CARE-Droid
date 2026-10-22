"""Audit and SHA-align existing multi-year P31, A0 Host, global24 and S7 caches; never extract APKs."""
from __future__ import annotations
import importlib.metadata,os,pickle,sys,time
from pathlib import Path
import numpy as np
from myd_common import *
from protocol import deduplicate,split_window,WINDOWS

def paths():
    r=read_json(DIAG/'resolved_inputs.json');fc=family('common')
    return dict(source_p31=fc.SOURCE_P31,future_p31=Path(r['future_p31_cache']),
        host=DIAG/'cache/a0_role_2017_2021.npz',future_s7=DIAG/'cache/future_s7.npz',
        source_s7=PROJECT/'artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz',
        sealed=Path(r['sealed_features']),schema=PROJECT/'artifacts/p1-features/feature_schema.json',
        endpoint=Path(r['endpoint_cache']),future_labels=fc.FUTURE_LABELS,
        future_reference=fc.FUTURE_REFERENCE,historical_direct=PROJECT/'outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv',
        source_contract=PROJECT/'artifacts/care-litelocal-semantic-source-v1/feature_contract.json',
        source_resolved=PROJECT/'artifacts/care-litelocal-semantic-source-v1/resolved_inputs.json',
        future_resolved=PROJECT/'artifacts/care-litelocal-p31-diagnostics-v1/preflight/resolved_inputs.json',
        mapping=Path(read_json(PROJECT/'artifacts/care-litelocal-p31-diagnostics-v1/preflight/resolved_inputs.json')['mapping']))

def finite(name,a,width):
    if a.ndim!=2 or a.shape[1]!=width or not np.isfinite(a).all():raise RuntimeError(f'invalid {name}: shape={a.shape}, width={width}')

def unique_archive(p,key='sha256'):
    a=np.load(p,allow_pickle=False);ids=[norm(s) for s in a[key]]
    if len(ids)!=len(set(ids)):raise RuntimeError('duplicate SHA inside cache '+str(p))
    return a,{s:i for i,s in enumerate(ids)}

def load_data():
    report=read_json(OUTPUT/'dataset_audit.json')
    if report['status']!='PASS':raise RuntimeError('dataset audit is not PASS')
    for p,h in report['bundle_hashes'].items():
        if sha256(p)!=h:raise RuntimeError('prepared data changed '+p)
    with np.load(OUTPUT/'data.npz',allow_pickle=False) as a:d={k:a[k] for k in a.files}
    with (OUTPUT/'feature_rows.pkl').open('rb') as f:d['feature_rows']=pickle.load(f)
    d['signature']=report['dataset_signature'];return d

def audit(resume=False):
    start=time.perf_counter();OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'dataset_audit.json').is_file():
        if not resume:raise RuntimeError('audit exists: use --resume to verify, never overwrite an earlier audit')
        d=load_data();verify_assets();print('MYD_AUDIT_RESUME_VERIFIED',flush=True);return d
    p=paths();missing=[str(v) for v in p.values() if not v.is_file()]
    if missing:
        dump(OUTPUT/'missing_assets.json',dict(missing=missing));raise RuntimeError('missing required caches '+repr(missing))
    v4,adapter,args=runtime()
    # Protect actual caches and every original Python dependency loaded in this process.
    originals={str(v.resolve()):str(k) for k,v in p.items()}
    for m in list(sys.modules.values()):
        fp=getattr(m,'__file__',None)
        if fp and str(PROJECT) in str(fp) and Path(fp).suffix=='.py' and HERE not in Path(fp).resolve().parents:originals[str(Path(fp).resolve())]='original dependency'
    originals[str((DIAG/'resolved_inputs.json').resolve())]='resolved inputs'
    dump(OUTPUT/'baseline_asset_manifest_before.json',[asset(k,v) for k,v in originals.items()])
    src,sp=unique_archive(p['source_p31']);fut,fp=unique_archive(p['future_p31']);host,hp=unique_archive(p['host']);s7f,s7p=unique_archive(p['future_s7'])
    roles=[str(x) for x in src['role_names']]
    if len(roles)!=31 or len(set(roles))!=31 or roles!=[str(x) for x in fut['role_names']]:raise RuntimeError('Role31 order differs')
    import yaml
    mapping=yaml.safe_load(p['mapping'].read_text())
    if roles!=list(mapping['roles']):raise RuntimeError('actual semantic mapping differs from both cache role columns')
    sr=read_json(p['source_resolved']);fr=read_json(p['future_resolved'])
    if Path(sr['mapping']).resolve()!=Path(fr['mapping']).resolve():raise RuntimeError('source/future producers use different Role mapping paths')
    manifest,source_rows,clean_rows,by_sha,schema,canonical=v4.audit.load_inputs(args)
    source_labels={norm(r['sha256']):int(r['label']) for r in manifest}
    future_labels={norm(r['sha256']):(int(r['year']),int(r['label'])) for r in read_csv(p['future_labels'])}
    records=[];p31=[]
    for year,a,index in [(2017,src,sp),(None,fut,fp)]:
        finite('P31',a['role_prevalence'],31)
        for sid,j in index.items():
            yr=year or int(a['year'][j]);lab=source_labels[sid] if yr==2017 else future_labels[sid][1]
            if yr!=2017 and future_labels[sid][0]!=yr:raise RuntimeError('future label year mismatch '+sid)
            values=a['role_prevalence'][j]
            if np.any((values<0)|(values>1)):raise RuntimeError('P31 fraction outside [0,1]')
            records.append(dict(sha256=sid,year=yr,label=lab,order=len(records)));p31.append(values)
    counts={y:sum(r['year']==y for r in records) for y in YEARS}
    if counts!=EXPECTED:raise RuntimeError('year count mismatch '+repr(counts))
    kept,excluded=deduplicate(records);ix=[r['order'] for r in kept];p31=np.asarray(p31,np.float32)[ix]
    source_order={norm(r['sha256']):i for i,r in enumerate(source_rows)}
    endpoint=v4.audit.exp.read_cache(args.cache)
    source_s7,_=v4.v3.load_source_s7(source_rows,endpoint,args)
    rows_by_sha={norm(r['sha256']):r for r in clean_rows};rows_by_sha.update({norm(r['sha256']):r for r in source_rows})
    selected_rows=[rows_by_sha[r['sha256']] for r in kept]
    _,all_global,names=v4.audit.compact_matrix_variant(v4.VARIANT,source_rows,selected_rows,by_sha,schema,canonical)
    gi,_=v4.evidence.split_view_indices(names);glob=np.asarray(all_global[:,gi],np.float32)
    finite('global24',glob,24)
    values=np.asarray(host['host_values'],np.float32);offsets=np.asarray(host['host_offsets'],int)
    finite('A0 host',values,169)
    if offsets.shape!=(len(hp)+1,) or offsets[0]!=0 or offsets[-1]!=len(values) or np.any(np.diff(offsets)<1):raise RuntimeError('invalid raw Host cache offsets')
    if np.any(values[:,105:]!=0):raise RuntimeError('A0 ROLE_ONLY compatibility64 is not zero')
    newvalues=[];newoff=[0];real=[];empty=[];s7=[]
    for r in kept:
        sid=r['sha256'];h=hp[sid];b=values[offsets[h]:offsets[h+1]];newvalues.append(b);newoff.append(newoff[-1]+len(b))
        real.append(int(host['real_host_count'][h]));empty.append(int(host['host_was_empty'][h]))
        if real[-1]==0 and not empty[-1]:raise RuntimeError('unmarked zero Host')
        s7.append(source_s7[source_order[sid]] if r['year']==2017 else s7f['s7'][s7p[sid]])
    s7=np.asarray(s7,np.float32);finite('S7',s7,1546)
    data=dict(ids=np.asarray([r['sha256'] for r in kept]),years=np.asarray([r['year'] for r in kept],np.int16),y=np.asarray([r['label'] for r in kept],np.int8),p31=p31,s7=s7,global24=glob,host_values=np.concatenate(newvalues),host_offsets=np.asarray(newoff),real_host_count=np.asarray(real),host_was_empty=np.asarray(empty))
    hist={norm(r['sha256']):r for r in read_csv(p['historical_direct'])}
    ref={norm(r['sha256']):r for r in read_csv(p['future_reference'])}
    data['historical_base']=np.asarray([float(hist[r['sha256']]['stage1_prob']) if r['year']>2017 else np.nan for r in kept])
    data['historical_expert']=np.asarray([float(hist[r['sha256']]['direct_c43_prob']) if r['year']>2017 else np.nan for r in kept])
    data['historical_members']=np.asarray([[float(ref[r['sha256']]['p_P31__'+m]) for m in MEMBERS] if r['year']>2017 else [np.nan]*4 for r in kept])
    save_npz(OUTPUT/'data.npz',**data);save_pickle(OUTPUT/'feature_rows.pkl',dict(rows=selected_rows,canonical_names=canonical,global_names=[names[i] for i in gi],compact_by_sha=by_sha,schema=schema))
    write_csv(OUTPUT/'sample_manifest.csv',kept);write_csv(OUTPUT/'dedup_exclusions.csv',excluded)
    windows={}
    for w in WINDOWS:
        tr,va=split_window(kept,w);windows[w]=dict(train_n=len(tr),validation_n=len(va),train_years=WINDOWS[w][0],validation_year=WINDOWS[w][1],sha_overlap=0)
    bundle={str(OUTPUT/f):sha256(OUTPUT/f) for f in ['data.npz','feature_rows.pkl','sample_manifest.csv']}
    stats=[dict(year=y,raw_n=counts[y],effective_n=sum(r['year']==y for r in kept),benign=sum(r['year']==y and r['label']==0 for r in kept),malware=sum(r['year']==y and r['label']==1 for r in kept),empty_hosts=sum(data['host_was_empty'][data['years']==y]),zero_p31=int(np.sum(data['p31'][data['years']==y].sum(1)==0))) for y in YEARS]
    producer_paths=[PROJECT/'scripts/care-litelocal-semantic-source-v1/extract_local_semantics.py',PROJECT/'scripts/care-litelocal-p31-diagnostics-v1/extract_future_p31.py']
    producers=[asset(q,'P31 producer version; same semantic Role mapping') for q in producer_paths]
    # The producer binaries differ; record this rather than claiming byte-identical extractors.
    report=dict(status='PASS',raw_n=len(records),effective_n=len(kept),duplicate_excluded=len(excluded),label_conflicts=0,by_year=stats,dimensions=dict(P31=31,raw_host=169,model_host=142,global_dim=24,S7=1546),role_names=roles,mapping_sha256=sha256(p['mapping']),producers=producers,feature_versions='same Role mapping and audited formal A0/S7 caches; source/future P31 producer scripts differ and are recorded',family_metadata='no validated family grouping used; not a family-disjoint test',bundle_hashes=bundle,dataset_signature=signature(bundle),audit_seconds=time.perf_counter()-start,original_apks_opened=0,new_extraction_seconds=0)
    dump(OUTPUT/'dataset_audit.json',report);dump(OUTPUT/'protocol_audit.json',dict(status='PASS',windows=windows,development_only=True,sha_dedup_rule='earliest year; conflicting labels fail',historical_asset_versions_preserved=True,git='NOT_APPLICABLE: user forbids Git commands',stage1='original calibrated SVM',router='original 4 member MeanOpposition; round(n*.25), score descending, SHA ascending',stage2='original A0/RGCondAttn Direct; equal BCE; no residual/gate; original internal early stopping',C2='uniform year quotas with replacement, exactly N draws/epoch; C1-selected epoch count, same data/prototypes/architecture/steps'))
    versions={k:importlib.metadata.version(k) for k in ['numpy','scikit-learn','torch','xgboost']}
    dump(OUTPUT/'environment.json',dict(python=sys.version,versions=versions,git='not applicable; no Git used'))
    extra=read_json(OUTPUT/'baseline_asset_manifest_before.json');known={r['path'] for r in extra}
    for q in producer_paths:
        if str(q.resolve()) not in known:extra.append(asset(q,'P31 producer'))
    dump(OUTPUT/'baseline_asset_manifest_before.json',extra)
    verify_assets();print('MYD_DATA_AUDIT_PASS',report['effective_n'],windows,flush=True);return load_data()

if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser();cli.add_argument('--resume',action='store_true');audit(cli.parse_args().resume)
