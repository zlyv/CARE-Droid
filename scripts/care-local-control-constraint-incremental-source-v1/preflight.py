#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,sys
from common import *

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    sys.dont_write_bytecode=True
    import sklearn,scipy,androguard,yaml
    required=[FORMAL,DIRECT,STORE,REPLAY,P31]+[PARENT/n for n in ('final_decision.json','coverage.json','baseline_integrity_check.json','protocol_identity.json','selection_label_free.csv','protocol.json')]
    for p in required:
        if not p.is_file():raise RuntimeError('BLOCKED: missing '+str(p))
    decision=json.loads((PARENT/'final_decision.json').read_text());integrity=json.loads((PARENT/'baseline_integrity_check.json').read_text())
    if decision['status']!='LCC_CONTROL_OBSERVABILITY_SUPPORTED' or integrity['status']!='PASS' or not decision['parent_event_inventory_equal']:raise RuntimeError('v2 gate/integrity/inventory did not pass')
    identity=json.loads((PARENT/'protocol_identity.json').read_text());parent_token=identity['scan_identity_sha256']
    for name in ('control_core.py','dex_adapter.py','scan_one.py'):
        if sha256(HERE/'engine'/name)!=identity['engine_source_sha256'][name]:raise RuntimeError('BLOCKED: extractor differs from passed v2: '+name)
    replay=json.loads(REPLAY.read_text())
    replay_error=float(replay.get('max_abs_probability_difference',1))
    if replay.get('status')!='PASS' or not np.isfinite(replay_error) or replay_error>1e-4 or not replay.get('checks',{}).get('nested_calls==5') or not replay.get('checks',{}).get('fold_calls==5'):raise RuntimeError('frozen C43 replay/nested upstream audit did not pass')
    ids,y,folds,base,route,expert,final=load_reference();data=load_store(ids,y,folds);m=metrics(y,base,final,route)
    if (m['fixed'],m['broken'],m['net'])!=(49,17,32) or route.sum()!=200 or ((base>=.5)!=y).sum()!=97:raise RuntimeError('frozen Direct-C43 reference changed')
    if [int((folds==f).sum()) for f in range(1,6)]!=[160,160,160,160,159]:raise RuntimeError('formal fold membership changed')
    if [int((route&(folds==f)).sum()) for f in range(1,6)]!=[40]*5:raise RuntimeError('formal route per-fold membership changed')
    for f in range(1,6):
        if np.max(np.abs(data[f'f{f}_test_base']-base[folds==f]))>1e-12:raise RuntimeError('stored test Stage-I probability mismatch')
    parent_ids={r['sha256'] for r in read_csv(PARENT/'selection_label_free.csv')}
    if parent_ids!=set(ids[route]):raise RuntimeError('v2 routed cohort differs')
    # Import the same unsupervised mapping/Host implementation; never edit it.
    backend=module('_lcip_engine_common',HERE/'engine/common.py');assets=backend.discover()
    endpoints=backend.endpoint_rows(assets['endpoint'],set(ids));apks,path_audit=backend.apk_paths(set(ids))
    hostrel_path=PROJECT/'scripts/care-di25-compact43-relationquery-source-v1/hostrel_features.py'
    hostrel=module('_lcip_formal_hostrel',hostrel_path)
    mapping=yaml.safe_load(assets['mapping'].read_text(encoding='utf-8'))
    if len(mapping['roles'])!=31:raise RuntimeError('Role vocabulary differs')
    with np.load(P31,allow_pickle=False) as a:
        pids=[norm(v) for v in a['sha256']];values=np.asarray(a['role_prevalence'],float)
    if len(set(pids))!=799 or set(pids)!=set(ids) or values.shape!=(799,31) or not np.isfinite(values).all() or (values<0).any():raise RuntimeError('P31 shape or membership contract')
    index={sid:i for i,sid in enumerate(pids)};aligned_p31=values[[index[s] for s in ids]]
    p=protocol();old=OUTPUT/'protocol.json'
    if old.exists() and json.loads(old.read_text())!=p:raise RuntimeError('existing output belongs to another protocol')
    engine={f.name:sha256(f) for f in sorted((HERE/'engine').glob('*.py'))}
    experiment={f.name:sha256(f) for f in sorted(HERE.glob('*.py'))}
    token=hashlib.sha256(json.dumps(dict(protocol_sha256=sha256(HERE/'protocol.json'),engine=engine,parent_token=parent_token),sort_keys=True).encode()).hexdigest()
    protected=[(r,'frozen reference '+r.name) for r in required]+[(assets['endpoint'],'formal source endpoint cache'),(assets['mapping'],'frozen Role31 mapping'),(hostrel_path,'formal Host reconstruction')]
    protected += [(apks[sid],'original APK '+sid) for sid in sorted(ids)]
    for sid in sorted(parent_ids):
        scan=PARENT/'cache/scans'/parent_token[:16]/f'{sid}.json.gz'
        if not scan.is_file():raise RuntimeError('passed v2 scan missing '+sid)
        protected.append((scan,'read-only v2 scan '+sid))
    before=[]
    for i,(path,purpose) in enumerate(protected,1):
        rec=asset(path,purpose);before.append(rec)
        if purpose.startswith('original APK ') and rec['sha256']!=purpose.split()[-1]:raise RuntimeError('APK SHA mismatch '+str(path))
        if i%50==0:print(f'LCIP_ASSET_BEFORE {i}/{len(protected)}',flush=True)
    bp=OUTPUT/'baseline_asset_manifest_before.json'
    if bp.exists() and json.loads(bp.read_text())!=before:raise RuntimeError('protected assets changed since first run')
    dump(bp,before);dump(old,p)
    rows=[];parent_paths={}
    for i,sid in enumerate(ids):
        frozen=backend.frozen_view(endpoints[sid])
        expected={str(int(r['host_id'])) for r in hostrel.reconstruct_host_relations(endpoints[sid]['payload'],np.arange(31))}
        if set(frozen['host_ids'])!=expected:raise RuntimeError('frozen Host set differs '+sid)
        dump(OUTPUT/'cache/frozen'/f'{sid}.json',frozen)
        rows.append(dict(sha256=sid,fold=int(folds[i]),routed=int(route[i])))
        if sid in parent_ids:
            record_path=PARENT/'cache/scans'/parent_token[:16]/f'{sid}.json.gz';record=load_gz(record_path)
            if record.get('schema')!='lcc-fa-v2.0' or record.get('scan_identity_sha256')!=parent_token or record.get('sha256')!=sid or record.get('status')!='OK':raise RuntimeError('v2 cache identity mismatch '+sid)
            if record['frozen_pairs_total']!=len(frozen['pairs']) or record['frozen_hosts_total']!=len(frozen['host_ids']):raise RuntimeError('v2 frozen graph count mismatch '+sid)
            parent_paths[sid]=str(record_path)
    write_csv(OUTPUT/'selection_label_free.csv',rows);dump(OUTPUT/'apk_paths.json',{s:str(apks[s]) for s in ids});dump(OUTPUT/'parent_scan_paths.json',parent_paths)
    dump(OUTPUT/'engine_identity.json',dict(scan_identity_sha256=token,engine_source_sha256=engine,experiment_source_sha256=experiment,parent_scan_identity=parent_token,mapping=str(assets['mapping'])))
    np.savez_compressed(safe(OUTPUT/'p31_aligned.npz'),ids=ids,p31=aligned_p31)
    dump(OUTPUT/'environment_inventory.json',dict(python=sys.version,python_executable=sys.executable,numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,
        androguard=getattr(androguard,'__version__','unknown'),source_rows=799,target_route=200,parent_scan_reuse=200,additional_label_free_scans=599,
        gpu_required=False,future_rows_loaded=0,path_sources=path_audit,hidden_dimensions={str(f):int(data[f'f{f}_train_hidden'].shape[1]) for f in range(1,6)},
        auxiliary_hyperparameter_search=False,upstream_train_hidden_in_sample=True,upstream_inner_selection_not_replayed=True))
    dump(OUTPUT/'preflight.json',dict(status='PASS',reference=m,source_rows=799,routed=200,all_source_features_required=True,
        fixed_auxiliary_parameters=True,outer_test_labels_used_for_training=False,formal_fold_coordinates_kept_separate=True,future_rows_loaded=0))
    print('LCIP_PREFLIGHT_PASS source=799 route=200 reuse=200 additional=599 fixed_readout=true',flush=True)
if __name__=='__main__':main()
