#!/usr/bin/env python3
"""LOCAL synthetic pipeline test, not a research experiment or accuracy result.

Requires the local previous fixture helper. Generated files stay in a new work
directory; no existing fixture, APK or formal result is changed or deleted.
"""
from __future__ import annotations
import importlib.util,json,os,shutil,subprocess,sys,uuid,hashlib
from pathlib import Path
import numpy as np
from common import protocol
from model_core import fit_predict

HERE=Path(__file__).resolve().parent

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def canonical_prior_fit(x,y,t,config):
    # Independently reproduce the OLD implementation; do not generate the
    # reference with the new helper being tested (which hid the v1 dtype bug).
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    model=make_pipeline(StandardScaler(),LogisticRegression(**{k:v for k,v in config.items() if k!='name'}))
    return model.fit(x,y).predict_proba(t)[:,1]

def main():
    old_test=load('_hbi_old_binding_tests',HERE/'binding_selftest.py')
    old_selftest=sys.modules.get('selftest');sys.modules['selftest']=old_test
    helper=load('_hbi_fixture_helper',HERE.parent/'care-host-control-binding-observability-v1/integration_selftest.py')
    if old_selftest is None:sys.modules.pop('selftest',None)
    else:sys.modules['selftest']=old_selftest
    holder=(HERE.parent/'work'/('hi'+uuid.uuid4().hex[:8])).resolve();holder.mkdir()
    root=holder/'real';root.mkdir()
    parent,endpoint,compact,manifest=helper.make_fixture(root,'heterogeneous')
    def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
    copy(HERE/'binding_core.py',root/'scripts/care-host-control-binding-observability-v1/binding_core.py')
    copy(HERE/'reference_core.py',root/'scripts/care-local-control-constraint-incremental-source-v1/common.py')
    helper.write_json(parent/'protocol.json',load('_hbi_prior_common',HERE.parent/'care-local-control-constraint-incremental-source-v1/common.py').protocol())
    import csv
    with (parent/'selection_label_free.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    ids=np.asarray([r['sha256'] for r in rows]);folds=np.asarray([int(r['fold']) for r in rows]);route=np.asarray([bool(int(r['routed'])) for r in rows])
    y=np.arange(799)%2;base=np.where(y,.8,.2);expert=np.where(y,.8,.2)
    for f,nerr,nfix,nbreak,noutside in zip(range(1,6),[17,17,17,17,16],[10,10,10,10,9],[4,4,3,3,3],[3,3,3,2,2]):
        selected=np.flatnonzero((folds==f)&route);outside=np.flatnonzero((folds==f)&~route)
        wrong=np.r_[selected[:nerr],outside[:noutside]];base[wrong]=1-base[wrong]
        expert[selected]=base[selected]
        expert[selected[:nfix]]=1-base[selected[:nfix]]
        expert[selected[nerr:nerr+nbreak]]=1-base[selected[nerr:nerr+nbreak]]
    formal=[dict(arm='SVM',sha256=s,fold=int(folds[i]),routed=int(route[i]),label=int(y[i]),stage1_prob=float(base[i])) for i,s in enumerate(ids)]
    helper.write_csv(root/'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv',formal)
    field='P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2__expert_prob'
    helper.write_csv(root/'outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv',
        [dict(sha256=s,**{field:float(expert[i])}) for i,s in enumerate(ids)])
    with np.load(parent/'control_features.npz',allow_pickle=False) as z:e=z['evidence'];cov=z['coverage']
    final=base.copy();final[route]=expert[route]
    store=dict(ids=ids,folds=folds);pred_a=base.copy();pred_h=base.copy();rng=np.random.default_rng(42)
    # A known separable synthetic signal makes the negative information gate test
    # deterministic. It is not evidence about actual C43 performance.
    raw=np.column_stack([2*y-1+rng.normal(0,.01,799),rng.normal(0,.01,799)]).astype(np.float32)
    for f in range(1,6):
        ti=np.flatnonzero(folds!=f);vi=np.flatnonzero(folds==f);h=raw.copy();h[:,0]*=(-1 if f%2 else 1)
        for split,ix in (('train',ti),('test',vi)):
            store[f'f{f}_{split}_index']=ix;store[f'f{f}_{split}_hidden']=h[ix]
            store[f'f{f}_{split}_base']=base[ix];store[f'f{f}_{split}_host_count']=np.full(len(ix),2)
        store[f'f{f}_train_y']=y[ti]
        a=canonical_prior_fit(np.column_stack([h[ti],cov[ti],e[ti]]),y[ti],np.column_stack([h[vi],cov[vi],e[vi]]),protocol()['auxiliary_model'])
        b=canonical_prior_fit(h[ti],y[ti],h[vi],protocol()['auxiliary_model']);pick=route[vi]
        pred_a[vi[pick]]=a[pick];pred_h[vi[pick]]=b[pick]
    sp=root/'outputs/care-svm-c43-representation-collision-audit-v1/representation_store.npz';sp.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(sp,**store)
    helper.write_json(sp.with_name('replay_validation.json'),dict(status='PASS',max_abs_probability_difference=0,
        checks={'nested_calls==5':True,'fold_calls==5':True},synthetic_only=True))
    np.savez_compressed(parent/'probe_predictions.npz',ids=ids,M0_DIRECT_C43=final,M2_HIDDEN_CONTROL=pred_a,M1_HIDDEN_READOUT=pred_h)
    env=os.environ.copy();env.update(HCB_PROJECT=str(root),HCB_PARENT=str(parent),HCB_ENDPOINT=str(endpoint),
        HCB_C43_CACHE=str(compact),HCB_ROLE_MANIFEST=str(manifest),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    obs_script=HERE.parent/'care-host-control-binding-observability-v1/run_audit.py'
    obsrun=subprocess.run([sys.executable,'-B',str(obs_script)],env=env,capture_output=True,text=True)
    if obsrun.returncode:print(obsrun.stdout);print(obsrun.stderr);raise RuntimeError('synthetic prerequisite failed')
    obs=root/'outputs/care-host-control-binding-observability-v1'
    env.update(HBI_PROJECT=str(root),HBI_OBSERVABILITY=str(obs))
    log=holder/'pipeline.log'
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-B',str(HERE/'pipeline.py')],env=env,stdout=stream,stderr=subprocess.STDOUT)
    out=root/'outputs/care-host-control-binding-information-source-v2';d=json.loads((out/'final_decision.json').read_text())
    if result.returncode or d['status']!='HOST_BINDING_INFORMATION_NOT_SUPPORTED' or d['formal_integrity']!='PASS':
        print(log.read_text(encoding='utf-8'));raise RuntimeError('synthetic information test differed '+repr(d))
    replay=json.loads((out/'aggregate_replay_validation.json').read_text())
    if replay['status']!='PASS':raise RuntimeError('synthetic comparator replay did not pass')
    if any(r['hidden_train_dtype']!='float32' or r['aggregate_train_dtype']!='float64' for r in replay['fold_details']):raise RuntimeError('float32/float64 replay path not exercised')
    with np.load(out/'probe_predictions.npz',allow_pickle=False) as z:
        if len(z.files)!=45:raise RuntimeError('arm count mismatch')
        for a in z.files:
            if a!='ids' and not np.array_equal(z[a][~route],base[~route]):raise RuntimeError('unrouted changed')
    digest=hashlib.sha256((out/'final_decision.json').read_bytes()).hexdigest()
    repeat=subprocess.run([sys.executable,'-B',str(HERE/'pipeline.py')],env=env,capture_output=True,text=True)
    if repeat.returncode==0 or digest!=hashlib.sha256((out/'final_decision.json').read_bytes()).hexdigest():raise RuntimeError('existing output not protected')
    missing=holder/'missing';missing.mkdir()
    copy(HERE/'binding_core.py',missing/'scripts/care-host-control-binding-observability-v1/binding_core.py')
    copy(HERE/'reference_core.py',missing/'scripts/care-local-control-constraint-incremental-source-v1/common.py')
    env['HBI_PROJECT']=str(missing)
    blocked=subprocess.run([sys.executable,'-B',str(HERE/'pipeline.py')],env=env,capture_output=True,text=True)
    md=json.loads((missing/'outputs/care-host-control-binding-information-source-v2/final_decision.json').read_text())
    if blocked.returncode==0 or md['status']!='HOST_BINDING_INFORMATION_BLOCKED' or md['stopped_at']!='preflight':raise RuntimeError('missing input not blocked')
    print(json.dumps(dict(status='HBI_INTEGRATION_SELFTEST_PASS',synthetic_only=True,source_rows=799,routed=200,
        comparator_replay='PASS',float32_hidden_and_float64_aggregate_checked=True,reference_generated_independently=True,
        all_44_prediction_arms_checked=True,unrouted_preserved=True,integrity='PASS',
        negative_information_gate='PASS',missing_input_blocks=True,existing_output_refused=True,fixture_directory=str(holder)),indent=2))
if __name__=='__main__':main()
