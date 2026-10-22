#!/usr/bin/env python3
"""Generate SYNTHETIC fixtures in a new temp directory and exercise the full pipeline.

No APK, real experiment output, label or network resource is used. Fixture files
remain for diagnosis; they are not included in the deployment zip.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, os, shutil, subprocess, sys, uuid
from pathlib import Path
import numpy as np
from binding_core import proof_tables, aggregates
from selftest import event

HERE=Path(__file__).resolve().parent

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def make_fixture(root, mode):
    parent=root/'outputs/care-local-control-constraint-incremental-source-v1'
    scripts=root/'scripts';source=HERE.parent/'care-local-control-constraint-incremental-source-v1'
    hsource=HERE.parent/'role_api_deps/scripts/care-di25-compact43-relationquery-source-v1/hostrel_features.py'
    files=[(source/'feature_core.py',scripts/'care-local-control-constraint-incremental-source-v1/feature_core.py'),
           (source/'extract_features.py',scripts/'care-local-control-constraint-incremental-source-v1/extract_features.py'),
           (hsource,scripts/'care-di25-compact43-relationquery-source-v1/hostrel_features.py')]
    for a,b in files:
        b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
    write_json(parent/'feature_validation.json',dict(status='PASS',source_rows=799,feature_extraction_labels_used=False))
    write_json(parent/'baseline_integrity_check.json',dict(status='PASS'))
    write_json(parent/'protocol.json',dict(synthetic_test_only=True))
    identity=dict(parent_scan_identity='synthetic-v2',scan_identity_sha256='synthetic-v2',
        experiment_source_sha256={a.name:hashlib.sha256(a.read_bytes()).hexdigest() for a,b in files[:2]})
    write_json(parent/'engine_identity.json',identity)
    assets=root/'artifacts/care-source-local-assets-v2'
    write_json(assets/'local_asset_manifest.json',dict(payload_to_model_role=list(range(31))))
    endpoint=assets/'endpoint_centered_tensor_cache_2017.jsonl.gz'
    base=np.zeros((3,15));base[2,4]=1
    payload=dict(node_ids=[1,2,9],node_role_ids=[[],[],[0]],node_base_features=base.tolist(),edge_index=[[0,1],[2,2]])
    frozen=dict(pairs={'1:9':[0],'2:9':[0]},host_ids=['1','2'])
    rows=[];official=[];status=[];ids=[];evidence=[];coverage=[];values=[]
    with gzip.open(endpoint,'wt',encoding='utf-8') as ef:
        index=0
        for fold,size in enumerate([160,160,160,160,159],1):
            for j in range(size):
                sid=hashlib.sha256(f'synthetic-{fold}-{j}'.encode()).hexdigest();ids.append(sid)
                r=dict(sha256=sid,fold=fold,routed=int(j<40));rows.append(r)
                official.append(dict(arm='SVM',**r,label='DO_NOT_READ_LABEL',stage1_prob='DO_NOT_READ_PROBABILITY'))
                write_json(parent/'cache/frozen'/(sid+'.json'),frozen)
                tier='VERIFIED_CONTROL' if mode=='homogeneous' else 'PARTIAL'
                record=dict(schema='lcc-fa-v2.0',scan_identity_sha256='synthetic-v2',status='OK',sha256=sid,
                    labels_used=False,future_rows_loaded=0,frozen_pairs_total=2,frozen_hosts_total=2,
                    events=[event(sid+'a'),event(sid+'b',host=2,tier=tier)])
                scan=parent/'cache/scans/synthetic'/(sid+'.json.gz');scan.parent.mkdir(parents=True,exist_ok=True)
                with gzip.open(scan,'wt',encoding='utf-8') as f:json.dump(record,f)
                status.append(dict(sha256=sid,status='OK',path=str(scan.resolve()),source='NEW_SOURCE_SCAN'))
                a,b=aggregates(proof_tables(record,frozen));evidence.append(a);coverage.append(b)
                ef.write(json.dumps(dict(sha256=sid,status='OK',payload=payload))+'\n')
                v=np.zeros((2,105));v[:,0]=1;v[:,43]=1;v[0,31]=1;values.append(v);index+=1
    write_csv(parent/'selection_label_free.csv',rows);write_csv(parent/'feature_scan_status.csv',status)
    write_csv(root/'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv',official)
    np.savez_compressed(parent/'control_features.npz',ids=np.asarray(ids),evidence=np.asarray(evidence),coverage=np.asarray(coverage))
    compact=root/'artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz'
    compact.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(compact,sha256=np.asarray(ids),host_values=np.concatenate(values),host_offsets=np.arange(800)*2,
        real_host_count=np.full(799,2),host_was_empty=np.zeros(799,int))
    if mode=='missing':
        # Preserve membership: corrupt the status path, never exclude its APK.
        status[0]['path']=str(parent/'cache/scans/never_created.json.gz')
        write_csv(parent/'feature_scan_status.csv',status)
    return parent, endpoint, compact, assets/'local_asset_manifest.json'

def main():
    fixture_parent=HERE.parent/'work'
    fixture_parent.mkdir(exist_ok=True)
    holder=(fixture_parent/('hb'+uuid.uuid4().hex[:8])).resolve()
    holder.mkdir()
    results=[]
    for mode,expected in [('heterogeneous','HOST_CONTROL_BINDING_OBSERVABLE'),
                          ('homogeneous','HOST_CONTROL_BINDING_COVERAGE_INSUFFICIENT'),
                          ('missing','HOST_CONTROL_BINDING_BLOCKED')]:
        root=holder/mode;root.mkdir()
        parent,endpoint,compact,manifest=make_fixture(root,mode)
        env=os.environ.copy();env.update(HCB_PROJECT=str(root),HCB_PARENT=str(parent),HCB_ENDPOINT=str(endpoint),
            HCB_C43_CACHE=str(compact),HCB_ROLE_MANIFEST=str(manifest),PYTHONDONTWRITEBYTECODE='1')
        run=subprocess.run([sys.executable,'-B',str(HERE/'run_audit.py')],env=env,text=True,capture_output=True)
        out=root/'outputs/care-host-control-binding-observability-v1'
        decision=json.loads((out/'final_decision.json').read_text(encoding='utf-8'))
        if decision['status']!=expected or run.returncode!=(1 if mode=='missing' else 0):
            print(run.stdout);print(run.stderr);raise RuntimeError('integration result differs '+mode+repr(decision))
        if mode!='missing' and decision['formal_integrity']!='PASS':raise RuntimeError('fixture integrity failed')
        first_hash=hashlib.sha256((out/'final_decision.json').read_bytes()).hexdigest()
        retry=subprocess.run([sys.executable,'-B',str(HERE/'run_audit.py')],env=env,text=True,capture_output=True)
        if retry.returncode==0 or hashlib.sha256((out/'final_decision.json').read_bytes()).hexdigest()!=first_hash:
            raise RuntimeError('existing-output protection failed')
        results.append(dict(mode=mode,status=decision['status'],integrity=decision['formal_integrity']))
    print(json.dumps(dict(status='HCB_INTEGRATION_SELFTEST_PASS',fixtures='SYNTHETIC_NOT_RESEARCH_RESULTS',
        modes=results,existing_output_refused=True,fixture_directory=str(holder)),indent=2))

if __name__=='__main__':main()
