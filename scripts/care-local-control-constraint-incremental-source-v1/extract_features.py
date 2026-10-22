#!/usr/bin/env python3
from __future__ import annotations
import concurrent.futures,subprocess,time
from common import *
from feature_core import extract

def one(row,apks,parent,identity):
    sid=row['sha256'];token=identity['scan_identity_sha256'];start=time.monotonic()
    if sid in parent:return dict(sha256=sid,status='OK',path=parent[sid],source='PASSED_V2_READ_ONLY',wall_seconds=0)
    out=OUTPUT/'cache/scans'/token[:16]/f'{sid}.json.gz';stat=OUTPUT/'checkpoints'/token[:16]/f'{sid}.json'
    if out.exists() and stat.exists():
        prior=json.loads(stat.read_text())
        if prior.get('status')=='OK' and prior.get('scan_identity_sha256')==token:
            r=load_gz(out)
            if r.get('schema')=='lcc-fa-v2.0' and r.get('scan_identity_sha256')==token and r.get('sha256')==sid and r.get('status')=='OK':return dict(sha256=sid,status='OK',path=str(out),source='NEW_SOURCE_RESUME',wall_seconds=prior['wall_seconds'])
    command=[sys.executable,'-B',str(HERE/'engine/scan_one.py'),'--apk',apks[sid],'--sha',sid,'--frozen',str(OUTPUT/'cache/frozen'/f'{sid}.json'),
        '--mapping',identity['mapping'],'--output',str(out),'--status',str(stat),'--scan-identity',token]
    try:
        log=safe(OUTPUT/'logs/workers'/f'{sid}.log')
        with log.open('a',encoding='utf-8') as f:
            result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=int(os.environ.get('LCIP_APK_TIMEOUT_SECONDS','1200')))
        state=json.loads(stat.read_text()) if stat.exists() else dict(status='ERROR',error='worker did not write status')
        if result.returncode!=0:state['status']='ERROR'
        return dict(sha256=sid,status=state['status'],path=str(out),source='NEW_SOURCE_SCAN',wall_seconds=state.get('wall_seconds',time.monotonic()-start),error=state.get('error',''))
    except Exception as exc:return dict(sha256=sid,status='TIMEOUT' if isinstance(exc,subprocess.TimeoutExpired) else 'ERROR',path=str(out),source='NEW_SOURCE_SCAN',wall_seconds=time.monotonic()-start,error=repr(exc))

def main():
    rows=read_csv(OUTPUT/'selection_label_free.csv');apks=json.loads((OUTPUT/'apk_paths.json').read_text());parent=json.loads((OUTPUT/'parent_scan_paths.json').read_text());identity=json.loads((OUTPUT/'engine_identity.json').read_text())
    workers=int(os.environ.get('LCIP_WORKERS','4'));start=time.monotonic();statuses=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(one,r,apks,parent,identity) for r in rows]
        for job in concurrent.futures.as_completed(jobs):
            statuses.append(job.result());n=len(statuses)
            if n%20==0 or n==799:print(f'LCIP_SOURCE_FEATURES {n}/799 failed={sum(s["status"]!="OK" for s in statuses)}',flush=True)
    write_csv(OUTPUT/'feature_scan_status.csv',statuses)
    dump(OUTPUT/'feature_timing.json',dict(actual_invocation_wall_seconds=time.monotonic()-start,new_scan_serial_equivalent_seconds=sum(s['wall_seconds'] for s in statuses if s['source']!='PASSED_V2_READ_ONLY'),workers=workers,reused_parent_scans=len(parent)))
    failures=[s for s in statuses if s['status']!='OK']
    if failures:raise RuntimeError(f'BLOCKED: source feature failures={len(failures)}/799; no APK exclusions allowed')
    status_by={s['sha256']:s for s in statuses};evidence=[];coverage=[];audits=[]
    for row in rows:
        sid=row['sha256'];record=load_gz(status_by[sid]['path']);frozen=json.loads((OUTPUT/'cache/frozen'/f'{sid}.json').read_text())
        if record.get('sha256')!=sid or record.get('status')!='OK' or record.get('schema')!='lcc-fa-v2.0':raise RuntimeError('scan SHA/schema/status mismatch '+sid)
        if record['frozen_pairs_total']!=len(frozen['pairs']) or record['frozen_hosts_total']!=len(frozen['host_ids']):raise RuntimeError('scan/frozen scope differs '+sid)
        e,c,a=extract(record,frozen);evidence.append(e);coverage.append(c);audits.append(dict(sha256=sid,fold=int(row['fold']),routed=int(row['routed']),source=status_by[sid]['source'],**a))
    routed=[r for r in audits if r['routed']]
    parent_cov=json.loads((PARENT/'coverage.json').read_text())
    if sum(r['verified_events'] for r in routed)!=parent_cov['verified_control_events'] or sum(r['aligned_events'] for r in routed)!=parent_cov['aligned_events']:raise RuntimeError('reused routed facts differ from passed v2')
    np.savez_compressed(safe(OUTPUT/'control_features.npz'),ids=np.asarray([r['sha256'] for r in rows]),evidence=np.asarray(evidence),coverage=np.asarray(coverage))
    write_csv(OUTPUT/'feature_audit.csv',audits)
    dump(OUTPUT/'feature_validation.json',dict(status='PASS',source_rows=799,routed=200,parent_verified_events=parent_cov['verified_control_events'],
        evidence_width=124,coverage_width=64,feature_extraction_labels_used=False,zero_frozen_host_apks=sum(r['frozen_edges']==0 for r in audits),future_rows_loaded=0))
    print('LCIP_FEATURES_PASS source=799 evidence_width=124 coverage_width=64',flush=True)
if __name__=='__main__':main()
