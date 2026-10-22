#!/usr/bin/env python3
from __future__ import annotations
import concurrent.futures,csv,json,os,subprocess,sys,time
from collections import Counter
from pathlib import Path
from common import *
HERE=Path(__file__).resolve().parent
def run_one(row,paths,mapping,token):
    sid=row['sha256'];out=OUTPUT/'cache/scans'/token[:16]/f'{sid}.json.gz';stat=OUTPUT/'checkpoints'/token[:16]/f'{sid}.json'
    if out.is_file() and stat.is_file():
        prior=json.loads(stat.read_text(encoding='utf-8'))
        if prior.get('status')=='OK' and prior.get('scan_identity_sha256')==token:
            record=load_gz(out)
            if record.get('schema')==SCHEMA and record.get('scan_identity_sha256')==token:return prior
    cmd=[sys.executable,str(HERE/'scan_one.py'),'--apk',paths[sid],'--sha',sid,'--mapping',str(mapping),
         '--frozen',str(OUTPUT/'cache/frozen'/f'{sid}.json'),'--output',str(out),'--status',str(stat),'--scan-identity',token]
    start=time.monotonic();timeout=int(os.environ.get('LCC_APK_TIMEOUT_SECONDS','1200'))
    try:
        log=safe(OUTPUT/'logs/workers'/f'{sid}.log')
        with log.open('a',encoding='utf-8') as f:process=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        if stat.exists():return json.loads(stat.read_text(encoding='utf-8'))
        status=dict(sha256=sid,status='ERROR',error='worker exit '+str(process.returncode),wall_seconds=time.monotonic()-start,scan_identity_sha256=token)
    except subprocess.TimeoutExpired:status=dict(sha256=sid,status='TIMEOUT',wall_seconds=time.monotonic()-start,scan_identity_sha256=token)
    except Exception as exc:status=dict(sha256=sid,status='ERROR',error=repr(exc),wall_seconds=time.monotonic()-start,scan_identity_sha256=token)
    dump(stat,status);return status
def main():
    selected=list(csv.DictReader((OUTPUT/'selection_label_free.csv').open(encoding='utf-8')));paths=json.loads((OUTPUT/'apk_paths.json').read_text())
    assets=discover();token=json.loads((OUTPUT/'protocol_identity.json').read_text())['scan_identity_sha256'];workers=int(os.environ.get('LCC_WORKERS','4'))
    start=time.monotonic();statuses=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(run_one,r,paths,assets['mapping'],token):r for r in selected}
        for job in concurrent.futures.as_completed(jobs):
            s=job.result();statuses.append(s)
            print(f"LCC_SCAN {len(statuses)}/200 status={s['status']}",flush=True)
    write_csv(OUTPUT/'scan_status.csv',statuses);dump(OUTPUT/'run_timing.json',dict(current_invocation_wall_seconds=time.monotonic()-start,
      all_scan_wall_seconds_sum=sum(float(s.get('wall_seconds',0)) for s in statuses),workers=workers,resume_supported=True))
    print('LCC_SCAN_ALL_DONE',flush=True)
if __name__=='__main__':main()
