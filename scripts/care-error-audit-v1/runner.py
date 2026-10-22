#!/usr/bin/env python3
"""CARE-EAD phase runner; reads frozen assets, writes state/logs only to outputs/care_error_audit_v1; no training except 2017 diagnostic probes; no APK reads."""
from __future__ import annotations
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from ead import OUT,assets,verify_assets,writejson

PHASES=['protocol','evidence','probe','representation','routing','future','finish']
def state(path,obj):
    tmp=path.with_name(path.name+'.new')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)
def main():
    resume='--resume' in sys.argv
    if not OUT.is_dir():OUT.mkdir(parents=True,exist_ok=False)
    manifest=OUT/'asset_manifest_before.json'
    if not manifest.exists():
        if resume:raise RuntimeError('resume requested but asset manifest missing')
        if any(p.name not in ('run.log','smoke.log') for p in OUT.iterdir()):raise RuntimeError('output occupied without manifest')
        writejson('asset_manifest_before.json',assets())
    elif not resume:raise RuntimeError('output exists: use --resume, never overwrite')
    verify_assets()
    statuspath=OUT/'phase_status.json';st=json.loads(statuspath.read_text()) if statuspath.exists() else {}
    runstate=OUT/'run_state.json'
    for ph in PHASES:
        if st.get(ph,{}).get('status')=='DONE':
            print('EAD_PHASE_RESUME',ph,flush=True);continue
        verify_assets();start=time.time();st[ph]=dict(status='RUNNING',started=start);state(statuspath,st);state(runstate,dict(status='RUNNING',phase=ph,epoch=start))
        print('EAD_PHASE_START',ph,flush=True)
        log=OUT/'logs'/f'{ph}.log';log.parent.mkdir(parents=True,exist_ok=True)
        if log.exists():
            if not resume or st.get(ph,{}).get('status')!='RUNNING' and st.get(ph,{}).get('status')!='FAILED':raise RuntimeError('existing unfinished phase log; inspect before resume '+str(log))
            attempt=1
            while (OUT/'logs'/f'{ph}.retry{attempt}.log').exists():attempt+=1
            log=OUT/'logs'/f'{ph}.retry{attempt}.log'
        with log.open('x',encoding='utf-8') as f:
            child=subprocess.run(['/home/boyuanjie/envs/tif/bin/python','-B',str(Path(__file__).with_name('ead.py')),'--phase',ph],cwd='/home/boyuanjie/projects/tasa-droid',stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        if child.returncode:
            detail=log.read_text(encoding='utf-8')[-4000:];st[ph]=dict(status='FAILED',exit_code=child.returncode,seconds=time.time()-start);state(statuspath,st);state(runstate,dict(status='FAILED',phase=ph,detail=detail));(OUT/'failed.flag').write_text(ph,encoding='utf-8');(OUT/'exit_code.txt').write_text(str(child.returncode),encoding='utf-8');raise RuntimeError(f'{ph} rc={child.returncode}: '+detail)
        st[ph]=dict(status='DONE',seconds=time.time()-start);state(statuspath,st);print('EAD_PHASE_DONE',ph,'seconds',round(time.time()-start,2),flush=True)
    verify_assets();state(runstate,dict(status='COMPLETE',phase='finish',epoch=time.time()));(OUT/'exit_code.txt').write_text('0',encoding='utf-8');(OUT/'completed.flag').write_text('complete',encoding='utf-8');
    failed=OUT/'failed.flag'
    if failed.exists():failed.rename(OUT/'failed_previous_attempt.flag')
    print('CARE_EAD_V1_DONE',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
