#!/usr/bin/env python3
"""LiteStage-V1 resumable phase runner. Reads source/future APKs only through extract.py; writes new output logs/state; trains only source 2017 RF."""
from __future__ import annotations
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from common import OUT,load_shards

ROOT=Path('/home/boyuanjie/projects/tasa-droid');HERE=ROOT/'scripts/care-lite-stage-v1';PY='/home/boyuanjie/envs/tif/bin/python'
PHASES=['audit','extract_source','train','structural','extract_future','future','cost','summarize']
def save(path,obj):
    t=path.with_name(path.name+'.new');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,path)
def command(name):
    if name=='extract_source':return [PY,'-B',str(HERE/'extract.py'),'--cohort','source','--level','L4']
    if name=='extract_future':
        selection=json.loads((OUT/'source_selection.json').read_text());arm=selection['global_final_arm'];level=arm.split('_')[0]
        if level not in ('L1','L2','L3','L4'):raise RuntimeError('selected arm not an extraction level')
        return [PY,'-B',str(HERE/'extract.py'),'--cohort','future','--level',level]
    return [PY,'-B',str(HERE/({'train':'train.py','future':'future.py','structural':'structural.py','cost':'cost.py','summarize':'summarize.py','audit':'audit.py'}[name]))]
def main():
    resume='--resume' in sys.argv
    OUT.mkdir(parents=True,exist_ok=True)
    status=OUT/'phase_status.json';runstate=OUT/'run_state.json'
    if (OUT/'final_decision.json').exists():raise RuntimeError('existing final experiment; refuse overwrite')
    if status.exists():
        if not resume:raise RuntimeError('experiment already started; use --resume')
        st=json.loads(status.read_text(encoding='utf-8'))
    else:
        if resume:raise RuntimeError('resume without status')
        st={}
    blocked=False
    for phase in PHASES:
        if phase in ('structural','extract_future','future') and not (OUT/'source_selection.json').exists():
            st[phase]=dict(status='SKIPPED_DEPENDENCY',reason='2017 source model unavailable');save(status,st);print('LITESTAGE_PHASE_SKIPPED',phase,flush=True);blocked=True;continue
        complete=st.get(phase,{}).get('status')=='DONE'
        if phase=='extract_source' and complete and resume:
            _,missing=load_shards('source')
            if missing:complete=False
        if complete:print('LITESTAGE_PHASE_RESUME',phase,flush=True);continue
        start=time.time();st[phase]=dict(status='RUNNING',start_epoch=start);save(status,st);save(runstate,dict(status='RUNNING',phase=phase,epoch=start))
        logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True);log=logs/(phase+'.log')
        if log.exists():
            if not resume:raise RuntimeError('phase log already exists '+str(log))
            attempt=1
            while (logs/f'{phase}.retry{attempt}.log').exists():attempt+=1
            log=logs/f'{phase}.retry{attempt}.log'
        print('LITESTAGE_PHASE_START',phase,'log',log,flush=True)
        with log.open('x',encoding='utf-8') as f:
            p=subprocess.run(command(phase),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        if p.returncode:
            tail=log.read_text(encoding='utf-8')[-5000:]
            st[phase]=dict(status='FAILED',rc=p.returncode,seconds=time.time()-start,log=str(log));save(status,st);save(runstate,dict(status='FAILED',phase=phase,detail=tail))
            print('LITESTAGE_PHASE_FAIL',phase,tail,flush=True)
            if phase=='train' and 'source shards missing' in tail:
                blocked=True;continue
            (OUT/'exit_code.txt').write_text(str(p.returncode),encoding='utf-8');(OUT/'failed.flag').write_text(phase,encoding='utf-8')
            raise RuntimeError(f'phase {phase} failed; see {log}')
        st[phase]=dict(status='DONE',seconds=time.time()-start,log=str(log));save(status,st)
        print('LITESTAGE_PHASE_DONE',phase,'seconds',round(time.time()-start,2),flush=True)
    future_ok=(OUT/'future_frozen_metrics.csv').exists()
    if blocked or not future_ok:
        save(runstate,dict(status='BLOCKED_PARTIAL',phase='summarize',future_complete=future_ok));(OUT/'exit_code.txt').write_text('2',encoding='utf-8');(OUT/'failed.flag').write_text('blocked_partial',encoding='utf-8');print('CARE_LITESTAGE_V1_PARTIAL',flush=True)
    else:
        save(runstate,dict(status='COMPLETE',phase='summarize',epoch=time.time()));(OUT/'exit_code.txt').write_text('0',encoding='utf-8');(OUT/'completed.flag').write_text('complete',encoding='utf-8');print('CARE_LITESTAGE_V1_DONE',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
