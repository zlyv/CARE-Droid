#!/usr/bin/env python3
"""LFC-V1 resumable tmux phase runner. Inputs: frozen old assets; outputs run/log/status files only under LFC_OUT. Runs new first-stage RF training; never parses APK or trains C43."""
from __future__ import annotations
import json,os,subprocess,sys,time,traceback
from core import OUT,ROOT

HERE=ROOT/'scripts/care-lfc-v1'
PY='/home/boyuanjie/envs/tif/bin/python'
PHASES=('audit','selftest','source','future','report')
def save(path,obj):
    t=path.with_name(path.name+'.next');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,path)
def main():
    resume='--resume' in sys.argv
    if not OUT.is_dir():raise RuntimeError('run output directory must pre-exist')
    status=OUT/'phase_status.json';state=OUT/'run_state.json'
    if (OUT/'completed.flag').exists():raise RuntimeError('already completed; no rerun')
    if status.exists():
        if not resume:raise RuntimeError('existing phase status; use --resume')
        phases=json.loads(status.read_text(encoding='utf-8'))
    else:
        if resume:raise RuntimeError('resume without phase status')
        phases={}
    for phase in PHASES:
        if phases.get(phase,{}).get('status')=='DONE':
            print('LFC_PHASE_RESUME',phase,flush=True);continue
        started=time.time();phases[phase]=dict(status='RUNNING',started=started);save(status,phases)
        save(state,dict(status='RUNNING',phase=phase,epoch=started))
        logs=OUT/'logs';logs.mkdir(exist_ok=True);log=logs/(phase+'.log');attempt=0
        while log.exists():
            attempt+=1;log=logs/f'{phase}.retry{attempt}.log'
        print('LFC_PHASE_START',phase,'log',log,flush=True)
        with log.open('x',encoding='utf-8') as f:
            p=subprocess.run([PY,'-B',str(HERE/(phase+'.py'))],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        if p.returncode:
            tail=log.read_text(encoding='utf-8')[-6000:]
            phases[phase]=dict(status='FAILED',rc=p.returncode,seconds=time.time()-started,log=str(log));save(status,phases)
            save(state,dict(status='FAILED',phase=phase,log=str(log),tail=tail));(OUT/'exit_code.txt').write_text(str(p.returncode),encoding='utf-8');(OUT/'failed.flag').write_text(phase,encoding='utf-8')
            print('LFC_PHASE_FAIL',phase,tail,flush=True)
            raise RuntimeError('phase failed '+phase)
        phases[phase]=dict(status='DONE',seconds=time.time()-started,log=str(log));save(status,phases)
        print('LFC_PHASE_DONE',phase,'seconds',round(time.time()-started,2),flush=True)
    if (OUT/'failed.flag').exists():
        old=OUT/'failed.flag';old.rename(OUT/'failed_previous_attempt.flag')
    save(state,dict(status='COMPLETE',phase='report',epoch=time.time()))
    (OUT/'exit_code.txt').write_text('0',encoding='utf-8');(OUT/'completed.flag').write_text('complete',encoding='utf-8')
    print('CARE_LFC_V1_DONE',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
