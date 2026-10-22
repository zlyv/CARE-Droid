#!/usr/bin/env python3
from __future__ import annotations
import ast,subprocess,time
from common import *

def main():
    if OUTPUT.exists():raise RuntimeError('existing experiment output: use a new version; never overwrite')
    safe(OUTPUT).mkdir(exist_ok=False);timing={};phase='syntax';error=None
    print('CARE_HOST_CONTROL_BINDING_INFORMATION_START output='+str(OUTPUT),flush=True)
    def run(name):
        nonlocal phase
        phase=name;t=time.monotonic();print('HBI_PHASE_START name='+name,flush=True)
        log=safe(OUTPUT/'logs'/(name+'.log'))
        with log.open('w',encoding='utf-8') as stream:
            worker=subprocess.Popen([sys.executable,'-B',str(HERE/(name+'.py'))],stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in worker.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
            code=worker.wait()
        if code:
            raise RuntimeError(f'{name} failed rc={code}; log={log}; tail='+log.read_text(encoding='utf-8')[-6000:])
        timing[name]=time.monotonic()-t;print(f'HBI_PHASE_DONE name={name} seconds={timing[name]:.3f}',flush=True)
    try:
        for f in HERE.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
        run('binding_selftest');run('selftest');run('preflight');run('prepare_features');run('run_probe')
    except Exception as exc:error=dict(phase=phase,error=repr(exc))
    finally:
        if (OUTPUT/'baseline_asset_manifest_before.json').is_file():
            t=time.monotonic()
            try:integrity()
            except Exception as exc:error=error or dict(phase='integrity',error=repr(exc))
            timing['integrity']=time.monotonic()-t
    if error is None:
        try:run('summarize')
        except Exception as exc:error=dict(phase=phase,error=repr(exc))
    dump(OUTPUT/'phase_timing.json',timing)
    safe(OUTPUT/'reproduce_commands.md').write_text(f'```bash\ncd {PROJECT}\nbash scripts/{NAME}/launch_tmux.sh\nbash scripts/{NAME}/collect_results.sh\n```\nFresh output required; no automatic future run or overwrite.\n',encoding='utf-8')
    if error:
        c=OUTPUT/'baseline_integrity_check.json';intact=read_json(c)['status'] if c.is_file() else 'UNAVAILABLE'
        d=dict(status='HOST_BINDING_INFORMATION_BLOCKED',stopped_at=error['phase'],reason=error['error'],formal_integrity=intact,
            auxiliary_fits_may_have_run=error['phase'] in ('run_probe','summarize'),future_rows_loaded=0,
            permit_specifying_separate_frozen_diagnostic=False,next_action='repair explicit path/implementation failure in a new version; no sample exclusions or parameter changes')
        dump(OUTPUT/'final_decision.json',d);safe(OUTPUT/'REPORT.md').write_text('# Host binding information test blocked\n\n'+json.dumps(d,indent=2)+'\n',encoding='utf-8')
        print('CARE_HOST_CONTROL_BINDING_INFORMATION_FAILED '+json.dumps(error),flush=True);raise SystemExit(1)
    print('CARE_HOST_CONTROL_BINDING_INFORMATION_DONE',flush=True)
    print('report='+str(OUTPUT/'REPORT.md'),flush=True)
if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
