from __future__ import annotations
import ast,os,subprocess,sys,time
from pilot_common import *

def main():
    if OUTPUT.exists(): raise RuntimeError('experiment output exists; use a new version, never overwrite')
    safe(OUTPUT).mkdir(exist_ok=False); timings={}; error=None; phase='syntax'
    print('CARE_DIRECT_C43_ROUTE_CONDITIONED_START output='+str(OUTPUT),flush=True)
    def run(name,script,*extra):
        nonlocal phase
        phase=name; t=time.monotonic(); print('RCT_PHASE_START name='+name,flush=True)
        log=safe(OUTPUT/'logs'/(name+'.log'))
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RCT_TEST_ROOT=str(OUTPUT/'selftest-fixtures'))
        with log.open('w',encoding='utf-8') as h:
            worker=subprocess.Popen([sys.executable,'-B',str(HERE/script),*extra],env=env,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace')
            for line in worker.stdout: h.write(line); h.flush(); print(line,end='',flush=True)
            code=worker.wait()
        timings[name]=time.monotonic()-t
        if code: raise RuntimeError(f'{name} failed rc={code}; log={log}')
        print(f'RCT_PHASE_DONE name={name} seconds={timings[name]:.2f}',flush=True)
    try:
        for f in HERE.glob('*.py'): ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
        phase='shell_syntax'
        for f in HERE.glob('*.sh'): subprocess.run(['bash','-n',str(f)],check=True)
        print('RCT_SHELL_SYNTAX_PASS',flush=True)
        run('selftest','selftest.py'); run('preflight','preflight.py'); run('prepare_routes','prepare_routes.py')
        for arm in ARMS: run(arm,'run_arm.py','--arm',arm)
    except BaseException as e: error=dict(phase=phase,reason=repr(e))
    finally:
        if (OUTPUT/'baseline_asset_manifest_before.json').is_file():
            t=time.monotonic()
            try: verify_integrity()
            except Exception as e: error=dict(phase='integrity',reason=repr(e),original_error=error)
            timings['integrity']=time.monotonic()-t
    if error is None:
        try: run('summary','summarize.py')
        except BaseException as e: error=dict(phase=phase,reason=repr(e))
    dump(OUTPUT/'phase_timing.json',timings)
    safe(OUTPUT/'reproduce_commands.md').write_text(f'```bash\ncd {PROJECT}\nRCT_GPU=1 bash scripts/{NAME}/launch_tmux.sh\nbash scripts/{NAME}/collect_results.sh\n```\nFresh experiment directory only. No automatic resume, future run, tuning or sample exclusion.\n',encoding='utf-8')
    if error:
        intact=read_json(OUTPUT/'baseline_integrity_check.json')['status'] if (OUTPUT/'baseline_integrity_check.json').is_file() else 'NOT_YET_AUDITED'
        d=dict(status='ROUTE_CONDITIONED_TRAINING_BLOCKED',stopped_at=error['phase'],reason=error['reason'],formal_integrity=intact,
            future_rows_loaded=0,automatic_future_run=False,tuning_authorized=False,next_action='inspect explicit implementation or input failure; retain partial logs; repair in a new version')
        dump(OUTPUT/'final_decision.json',d)
        safe(OUTPUT/'REPORT.md').write_text('# Route-conditioned training blocked\n\n'+json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('CARE_DIRECT_C43_ROUTE_CONDITIONED_FAILED '+json.dumps(error),flush=True); raise SystemExit(1)
    print('CARE_DIRECT_C43_ROUTE_CONDITIONED_DONE',flush=True)
    print('decision='+str(OUTPUT/'final_decision.json'),flush=True)
if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
