from __future__ import annotations
import ast,os,subprocess,sys,time
from diag_common import *

def main():
    if OUTPUT.exists():raise RuntimeError('diagnostic output exists; choose a new version')
    safe(OUTPUT).mkdir(exist_ok=False);timing={};error=None;phase='syntax'
    def run(name,script,*extra):
        nonlocal phase
        phase=name;t=time.monotonic();print('RCTD_PHASE_START name='+name,flush=True)
        log=safe(OUTPUT/'logs'/(name+'.log'));env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        with log.open('w',encoding='utf-8') as f:
            p=subprocess.Popen([sys.executable,'-B',str(HERE/script),*extra],cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in p.stdout:f.write(line);f.flush();print(line,end='',flush=True)
            rc=p.wait()
        timing[name]=time.monotonic()-t
        if rc:raise RuntimeError(f'{name} failed rc={rc}; log={log}')
        print(f'RCTD_PHASE_DONE name={name} seconds={timing[name]:.2f}',flush=True)
    print('CARE_ROUTE_CONDITIONED_DIAGNOSTICS_START',flush=True)
    try:
        for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        for p in HERE.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
        run('selftest','selftest.py');run('preflight','preflight.py');run('frozen','run_mode.py','--mode','frozen')
        run('frozen_summary','summarize.py','--frozen-only')
        run('within','run_mode.py','--mode','within')
    except BaseException as e:error=dict(phase=phase,reason=repr(e))
    finally:
        if (OUTPUT/'baseline_asset_manifest_before.json').is_file():
            t=time.monotonic()
            try:integrity()
            except Exception as e:error=dict(phase='integrity',reason=repr(e),earlier_error=error)
            timing['integrity']=time.monotonic()-t
    if error is None:
        try:run('summary','summarize.py')
        except BaseException as e:error=dict(phase=phase,reason=repr(e))
    dump(OUTPUT/'phase_timing.json',timing)
    safe(OUTPUT/'reproduce_commands.md').write_text(f'```bash\ncd {PROJECT}\nRCTD_GPU=1 bash scripts/{NAME}/launch_tmux.sh\nbash scripts/{NAME}/collect_results.sh\n```\nExplicitly authorized diagnostics; fresh directory required; no tuning.\n',encoding='utf-8')
    if error:
        intact=read_json(OUTPUT/'baseline_integrity_check.json')['status'] if (OUTPUT/'baseline_integrity_check.json').is_file() else 'NOT_YET_AUDITED'
        d=dict(status='ROUTE_CONDITIONED_DIAGNOSTICS_BLOCKED',source_status='ROUTE_CONDITIONED_TRAINING_NOT_SUPPORTED',stopped_at=error['phase'],reason=error['reason'],
            formal_integrity=intact,formal_model_changed=False,future_used_for_selection=False,partial_results_preserved=True)
        dump(OUTPUT/'final_decision.json',d);safe(OUTPUT/'REPORT.md').write_text('# Diagnostics blocked\n\n'+json.dumps(d,indent=2)+'\n',encoding='utf-8')
        print('CARE_ROUTE_CONDITIONED_DIAGNOSTICS_FAILED '+json.dumps(error),flush=True);raise SystemExit(1)
    print('CARE_ROUTE_CONDITIONED_DIAGNOSTICS_DONE',flush=True)
if __name__=='__main__':sys.dont_write_bytecode=True;main()
