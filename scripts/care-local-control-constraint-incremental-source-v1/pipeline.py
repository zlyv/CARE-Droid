#!/usr/bin/env python3
from __future__ import annotations
import ast,subprocess,time
from common import *
def main():
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True;timing={};phase='syntax';error=None
    def run(name,path=None):
        nonlocal phase
        phase=name;start=time.monotonic();print('LCIP_PHASE_START name='+name,flush=True)
        subprocess.run([sys.executable,'-B',str(path or HERE/(name+'.py'))],check=True)
        timing[name]=time.monotonic()-start;print(f'LCIP_PHASE_DONE name={name} seconds={timing[name]:.3f}',flush=True)
    print('CARE_LCC_INCREMENTAL_SOURCE_START output='+str(OUTPUT),flush=True)
    try:
        for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        run('engine_selftest',HERE/'engine/selftest.py');run('selftest');run('preflight');run('extract_features');run('run_probe')
    except Exception as exc:error=dict(phase=phase,error=repr(exc))
    finally:
        if (OUTPUT/'baseline_asset_manifest_before.json').is_file():
            try:run('integrity')
            except Exception as exc:error=error or dict(phase='integrity',error=repr(exc))
    if error is None:
        try:run('summarize')
        except Exception as exc:error=dict(phase=phase,error=repr(exc))
    dump(OUTPUT/'phase_timing.json',timing)
    safe(OUTPUT/'reproduce_commands.md').write_text(f'cd {PROJECT}\nLCIP_WORKERS=4 bash scripts/{NAME}/run_on_server.sh\n',encoding='utf-8')
    if error:
        integrity=json.loads((OUTPUT/'baseline_integrity_check.json').read_text())['status'] if (OUTPUT/'baseline_integrity_check.json').exists() else 'UNAVAILABLE'
        d=dict(status='CONTROL_INCREMENT_BLOCKED',stopped_at=error['phase'],reason=error['error'],formal_integrity=integrity,future_rows_loaded=0,
            continue_to_specify_frozen_diagnostic=False,next_action='repair only the explicit path/implementation failure without changing frozen protocol; no sample exclusions')
        dump(OUTPUT/'final_decision.json',d);safe(OUTPUT/'REPORT.md').write_text('# Incremental source test blocked\n\n'+json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('CARE_LCC_INCREMENTAL_SOURCE_FAILED '+json.dumps(error),flush=True);raise SystemExit(1)
    print('CARE_LCC_INCREMENTAL_SOURCE_DONE',flush=True)
    print('report='+str(OUTPUT/'REPORT.md'),flush=True)
if __name__=='__main__':main()
