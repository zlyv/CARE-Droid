#!/usr/bin/env python3
from __future__ import annotations
import ast,json,os,subprocess,sys,time
from pathlib import Path
from common import *
HERE=Path(__file__).resolve().parent
def main():
    sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1';phase='syntax';timings={};error=None
    def run(name):
        nonlocal phase
        phase=name;t=time.monotonic();print(f'LCC_PHASE_START name={name}',flush=True)
        subprocess.run([sys.executable,str(HERE/(name+'.py'))],check=True)
        timings[name]=time.monotonic()-t;print(f'LCC_PHASE_DONE name={name} wall_seconds={timings[name]:.3f}',flush=True)
    print('CARE_LCC_FEASIBILITY_START output='+str(OUTPUT),flush=True)
    try:
        for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        for name in ('selftest','preflight','run_audit'):run(name)
    except Exception as exc:error=dict(phase=phase,error=repr(exc))
    finally:
        if (OUTPUT/'baseline_asset_manifest_before.json').is_file():
            try:run('integrity')
            except Exception as exc:error=error or dict(phase='integrity',error=repr(exc))
    if error is None:
        try:run('summarize')
        except Exception as exc:error=dict(phase='summarize',error=repr(exc))
    dump(OUTPUT/'phase_timing.json',timings)
    if error:
        integrity='UNAVAILABLE'
        ip=OUTPUT/'baseline_integrity_check.json'
        if ip.is_file():integrity=json.loads(ip.read_text())['status']
        d=dict(status='LCC_FEASIBILITY_BLOCKED',stopped_at=error['phase'],error=error['error'],formal_integrity=integrity,
           labels_used=False,future_rows_loaded=0,model_training_authorized=False,continue_to_design_conditional_probe=False,
           next_action='repair the explicit implementation, path or permission error without changing the protocol; resume completed scans')
        dump(OUTPUT/'final_decision.json',d)
        safe(OUTPUT/'REPORT.md').write_text('# LCC-FA v1\n\nStatus: **LCC_FEASIBILITY_BLOCKED**.\n\n'+json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        safe(OUTPUT/'reproduce_commands.md').write_text(f'cd {PROJECT}\nLCC_WORKERS=4 bash scripts/care-local-control-constraint-feasibility-v1/run_on_server.sh\n',encoding='utf-8')
        print('CARE_LCC_FEASIBILITY_FAILED '+json.dumps(error),flush=True);raise SystemExit(1)
    print('CARE_LCC_FEASIBILITY_DONE',flush=True)
    print('report='+str(OUTPUT/'REPORT.md'),flush=True)
if __name__=='__main__':main()
