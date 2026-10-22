from __future__ import annotations
import sys
from common import *

PRE = ROOT/'scripts/care-cfg-evidence-audit-2017-v3/prelaunch_integrity.json'

def assets():
    paths=[CFG_FEATURES,EGO_SOURCE,ENDPOINT_CACHE,H1_OUT/'protocol.json',H1_OUT/'baseline_integrity_check.json']
    for split in SPLITS:
        for seed in SEEDS:
            paths.append(SCBE/f'models/{split}/{seed}/M0_REPRODUCED.pkl')
            paths.append(H1_OUT/f'models/{split}/{seed}/H1_HOST_BOUND_CFG.pkl')
    return paths

def snapshot():
    return {str(p):{'size':p.stat().st_size,'sha256':sha256(p)} for p in assets()}

mode=sys.argv[1]
if mode=='pre':
    write_json_new(PRE,{'status':'CAPTURED','assets':snapshot()})
elif mode=='post':
    before=read_json(PRE)['assets'];after=snapshot()
    changed=[p for p in before if before[p]!=after.get(p)]
    write_json_new(OUT/'integrity_check.json',{'status':'PASS' if not changed else 'FAIL','checked_assets':len(before),'changed_assets':changed,'before':before,'after':after})
else: raise SystemExit('use pre or post')
