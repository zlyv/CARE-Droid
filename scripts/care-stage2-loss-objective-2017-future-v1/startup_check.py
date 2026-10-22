#!/usr/bin/env python3
"""Run one complete T0 fold/seed retraining check before the formal tmux launch.

Inputs are the frozen S1/900000 W0 tensors and checkpoint. Output is one smoke JSON
inside the new artifact. The check proves custom objective plumbing reproduces original
epoch selection and per-APK probabilities before launching all T0/T1/T2 jobs.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import run_experiment as r
from common import OUT,atomic_json,load_data
from bfa_core import formal_split
data=load_data();pack=formal_split(data,'S1',900000,device='cuda',train_data=True)
pred,_,audit=r.fit_arm(pack,data,'T0_W0_BCE',900000)
diff=float(np.max(np.abs(pred-np.asarray(pack['old']['probability']))))
result={'status':'PASS' if diff<=1e-6 and audit['selected_epochs']==int(pack['old']['selected_epochs']) else 'FAIL','split':'S1','seed':900000,'selected_epochs':audit['selected_epochs'],'formal_selected_epochs':int(pack['old']['selected_epochs']),'max_probability_difference':diff}
atomic_json(OUT/'smoke/t0_full_startup_check.json',result,replace=True)
if result['status']!='PASS':raise RuntimeError(json.dumps(result))
print('T0_FULL_STARTUP_CHECK_PASS',result,flush=True)
