"""Unchanged SVM-anchored MeanOpposition; compatible reanchoring isolates auxiliary-member retraining."""
from __future__ import annotations
import numpy as np
from myd_common import *
from protocol import mean_opposition,top_budget
from train_stage1 import run

def evaluate(data,window,jobs=4):
    p=run(data,window,jobs,committee=True);old=p['old_member'];new=p['new_member'];reanchor=old.copy();reanchor[:,3]=p['s1']
    scores=[mean_opposition(old,p['s0']),mean_opposition(reanchor,p['s1']),mean_opposition(new,p['s1'])]
    save_npz(OUTPUT/'predictions'/window/'router.npz',ids=p['ids'],val_index=p['val_index'],train_index=p['train_index'],s0=p['s0'],s1=p['s1'],old_score=scores[0],reanchor_score=scores[1],new_score=scores[2],old_route=top_budget(scores[0],p['ids']),reanchor_route=top_budget(scores[1],p['ids']),new_route=top_budget(scores[2],p['ids']))
    print('MYD_ROUTER_DONE',window,flush=True)
