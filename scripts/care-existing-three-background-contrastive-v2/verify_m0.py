"""Require exact historical training replay before any matched-seed contrastive arm."""
from __future__ import annotations
import argparse
import numpy as np
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('seed',type=int);a=p.parse_args()
    if a.tag not in ['T2_C1','T3_C1','T4_C1'] or a.seed not in [900000,900001,900002]:raise RuntimeError('invalid frozen window or seed')
    mc,v4,args,data=original();original_checkpoint=checkpoint(BASE/'models/stage2'/a.tag/(str(a.seed)+'.pkl'),mc,data)
    path=OUTPUT/'full'/a.tag/str(a.seed)/'M0_predictions.npz'
    with np.load(path,allow_pickle=False) as f:
        if list(f['sha256'])!=list(original_checkpoint['metadata']['validation_sha']):raise RuntimeError('M0 validation order differs')
        pp=np.asarray(f['probability'])
    diff=float(np.max(np.abs(pp-original_checkpoint['probability'])))
    if diff>1e-6:raise RuntimeError(f'M0 training replay failed: {diff}')
    write_json(path.with_name('M0_exact_retrain_check.json'),dict(status='PASS',tag=a.tag,seed=a.seed,max_probability_difference=diff,tolerance=1e-6,training_replay_not_just_checkpoint_inference=True))
    print('ETBC_M0_EXACT',a.tag,a.seed,diff,flush=True)

if __name__=='__main__':main()
