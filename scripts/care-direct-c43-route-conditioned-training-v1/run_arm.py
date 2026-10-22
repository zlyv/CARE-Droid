from __future__ import annotations
import argparse,os,sys
import numpy as np
from pilot_common import *
from runtime import load_runtime,arguments,validate_loaded_inputs
from train_adapter import TrainingState

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--arm',choices=ARMS,required=True); cli=ap.parse_args(); arm=cli.arm
    if arm!='B0_ORIGINAL' and read_json(OUTPUT/'baseline_replay_validation.json').get('status')!='PASS': raise RuntimeError('B0 did not replay; weighted fits prohibited')
    v4,_,_=load_runtime(arm)
    from nested_stage1 import NestedStage1
    with np.load(OUTPUT/'source_index.npz',allow_pickle=False) as a: index={k:a[k] for k in a.files}
    original_loader=v4.audit.load_inputs
    def checked_loader(args):
        loaded=original_loader(args); ids,*_=validate_loaded_inputs(loaded)
        if not np.array_equal(ids,index['ids']): raise RuntimeError('runtime manifest order differs from preflight')
        return loaded
    v4.audit.load_inputs=checked_loader
    # Cache the original inner-OOF SVM from the identical nested committee fit.
    # This reproduces NestedStage1's 4 folds and 3-fold calibration exactly.
    if arm=='B0_ORIGINAL':
        nested=NestedStage1('SVM',OUTPUT/'audits'/f'{arm}_nested.json',int(os.environ.get('RCT_JOBS','4')))
        v4.inner_oof_global24=nested.inner_oof
    else:
        calls=[]
        def cached_inner(rows,labels,*unused):
            f=len(calls)+1
            with np.load(OUTPUT/'routes'/f'f{f}.npz',allow_pickle=False) as a:
                if not np.array_equal(a['ids'],np.asarray([norm(r['sha256']) for r in rows])): raise RuntimeError('cached nested training IDs')
                ix=a['train_index']
                if not np.array_equal(labels,index['y'][ix]): raise RuntimeError('cached nested training labels')
                p=np.asarray(a['member'][:,3],float)
            calls.append(dict(fold=f,train_n=len(rows),strict_inner_OOF=True,outer_test_excluded=True))
            dump(OUTPUT/'audits'/f'{arm}_nested.json',calls)
            return p
        v4.inner_oof_global24=cached_inner
    state=TrainingState(v4,arm,index); v4.fit_predict_variant=state.fit_predict
    sys.argv=[sys.argv[0]]+arguments(arm,os.environ.get('RCT_DEVICE','cuda')); v4.main()
    if state.fold!=5 or (arm=='B0_ORIGINAL' and len(nested.calls)!=5): raise RuntimeError('five outer-fold calls required')
    architecture_path=OUTPUT/'source'/arm/'architecture.json'
    architecture=read_json(architecture_path)
    architecture.update(first_layer='frozen calibrated P31-SVM',router='frozen SVM-anchored four-member MeanOpposition Top25%',
        candidate_base='Direct head receives zero Stage-I logit; training weights use independent nested committee',
        bounded_correction=None,decision_threshold=.5,arm=arm,loss_weighting=arm,
        epoch_policy=protocol()['epoch_policy'],future_rows_loaded=0)
    dump(architecture_path,architecture)
    path=OUTPUT/'source'/arm/'source_predictions.csv'; rows=read_csv(path); by={norm(r['sha256']):r for r in rows}; key=METHOD+'__'+arm+'__expert_prob'
    if len(rows)!=799 or set(by)!=set(index['ids']): raise RuntimeError('expert output membership')
    p=np.asarray([float(by[s][key]) for s in index['ids']])
    if not np.isfinite(p).all(): raise RuntimeError('nonfinite expert')
    save_npz(OUTPUT/'predictions'/f'{arm}.npz',ids=index['ids'],expert=p)
    if arm=='B0_ORIGINAL':
        diff=float(np.max(np.abs(p-index['expert']))); m=metrics(index['y'],index['base'],p,index['route'])
        checks=dict(probability_within_tolerance=diff<=protocol()['source_replay_probability_tolerance'],
            all_decisions_exact=bool(np.array_equal(p>=.5,index['expert']>=.5)),fixed_broken_net_exact=(m['fixed'],m['broken'],m['net'])==(49,17,32))
        dump(OUTPUT/'baseline_replay_validation.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
            max_probability_difference=diff,metrics=m,future_rows_loaded=0))
        if not all(checks.values()): raise RuntimeError('B0 original replay failed; stop before B1')
    print('RCT_ARM_DONE arm='+arm,flush=True)
if __name__=='__main__': main()
