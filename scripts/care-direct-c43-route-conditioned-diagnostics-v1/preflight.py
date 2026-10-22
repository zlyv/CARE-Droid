from __future__ import annotations
import os,sys
from diag_common import *
from diagnostic_runtime import create_runtime

def main():
    decision=read_json(PARENT_OUTPUT/'final_decision.json')
    if decision.get('status')!='ROUTE_CONDITIONED_TRAINING_NOT_SUPPORTED' or decision.get('baseline_replay')!='PASS' or decision.get('formal_integrity')!='PASS':
        raise RuntimeError('expected completed NOT_SUPPORTED source Pilot with exact B0 and intact assets')
    policy=read_json(PARENT_OUTPUT/'protocol.json')
    for k,v in dict(route_weight=4.,other_weight=1.,threshold=.5,routing_crossfit_folds=4,member_calibration_folds=3,shuffle_seed=20261017,seeds=[0,1,2]).items():
        if policy.get(k)!=v:raise RuntimeError('frozen source setting changed '+k)
    if read_json(PARENT_SCRIPT/'protocol.json')!=policy:raise RuntimeError('parent implementation protocol no longer matches completed experiment')
    resolved=read_json(FORMAL_DIAG/'resolved_inputs.json')
    required=[(PARENT_OUTPUT/n,'completed source Pilot '+n) for n in ('final_decision.json','protocol.json','baseline_replay_validation.json','baseline_integrity_check.json','source_results.csv')]
    required += [(PARENT_SCRIPT/n,'passed source implementation') for n in ('runtime.py','pilot_common.py','weighting.py','train_adapter.py','protocol.json')]
    required += [(FORMAL_DIAG/'resolved_inputs.json','formal resolved paths'),(FORMAL_DIAG/'cache/future_s7.npz','cached future S7'),
        (FORMAL_DIAG/'cache/a0_role_2017_2021.npz','cached source/future A0 bags'),(FORMAL_DIAG/'references/future_label_free.csv','frozen Stage-I committee'),
        (FORMAL_DIAG/'frozen_predictions_with_labels.csv','frozen labels: summary only'),
        (PROJECT/'outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv','historical frozen Direct baseline replay'),
        (OLD_FUTURE/'run_future.py','historical frozen Direct runner'),(OLD_FUTURE/'future_common.py','historical future utilities')]
    required += [(Path(resolved[k]),'resolved input '+k) for k in ('sealed_features','endpoint_cache','future_inter_cache','future_p31_cache')]
    required += [(FORMAL_DIAG/n,'fixed within-year folds/labels') for n in ('within_2018_2019/predictions.csv','within_2020_2021/predictions.csv')]
    core,rt,w,ta,v4,adapter,runner=create_runtime('frozen')
    args=rt.family_module('run_frozen_arm').build_parser().parse_args(['--arm','SVM','--device',os.environ.get('RCTD_DEVICE','cuda')])
    required += [(getattr(args,k),'source default input '+k) for k in ('schema','p72_oof','source_multiscale_features')]
    common=rt.family_module('common');required.append((common.SOURCE_P31,'2017 P31'))
    required += [(Path(r['absolute_path']),r['purpose']) for r in read_json(PARENT_OUTPUT/'baseline_asset_manifest_after.json')]
    for root in (FAMILY,OLD_FUTURE,PARENT_SCRIPT,PROJECT/'scripts/care-svm-c43-remove-bounded-correction-v1'):
        required += [(p,'dependency source') for p in root.rglob('*.py')]
    checkpoints=PROJECT/'outputs/care-p31-classifier-family-v1/checkpoints'
    required += [(p,'within-year frozen Stage-I checkpoint') for p in checkpoints.glob('within_*_fold_*.npz')]
    missing=[str(p) for p,_ in required if not Path(p).is_file()]
    if missing:raise RuntimeError('missing read-only diagnostic dependency '+repr(missing[:15]))
    records={str(Path(p).resolve()):asset(p,purpose) for p,purpose in required}
    dump(OUTPUT/'baseline_asset_manifest_before.json',list(records.values()))
    for old in read_json(PARENT_OUTPUT/'baseline_asset_manifest_after.json'):
        now=records[str(Path(old['absolute_path']).resolve())]
        if any(now[k]!=old[k] for k in ('size_bytes','mtime_ns','sha256')):raise RuntimeError('source asset changed since Pilot '+old['absolute_path'])
    manifest=read_json(PARENT_OUTPUT/'implementation_manifest.json')
    for name in ('runtime.py','pilot_common.py','weighting.py','train_adapter.py','protocol.json'):
        old=next((r for r in manifest if Path(r['absolute_path']).name==name),None)
        if old is None or sha256(PARENT_SCRIPT/name)!=old['sha256']:raise RuntimeError('parent implementation changed '+name)
    meta=read_csv(FORMAL_DIAG/'within_2018_2019/predictions.csv')+read_csv(FORMAL_DIAG/'within_2020_2021/predictions.csv')
    ids=[norm(r['sha256']) for r in meta]
    if len(ids)!=1989 or len(set(ids))!=1989 or {y:sum(int(r['year'])==y for r in meta) for y in YEARS}!=COUNTS:raise RuntimeError('diagnostic cohort changed')
    frozen=read_csv(FORMAL_DIAG/'references/future_label_free.csv')
    if len(frozen)!=1989 or {norm(r['sha256']) for r in frozen}!=set(ids):raise RuntimeError('frozen/within diagnostic membership differs')
    v4.resolve_device(args.device)
    dump(OUTPUT/'protocol.json',dict(name=NAME,authorization='explicit user request after source NOT_SUPPORTED',source_status=decision['status'],
        frozen_source_policy=policy,scope=['2017 -> 2018..2021','independent within-year fixed five-fold 2018..2021'],
        contexts=21,arms=list(ARMS),within_test_labels_excluded_from_fit=True,future_labels_for_frozen_summary_only=True,
        outer_fold_weight_seed_rule='20261017 + within fold (same as source)',full_source_weight_seed=20261017,
        automatic_method_selection=False,threshold_changed=False,weight_search=False,formal_model_changed=False))
    dump(OUTPUT/'preflight.json',dict(status='PASS',source_status_preserved=decision['status'],future_n=1989,year_counts=COUNTS,
        caches_reused=True,new_APK_feature_extraction=False,neural_runtime='server',python=sys.version))
    print('RCTD_PREFLIGHT_PASS source_status=NOT_SUPPORTED future=1989 diagnostics_only=true',flush=True)
if __name__=='__main__':main()
