#!/usr/bin/env python3
"""Recover the completed v1 branch-audit intermediates into immutable v2.

Inputs are v1 files produced after all 15 W0 replays, counterfactuals, matching,
and bootstrap completed. Outputs are a new v2 tree with final reports and final
asset verification. It performs no model inference, fitting, or APK access.
"""
from __future__ import annotations
import importlib.util,json,os,shutil,sys,time
from pathlib import Path
import pandas as pd

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=ROOT/'scripts/care-host-branch-attribution-audit-2017-v1'
SRC=ROOT/'artifacts/care_host_branch_attribution_audit_2017_v1'
OUT=ROOT/'artifacts/care_host_branch_attribution_audit_2017_v2'
V3=ROOT/'artifacts/care_stage2_error_complementarity_audit_2017_v3'

spec=importlib.util.spec_from_file_location('branch_audit',HERE/'audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT

def copy_required():
    rel=[
      'config/audit_config.json','config/asset_manifest.json','integrity/sha256_before.json','integrity/w0_replay_check.json','integrity/readout_layout_check.json',
      'features/host_branch_embeddings.parquet','features/critical_attended_similarity.csv','features/attention_descriptive_stats.csv',
      'interventions/branch_2x2_counterfactuals.csv','interventions/branch_contribution_summary.csv','interventions/branch_gradient_sensitivity.csv','interventions/critical_topk_substitution.csv','interventions/attended_pooling_interventions.csv',
      'groups/outcome_groups.csv','groups/matched_group_comparison.csv','groups/matching_balance_report.csv','groups/class_stratified_results.csv',
      'statistics/bootstrap_intervals.csv','statistics/fold_seed_stability.csv']
    missing=[x for x in rel if not (SRC/x).is_file()]
    if missing:raise RuntimeError('v1 recovery source incomplete '+repr(missing))
    for x in rel:
        dst=OUT/x;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SRC/x,dst)
    return rel

def run():
    if not OUT.exists():raise RuntimeError('launcher must create v2 output')
    m.status('recover_completed_v1_intermediates')
    copied=copy_required();replay=json.loads((OUT/'integrity/w0_replay_check.json').read_text());layout=json.loads((OUT/'integrity/readout_layout_check.json').read_text())
    if replay.get('status')!='PASS' or max(float(x['max_probability_diff']) for x in replay['checks'])>m.TOL:raise RuntimeError('v1 replay is not PASS')
    groups=pd.read_csv(OUT/'groups/outcome_groups.csv');matched=pd.read_csv(OUT/'groups/matched_group_comparison.csv');balance=pd.read_csv(OUT/'groups/matching_balance_report.csv').to_dict('records');boots=pd.read_csv(OUT/'statistics/bootstrap_intervals.csv').to_dict('records')
    decision=m.report(groups,matched,balance,boots,replay['checks'],layout['actual_layout'])
    baseline={'status':'PASS','source_endpoint':str(V3),'recovered_completed_intermediates_from':str(SRC),'recovery_reason':'v1 failed only in final report DataFrame truth-value check after all scientific computation completed','rows':2397,'apks':799,'manifest_seeds':m.MANIFEST_SEEDS,'W0_model_seeds':m.MODEL_SEEDS,'folds':m.SPLITS,'endpoint_BA':json.loads((V3/'config/baseline_manifest.json').read_text())['mean_2017_end_to_end_BA'],'max_W0_probability_diff':replay['max_probability_diff'],'endpoint_stage2_max_diff':replay['endpoint_stage2_max_diff'],'endpoint_final_max_diff':replay['endpoint_final_max_diff'],'different_historical_endpoint_used':False}
    m.wjson(OUT/'config/baseline_manifest.json',baseline);m.wjson(OUT/'config/recovery_provenance.json',{'status':'PASS','source':str(SRC),'copied_files':copied,'scientific_steps_reused':['15 fold-seed W0 replays','readout traces','2x2 counterfactuals','gradients','Top-K substitutions','pooling interventions','matching','bootstrap'],'scientific_results_recomputed':False,'report_logic_fixed':'DataFrame.empty used instead of ambiguous DataFrame truth test'})
    before=json.loads((OUT/'integrity/sha256_before.json').read_text());m.integrity_after(before)
    required=['config/audit_config.json','config/baseline_manifest.json','config/asset_manifest.json','config/recovery_provenance.json','integrity/sha256_before.json','integrity/sha256_after.json','integrity/w0_replay_check.json','integrity/readout_layout_check.json','features/host_branch_embeddings.parquet','features/critical_attended_similarity.csv','features/attention_descriptive_stats.csv','interventions/branch_2x2_counterfactuals.csv','interventions/branch_contribution_summary.csv','interventions/branch_gradient_sensitivity.csv','interventions/critical_topk_substitution.csv','interventions/attended_pooling_interventions.csv','groups/outcome_groups.csv','groups/matched_group_comparison.csv','groups/matching_balance_report.csv','groups/class_stratified_results.csv','statistics/bootstrap_intervals.csv','statistics/fold_seed_stability.csv','statistics/mechanism_evidence_summary.csv','reports/final_report.md','reports/final_decision.json']
    missing=[x for x in required if not (OUT/x).is_file()]
    if missing:raise RuntimeError('recovered output missing '+repr(missing))
    m.status('complete','COMPLETE',decision=decision['status'],next_experiment_decision=decision['next_experiment_decision'],recovered_from_v1=True,rows=2397,apks=799)

if __name__=='__main__':run()
