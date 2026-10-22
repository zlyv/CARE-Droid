#!/usr/bin/env python3
"""Audit and freeze the baseline/semantic contracts before Adapter training."""
from __future__ import annotations
import argparse,json,os
from semantic_common import *

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--resume',action='store_true'); a=ap.parse_args()
 if OUT.exists() and not a.resume: raise RuntimeError('output exists; use --resume and never overwrite')
 for p in [OUT/'config',OUT/'audit',OUT/'source_2017',OUT/'future',OUT/'summary',OUT/'logs',OUT/'models',OUT/'future_cache',OUT/'future_frozen',OUT/'future_status']: p.mkdir(parents=True,exist_ok=True)
 required=[SEM_V2/'final_decision.json',SEM_V2/'host_semantic_features.csv.gz',AUTO/'per_apk_predictions.csv.gz',AUTO/'asset_integrity_check.json',ROOT/'outputs/care_multiyear_development_v1/data.npz']
 required += [SCBE/'models'/split/str(seed)/'M0_REPRODUCED.pkl' for split in TRAIN_SPLITS for seed in SEEDS]
 missing=[str(p) for p in required if not p.is_file()]
 if missing: raise RuntimeError('missing required assets '+repr(missing))
 sem=json.loads((SEM_V2/'final_decision.json').read_text()); status=sem.get('status') or sem.get('decision')
 if status!='HOST_SEMANTIC_INPUT_FEASIBLE': raise RuntimeError(f'v2 semantic decision mismatch {status}')
 auto=json.loads((AUTO/'asset_integrity_check.json').read_text());
 if auto.get('status')!='PASS': raise RuntimeError('router integrity is not PASS')
 baseline={'status':'FROZEN','baseline':'W0 Direct-C43 SCBE/HPRN compatible endpoint','reason':'complete replay with current Manifest135 autonomous router and T1 future assets','2017_W0_reported_BA':0.919465,'model_seeds':SEEDS,'manifest_seeds':MANIFEST_SEEDS,'folds':SOURCE_SPLITS,'W0_checkpoints':[str(SCBE/'models'/s/str(seed)/'M0_REPRODUCED.pkl') for s in TRAIN_SPLITS for seed in SEEDS],'stage1':'Manifest135 RF probability','router_file':str(AUTO/'per_apk_predictions.csv.gz'),'router':'AUTO_ERROR_LR P(error)>=0.5; no fixed budget','semantic_source':str(SEM_V2),'semantic_v1_forbidden':True}
 config={'status':'FROZEN_BEFORE_TRAINING','arms':ARMS,'feature_width':FEATURE_WIDTH,'sidecar_width':17,'adapter':'Linear(16,16)-ReLU-Dropout-Linear(16,24,bias=False), final zero init','mask':'zero residual without valid evidence','training':'APK BCE; W0 selected epochs; same optimizer/batches/seeds/folds','S1':'v2 aggregate fixed category counts; cache lacks Role-by-category event binding','S3':'fixed label-free hashed embedding of source role, target role and dependency type weighted by log1p support','C3':'within-APK exact stratum shuffle by SelfRole31, endpoint bin, instruction bin, parse status','C4':'within-Host target-role rewiring by dependency type, preserving source/target marginals and support','future':'one-shot 2018-2021 after source freeze; no selection','threshold':0.5}
 atomic_json(OUT/'config/baseline_manifest.json',baseline); atomic_json(OUT/'config/experiment_config.json',config); records=[digest(p) for p in required]; atomic_json(OUT/'audit/asset_integrity_before.json',{'status':'PASS','assets':records}); atomic_json(OUT/'logs/run_status.json',{'status':'PREFLIGHT_PASS','stage':'preflight'}); print('PREFLIGHT_PASS',flush=True)
if __name__=='__main__': main()
