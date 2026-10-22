"""Execute the multi-year same-APK forensic audit; replay frozen models, extract targeted code, preserve UNKNOWN causes."""
from __future__ import annotations
import argparse,concurrent.futures,gzip,subprocess,traceback
from collections import Counter
from audit_common import *
from cohort import prepare
from replay import run as replay

def protect(mc,data,members):
    paths={BASE/'data.npz':'frozen multi-year inputs',BASE/'feature_rows.pkl':'frozen raw feature rows',BASE/'dataset_audit.json':'dataset audit',BASE/'per_apk_predictions.csv':'frozen predictions',BASE/'final_decision.json':'completed experiment decision',PROJECT/'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz':'endpoint payloads',Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json'):'payload to model Role permutation',PROJECT/'artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_lineage_2018_2021_exact.csv':'APK lineage'}
    for p in SOURCE.glob('*.py'):paths[p]='original development implementation'
    for tag in ['C0R_2017']+[w+'_'+a for w in ['T2','T3','T4'] for a in ['C1','C2']]:
        for seed in SEEDS:paths[BASE/'models/stage2'/tag/(str(seed)+'.pkl')]='saved model and preprocessing'
    for r in members:
        if r['raw_apk_present']:paths[Path(r['apk_path'])]='original APK, exact canonical-Host static inspection only'
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if p and str(PROJECT) in str(p) and Path(p).suffix=='.py' and HERE not in Path(p).resolve().parents:paths[Path(p).resolve()]='original inference dependency'
    manifest=[]
    for i,(p,purpose) in enumerate(paths.items(),1):
        manifest.append(asset(p,purpose))
        if i%25==0:print('MEEA_PROTECT',i,len(paths),flush=True)
    dump(OUTPUT/'baseline_asset_manifest_before.json',manifest)

def scan_one(jobpath,seconds=180):
    job=read_json(jobpath);log=safe(OUTPUT/'logs'/('code_'+job['member_id']+'.log'));start=time.monotonic()
    try:
        with log.open('w') as f:p=subprocess.run([sys.executable,'-B',str(HERE/'scan_code.py'),'--job',str(jobpath)],stdout=f,stderr=subprocess.STDOUT,timeout=seconds)
        status=OUTPUT/'private/scan_status'/(job['sha256']+'.json')
        if p.returncode!=0 or not status.is_file():raise RuntimeError('static parser failed rc='+str(p.returncode))
        return read_json(status)
    except Exception as e:
        s=dict(sha256=job['sha256'],member_id=job['member_id'],status='UNKNOWN_STATIC_SCAN_TIMEOUT_OR_FAILURE',error=repr(e),wall_seconds=time.monotonic()-start)
        dump(OUTPUT/'private/scan_status'/(job['sha256']+'.json'),s);dump(OUTPUT/'blind/code'/(job['member_id']+'.failure.json'),dict(member_id=job['member_id'],status=s['status']));return s

def dossiers(members,pairs,behavior):
    mid={r['sha256']:r['member_id'] for r in members};target_lookup={r['sha256']:r for r in members if r['target']};forms=[];keys=[]
    for pair in pairs:
        sid=pair['target_sha256'];cid='C_'+signature(['MEEA_CASE',sid])[:16];blocks=[sid,pair.get('opposite_label_control',''),pair.get('same_label_anchor','')];blocks=[s for s in blocks if s]
        blocks=sorted(blocks,key=lambda s:signature(['BLIND_ORDER',cid,s]));letters={s:chr(65+i) for i,s in enumerate(blocks)}
        facts=[]
        for s in blocks:
            member=mid[s];code=OUTPUT/'blind/code'/(member+'.json.gz');facts.append(dict(member=letters[s],member_id=member,code_file='code/'+member+'.json.gz' if code.exists() else 'code/'+member+'.failure.json'))
        dump(OUTPUT/'blind/cases'/(cid+'.json'),dict(case_id=cid,members=facts,labels_predictions_and_target_identity_hidden=True,code_identity_may_reveal_application_function=True,review_instruction='Cite exact method and DEX offset. Code presence/permission/attention is not benign or malicious proof. No general benign certificate; unknown is valid.'))
        forms.append(dict(case_id=cid,member_A=mid[blocks[0]],member_B=mid[blocks[1]] if len(blocks)>1 else '',member_C=mid[blocks[2]] if len(blocks)>2 else '',verified_program_fact='',method_signature='',dex_offset_bytes='',supporting_code_member='',observed_difference='',direction='UNKNOWN',scope='UNKNOWN',fact_already_in_frozen_Role_input='UNKNOWN',explanation='',review_confidence='UNKNOWN',additional_evidence_needed=''))
        keys.append(dict(case_id=cid,target_member=letters[sid],target_sha256=sid,target_category=target_lookup[sid]['category'],year=pair['year'],label=target_lookup[sid]['label'],same_label_anchor_member=letters.get(pair.get('same_label_anchor',''),''),opposite_label_control_member=letters.get(pair.get('opposite_label_control',''),''),match_rule_same=pair.get('same_label_anchor_match',''),match_rule_opposite=pair.get('opposite_label_control_match','')))
    write_csv(OUTPUT/'blind/review_form.csv',sorted(forms,key=lambda r:r['case_id']));write_csv(OUTPUT/'private/blind_review_key.csv',keys)

def summarize(members,behavior,status,timing):
    targets=[r for r in behavior if r['target']];reg=[r for r in targets if r['category']=='REGRESSED_FN'];persistent=[r for r in targets if r['category']!='REGRESSED_FN']
    observed=dict(regressed_FN_cases=len(reg),regressed_FN_all_C0_seeds_correct_all_C1_wrong=sum(r['C0_correct_seeds']==3 and r['C1_correct_seeds']==0 for r in reg),regressed_FN_C1_mixed_seed_predictions=sum(r['C1_correct_seeds'] in [1,2] for r in reg),regressed_FN_same_mean_critical_host=sum(r['C0_C1_mean_critical_same'] for r in reg),persistent_cases=len(persistent),persistent_all_9_seed_predictions_wrong=sum(r['C0_correct_seeds']==0 and r['C1_correct_seeds']==0 and r['C2_correct_seeds']==0 for r in persistent),cause_confirmed_count=0,UNKNOWN_cause_count=len(targets),boundary='seed agreement and critical-Host changes are observations, not root causes; no hidden-coordinate comparison or hybrid preprocessing swaps')
    dump(OUTPUT/'behavior_observations.json',observed);write_csv(OUTPUT/'static_code_status.csv',status)
    exact=sum(int(s.get('exact_host_bodies',0)) for s in status);selected=sum(int(s.get('selected_hosts',0)) for s in status)
    code_summary=dict(unique_APKs_inspected=len(status),status_counts=dict(Counter(s['status'] for s in status)),selected_Hosts=selected,exact_Host_bodies=exact,exact_selected_Host_fraction=exact/selected if selected else None,selection='all per-seed critical Hosts plus ensemble top3 per model',unselected_code_not_claimed_absent=True,literals_redacted=True,APK_execution_performed=False)
    dump(OUTPUT/'code_observability.json',code_summary);dump(OUTPUT/'phase_timing.json',timing)
    decision=dict(status='FROZEN_REPLAY_AND_TARGETED_CODE_COMPLETE_CAUSES_UNRESOLVED',targets=len(targets),matched_unique_members=len(members),model_replay='PASS',formal_integrity=read_json(OUTPUT/'baseline_integrity_check.json')['status'],verified_error_mechanisms=[],cause_confirmed_count=0,model_training_performed=False,model_training_authorized=False,threshold_changed=False,new_feature_added=False,next_action='review blind case dossiers and cite independent concrete code facts before opening private key; no new model from these observational flags',no_more_2017_generic_modality_screen=True)
    dump(OUTPUT/'decision.json',decision)
    lines=['# CARE-Droid Multi-Year Error Evidence Audit V1','','Status: '+decision['status'],'','## Scope','','73 target APKs: 35 persistent FN, 18 persistent FP, 3 persistent broken TP, 7 persistent broken TN, 10 regressed FN. 2018 excluded because its three expert configurations are identical. All comparisons use the same Stage-I, Top25 mask and raw frozen Host slots. Controls are same-year routed C1-correct APKs, with matching relaxation/reuse explicitly recorded. This is posthoc development forensics, not a new generalization experiment.','','## Actual model behavior','','```json',json.dumps(plain(observed),ensure_ascii=False,indent=2),'```','','## Actual static code coverage','','```json',json.dumps(plain(code_summary),ensure_ascii=False,indent=2),'```','','## What is and is not established','','- Saved model probabilities were replayed under original preprocessing; full per-seed trace data stay in private/traces. No fitting occurred.','- Raw Host identity, Role presence, endpoint counts and directed Relation31 were matched exactly to retained endpoint payloads.','- Same raw code can yield different selected Hosts/scores after training changes. This does not identify whether training coverage, normalization, background fitting or learned parameters caused a label error.','- Neighbor label composition is a diagnostic raw-sketch statistic, depends on training N, and is not semantic supervision or a cause.','- Different trained hidden vectors are not compared as if their coordinates were aligned. Old/new scalers or heads are never swapped into incompatible models.','- APK inspection is static DEX parsing only. Missing/ambiguous Hosts, target declarations and truncated code stay unresolved. Native, reflection and loader presence are not screened as a new classifier.','- Attention is a model preference, not a malicious/benign Host label. API occurrence, branch presence and permission names are not class direction or runtime causality.','- No verified error mechanism has yet been established. Unknown is not absence of a guard, not benign evidence and not static impossibility.','','## Review files','','Open blind/review_form.csv and blind/cases only. Do not open private/blind_review_key.csv or private traces while doing blinded semantic review. Application code names may naturally reveal function, so blinding is to dataset labels/predictions, not to all application identity. Controls may repeat; they are not independent replicate APKs. Cite DEX name/hash, method and instruction offset for every claim. A semantic fact must be externally grounded before attributing a classifier error to extraction or representation.','','## Integrity and timing','','```json',json.dumps(read_json(OUTPUT/'baseline_integrity_check.json'),ensure_ascii=False,indent=2),'```','','```json',json.dumps(timing,indent=2),'```','']
    safe(OUTPUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if OUTPUT.exists():raise RuntimeError('independent output already exists; use a new version, never overwrite')
    OUTPUT.mkdir(parents=True);started=time.monotonic();timing={}
    mc,data=old();v4,_,_=mc.runtime();t=time.monotonic();members,pairs,desc,roles=prepare(data);protect(mc,data,members);timing['preflight_and_hash_before_seconds']=time.monotonic()-t
    dump(OUTPUT/'protocol.json',dict(version=1,target_definition='fixed before code inspection: persistent three-expert errors and C0-correct/C1-wrong FN; 2019-2021 routed only',model_training=False,threshold_tuning=False,dynamic_execution=False,blind_review_required_for_class_semantics=True,replay_tolerance=1e-6,selected_host_code_cap_rule='all seed critical plus mean-attention top3 per model',per_APK_timeout_seconds=180,max_DEX_bytes=128*1024*1024,max_total_DEX_bytes=512*1024*1024,arbitrary_string_literals_redacted=True,causal_interpretation_authorized=False))
    t=time.monotonic();slots,behavior=replay(mc,data,members,desc,roles,a.device);timing['model_replay_seconds']=time.monotonic()-t
    jobs=[]
    for r in members:
        path=OUTPUT/'private/jobs'/(r['sha256']+'.json');dump(path,dict(sha256=r['sha256'],member_id=r['member_id'],apk_path=r['apk_path'],hosts=slots[r['sha256']]));jobs.append(path)
    t=time.monotonic();status=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(a.workers,2))) as pool:
        futures=[pool.submit(scan_one,j) for j in jobs]
        for f in concurrent.futures.as_completed(futures):
            status.append(f.result());print('MEEA_CODE_PROGRESS',len(status),len(jobs),status[-1]['status'],flush=True)
    timing['targeted_DEX_seconds']=time.monotonic()-t;dossiers(members,pairs,behavior);t=time.monotonic();integrity();timing['hash_after_seconds']=time.monotonic()-t;timing['total_seconds']=time.monotonic()-started;summarize(members,behavior,status,timing)
    print('MEEA_AUDIT_COMPLETE',str(OUTPUT/'report.md'),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUTPUT.exists():
            dump(OUTPUT/'decision.json',dict(status='AUDIT_BLOCKED',error=repr(e),traceback=traceback.format_exc(),model_training_performed=False,automatic_continuation=False))
            if (OUTPUT/'baseline_asset_manifest_before.json').exists():integrity()
        raise
