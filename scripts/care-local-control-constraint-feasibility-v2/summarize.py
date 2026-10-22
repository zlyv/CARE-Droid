#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from common import *
from comparison import compare_events,aggregate
VERIFIED={'VERIFIED_CONTROL','VERIFIED_PERMISSION_GUARD'}
def ratio(a,b):return a/b if b else 0.0
def main():
    selected=list(csv.DictReader((OUTPUT/'selection_label_free.csv').open(encoding='utf-8')));protocol=json.loads((OUTPUT/'protocol.json').read_text())
    statuses={r['sha256']:r for r in csv.DictReader((OUTPUT/'scan_status.csv').open(encoding='utf-8'))}
    token=json.loads((OUTPUT/'protocol_identity.json').read_text())['scan_identity_sha256']
    parent_token=json.loads((BASELINE_OUTPUT/'protocol_identity.json').read_text())['scan_identity_sha256']
    apkrows=[];eventrows=[];methodrows=[];proofrows=[];edgerows=[];missingrows=[];transitions=[]
    align=Counter();unknown=Counter();partial=Counter();permission_partial=Counter()
    for r in selected:
        sid=r['sha256'];f=int(r['fold']);stat=statuses[sid];frozen=json.loads((OUTPUT/'cache/frozen'/f'{sid}.json').read_text())
        record=load_gz(OUTPUT/'cache/scans'/token[:16]/f'{sid}.json.gz') if stat['status']=='OK' else None
        events=record['events'] if record else [];aligned=[e for e in events if e['frozen_aligned']]
        old=load_gz(BASELINE_OUTPUT/'cache/scans'/parent_token[:16]/f'{sid}.json.gz')
        if old.get('sha256')!=sid or old.get('status')!='OK':raise RuntimeError('v1 comparison cache invalid '+sid)
        transitions+=compare_events(sid,f,old['events'],events)
        verified=[e for e in aligned if e['evidence_tier'] in VERIFIED];permissions=[e for e in aligned if e['evidence_tier']=='VERIFIED_PERMISSION_GUARD']
        tiers=Counter(e['evidence_tier'] for e in aligned)
        local=dict(sha256=sid,fold=f,status=stat['status'],frozen_hosts=len(frozen['host_ids']),frozen_pairs=len(frozen['pairs']),
          recovered_hosts=record['frozen_hosts_recovered'] if record else 0,recovered_pairs=record['frozen_pairs_recovered'] if record else 0,
          raw_sensitive_events=len(events),aligned_sensitive_events=len(aligned),cfg_parsed_events=sum(e.get('cfg_status')=='PARSED' for e in aligned),
          verified_control_events=len(verified),verified_permission_events=len(permissions),unknown_events=tiers['UNKNOWN'],partial_events=tiers['PARTIAL'],
          has_verified_control=int(bool(verified)),has_verified_permission=int(bool(permissions)),wall_seconds=float(stat.get('wall_seconds') or 0),error=stat.get('error',''))
        apkrows.append(local)
        for e in events:
            align[e['alignment_status']]+=1
            if e['frozen_aligned'] and e['evidence_tier']=='UNKNOWN':unknown[e['reason']]+=1
            if e['frozen_aligned'] and e['evidence_tier']=='PARTIAL':partial[e['reason']]+=1
            if e['frozen_aligned']:
                permission_partial.update(p['reason'] for p in e['permissions'] if p['status']=='PARTIAL')
            eventrows.append({**e,'fold':f,'controls':json.dumps(e['controls'],ensure_ascii=False),'permissions':json.dumps(e['permissions'],ensure_ascii=False),
                              'instruction_evidence':json.dumps(e.get('instruction_evidence',[]),ensure_ascii=False)})
            for c in e['controls']:
                proofrows.append(dict(sha256=sid,fold=f,event_id=e['event_id'],host_node_id=e['host_node_id'],endpoint_node_id=e['endpoint_node_id'],
                  invoke_offset_bytes=e['invoke_offset_bytes'],branch_offset_bytes=c['branch_offset'],proof_json=json.dumps(c),evidence_tier=e['evidence_tier']))
        if record:
            methodrows += [dict(sha256=sid,fold=f,**m) for m in record['methods']]
            missingrows += [dict(sha256=sid,fold=f,**m) for m in record['missing_frozen_pairs']]
        else:missingrows += [dict(sha256=sid,fold=f,pair_key=p,reason='APK_SCAN_'+stat['status']) for p in frozen['pairs']]
        by_pair=defaultdict(list)
        for e in aligned:by_pair[str(e['host_node_id'])+':'+str(e['endpoint_node_id'])].append(e)
        for pair in frozen['pairs']:
            matching=by_pair.get(pair,[])
            edgerows.append(dict(sha256=sid,fold=f,pair_key=pair,observed_calls=len(matching),
              any_verified_control=int(any(e['evidence_tier'] in VERIFIED for e in matching)),
              all_observed_calls_verified_control=int(bool(matching) and all(e['evidence_tier'] in VERIFIED for e in matching)),
              any_verified_permission=int(any(e['evidence_tier']=='VERIFIED_PERMISSION_GUARD' for e in matching)),
              unknown_calls=sum(e['evidence_tier']=='UNKNOWN' for e in matching),
              scope='OBSERVED_EXACT_DEX_INVOKES_ONLY; NOT_ALL_RUNTIME_DISPATCH_TARGETS'))
    foldrows=[]
    for f in range(1,6):
        local=[r for r in apkrows if r['fold']==f]
        foldrows.append(dict(fold=f,apks=len(local),scan_success=sum(r['status']=='OK' for r in local),
           verified_control_apks=sum(r['has_verified_control'] for r in local),verified_control_apk_rate=ratio(sum(r['has_verified_control'] for r in local),len(local)),
           verified_permission_apks=sum(r['has_verified_permission'] for r in local),
           frozen_pair_recovery=ratio(sum(r['recovered_pairs'] for r in local),sum(r['frozen_pairs'] for r in local)),
           aligned_events=sum(r['aligned_sensitive_events'] for r in local),verified_control_events=sum(r['verified_control_events'] for r in local)))
    total=lambda key:sum(r[key] for r in apkrows)
    cov=dict(selected_apks=200,successful_scans=sum(r['status']=='OK' for r in apkrows),scan_success_rate=ratio(sum(r['status']=='OK' for r in apkrows),200),
       frozen_host_method_recovery=ratio(total('recovered_hosts'),total('frozen_hosts')),
       frozen_host_endpoint_edge_recovery=ratio(total('recovered_pairs'),total('frozen_pairs')),
       frozen_hosts=total('frozen_hosts'),frozen_pairs=total('frozen_pairs'),raw_sensitive_events=total('raw_sensitive_events'),aligned_events=total('aligned_sensitive_events'),
       whole_apk_event_alignment_rate=ratio(total('aligned_sensitive_events'),total('raw_sensitive_events')),
       aligned_event_cfg_parse_rate=ratio(total('cfg_parsed_events'),total('aligned_sensitive_events')),
       aligned_event_unknown_rate=ratio(total('unknown_events'),total('aligned_sensitive_events')),
       aligned_event_partial_rate=ratio(total('partial_events'),total('aligned_sensitive_events')),
       verified_control_apk_rate=ratio(total('has_verified_control'),200),verified_control_event_rate=ratio(total('verified_control_events'),total('aligned_sensitive_events')),
       verified_permission_apk_rate=ratio(total('has_verified_permission'),200),verified_permission_events=total('verified_permission_events'),
       verified_control_events=total('verified_control_events'),folds_with_control_apk_rate_at_least_0_15=sum(r['verified_control_apk_rate']>=.15 for r in foldrows),
       folds_with_verified_permission=sum(r['verified_permission_apks']>0 for r in foldrows),
       serial_equivalent_scan_wall_seconds=total('wall_seconds'),per_apk_scan_seconds_median=(sorted(r['wall_seconds'] for r in apkrows)[99]+sorted(r['wall_seconds'] for r in apkrows)[100])/2,
       alignment_reasons=dict(align),unknown_reasons=dict(unknown),missing_frozen_pair_reasons=dict(Counter(r['reason'] for r in missingrows)),
       partial_reasons=dict(partial),permission_partial_reasons=dict(permission_partial),
       aligned_try_events=sum(e['frozen_aligned'] and bool(e.get('try_region_count')) for e in eventrows),
       aligned_cyclic_events=sum(e['frozen_aligned'] and bool(e.get('has_reachable_cycle')) for e in eventrows),
       parsed_try_methods=sum(m['cfg_status']=='PARSED' and bool(m.get('tries_size')) for m in methodrows),
       parsed_cyclic_methods=sum(m['cfg_status']=='PARSED' and bool(m.get('has_reachable_cycle')) for m in methodrows),
       permission_check_invokes_in_frozen_methods=sum(m.get('permission_check_invokes',0) for m in methodrows if m['frozen_host']),
       permission_def_use_method_reasons=dict(Counter(m['permission_def_use_error'] for m in methodrows if m.get('permission_def_use_error'))),
       labels_used=False,future_rows_loaded=0)
    g=protocol['stage1_gate'];integrity=json.loads((OUTPUT/'baseline_integrity_check.json').read_text())
    criteria={
      'scan_success_rate>=0.95':cov['scan_success_rate']>=g['scan_success_rate_min'],
      'frozen_host_method_recovery>=0.95':cov['frozen_host_method_recovery']>=g['frozen_host_method_recovery_min'],
      'frozen_host_endpoint_edge_recovery>=0.90':cov['frozen_host_endpoint_edge_recovery']>=g['frozen_host_endpoint_edge_recovery_min'],
      'aligned_event_cfg_parse_rate>=0.60':cov['aligned_event_cfg_parse_rate']>=g['aligned_event_cfg_parse_rate_min'],
      'aligned_event_unknown_rate<=0.40':cov['aligned_event_unknown_rate']<=g['aligned_event_unknown_rate_max'],
      'verified_control_apk_rate>=0.20':cov['verified_control_apk_rate']>=g['verified_control_apk_rate_min'],
      'verified_control_event_rate>=0.05':cov['verified_control_event_rate']>=g['verified_control_event_rate_min'],
      'at_least_4_folds_control_apk_rate>=0.15':cov['folds_with_control_apk_rate_at_least_0_15']>=g['folds_with_control_apk_rate_at_least_0_15_min'],
      'serial_scan_cost<=4h':cov['serial_equivalent_scan_wall_seconds']<=g['serial_equivalent_scan_wall_seconds_max'],
      'formal_integrity=PASS':integrity['status']=='PASS'}
    passed=all(criteria.values());perm=cov['verified_permission_apk_rate']>=.05 and cov['folds_with_verified_permission']>=4
    matched=aggregate(json.loads((BASELINE_OUTPUT/'coverage.json').read_text()),cov,transitions)
    inventory_ok=(matched['event_inventory'].get('V1_ONLY',0)==0 and matched['event_inventory'].get('V2_ONLY',0)==0 and matched['alignment_changes']==0)
    # An inventory change cannot be silently attributed to CFG coverage.
    passed=passed and inventory_ok
    decision=dict(status='LCC_CONTROL_OBSERVABILITY_SUPPORTED' if passed else 'LCC_OBSERVABILITY_GATE_FAILED',stopped_at='STAGE1_LABEL_FREE_FULL_200',
      criteria=criteria,permission_constraints_ready_for_separate_probe=bool(passed and perm),labels_used=False,future_rows_loaded=0,
      v1_gate_thresholds_unchanged=True,parent_event_inventory_equal=inventory_ok,
      model_training_authorized=False,continue_to_design_conditional_probe=passed,formal_integrity=integrity['status'],
      next_action='specify a separate strict source incremental test; permission features only if separately ready' if passed else 'stop this version; inspect coverage and unknown reasons without class-driven repairs',
      semantic_boundary='static local control proof is not runtime execution, user consent or benign evidence')
    for name,rows in [('apk_coverage.csv',apkrows),('sensitive_events.csv',eventrows),('method_cfg_audit.csv',methodrows),('control_proofs.csv',proofrows),
                      ('frozen_edge_evidence.csv',edgerows),('missing_frozen_pairs.csv',missingrows),('fold_coverage.csv',foldrows)]:write_csv(OUTPUT/name,rows)
    dump(OUTPUT/'coverage.json',cov);dump(OUTPUT/'final_decision.json',decision)
    write_csv(OUTPUT/'v1_v2_event_transitions.csv',transitions);dump(OUTPUT/'v1_v2_comparison.json',matched)
    lines=['# CARE-Droid Local Control Constraint Feasibility Audit v2','',f"Final status: **{decision['status']}**.",'',
      'All 200 frozen 2017 SVM-routed APKs were selected without accessing labels or error categories. No classifier or future diagnostic was run.','',
      '| Coverage | Value |','|---|---:|']
    for key in ('successful_scans','frozen_host_method_recovery','frozen_host_endpoint_edge_recovery','whole_apk_event_alignment_rate','aligned_event_cfg_parse_rate',
                'aligned_event_unknown_rate','aligned_event_partial_rate','verified_control_apk_rate','verified_control_event_rate','verified_permission_apk_rate','serial_equivalent_scan_wall_seconds'):
        lines.append(f'| {key} | {cov[key]} |')
    lines += ['', '## Gate criteria','']+[f'- {k}: {v}' for k,v in criteria.items()]
    lines += ['', '## Scope and limitations','',
      'Control proofs use strong postdominance plus exclusive branch-successor reachability. DEX try/catch edges are conservatively included; every potentially throwing instruction retains EXIT, including catch-all. No exception-type pruning or termination assumption is made. Infinite paths avoiding a call invalidate postdominance. Malformed targets/tables, work caps and unsupported instructions remain UNKNOWN.',
      'Permission proofs trace a unique local return value to equality/inequality against zero, require resolved framework dispatch and reject a bypass. Exception edges erase symbolic values. Cyclic permission def-use remains unsupported and PARTIAL; zero verified permission facts is not proof that no checks exist.',
      'Whole-APK event alignment and frozen Host-edge recovery have different denominators. Cap-related nonalignment reasons are possibilities supported by cache flags, not proof that a particular event was cropped.',
      'An edge with one proven call is reported separately from an edge with all observed calls proven. Neither is proof of every runtime dispatch target.',
      'The restricted algorithm can underestimate coverage. A failed gate means this v2 observation is insufficient, not that local constraints never exist.',
      '', '## Matched v1/v2 change','',f"Event inventory equal: {inventory_ok}; newly verified events: {matched['newly_verified_events']}; lost verified events: {matched['lost_verified_events']}.",
      'See v1_v2_comparison.json for metric differences and v1_v2_event_transitions.csv for every matched event. Coverage changes are not evidence of improved malware classification.',
      '',f"Formal asset integrity: **{integrity['status']}**.",f"Next action: {decision['next_action']}",'']
    safe(OUTPUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    safe(OUTPUT/'reproduce_commands.md').write_text(f"# Reproduction\n\n```bash\ncd {PROJECT}\nLCC_WORKERS=4 bash scripts/care-local-control-constraint-feasibility-v2/run_on_server.sh\n```\n\nThis package ends after stage 1 and never launches supervised training.\n",encoding='utf-8')
    print('LCC_FINAL_STATUS='+decision['status'],flush=True)
if __name__=='__main__':main()
