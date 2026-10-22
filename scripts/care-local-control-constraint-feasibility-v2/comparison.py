#!/usr/bin/env python3
"""Matched event comparison, strictly label-free; never filters the cohort."""
from collections import Counter

VERIFIED={'VERIFIED_CONTROL','VERIFIED_PERMISSION_GUARD'}
METRICS=('scan_success_rate','frozen_host_method_recovery','frozen_host_endpoint_edge_recovery',
         'aligned_event_cfg_parse_rate','aligned_event_unknown_rate','verified_control_apk_rate',
         'verified_control_event_rate','verified_permission_apk_rate','serial_equivalent_scan_wall_seconds')

def compare_events(sid,fold,old,new):
    before={e['event_id']:e for e in old};after={e['event_id']:e for e in new}
    if len(before)!=len(old) or len(after)!=len(new):raise RuntimeError('duplicate event identity in comparison '+sid)
    rows=[]
    for eid in sorted(set(before)|set(after)):
        a=before.get(eid);b=after.get(eid)
        # Full event inventory changes are recorded, including non-aligned calls.
        rows.append(dict(sha256=sid,fold=fold,event_id=eid,
            inventory_status='MATCHED' if a and b else 'V1_ONLY' if a else 'V2_ONLY',
            aligned_v1=bool(a and a['frozen_aligned']),aligned_v2=bool(b and b['frozen_aligned']),
            tier_v1=a['evidence_tier'] if a else 'MISSING',tier_v2=b['evidence_tier'] if b else 'MISSING',
            reason_v1=a['reason'] if a else '',reason_v2=b['reason'] if b else '',
            verified_gained=bool(a and b and a['frozen_aligned'] and b['frozen_aligned'] and
                                 a['evidence_tier'] not in VERIFIED and b['evidence_tier'] in VERIFIED),
            verified_lost=bool(a and b and a['frozen_aligned'] and b['frozen_aligned'] and
                               a['evidence_tier'] in VERIFIED and b['evidence_tier'] not in VERIFIED)))
    return rows

def aggregate(old_coverage,new_coverage,rows):
    aligned=[r for r in rows if r['aligned_v1'] or r['aligned_v2']]
    return dict(labels_used=False,future_rows_loaded=0,
        metric_changes={key:dict(v1=old_coverage[key],v2=new_coverage[key],
                                difference=new_coverage[key]-old_coverage[key]) for key in METRICS},
        event_inventory= dict(Counter(r['inventory_status'] for r in rows)),
        alignment_changes=sum(r['aligned_v1']!=r['aligned_v2'] for r in rows),
        aligned_tier_transitions=dict(Counter(r['tier_v1']+' -> '+r['tier_v2'] for r in aligned)),
        newly_verified_events=sum(r['verified_gained'] for r in aligned),
        lost_verified_events=sum(r['verified_lost'] for r in aligned),
        labels_or_predictions_accessed=False,
        interpretation='More parsed CFGs or fewer UNKNOWNs do not establish class direction or correction gains.')
