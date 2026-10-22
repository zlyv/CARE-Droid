#!/usr/bin/env python3
"""Positive proof coverage, not absence-based unguarded or benign flags."""
from collections import defaultdict
import numpy as np
VERIFIED={'VERIFIED_CONTROL','VERIFIED_PERMISSION_GUARD'}

def extract(record,frozen):
    by_pair=defaultdict(list);seen=set()
    for e in record['events']:
        if e['event_id'] in seen:raise RuntimeError('duplicate sensitive event ID')
        seen.add(e['event_id'])
        if e['frozen_aligned']:
            key=str(e['host_node_id'])+':'+str(e['endpoint_node_id'])
            if key not in frozen['pairs'] or set(e['role_ids'])!=set(frozen['pairs'][key]):raise RuntimeError('event/frozen edge Role mismatch')
            by_pair[key].append(e)
    evidence=np.zeros((4,31),float);den=np.zeros(31);unknown=np.zeros(31);event_den=np.zeros(31);frozen_den=np.zeros(31)
    for pair,roles in frozen['pairs'].items():
        if any(not 0<=int(r)<31 for r in roles):raise RuntimeError('invalid Role31 index')
        for role in roles:frozen_den[int(role)]+=1
        events=by_pair.get(pair,[])
        if not events:continue
        verified=[e for e in events if e['evidence_tier'] in VERIFIED]
        for e in verified:
            if not e.get('controls'):raise RuntimeError('verified tier lacks proof')
            for c in e['controls']:
                if not c.get('event_postdominates_allowing_successor') or not c.get('blocked_successors_cannot_reach_event'):raise RuntimeError('invalid strict proof')
        dominant=any(any(c.get('branch_dominates_event') and not c.get('bypass_path_possible') for c in e['controls']) for e in verified)
        bypass=any(any(c.get('bypass_path_possible') for c in e['controls']) for e in verified)
        values=(bool(verified),len(verified)==len(events),dominant,bypass)
        for role in roles:
            r=int(role);den[r]+=1;evidence[:,r]+=values
            event_den[r]+=len(events);unknown[r]+=sum(e['evidence_tier']=='UNKNOWN' for e in events)
    evidence/=np.maximum(den,1)[None,:]
    coverage=np.concatenate([(den>0).astype(float),unknown/np.maximum(event_den,1),
                             [len(by_pair)/max(len(frozen['pairs']),1),sum(e['evidence_tier']=='UNKNOWN' for es in by_pair.values() for e in es)/max(sum(map(len,by_pair.values())),1)]])
    if evidence.shape!=(4,31) or coverage.shape!=(64,) or not np.isfinite(coverage).all():raise RuntimeError('feature width/finiteness contract')
    audit=dict(frozen_edges=len(frozen['pairs']),observed_edges=len(by_pair),aligned_events=sum(map(len,by_pair.values())),
        verified_events=sum(e['evidence_tier'] in VERIFIED for es in by_pair.values() for e in es),unknown_events=sum(e['evidence_tier']=='UNKNOWN' for es in by_pair.values() for e in es),
        active_observed_roles=int((den>0).sum()),has_verified_fact=bool(evidence.any()))
    return evidence.reshape(-1),coverage,audit

def apply_role_mask(values,coverage):
    a=np.asarray(values,float).reshape(len(values),4,31).copy()
    a*=np.asarray(coverage)[:,:31][:,None,:]
    return a.reshape(len(values),124)
