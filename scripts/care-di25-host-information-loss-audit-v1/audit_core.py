#!/usr/bin/env python3
"""Label-free reconstruction utilities for the Host information-loss audit."""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np

ROLE_COUNT=31

def entropy(counts):
    x=np.asarray(counts,dtype=float);s=float(x.sum())
    if s<=0:return 0.0
    p=x[x>0]/s
    return float(-np.sum(p*np.log(p))/math.log(ROLE_COUNT))

def reconstruct(payload):
    ids=list(payload.get("node_ids") or []);roles=[tuple(sorted({int(v) for v in (x or []) if 0<=int(v)<ROLE_COUNT})) for x in (payload.get("node_role_ids") or [])]
    base=np.asarray(payload.get("node_base_features") or [],dtype=float)
    if base.size==0:base=np.zeros((0,15),dtype=float)
    ei=payload.get("edge_index") or [[],[]];ef=np.asarray(payload.get("edge_features") or [],dtype=float)
    if len(ei)!=2:raise RuntimeError("edge_index schema")
    src=np.asarray(ei[0],dtype=int);dst=np.asarray(ei[1],dtype=int)
    if ef.size==0:ef=np.zeros((len(src),7),dtype=float)
    if base.ndim!=2 or base.shape[1]!=15 or len(ids)!=len(base) or len(roles)!=len(base):raise RuntimeError("node schema")
    if len(src)!=len(dst) or ef.shape!=(len(src),7):raise RuntimeError("edge schema")
    endpoints=set(map(int,np.flatnonzero(base[:,4]>.5)));targets=defaultdict(set);endpoint_edge_rows=defaultdict(list)
    for j,(s,t) in enumerate(zip(src,dst)):
        s=int(s);t=int(t)
        if t in endpoints:
            targets[s].add(t);endpoint_edge_rows[s].append(j)
    host_roles={}
    for h,ts in targets.items():
        merged=tuple(sorted({r for t in ts for r in roles[t]}))
        if merged:host_roles[h]=merged
    hosts=sorted(host_roles);hs=set(hosts);host_edges={(int(s),int(t)) for s,t in zip(src,dst) if int(s) in hs and int(t) in hs and int(s)!=int(t)}
    incoming={h:set() for h in hosts};outgoing={h:set() for h in hosts}
    for s,t in host_edges:outgoing[s].add(t);incoming[t].add(s)
    result=[]
    for h in hosts:
        ts=sorted(targets[h]);counts=np.zeros(ROLE_COUNT,dtype=int)
        for t in ts:
            for r in roles[t]:counts[r]+=1
        erows=endpoint_edge_rows[h];ctx=np.mean(ef[erows],axis=0) if erows else np.zeros(7)
        reciprocal=sum(1 for n in outgoing[h] if h in outgoing.get(n,set()))
        result.append({"host_index":h,"host_id":ids[h],"role_presence":(counts>0).astype(int),"role_endpoint_counts":counts,
          "unique_endpoint_count":len(ts),"role_assignment_count":int(counts.sum()),"endpoint_role_entropy":entropy(counts),
          "multi_role_endpoint_ratio":float(np.mean([len(roles[t])>=2 for t in ts])) if ts else 0.0,
          "incoming_host_degree":len(incoming[h]),"outgoing_host_degree":len(outgoing[h]),
          "reciprocal_host_ratio":reciprocal/max(len(outgoing[h]),1),
          "icc_ratio":float(ctx[0]),"reflection_ratio":float(ctx[1]),"non_direct_ratio":float(ctx[2]),
          "upstream_ratio":float(ctx[5]),"downstream_ratio":float(ctx[6])})
    return result,len(ids),len(src)
