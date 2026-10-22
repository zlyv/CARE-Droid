#!/usr/bin/env python3
"""HDBC-V1: read existing endpoint graph caches, not APKs; write nowhere (caller writes HDBC output only)."""
from __future__ import annotations
import gzip,hashlib,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

ROLE_N=31
DIRECT_N=2*ROLE_N*ROLE_N
UNDIR_N=2*(ROLE_N*(ROLE_N+1)//2)

def cache_rows(path):
    out={}
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row.get('status')!='OK':continue
            sid=str(row['sha256']).lower()
            if sid in out:raise RuntimeError('duplicate successful graph '+sid)
            out[sid]=row
    return out

def undirected_index(a,b):
    a,b=sorted((int(a),int(b)))
    return a*ROLE_N-a*(a-1)//2+(b-a)

def extract_one(sid,payload,shuffled=False):
    """Only DIRECT_CALL edges; true caller->callee paths, at most two edges.

    The H5 null reverses every cached direct edge. It preserves the undirected
    graph and Role multiset, not in/out-degree; it is NEVER program evidence.
    """
    ids=payload.get('node_ids') or []
    roles=payload.get('node_role_ids') or []
    base=payload.get('node_base_features') or []
    ei=payload.get('edge_index') or [[],[]]
    ef=payload.get('edge_features') or []
    n=len(ids)
    if len(roles)!=n or len(base)!=n or len(ei)!=2 or len(ei[0])!=len(ei[1]) or len(ef)!=len(ei[0]):
        raise RuntimeError('invalid endpoint graph schema '+sid)
    endpoint={i for i,b in enumerate(base) if len(b)==15 and float(b[4])>.5}
    edges=set()
    for a,b,f in zip(ei[0],ei[1],ef):
        a=int(a);b=int(b)
        if not (0<=a<n and 0<=b<n and len(f)==7):raise RuntimeError('bad edge '+sid)
        if float(f[2])==0.:edges.add((a,b))
    if shuffled:edges={(b,a) for a,b in edges}
    outgoing=defaultdict(set)
    for a,b in edges:outgoing[a].add(b)
    # A Host is an ordinary method directly calling a Role-bearing endpoint.
    # H5 must preserve the ORIGINAL Host and Role assignments; only relation
    # direction is reversed. Do not derive Host identity from synthetic edges.
    original_edges=set()
    for a,b,f in zip(ei[0],ei[1],ef):
        if float(f[2])==0.:original_edges.add((int(a),int(b)))
    hostroles=defaultdict(set)
    for a,b in original_edges:
        if b in endpoint and a not in endpoint:
            hostroles[a].update(int(r) for r in roles[b] if 0<=int(r)<ROLE_N)
    hosts={h for h,r in hostroles.items() if r}
    pairs=set();proof=[];h1=0;h2=0
    for h in sorted(hosts):
        for mid in outgoing.get(h,()):
            if mid in hosts and mid!=h:
                pairs.add((h,mid,1,mid))
            for target in outgoing.get(mid,()):
                if target in hosts and target not in (h,mid):
                    pairs.add((h,target,2,mid))
    # One witness per Host pair/hop, deterministic hashed-ID tie-break.
    chosen={}
    for a,b,hop,mid in sorted(pairs,key=lambda x:(x[0],x[1],x[2],str(ids[x[3]]))):
        chosen.setdefault((a,b,hop),(a,b,hop,mid))
    directed=Counter();undirected=Counter()
    for a,b,hop,mid in chosen.values():
        if hop==1:h1+=1
        else:h2+=1
        for ra in sorted(hostroles[a]):
            for rb in sorted(hostroles[b]):
                directed[(hop-1)*ROLE_N*ROLE_N+ra*ROLE_N+rb]+=1
                undirected[(hop-1)*(ROLE_N*(ROLE_N+1)//2)+undirected_index(ra,rb)]+=1
                if not shuffled:
                    proof.append(dict(sha256=sid,source_host_id=str(ids[a]),target_host_id=str(ids[b]),
                        source_role=ra,target_role=rb,hop=hop,intermediate_method_id=str(ids[mid]) if hop==2 else '',
                        edge_source='endpoint_centered_cache DIRECT_CALL only',synthetic_null=0))
    xdir=np.zeros(DIRECT_N,np.float32);xund=np.zeros(UNDIR_N,np.float32)
    denom=max(len(hosts),1)
    for k,v in directed.items():xdir[k]=np.log1p(v)/np.sqrt(denom)
    for k,v in undirected.items():xund[k]=np.log1p(v)/np.sqrt(denom)
    diag=np.asarray([np.log1p(len(hosts)),np.log1p(len(edges)),
        h1/max(len(hosts)*(len(hosts)-1),1),h2/max(len(hosts)*(len(hosts)-1),1)],np.float32)
    coverage=dict(sha256=sid,graph_available=1,node_count=n,direct_method_edges=len(edges),
        host_count=len(hosts),one_hop_host_pairs=h1,two_hop_host_pairs=h2,
        usable_relation=int(h1+h2>0),endpoint_count=len(endpoint))
    return xdir,xund,diag,coverage,proof

def select_vocab(matrix,train,width=128):
    support=np.count_nonzero(matrix[train]>0,axis=0)
    # No labels and no outer validation APKs participate in selection.
    order=np.lexsort((np.arange(matrix.shape[1]),-support))
    chosen=np.asarray([i for i in order if support[i]>0][:width],int)
    if len(chosen)<width:
        fill=[i for i in range(matrix.shape[1]) if i not in set(chosen)]
        chosen=np.r_[chosen,np.asarray(fill[:width-len(chosen)],int)]
    return chosen,support[chosen]

def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def selftest():
    base=[[0.]*15 for _ in range(7)]
    for i in (4,5,6):base[i][4]=1.
    edges=[(0,4),(1,5),(2,6),(0,1),(0,3),(3,2)]
    payload=dict(node_ids=list(range(7)),node_role_ids=[[],[],[],[],[0],[1],[2]],
        node_base_features=base,edge_index=[[a for a,b in edges],[b for a,b in edges]],
        edge_features=[[0.]*7 for _ in edges])
    d,u,_,c,proof=extract_one('toy',payload)
    assert c['host_count']==3 and c['one_hop_host_pairs']==1 and c['two_hop_host_pairs']==1
    assert d[0*961+0*31+1]>0 and d[1*961+0*31+2]>0
    assert d[0*961+1*31+0]==0 and d[0*961+0*31+0]==0
    assert any(r['hop']==2 and r['intermediate_method_id']=='3' for r in proof)
    rev,_,_,cr,_=extract_one('toy',payload,shuffled=True)
    assert cr['host_count']==3 and rev[0*961+1*31+0]>0 and rev[0*961+0*31+1]==0
    return True
