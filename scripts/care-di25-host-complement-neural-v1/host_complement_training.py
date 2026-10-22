#!/usr/bin/env python3
from __future__ import annotations
import gzip,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from host_position_encoder import structural_vector,POSITION_DIM

RAW_DIM=74;META_DIM=3
def norm(x):return str(x or "").strip().lower()

def pack(host,roles,position,edges):
    host=np.asarray(host,np.float32);roles=np.asarray(roles,np.float32);position=np.asarray(position,np.float32);width=host.shape[1]+roles.shape[1]+position.shape[1]
    out=np.zeros((len(host)+len(edges),width+3),np.float32);out[:len(host),:host.shape[1]]=host;out[:len(host),host.shape[1]:host.shape[1]+roles.shape[1]]=roles;out[:len(host),host.shape[1]+roles.shape[1]:width]=position
    for k,(s,t) in enumerate(edges,len(host)):out[k,width]=1;out[k,width+1]=s;out[k,width+2]=t
    return out
def unpack(values,base_dim,role_count):
    values=np.asarray(values,np.float32);width=values.shape[1]-3;mask=values[:,width]<.5;host=values[mask,:width];er=values[~mask]
    edges=[(int(round(r[width+1])),int(round(r[width+2]))) for r in er]
    return host[:,:base_dim],host[:,base_dim:base_dim+role_count],host[:,base_dim+role_count:base_dim+role_count+POSITION_DIM],edges

def make_fit_adapter(original_fit,role_count):
    def fit(train_bags,test_bags,labels,args,seed):
        tr=[unpack(x,RAW_DIM,role_count) for x in train_bags];te=[unpack(x,RAW_DIM,role_count) for x in test_bags]
        train_aug,test_aug,audit=original_fit([x[0] for x in tr],[x[0] for x in te],labels,args,seed)
        train=[pack(a,x[1],x[2],x[3]) for a,x in zip(train_aug,tr)];test=[pack(a,x[1],x[2],x[3]) for a,x in zip(test_aug,te)]
        audit=dict(audit);audit.update({"complement_changes_anomaly_fit":False,"anomaly_fit_input_dim":RAW_DIM})
        return train,test,audit
    return fit

def load_extended_bags(expected,host_path,ego_path,endpoint_path,manifest_path):
    manifest=json.loads(Path(manifest_path).read_text(encoding="utf-8"));role_names=list(manifest["model_role_order"]);remap=np.asarray(manifest["payload_to_model_role"],int);R=len(role_names)
    payloads={}
    with gzip.open(endpoint_path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row=json.loads(line);payloads[norm(row["sha256"])]=row["payload"]
    with np.load(host_path,allow_pickle=False) as h,np.load(ego_path,allow_pickle=False) as e:
        ids=[norm(x) for x in h["sha256"].astype(str)];eids=[norm(x) for x in e["sha256"].astype(str)];lookup={s:i for i,s in enumerate(ids)};elookup={s:i for i,s in enumerate(eids)}
        hv=np.asarray(h["host_values"],np.float32);ho=np.asarray(h["host_offsets"],int);real=np.asarray(h["real_host_count"],int);eo=np.asarray(e["ego_offsets"],int);hid=e["host_ids"].astype(str)
        ino=np.asarray(e["incoming_offsets"],int);outo=np.asarray(e["outgoing_offsets"],int);inp=np.asarray(e["incoming_host_positions"],int);outp=np.asarray(e["outgoing_host_positions"],int);status=np.asarray(e["ego_status"],int)
        bags=[];usage=[]
        for sha in expected:
            i,ei=lookup[sha],elookup[sha];hb,he=ho[i],ho[i+1];eb,ee=eo[ei],eo[ei+1];host=hv[hb:he];n=int(real[i]);edges=set()
            for g in range(eb,ee):
                for t in outp[outo[g]:outo[g+1]]:edges.add((g-eb,int(t)-eb))
            roles=np.zeros((len(host),R),np.float32);pos=np.zeros((len(host),POSITION_DIM),np.float32);multi=0;duplicates=0
            if n:
                p=payloads[sha];nodes=[str(x) for x in p["node_ids"]];idx={x:k for k,x in enumerate(nodes)};base=np.asarray(p["node_base_features"],float);rawroles=p["node_role_ids"];src,dst=map(lambda x:list(map(int,x)),p["edge_index"]);edge=set(zip(src,dst));endpoints=set(np.flatnonzero(base[:,4]>.5).astype(int));targets=defaultdict(set)
                for s,t in edge:
                    if t in endpoints:targets[s].add(t)
                hostnodes=set(targets);ordinary=set(range(len(nodes)))-hostnodes-endpoints;incoming=defaultdict(set);outgoing=defaultdict(set)
                for s,t in edge:
                    incoming[t].add(s);outgoing[s].add(t)
                for j,name in enumerate(hid[eb:ee]):
                    node=idx[str(name)];rr=[]
                    for endpoint in targets[node]:rr.extend(int(remap[int(v)]) for v in rawroles[endpoint])
                    duplicates+=len(rr)-len(set(rr));unique=sorted(set(rr));roles[j,unique]=1;multi+=int(len(unique)>=2)
                    hp=len(incoming[node]&hostnodes);hs=len(outgoing[node]&hostnodes);op=len(incoming[node]&ordinary);os=len(outgoing[node]&ordinary)
                    pos[j]=structural_vector(hp,hs,op,os,len(incoming[node]),len(outgoing[node]))
            flags=pos[:,7:11] if n else np.zeros((0,4),np.float32)
            bags.append(pack(host,roles,pos,sorted(edges)));usage.append({
                "sha256":sha,"real_host_count":n,"multi_role_host_count":multi,
                "duplicate_role_calls":duplicates,"nontrivial_ego_count":int(np.sum(status[eb:ee]==1)),
                "source_like_host_count":int(flags[:,0].sum()) if len(flags) else 0,
                "sink_like_host_count":int(flags[:,1].sum()) if len(flags) else 0,
                "bridge_like_host_count":int(flags[:,2].sum()) if len(flags) else 0,
                "isolated_like_host_count":int(flags[:,3].sum()) if len(flags) else 0,
            })
    return bags,usage,role_names
