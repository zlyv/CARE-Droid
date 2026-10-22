#!/usr/bin/env python3
from __future__ import annotations
import csv, gzip, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

ROLE_N=31; API_DIM=16; MIN_APK_DF=2; MAX_VOCAB=2048
ROLE_NAMES=("command_execution","dynamic_code_loading","native_library_loading","reflective_invocation","package_installation","package_uninstallation","device_administration","accessibility_control","sms_send_receive","call_control_or_recording","account_or_credential_access","contacts_or_calendar_access","device_identifier_access","location_access","microphone_access","camera_access","clipboard_access","network_transmission_or_upload","sensitive_file_or_content_write","boot_persistence","task_or_alarm_scheduling","background_service_control","overlay_or_ui_deception","icon_or_component_hiding","process_or_wakelock_control","cryptography","encoding_or_compression","url_or_http_setup","file_read","database_access","other")

def norm(v): return str(v or "").strip().casefold()
def write_csv(path,rows):
    rows=list(rows); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text("",encoding="utf-8"); return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)

def read_endpoint(path):
    out={}
    with gzip.open(path,"rt",encoding="utf-8") as h:
        for line in h:
            row=json.loads(line)
            if row.get("status")!="OK": continue
            key=norm(row.get("sha256")); payload=row.get("payload")
            if key and isinstance(payload,dict):
                if key in out and out[key]!=payload: raise RuntimeError("duplicate successful endpoint payload differs: "+key)
                out[key]=payload
    return out

def endpoint_hosts(payload):
    ids=list(payload.get("node_ids") or []); roles=list(payload.get("node_role_ids") or [])
    base=np.asarray(payload.get("node_base_features") or [],np.float32); edge=payload.get("edge_index") or [[],[]]
    if base.size==0: base=np.empty((0,15),np.float32)
    if base.ndim!=2 or base.shape[1]!=15 or len(ids)!=len(base) or len(roles)!=len(base) or len(edge)!=2: raise RuntimeError("endpoint payload schema mismatch")
    endpoints=set(map(int,np.flatnonzero(base[:,4]>.5))); calls=defaultdict(set)
    for s,t in zip(edge[0],edge[1]):
        if int(t) in endpoints: calls[int(s)].add(int(t))
    result=[]
    for host in sorted(calls):
        pairs=[]; merged=set()
        for endpoint in sorted(calls[host]):
            valid=sorted({int(r) for r in (roles[endpoint] or []) if 0<=int(r)<ROLE_N})
            merged.update(valid)
            for role in valid: pairs.append((str(ids[endpoint]),role))
        if merged: result.append({"host_node_index":host,"host_node_id":str(ids[host]),"pairs":tuple(sorted(set(pairs))),"roles":tuple(sorted(merged))})
    return result

class Registry:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            self.offset=np.asarray(z["api_offsets"],np.int64); self.api=np.asarray(z["api_ids"]).astype(str)
            self.role=np.asarray(z["api_roles"],np.int16); self.host_apk=np.asarray(z["host_apk_index"],np.int32)
        if len(self.offset)!=len(self.host_apk)+1 or len(self.api)!=len(self.role) or int(self.offset[-1])!=len(self.api): raise RuntimeError("role-API registry schema mismatch")
    def pairs(self,row_id):
        row_id=int(row_id)
        if row_id<0: return []
        a,b=int(self.offset[row_id]),int(self.offset[row_id+1]); return [(self.api[i],int(self.role[i])) for i in range(a,b)]

def bag_tokens(bags,registry):
    result=[]
    for bag in bags:
        apk=[]
        for row in np.asarray(bag):
            row_id=int(round(float(row[-1]))); by=[[] for _ in range(ROLE_N)]
            for api,role in registry.pairs(row_id): by[role].append(api)
            apk.append(by)
        result.append(apk)
    return result

def train_vocabulary(token_bags):
    df=Counter()
    for apk in token_bags: df.update({api for host in apk for role in host for api in role})
    kept=sorted(((api,n) for api,n in df.items() if n>=MIN_APK_DF),key=lambda x:(-x[1],x[0]))[:MAX_VOCAB]
    return {api:i for i,(api,_) in enumerate(kept)},dict(kept),df

def shuffled_within_role(token_bags,seed):
    rng=np.random.default_rng(int(seed)); out=[[ [list(v) for v in host] for host in apk] for apk in token_bags]
    slots=defaultdict(list); values=defaultdict(list)
    for ai,apk in enumerate(out):
        for hi,host in enumerate(apk):
            for role,tokens in enumerate(host):
                if tokens: slots[role].append((ai,hi,len(tokens))); values[role].extend(tokens)
    for role,items in slots.items():
        flat=np.asarray(values[role],dtype=object); flat=flat[rng.permutation(len(flat))].tolist(); cursor=0
        for ai,hi,n in items: out[ai][hi][role]=flat[cursor:cursor+n]; cursor+=n
    return out

def matrices(token_bags,vocab):
    rows=[]; cols=[]; data=[]; keys=[]; unknown=0; total=0; row=0
    for ai,apk in enumerate(token_bags):
        for hi,host in enumerate(apk):
            for role,tokens in enumerate(host):
                if not tokens: continue
                count=Counter(tokens); known=0
                for api,n in count.items():
                    total+=n
                    if api in vocab: rows.append(row); cols.append(vocab[api]); data.append(float(n)); known+=n
                    else: unknown+=n
                if len(tokens)-known: rows.append(row); cols.append(len(vocab)+role); data.append(float(len(tokens)-known))
                keys.append((ai,hi,role)); row+=1
    x=sparse.csr_matrix((data,(rows,cols)),shape=(row,len(vocab)+ROLE_N),dtype=np.float64)
    if x.shape[0] and x.shape[1]: x=normalize(x,norm="l2",axis=1,copy=False)
    return x,keys,unknown,total

def project(train_tokens,test_tokens,shuffle,seed):
    vocab,freq,df=train_vocabulary(train_tokens)
    if shuffle:
        train_tokens=shuffled_within_role(train_tokens,seed+11); test_tokens=shuffled_within_role(test_tokens,seed+29)
    xt,kt,_,_=matrices(train_tokens,vocab); xv,kv,unk,total=matrices(test_tokens,vocab)
    ncomp=min(API_DIM,max(xt.shape[0]-1,0),max(xt.shape[1]-1,0))
    svd=None
    if ncomp>=1 and xt.nnz:
        svd=TruncatedSVD(n_components=ncomp,random_state=17031); svd.fit(xt)
    def make(tokens,x,keys):
        out=[np.zeros((len(apk),ROLE_N,API_DIM),np.float32) for apk in tokens]
        emb=np.zeros((len(keys),API_DIM),np.float32)
        if svd is not None and len(keys): emb[:,:ncomp]=svd.transform(x).astype(np.float32)
        for value,(ai,hi,role) in zip(emb,keys): out[ai][hi,role]=value
        return out
    ordered=sorted(vocab,key=vocab.get); vh=hashlib.sha256("\n".join(ordered).encode()).hexdigest()
    return make(train_tokens,xt,kt),make(test_tokens,xv,kv),{"vocabulary_size":len(vocab),"vocabulary_sha256":vh,"svd_components":ncomp,"test_unknown_occurrences":int(unk),"test_api_occurrences":int(total),"test_unknown_rate":float(unk/max(total,1)),"singleton_train_apis":int(sum(n==1 for n in df.values())),"train_api_ge2":int(sum(n>=2 for n in df.values())),"train_api_ge5":int(sum(n>=5 for n in df.values())),"train_api_ge10":int(sum(n>=10 for n in df.values())),"_tokens":ordered,"_frequency":{api:int(freq[api]) for api in ordered}}
