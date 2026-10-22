#!/usr/bin/env python3
from __future__ import annotations
import gzip,hashlib,json,math
from collections import Counter
from pathlib import Path
import joblib,numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,f1_score,roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler,normalize as sparse_normalize

ROLE_N=31;CHANNEL_DIM=32;HASH_DIM=4096;EXACT_MIN_DF=2
ARMS=("H0_ROLE","H1_ROLE_API","H2_SAME_HOST","H3_HOST_CALL","H2_SHUFFLE","H3_SHUFFLE","COUNT_ONLY")
def norm(v):return str(v or "").strip().casefold()
def read_csv(path):
    import csv
    with Path(path).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write_csv(path,rows):
    import csv
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def load_graphs(lineage,root):
    graphs=[];fail=[]
    for row in lineage:
        sha=norm(row["sha256"]);path=Path(root)/f"{sha}.json.gz"
        try:
            with gzip.open(path,"rt",encoding="utf-8") as h:g=json.load(h)
            if norm(g.get("sha256"))!=sha:raise RuntimeError("SHA mismatch")
        except Exception as exc:g={"sha256":sha,"status":"ERROR","nodes":[],"same_host_edges":[],"host_call_edges":[],"stats":{}};fail.append({"sha256":sha,"error_type":type(exc).__name__,"error_message":str(exc)})
        if g.get("status")!="OK":fail.append({"sha256":sha,"error_type":g.get("error_type","EXTRACTION_ERROR"),"error_message":g.get("error_message","")})
        graphs.append(g)
    return graphs,fail
def role_matrix(graphs):
    x=np.zeros((len(graphs),ROLE_N),float)
    for i,g in enumerate(graphs):
        nodes=g.get("nodes",[])
        for n in nodes:
            for r in n.get("roles",[]):
                try:j=ROLE_NAMES.index(r);x[i,j]+=1
                except ValueError:pass
        if len(nodes):x[i]/=len(nodes)
    return x
ROLE_NAMES=("command_execution","dynamic_code_loading","native_library_loading","reflective_invocation","package_installation","package_uninstallation","device_administration","accessibility_control","sms_send_receive","call_control_or_recording","account_or_credential_access","contacts_or_calendar_access","device_identifier_access","location_access","microphone_access","camera_access","clipboard_access","network_transmission_or_upload","sensitive_file_or_content_write","boot_persistence","task_or_alarm_scheduling","background_service_control","overlay_or_ui_deception","icon_or_component_hiding","process_or_wakelock_control","cryptography","encoding_or_compression","url_or_http_setup","file_read","database_access","other")
def bin_gap(v):
    v=max(0,int(v));return "0" if v==0 else ("1_4" if v<=4 else ("5_16" if v<=16 else ("17_64" if v<=64 else "65p")))
def graph_tokens(g,channel,shuffle=False):
    nodes=g.get("nodes",[]);by={int(n["node_id"]):n for n in nodes}
    semantic={k:(v.get("api_hash","UNK"),v.get("api_family","unknown"),v.get("primary_role","other")) for k,v in by.items()}
    # Relation null: preserve topology, node/edge counts and the APK-level API
    # marginal, while breaking which API/family/role tuple occupies each endpoint.
    if shuffle and len(semantic)>1:
        ids=sorted(semantic);values=[semantic[i] for i in ids];seed=int(hashlib.sha256((g.get("sha256","")+channel).encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed);rng.shuffle(values);semantic=dict(zip(ids,values))
    if channel=="api":
        out=[]
        for n in nodes:
            out.extend(("api:"+n.get("api_hash","UNK"),"fam:"+n.get("api_family","unknown")))
        return out
    edges=g.get("same_host_edges",[]) if channel=="same" else g.get("host_call_edges",[]);prefix="same" if channel=="same" else "call";out=[]
    for e in edges:
        sa,sf,sr=semantic.get(int(e["src_node"]),("UNK","unknown","other"));ta,tf,tr=semantic.get(int(e["dst_node"]),("UNK","unknown","other"))
        out.append(f"{prefix}:api:{sa}>{ta}");out.append(f"{prefix}:fam:{sf}>{tf}");out.append(f"{prefix}:role:{sr}>{tr}")
        if channel=="same":out.append(f"same:gap:{bin_gap(e.get('instruction_gap',0))}:mid:{bin_gap(e.get('intermediate_invoke_count',0))}")
    return out
def count_matrix(graphs):
    rows=[]
    for g in graphs:
        s=g.get("stats",{});n=max(1,float(s.get("num_api_nodes",0)));rows.append([math.log1p(s.get("num_api_nodes",0)),math.log1p(s.get("num_hosts",0)),math.log1p(s.get("num_same_host_edges",0)),math.log1p(s.get("num_host_call_edges",0)),math.log1p(s.get("unique_api_count",0)),math.log1p(s.get("unique_pair_count",0)),float(s.get("num_same_host_edges",0))/n,float(s.get("num_host_call_edges",0))/n])
    return np.asarray(rows,float)
class ChannelProjector:
    def __init__(self,seed):self.seed=seed;self.hasher=FeatureHasher(n_features=HASH_DIM,input_type="string",alternate_sign=False);self.svd=None;self.components=0;self.allowed_exact=set()
    @staticmethod
    def is_exact(token):return token.startswith("api:") or ":api:" in token
    def clean(self,tokens):return [[t for t in row if not self.is_exact(t) or t in self.allowed_exact] for row in tokens]
    def fit(self,tokens):
        df=Counter(t for row in tokens for t in set(row) if self.is_exact(t));self.allowed_exact={t for t,n in df.items() if n>=EXACT_MIN_DF};x=sparse_normalize(self.hasher.transform(self.clean(tokens)),norm="l2",copy=False);maximum=min(CHANNEL_DIM,max(x.shape[0]-1,0),max(x.shape[1]-1,0))
        if maximum>0 and x.nnz:self.svd=TruncatedSVD(n_components=maximum,random_state=self.seed).fit(x);self.components=maximum
        return self
    def transform(self,tokens):
        out=np.zeros((len(tokens),CHANNEL_DIM),float)
        if self.svd is not None:out[:,:self.components]=self.svd.transform(sparse_normalize(self.hasher.transform(self.clean(tokens)),norm="l2",copy=False))
        return out
    def backoff_rate(self,tokens):
        exact=[t for row in tokens for t in row if self.is_exact(t)];return float(np.mean([t not in self.allowed_exact for t in exact])) if exact else 0.0
class Encoder:
    def __init__(self,seed):self.seed=seed;self.projectors={}
    def fit(self,graphs):
        for j,ch in enumerate(("api","same","call")):self.projectors[ch]=ChannelProjector(self.seed+j).fit([graph_tokens(g,ch) for g in graphs])
        return self
    def blocks(self,graphs,shuffle_same=False,shuffle_call=False):
        return {"role":role_matrix(graphs),"api":self.projectors["api"].transform([graph_tokens(g,"api") for g in graphs]),"same":self.projectors["same"].transform([graph_tokens(g,"same",shuffle_same) for g in graphs]),"call":self.projectors["call"].transform([graph_tokens(g,"call",shuffle_call) for g in graphs]),"count":count_matrix(graphs)}
    def vocabulary_audit(self,graphs):
        out={}
        for ch,p in self.projectors.items():
            tok=[graph_tokens(g,ch) for g in graphs];ordered=sorted(p.allowed_exact);out[ch]={"size":len(ordered),"sha256":hashlib.sha256("\n".join(ordered).encode()).hexdigest(),"backoff_rate":p.backoff_rate(tok),"tokens":ordered}
        return out
def arm_matrix(blocks,arm):
    z=np.zeros_like(blocks["api"]);same=blocks["same"];call=blocks["call"]
    if arm=="H0_ROLE":return np.c_[blocks["role"],z,z,z]
    if arm=="H1_ROLE_API":return np.c_[blocks["role"],blocks["api"],z,z]
    if arm in ("H2_SAME_HOST","H2_SHUFFLE"):return np.c_[blocks["role"],blocks["api"],same,z]
    if arm in ("H3_HOST_CALL","H3_SHUFFLE"):return np.c_[blocks["role"],blocks["api"],same,call]
    if arm=="COUNT_ONLY":return blocks["count"]
    raise KeyError(arm)
class Ensemble:
    def __init__(self,arm,seeds=(31001,31002,31003)):self.arm=arm;self.seeds=seeds;self.scalers=[];self.models=[]
    def fit(self,x,y):
        if x.ndim!=2 or len(x)!=len(y) or not np.all(np.isfinite(x)):raise RuntimeError("BLOCKED_BY_CONTRACT: invalid training matrix")
        for seed in self.seeds:
            sc=StandardScaler().fit(x);xs=sc.transform(x)
            if self.arm=="COUNT_ONLY":m=LogisticRegression(C=1,class_weight="balanced",solver="liblinear",max_iter=5000,random_state=seed)
            else:m=MLPClassifier(hidden_layer_sizes=(64,16),activation="relu",alpha=.01,batch_size=min(64,len(x)),learning_rate_init=.001,max_iter=500,early_stopping=True,validation_fraction=.15,n_iter_no_change=30,random_state=seed)
            m.fit(xs,y);self.scalers.append(sc);self.models.append(m)
        return self
    def predict(self,x):
        if x.ndim!=2 or not np.all(np.isfinite(x)):raise RuntimeError("BLOCKED_BY_CONTRACT: invalid inference matrix")
        p=np.mean([m.predict_proba(s.transform(x))[:,1] for s,m in zip(self.scalers,self.models)],axis=0)
        if not np.all(np.isfinite(p)):raise RuntimeError("BLOCKED_BY_CONTRACT: NaN/Inf prediction")
        return p
def fit_bundle(graphs,y,seed):
    enc=Encoder(seed).fit(graphs);normal=enc.blocks(graphs);ss=enc.blocks(graphs,shuffle_same=True);cs=enc.blocks(graphs,shuffle_call=True);models={}
    for arm in ARMS:
        b=ss if arm=="H2_SHUFFLE" else (cs if arm=="H3_SHUFFLE" else normal);models[arm]=Ensemble(arm).fit(arm_matrix(b,arm),y)
    return {"encoder":enc,"models":models}
def predict_bundle(bundle,graphs):
    enc=bundle["encoder"];normal=enc.blocks(graphs);ss=enc.blocks(graphs,shuffle_same=True);cs=enc.blocks(graphs,shuffle_call=True);out={};norms={}
    for arm,m in bundle["models"].items():
        b=ss if arm=="H2_SHUFFLE" else (cs if arm=="H3_SHUFFLE" else normal);x=arm_matrix(b,arm);out[arm]=m.predict(x);norms[arm]=np.linalg.norm(x,axis=1)
    return out,norms
def metric(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if len(y)!=len(p) or not len(y):raise RuntimeError(f"BLOCKED_BY_CONTRACT: metric shape mismatch y={len(y)} p={len(p)}")
    if not np.all(np.isfinite(p)):raise RuntimeError("BLOCKED_BY_CONTRACT: metric received NaN/Inf prediction")
    return {"BA":float(balanced_accuracy_score(y,p>=.5)),"AUC":float(roc_auc_score(y,p)) if len(set(y))==2 else 0.5,"F1":float(f1_score(y,p>=.5,zero_division=0))}
def direction(y,base,route,p):
    err=route&((base>=.5)!=y);fn=err&(y==1);fp=err&(y==0);correct=(p>=.5)==y
    return {"routed_errors":int(err.sum()),"FN":int(fn.sum()),"FP":int(fp.sum()),"overall_direction_acc":float(np.mean(correct[err])) if err.any() else 0.0,"FN_direction_acc":float(np.mean(correct[fn])) if fn.any() else 0.0,"FP_direction_acc":float(np.mean(correct[fp])) if fp.any() else 0.0,"wrong_direction":int(np.sum(err&~correct)),"correct_direction":int(np.sum(err&correct))}
