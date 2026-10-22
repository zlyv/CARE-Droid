#!/usr/bin/env python3
from __future__ import annotations
import gzip,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize

DIM=32
ROLE_NAMES=("command_execution","dynamic_code_loading","native_library_loading","reflective_invocation","package_installation","package_uninstallation","device_administration","accessibility_control","sms_send_receive","call_control_or_recording","account_or_credential_access","contacts_or_calendar_access","device_identifier_access","location_access","microphone_access","camera_access","clipboard_access","network_transmission_or_upload","sensitive_file_or_content_write","boot_persistence","task_or_alarm_scheduling","background_service_control","overlay_or_ui_deception","icon_or_component_hiding","process_or_wakelock_control","cryptography","encoding_or_compression","url_or_http_setup","file_read","database_access","other")
ROLE_INDEX={name:i for i,name in enumerate(ROLE_NAMES)}
def norm(v):return str(v or "").strip().casefold()
def hashed(tokens,dim=DIM):
    tokens=list(tokens)
    if not tokens:return np.zeros(dim,np.float32)
    h=FeatureHasher(n_features=dim,input_type="string",alternate_sign=True)
    x=normalize(h.transform([tokens]),norm="l2",copy=False).toarray()[0]
    return np.asarray(x,np.float32)
def deterministic_shuffle(n,sha):
    rng=np.random.default_rng(int(hashlib.sha256((sha+"|host-call-null").encode()).hexdigest()[:16],16))
    return rng.permutation(n)
def sha_alias(reference,length):return hashlib.sha256(norm(reference).encode()).hexdigest()[:length]
def endpoint_stable_id(reference):
    """Exact endpoint_centered_view._stable_id contract (unsigned uint64)."""
    digest=hashlib.sha256(str(reference).encode("utf-8")).digest()
    return str(int.from_bytes(digest[:8],byteorder="big",signed=False))
def read_endpoint(path):
    out={};skipped=0;duplicates=0
    with gzip.open(path,"rt",encoding="utf-8") as h:
        for line in h:
            r=json.loads(line)
            # Match the already-validated API-identity reader: only successful
            # cache records are admissible. Retry/error wrappers can share a SHA
            # and must never overwrite a valid payload.
            if r.get("status")!="OK":skipped+=1;continue
            key=norm(r.get("sha256"));payload=r.get("payload")
            if not key or not isinstance(payload,dict):skipped+=1;continue
            if key in out:
                duplicates+=1
                if out[key]!=payload:raise RuntimeError(f"duplicate successful endpoint payload differs for {key}")
                continue
            out[key]=payload
    if not out:raise RuntimeError("no successful endpoint payloads loaded")
    return out
def endpoint_hosts(payload):
    ids=list(payload.get("node_ids") or []);roles=list(payload.get("node_role_ids") or []);base=np.asarray(payload.get("node_base_features") or [],np.float32);edge=payload.get("edge_index") or [[],[]]
    # A valid endpoint-cache record may represent an APK with no recovered
    # graph nodes. Frozen C43 stores one zero placeholder for it, while its
    # real_host_count remains zero. Normalize that legitimate empty payload.
    if base.size==0:base=np.empty((0,15),np.float32)
    if base.ndim!=2 or base.shape[1]!=15 or len(ids)!=len(base) or len(roles)!=len(base) or len(edge)!=2:
        raise RuntimeError(f"endpoint payload schema mismatch ids={len(ids)} roles={len(roles)} base_shape={base.shape} edge_outer={len(edge)} keys={sorted(payload)}")
    if len(base)==0:
        if len(edge[0]) or len(edge[1]):raise RuntimeError("empty endpoint node table has nonempty edges")
        return []
    endpoints=set(map(int,np.flatnonzero(base[:,4]>.5)));calls=defaultdict(set)
    for s,t in zip(edge[0],edge[1]):
        if int(t) in endpoints:calls[int(s)].add(int(t))
    out=[]
    for host in sorted(calls):
        merged=sorted({int(r) for t in calls[host] for r in (roles[t] or []) if 0<=int(r)<31})
        if merged:out.append({"node_index":host,"node_id":str(ids[host]),"api_ids":tuple(sorted({str(ids[t]) for t in calls[host]})),"primary_role":merged[0]})
    return out
def load_graph(path):
    with gzip.open(path,"rt",encoding="utf-8") as h:return json.load(h)
def haig_hosts(graph,endpoint_api_ids):
    nodes=graph.get("nodes",[]);by_node={int(n["node_id"]):n for n in nodes};by_host=defaultdict(list)
    for n in nodes:by_host[str(n["host_id"])].append(n)
    # Endpoint IDs are decimal JSON encodings of the first eight SHA-256 bytes,
    # not variable-width hexadecimal hashes. Reproduce the frozen extractor's
    # exact unsigned-64-bit identity contract.
    universe={str(v) for v in endpoint_api_ids}
    def endpoint_token(node):
        token=endpoint_stable_id(node["canonical_api"])
        return token if token in universe else "__UNMAPPED__:"+token
    incident=defaultdict(list)
    for e in graph.get("host_call_edges",[]):
        left=by_node[int(e["src_node"])];right=by_node[int(e["dst_node"])]
        lf=left.get("api_family","unknown");rf=right.get("api_family","unknown")
        # Interaction features use a fixed canonical 32-hex identity. Endpoint
        # IDs are needed only to align the Host with the frozen C43 cache.
        la=sha_alias(left["canonical_api"],32);ra=sha_alias(right["canonical_api"],32)
        incident[str(left["host_id"])].extend((f"out:api:{la}>{ra}",f"out:fam:{lf}>{rf}"))
        incident[str(right["host_id"])].extend((f"in:api:{la}>{ra}",f"in:fam:{lf}>{rf}"))
    out=[]
    for hid,block in by_host.items():
        api=tuple(sorted({endpoint_token(n) for n in block}));tokens=[]
        for n in block:tokens.extend(("api:"+sha_alias(n["canonical_api"],32),"fam:"+n.get("api_family","unknown")))
        roles=[ROLE_INDEX.get(str(n.get("primary_role","other")),-1) for n in block]
        roles=[r for r in roles if r>=0]
        # HAIG host_id is SHA256(signature)[:32] in hexadecimal. The frozen
        # endpoint cache stores the same digest's first eight bytes as an
        # unsigned decimal uint64, so this conversion is an exact identity
        # bridge and does not expose or guess raw method signatures.
        endpoint_host_id=str(int(hid[:16],16))
        out.append({"host_id":hid,"endpoint_host_id":endpoint_host_id,"api_ids":api,"api_vec":hashed(tokens),"call_vec":hashed(incident.get(hid,[])),"primary_role":min(roles) if roles else -1})
    return out
def align_one(payload,graph):
    eh=endpoint_hosts(payload);endpoint_api_ids={x for h in eh for x in h["api_ids"]}
    hh=haig_hosts(graph,endpoint_api_ids);by=defaultdict(list)
    for i,h in enumerate(hh):by[h["endpoint_host_id"]].append(i)
    used=set();api=[];call=[];role=[];matched=0;ambiguous=0;fingerprint_equal=0
    for h in eh:
        cand=[i for i in by.get(str(h["node_id"]),[]) if i not in used]
        if len(cand)==1:
            j=cand[0];used.add(j);api.append(hh[j]["api_vec"]);call.append(hh[j]["call_vec"]);role.append(h["primary_role"]);matched+=1;fingerprint_equal+=int(hh[j]["api_ids"]==h["api_ids"])
        else:
            api.append(np.zeros(DIM,np.float32));call.append(np.zeros(DIM,np.float32));role.append(h["primary_role"]);ambiguous+=1
    return np.asarray(api,np.float32),np.asarray(call,np.float32),np.asarray(role,int),{"endpoint_hosts":len(eh),"haig_hosts":len(hh),"matched":matched,"ambiguous_or_unmatched":ambiguous,"matched_api_fingerprint_equal":fingerprint_equal}

