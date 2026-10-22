#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path("artifacts/care-droid-formal-v1")
SOURCE=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv")
CACHE=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz")
METHOD="P31RF__RFRoute__A0_ROLE_ONLY__Bounded"
BASE_FIELD="Global24__prob" # legacy column name; formal provenance is P31-RF
ROUTE_FIELD="A25__routed"
ROUTER_SCORE_FIELD="A25__score"
RANDOM_SEEDS=(20271001,20271002,20271003,20271004,20271005)
BUDGETS=(.10,.15,.20,.25,.30,.40,.50,1.00)
MAX_DELTA=2.25
ROLE_NAMES=("command_execution","dynamic_code_loading","native_library_loading","reflective_invocation","package_installation","package_uninstallation","device_administration","accessibility_control","sms_send_receive","call_control_or_recording","account_or_credential_access","contacts_or_calendar_access","device_identifier_access","location_access","microphone_access","camera_access","clipboard_access","network_transmission_or_upload","sensitive_file_or_content_write","boot_persistence","task_or_alarm_scheduling","background_service_control","overlay_or_ui_deception","icon_or_component_hiding","process_or_wakelock_control","cryptography","encoding_or_compression","url_or_http_setup","file_read","database_access","other")

def norm(x):return str(x).strip().casefold()
def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as handle:return list(csv.DictReader(handle))
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def sha256_text(value):return hashlib.sha256(value.encode()).hexdigest()
def binary_auc(y,score):
    y=np.asarray(y,int);score=np.asarray(score,float);pos=int(y.sum());neg=len(y)-pos
    if pos==0 or neg==0:return float("nan")
    order=np.argsort(score,kind="mergesort");ranks=np.empty(len(score),float);sorted_score=score[order];start=0
    while start<len(score):
        end=start+1
        while end<len(score) and sorted_score[end]==sorted_score[start]:end+=1
        ranks[order[start:end]]=(start+1+end)/2;start=end
    return float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg))
def metrics(y,base,p,route):
    y=np.asarray(y,int);base=np.asarray(base,float);p=np.asarray(p,float);route=np.asarray(route,bool);bp=base>=.5;pp=p>=.5;base_ok=bp==y;ok=pp==y
    tn=int(np.sum((y==0)&(~pp)));fp=int(np.sum((y==0)&pp));fn=int(np.sum((y==1)&(~pp)));tp=int(np.sum((y==1)&pp));tpr=tp/max(tp+fn,1);tnr=tn/max(tn+fp,1);f1=2*tp/max(2*tp+fp+fn,1)
    fixed=(~base_ok)&ok;broken=base_ok&(~ok)
    return {"BA":float((tpr+tnr)/2),"AUC":binary_auc(y,p),"F1":float(f1),"TN":tn,"FP":fp,"FN":fn,"TP":tp,"wrong":int(fp+fn),"fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),"routed":int(route.sum()),"prediction_class_flips":int(np.sum(bp!=pp))}
def deterministic_route(indices,sha,count,seed):
    indices=np.asarray(indices,int);rank=sorted(indices.tolist(),key=lambda i:(sha256_text(f"{seed}:{sha[i]}"),sha[i]));chosen=np.asarray(rank[:count],int);out=np.zeros(len(sha),bool);out[chosen]=True;return out

