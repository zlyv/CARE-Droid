#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path("artifacts/care-local-role-neighborhood-v1")
BASE=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv")
CACHE=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz")
METHOD="P31RF__RFRoute__A0_ROLE_ONLY__Bounded"
K=5
MIN_SUPPORT=5
CONTROL_REPETITIONS=20
NEAR_DUPLICATE_CUTOFF=.99
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
def unit(x,axis=-1):
    x=np.asarray(x,float);den=np.linalg.norm(x,axis=axis,keepdims=True)
    return np.divide(x,den,out=np.zeros_like(x),where=den>1e-12)
def logit(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8);return np.log(p/(1-p))

