#!/usr/bin/env python3
from common import ROOT,dump
dump(ROOT/"protocol.json",{"experiment":"CARE-Droid Role Evidence Reliability / Correction Acceptance Gate v1","source":{"year":2017,"rows":799,"folds":[160,160,160,160,159]},"fixed_line":{"stage1":"P31 calibrated RF","router":"RF-anchored MeanOpposition DI25 Top25%","stage2":"A0 Role-only C43 Relation-Query","max_delta":2.25,"threshold":0.5},"gate_target":"FIXED=1 vs BROKEN=0, learned only on prediction-changed strict nested inner-OOF candidates","arms":{"G0":"current correction","G1":"Stage-I uncertainty/router only","G2":"label-free Role reliability only","G3":"G1+G2","C1":"within-inner-fold shuffled Role reliability","C2":"count-only"},"gate_model":"StandardScaler + class-weighted LogisticRegression","thresholds":[.5,.6,.7,.8,.9],"threshold_utility":"fixed - 2*broken; inner source only","future_rows_loaded":0,"forbidden":["exact API","HAIG","Host-call","future selection"]})
print("ROLE_RELIABILITY_PROTOCOL_WRITTEN")

