#!/usr/bin/env python3
from common import ROOT,dump
dump(ROOT/"protocol.json",{
 "experiment":"D0 Correction-Score Anatomy + conditional Monotonic Bounded Correction Calibration v1",
 "stage1":"P31-RF only","router":"RF-anchored MeanOpposition DI25 Top25%","route":200,
 "stage2":"locked A0 Role C43-RQ","max_delta":2.25,"source":{"year":2017,"rows":799,"fixed_folds":[160,160,160,160,159]},
 "primary_strength_score":"abs_bounded_delta = abs(mean over three seeds of 2.25*tanh(raw_delta_seed))",
 "score_orientation":"larger means stronger C43 correction; no true-label sign is used",
 "auc_positive_class":"FIXED","auc_cohort":"D0 prediction-changed samples = FIXED union BROKEN",
 "stability_gate":"label-free, deployable, pooled raw AUC>=0.65, >=3 eligible folds, strict majority of eligible folds with raw AUC>0.5, top-strength enrichment, not a Stage-I-uncertainty proxy",
 "calibration_policy":"M1-M3 may run only after the stability gate; direction and max_delta must remain unchanged",
 "future_rows_loaded":0,"selection_uses_future":False
})
print("D0_SCORE_CALIBRATION_PROTOCOL_PASS")

