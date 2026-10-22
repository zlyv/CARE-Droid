#!/usr/bin/env python3
from common import ROOT,K,MIN_SUPPORT,CONTROL_REPETITIONS,NEAR_DUPLICATE_CUTOFF,dump
dump(ROOT/"protocol.json",{
 "experiment":"Local Role Neighborhood Probe v1","source":{"year":2017,"rows":799,"folds":[160,160,160,160,159]},
 "stage1":"P31-RF","router":"RF-anchored MeanOpposition DI25 Top25%","stage2_baseline":"A0 Role C43-RQ bounded","route":200,"max_delta":2.25,
 "representation":"label-free Compact43 Host content (43-D) grouped within APK by SelfRole31; learned hidden C43 embeddings are not saved and are not claimed",
 "memory_contract":"outer-train only; outer-test is query only; no test-test retrieval","metric":"cosine","k":K,"k_selection":"fixed a priori; no outer-test tuning","minimum_per_class_role_support":MIN_SUPPORT,
 "role_aggregation":"query Role-mass-weighted mean; no new attention","controls":{"label_shuffle_repetitions":CONTROL_REPETITIONS,"random_neighbor_repetitions":CONTROL_REPETITIONS,"role_shuffle_repetitions":CONTROL_REPETITIONS},
 "near_duplicate_cutoff":NEAR_DUPLICATE_CUTOFF,
 "advance_rule":"candidate must improve FP-vs-TP by >=0.03 over L0, retain FN-vs-TN within 0.03, have strict-majority valid-fold support for FP and FN, beat matched label/random controls by >=0.03, retain FP signal after >=0.99 neighbor exclusion, and have max absolute size correlation <0.80",
 "integration":"only if advance rule passes; preserve C43 sign, scale disagreement to 0.5 and agreement to 1.0","future_rows_loaded":0,"selection_uses_future":False
})
print("LOCAL_ROLE_NEIGHBOR_PROTOCOL_PASS")

