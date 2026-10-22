#!/usr/bin/env python3
from common import ROOT,dump

dump(ROOT/"protocol.json",{
  "experiment":"CARE-Droid class-conditional Role direction v1",
  "source":{"year":2017,"rows":799,"class_counts":{"benign":400,"malware":399},"fixed_folds":[160,160,160,160,159]},
  "stage1":"P31-RF",
  "router":"RF-anchored MeanOpposition DI25 Top25%",
  "route_budget":200,
  "baseline":"A0 Role-main C43-RQ bounded correction",
  "arms":{
    "D0_CURRENT_C43":"locked A0 replay, not retrained",
    "D1_GLOBAL":"outer-train global benign/malware Role-content cosine prototypes",
    "D2_PER_ROLE":"outer-train per-Role benign/malware cosine prototypes; Role-mass weighted mean",
    "D3_STAGE_CONDITIONED":"D2 direction with independently inner-selected strength for current benign/malware predictions"
  },
  "controls":{"C1_CLASS_SHUFFLE":"outer-train class assignment shuffled","C2_ROLE_SHUFFLE":"Role identity independently permuted within APK"},
  "prototype_min_class_support":5,
  "similarity":"cosine; positive means malware support",
  "bounded_integration":"stage1_logit + 2.25*tanh(gamma*direction_score)",
  "gamma_candidates":[0.5,1.0,2.0],
  "gamma_selection":"outer-train inner OOF only; maximize net, then minimize broken, then BA",
  "stage1_training_context":"strict nested OOF when selecting gamma",
  "future_rows_loaded":0,
  "selection_uses_future":False,
  "new_API_or_Host_features":False,
  "random_seeds":{"class_shuffle":"20300000+fold","role_shuffle":"20400000+fold"}
})
print("CLASS_ROLE_DIRECTION_PROTOCOL_PASS")
