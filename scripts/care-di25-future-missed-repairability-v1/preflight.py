#!/usr/bin/env python3
from audit_core import load_and_validate
ids,years,y,base,route,expert,audit=load_and_validate()
print("DI25_FUTURE_MISSED_REPAIRABILITY_PREFLIGHT_PASS")
print("future_rows=1989")
print("analysis_years=2020,2021")
print("Q_QUERY_ONLY_source_only_training=true")
print("Q_QUERY_ONLY_router_applied_during_model_fit=false")
print("Q_QUERY_ONLY_ego_consumed=false")
print("p_expert_max_abs_diff="+str(audit["p_expert_max_abs_diff"]))
print("p_final_max_abs_diff="+str(audit["p_final_max_abs_diff"]))
print("training_invoked=false")
print("inference_invoked=false")

