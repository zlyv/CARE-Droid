#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from viewdi_core import decision_instability,exact_top_fraction,exact_binomial_two_sided,feature_groups

names=[
"manifest_permission_count","manifest_permission_count_log","manifest_activity_count","manifest_service_count","manifest_receiver_count","manifest_provider_count","manifest_component_count","manifest_component_count_log","min_sdk","target_sdk",
"method_count","graph_node_count","graph_edge_count","context_role__boot_persistence","mapped_api_node_count","sensitive_node_count","sensitive_edge_count","manifest_context_match_count","internal_node_count","internal_edge_count","external_node_count","external_edge_count","unmapped_external_node_count","mapped_external_ratio"]
groups=feature_groups(names)
assert {k:len(v) for k,v in groups.items()}=={"drop_manifest":14,"drop_code_call":21,"drop_sensitive":19,"drop_graph":18}
base=np.asarray([.9,.1]);members=np.asarray([[.9,.8,.7,.6],[.9,.8,.7,.6]])
d=decision_instability(base,members)
assert np.allclose(d["score"],[.25,.75])
assert exact_top_fraction(np.arange(8),.25).tolist()==[False]*6+[True,True]
assert abs(exact_binomial_two_sided(1,5)-.21875)<1e-12
print("VIEWDI_SOURCE_SELFTEST_PASS")
print("members=drop_manifest,drop_code_call,drop_sensitive,drop_graph")
print("future_rows_loaded=0")

