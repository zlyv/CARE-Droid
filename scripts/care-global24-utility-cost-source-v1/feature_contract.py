#!/usr/bin/env python3
"""Frozen semantic grouping and conservative acquisition-cost provenance for Global24."""
from __future__ import annotations

GROUPS={
 "manifest":(
  "manifest_permission_count","manifest_permission_count_log","manifest_activity_count",
  "manifest_service_count","manifest_receiver_count","manifest_provider_count",
  "manifest_component_count","manifest_component_count_log","min_sdk","target_sdk"),
 "code_call":("method_count","graph_node_count","graph_edge_count"),
 "sensitive":("context_role__boot_persistence","mapped_api_node_count","sensitive_node_count",
              "sensitive_edge_count","manifest_context_match_count"),
 "graph":("internal_node_count","internal_edge_count","external_node_count","external_edge_count",
          "unmapped_external_node_count","mapped_external_ratio"),
}

# This is a contract-level provenance classification, not a measured runtime claim.
ACQUISITION={
 "manifest":("manifest_parse",False,"Manifest-only counts/SDK metadata"),
 "code_call":("dex_and_callgraph",True,"method_count needs code scan; graph node/edge counts require graph construction"),
 "sensitive":("sensitive_mapping_and_graph",True,"sensitive nodes/edges and context matching require mapped code/graph evidence"),
 "graph":("callgraph",True,"internal/external graph counts and ratios require graph construction"),
}


def indices(names,mode):
    if len(names)!=24 or len(set(names))!=24:raise RuntimeError("Global24 dimension/name contract failed")
    expected=set().union(*(set(v) for v in GROUPS.values()))
    if set(names)!=expected:raise RuntimeError("Global24 grouping mismatch: "+repr({"missing":sorted(expected-set(names)),"unexpected":sorted(set(names)-expected)}))
    if mode=="FULL24":keep=set(names)
    elif mode.startswith("DROP_"):keep=set(names)-set(GROUPS[mode[5:].lower()])
    elif mode.startswith("ONLY_"):keep=set(GROUPS[mode[5:].lower()])
    else:raise KeyError(mode)
    return [i for i,name in enumerate(names) if name in keep]


STAGE1_VARIANTS=("FULL24","DROP_MANIFEST","DROP_CODE_CALL","DROP_SENSITIVE","DROP_GRAPH","ONLY_MANIFEST","ONLY_CODE_CALL","ONLY_SENSITIVE","ONLY_GRAPH")
ROUTER_VARIANTS=("FULL24","DROP_MANIFEST","DROP_CODE_CALL","DROP_SENSITIVE","DROP_GRAPH")
