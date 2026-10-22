#!/usr/bin/env python3
"""Frozen Global24 contract for the Code/Call confirmation experiment."""
from __future__ import annotations

CODE_CALL=("method_count","graph_node_count","graph_edge_count")
VARIANTS=(
    "FULL24",
    "DROP_CODE_CALL",
    "DROP_METHOD_COUNT",
    "DROP_GRAPH_NODE_COUNT",
    "DROP_GRAPH_EDGE_COUNT",
)


def retained_indices(names, variant):
    names=list(names)
    if len(names)!=24 or len(set(names))!=24:
        raise RuntimeError("Global24 dimension/name contract failed")
    missing=sorted(set(CODE_CALL)-set(names))
    if missing:
        raise RuntimeError("Global24 Code/Call fields missing: "+repr(missing))
    if variant=="FULL24":
        dropped=set()
    elif variant=="DROP_CODE_CALL":
        dropped=set(CODE_CALL)
    elif variant=="DROP_METHOD_COUNT":
        dropped={"method_count"}
    elif variant=="DROP_GRAPH_NODE_COUNT":
        dropped={"graph_node_count"}
    elif variant=="DROP_GRAPH_EDGE_COUNT":
        dropped={"graph_edge_count"}
    else:
        raise KeyError(variant)
    return [i for i,name in enumerate(names) if name not in dropped]
