#!/usr/bin/env python3
import warnings,numpy as np
from haig_model import *
def graph(sha):
    return {"sha256":sha,"nodes":[{"node_id":0,"api_hash":"a","api_family":"android.telephony","primary_role":"device_identifier_access","roles":["device_identifier_access"]},{"node_id":1,"api_hash":"b","api_family":"java.net","primary_role":"network_transmission_or_upload","roles":["network_transmission_or_upload"]}],"same_host_edges":[{"src_node":0,"dst_node":1,"instruction_gap":4,"intermediate_invoke_count":1}],"host_call_edges":[{"src_node":1,"dst_node":0}],"stats":{"num_api_nodes":2,"num_hosts":1,"num_same_host_edges":1,"num_host_call_edges":1,"unique_api_count":2,"unique_pair_count":2}}
def main():
    gs=[graph("x"),graph("y"),graph("z")]
    assert role_matrix(gs).shape==(3,31);assert count_matrix(gs).shape==(3,8)
    assert len(graph_tokens(gs[0],"same"))==len(graph_tokens(gs[0],"same",True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning);e=Encoder(7).fit(gs)
    b=e.blocks(gs)
    assert arm_matrix(b,"H0_ROLE").shape==(3,127);assert arm_matrix(b,"H3_HOST_CALL").shape==(3,127)
    for x in b.values():assert np.all(np.isfinite(x))
    print("CARE_HAIG_SELFTEST_PASS width=127 relation_null=topology_preserving future_rows_loaded=0")
if __name__=="__main__":main()
