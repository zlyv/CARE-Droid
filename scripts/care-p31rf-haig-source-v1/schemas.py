#!/usr/bin/env python3
import json
from pathlib import Path
ROOT=Path("artifacts/care-p31rf-haig-source-v1")
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    node={"schema_version":"HAIG-v1","fields":{"node_id":"APK-local integer","canonical_api":"canonical class->method(descriptor)return signature","api_hash":"SHA-256 prefix of canonical API","roles":"one or more mapped semantic roles","primary_role":"deterministic first mapped role","api_family":"package-family backoff","host_id":"hash of Host method; construction only","local_order":"sensitive occurrence order inside Host","instruction_position":"position among all Host bytecode instructions","instruction_offset":"DEX offset","invoke_position":"order among all invokes"},"raw_app_method_identity_enters_model":False}
    edge={"schema_version":"HAIG-v1","SAME_HOST_ORDER":{"meaning":"linear bytecode order inside one Host; not data flow","attributes":["instruction_gap","bytecode_offset_gap","intermediate_invoke_count","same_basic_block(null in v1)"]},"HOST_CALL":{"meaning":"direct resolved invocation between sensitive Hosts; not data flow","attributes":["host_call_distance=1"]},"two_hop_disabled":True}
    (ROOT/"node_schema.json").write_text(json.dumps(node,indent=2)+"\n");(ROOT/"edge_schema.json").write_text(json.dumps(edge,indent=2)+"\n")
    print("CARE_HAIG_SCHEMAS_WRITTEN")
if __name__=="__main__":main()
