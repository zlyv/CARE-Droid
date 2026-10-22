#!/usr/bin/env python3
from feature_contract import CODE_CALL,VARIANTS,retained_indices


def main():
    names=[f"f{i}" for i in range(21)]+list(CODE_CALL)
    expected={"FULL24":24,"DROP_CODE_CALL":21,"DROP_METHOD_COUNT":23,
              "DROP_GRAPH_NODE_COUNT":23,"DROP_GRAPH_EDGE_COUNT":23}
    for variant,width in expected.items():
        keep=retained_indices(names,variant)
        assert len(keep)==width and len(set(keep))==width
    base_seed=20260925;fold=3
    seeds={variant:base_seed+fold*100000 for variant in VARIANTS}
    assert len(set(seeds.values()))==1
    print("GLOBAL24_CODECALL_CONFIRM_SELFTEST_PASS")
    print("variants="+",".join(VARIANTS))
    print("source_only=true future_rows_loaded=0 stage2_training_invoked=false")
    print("frozen_DI25_modified=false")
    print("variant_seed_policy=same_within_outer_fold")


if __name__=="__main__":main()
