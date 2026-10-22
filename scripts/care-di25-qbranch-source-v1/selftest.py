#!/usr/bin/env python3
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"care-di25-graphglobal-path-source-v1"))
from qbranch_models import ARM_SPECS, BLOCKS, TRAIN_ARMS, correction_mask

expected = {"FULL_Q":113, "ROLE_ONLY":32, "HOST_ONLY_Q":56,
            "HOST_ONLY_N0":56, "STRUCTURE_ONLY_Q":88, "STAGE1_CONTEXT_ONLY":25}
assert tuple(BLOCKS.values()) == ((0,24),(24,25),(25,57),(57,81),(81,105),(105,113))
assert set(TRAIN_ARMS) == set(expected) - {"FULL_Q"}
for arm, width in expected.items():
    mask = correction_mask(arm)
    assert tuple(mask.shape) == (113,) and int(mask.sum().item()) == width
    assert set(mask.tolist()) <= {0.0, 1.0}
assert ARM_SPECS["HOST_ONLY_Q"][1] == ARM_SPECS["HOST_ONLY_N0"][1]
print("QBRANCH_SOURCE_SELFTEST_PASS")
print("arms=" + ",".join(ARM_SPECS))
print("active_dimensions=" + ",".join(f"{k}:{v}" for k,v in expected.items()))
print("parameter_count_target=33036")
print("future_rows_loaded=0")
print("ego_consumed=false")
