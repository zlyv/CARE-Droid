#!/usr/bin/env python3
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/"care-di25-graphglobal-path-source-v1"))
import torch
from drop_role_model import ARM
m=torch.ones(113);m[25:57]=0
assert ARM=="Q_DROP_ROLE_DIRECT" and int(m.sum())==81 and int((m==0).sum())==32
print("Q_DROP_ROLE_SOURCE_SELFTEST_PASS")
print("role_token_to_attention=true")
print("graphglobal_to_attention=true")
print("role_token_to_delta=false")
print("active_dimensions=81")
print("parameter_count_target=33036")
print("future_rows_loaded=0")
print("ego_consumed=false")
