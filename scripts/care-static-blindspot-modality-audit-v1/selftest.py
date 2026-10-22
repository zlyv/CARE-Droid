#!/usr/bin/env python3
from __future__ import annotations
from feature_worker import component_name,entropy
from run_audit import binom_two_sided,holm

def main():
    assert component_name("com.demo",".MainActivity")=="lcom/demo/mainactivity;"
    assert component_name("com.demo","MainActivity")=="lcom/demo/mainactivity;"
    assert entropy(bytes(range(256))*4)>7.99
    assert binom_two_sided(10,0)<.002
    values=holm([.01,.04,.2]);assert values[0]<=values[1]<=values[2]
    print("STATIC_BLINDSPOT_SELFTEST_PASS modality=true matching_stats=true",flush=True)
if __name__=="__main__":main()
