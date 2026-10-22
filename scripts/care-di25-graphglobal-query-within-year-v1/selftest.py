#!/usr/bin/env python3
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    required={
        "preflight.py":["same_year_rows=1989","ego_consumed=false","route_DI"],
        "run_within_year_arm.py":["inner_oof_base","make_graphglobal_path_class","parameter count changed","same-year five-fold OOF"],
        "summarize.py":["N0_NO_GRAPHGLOBAL","Q_QUERY_ONLY","temporal generalization"],
        "run_on_server.sh":["run_arm N0_NO_GRAPHGLOBAL 0","run_arm Q_QUERY_ONLY 1","wait"],
    }
    for name,markers in required.items():
        text=(root/name).read_text(encoding="utf-8")
        for marker in markers:assert marker in text,(name,marker)
    shell=(root/"run_on_server.sh").read_text(encoding="utf-8")
    assert 'local arm="$1" gpu="$2"' not in shell
    print("GRAPHGLOBAL_QUERY_WITHIN_YEAR_SELFTEST_PASS")
    print("arms=N0_NO_GRAPHGLOBAL,Q_QUERY_ONLY")
    print("protocol=independent same-year five-fold OOF")
    print("ego_consumed=false")

if __name__=="__main__":main()
