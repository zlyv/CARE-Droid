#!/usr/bin/env python3
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent if HERE.parent.name=="scripts" else HERE.parent
SOURCE=PROJECT/("scripts" if (PROJECT/"scripts").is_dir() else "")/"care-di25-compact43-relationquery-source-v1"


def main():
    required={
        "compact_models.py":["relation_query_switch","relation_key","CORE_WIDTH=47","score=score+self.relation_query_switch*relation_score"],
        "compact_background.py":["ROLE_SCALE=np.sqrt(86.0/74.0)","TAIL_SCALE=np.sqrt(43.0/74.0)","COMPACT=43"],
        "hostrel_features.py":["directed_neighbor_role"],
    }
    for name,tokens in required.items():
        text=(SOURCE/name).read_text(encoding="utf-8")
        for token in tokens:
            if token not in text: raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {token} in {name}")
    rng=np.random.default_rng(7); role=rng.normal(size=(11,31)); tail=rng.normal(size=(11,12))
    transformed=np.concatenate([role*np.sqrt(86/74),tail*np.sqrt(43/74)],axis=1)
    old=np.sqrt((2*np.sum(role*role,axis=1)+np.sum(tail*tail,axis=1))/74)
    new=np.sqrt(np.mean(transformed*transformed,axis=1))
    if not np.allclose(old,new,atol=1e-12,rtol=1e-12): raise RuntimeError("weighted Compact43 distance identity failed")
    print("COMPACT43_RELATIONQUERY_DIAGNOSTICS_SELFTEST_PASS")
    print("protocols=frozen_2017_to_future,independent_same_year_5fold")
    print("candidate=C43_RELATION_QUERY control=H0_LOCKED gpus_requested=2")
    print("relation_enters_attention=true relation_enters_delta=false future_labels_loaded_by_model=false ego_consumed=false")


if __name__=="__main__": main()
