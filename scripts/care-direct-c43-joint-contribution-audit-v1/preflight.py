#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from audit_common import *
from interventions import ARMS, intervention_semantics

HISTORY = [
  {"question":"Host content beyond Role","evidence":"A0 Role-only vs API/HAIG/Host-call arms","answer":"tested; HAIG did not add stable net benefit in full C43","remaining_gap":"not a Direct-C43 joint-pair attribution"},
  {"question":"Relation31 marginal addition","evidence":"C43_COMPACT vs C43_RELATION_QUERY","answer":"tested under bounded protocol","remaining_gap":"which Relation-to-Host pairings drive Direct-C43 fixed/broken"},
  {"question":"RoleGraph conditioning","evidence":"RGConcat/RGCondAttn/RGCondDual and GraphGlobal Q/D path","answer":"tested; query-only boundary preferred","remaining_gap":"sample-level query/local misalignment in Direct-C43"},
  {"question":"background representation","evidence":"hard/soft prototype, signed/unsigned first/second-order TB-FRS","answer":"background alone is much weaker than Direct-C43","remaining_gap":"background may matter only through pairing with Host/Relation"},
  {"question":"readout/threshold","evidence":"collision probes and global threshold frontier","answer":"simple readout and threshold tuning unsupported","remaining_gap":"upstream joint attribution remains unresolved"},
  {"question":"Critical/Attended Host localization","evidence":"argmax/TopK/aligned attention and contextual witness audits","answer":"several localization variants already failed","remaining_gap":"whether frozen query selects the wrong Host because cross-block pairing is wrong"},
]

def main():
    if not OUTPUT.is_relative_to((PROJECT/"outputs").resolve()): raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output")
    required=[FORMAL,DIRECT,P31,FAMILY_SCRIPT/"model_adapter.py",DIRECT_SCRIPT/"model_variants.py"]
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing assets "+repr(missing))
    ids,y,fold,base,route,expert,final,field=load_reference(False); value=metrics(y,base,final,route)
    checks={"rows==799":len(ids)==799,"unique_sha==799":len(set(ids.tolist()))==799,
      "fold_sizes":[int(np.sum(fold==f)) for f in FOLDS]==[160,160,160,160,159],"route==200":int(route.sum())==200,
      "stage1_errors==97":int(np.sum((base>=.5)!=y))==97,"direct_49_17_32":(value["fixed"],value["broken"],value["net"])==(49,17,32)}
    if not all(checks.values()): raise RuntimeError("BLOCKED_BY_CONTRACT: frozen reference changed "+repr(checks))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    before=[asset(FORMAL,"frozen SVM Stage-I and route"),asset(DIRECT,"frozen Direct-C43 predictions"),asset(P31,"frozen source P31")]
    dump_json(OUTPUT/"baseline_asset_manifest_before.json",{"assets":before})
    protocol={"name":"SVM Direct-C43 Joint Contribution Audit v1","domain":"2017 fixed outer-fold OOF only",
      "reference":{"threshold":0.5,"fixed":49,"broken":17,"net":32},"repeats":12,
      "method":"train the exact frozen Direct-C43 replay once; perform label-free inference-time pairing interventions on the same fitted models",
      "arms":intervention_semantics(),"historical_coverage":HISTORY,
      "donor_matching":"one-to-one cyclic matched shuffle within Stage-I prediction direction, ordered by Stage-I probability and Host count; labels unused",
      "host_interventions":"within-APK row permutation preserves each feature-block marginal and Host count",
      "formal_A0_layout":"142 = Compact43 + 64 zero HAIG compatibility slots + Background4 + Relation31; zero slots are asserted at runtime",
      "prohibited":["new model","retraining per arm","new threshold","gate","future data","hyperparameter selection","claiming intervention as causal program semantics"],
      "future_rows_loaded":0}
    dump_json(OUTPUT/"protocol.json",protocol); (OUTPUT/"protocol_hash.txt").write_text(hashlib.sha256((OUTPUT/"protocol.json").read_bytes()).hexdigest()+"\n",encoding="utf-8")
    dump_json(OUTPUT/"preflight.json",{"status":"PASS","checks":checks,"direct":value,"probability_field":field,"future_rows_loaded":0})
    lines=["# Historical ablation coverage matrix","", "This audit distinguishes marginal module tests from conditional joint contribution inside the frozen SVM Direct-C43 model.","",
      "| Question | Existing evidence | Already answered | Remaining gap |","|---|---|---|---|"]
    for row in HISTORY: lines.append("| {question} | {evidence} | {answer} | {remaining_gap} |".format(**row))
    lines += ["","## Decision","","A genuinely missing question remains: no prior arm destroys only the cross-block correspondence while holding the fitted Direct-C43 model, feature marginals, route, threshold and parameter count fixed. The intervention replay is therefore not a renamed repetition of the historical ablations."]
    (OUTPUT/"historical_coverage_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("DIRECT_C43_JOINT_AUDIT_PREFLIGHT_PASS",flush=True)
if __name__=="__main__": main()

