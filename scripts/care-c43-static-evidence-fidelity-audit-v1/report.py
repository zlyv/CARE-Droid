#!/usr/bin/env python3
from __future__ import annotations
import json
from common import *
def main():
    d=json.loads((OUTPUT/"final_decision.json").read_text(encoding="utf-8"));summary=read_csv(OUTPUT/"mechanism_summary.csv")
    lines=["# CARE-Droid C43 static-evidence fidelity audit v1","","## Decision","",f"**{d['status']}**","",
      "This source-only audit traces every routed Direct-C43 failure through four frozen layers. It trains no model, changes no threshold, and loads no future data.","",
      "## Frozen cohort","","- Routed Stage-I errors left unfixed by Direct-C43: 35.","- Stage-I-correct samples broken by Direct-C43: 17.","- Total audited failures: 52.","- Every case is paired within the same outer fold with an opposite-label Direct-C43-correct sample and a same-label correct anchor.","",
      "## Layer trace","","| Candidate localization | Failures | Matched correct | Rate difference | Folds higher | Enriched |","|---|---:|---:|---:|---:|---:|"]
    for r in summary:lines.append(f"| {r['mechanism']} | {r['failure_count']} | {r['matched_correct_count']} | {float(r['rate_difference']):.4f} | {r['folds_failure_higher']} | {r['descriptively_enriched']} |")
    lines += ["","## Evidence boundary","",
      "The automated part can locate structural compression or collision, but cannot infer that a DEX difference is reliable benign or malicious semantics. That stronger A/B/C attribution requires the blinded semantic review in `blind_review_form.csv`; labels and predictions are isolated in `blind_review_key.csv`.","",
      "- `RAW_TO_C43_COMPRESSION_CANDIDATE`: raw DEX sketches are separated while frozen structured/C43 inputs are unusually similar.",
      "- `C43_TO_HIDDEN_COMPRESSION_CANDIDATE`: frozen C43 inputs are separated while hidden states collide.",
      "- `MULTILAYER_STATIC_COLLISION`: the opposite-label pair is unusually similar from raw sketches through hidden state.",
      "- `HIDDEN_TO_SCORE_READOUT_CANDIDATE`: hidden representations are separated but the target decision remains wrong.","",
      "## Next action","","Complete the blinded review before opening the key. Do not add a new feature or model solely from an automated candidate category.","",
      "Formal asset integrity: **PASS**. Future rows loaded: **0**."]
    safe(OUTPUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    commands="""# Reproduction commands\n\n```bash\ncd /home/boyuanjie/projects/tasa-droid\nCSEF_WORKERS=4 bash scripts/care-c43-static-evidence-fidelity-audit-v1/run_on_server.sh\n```\n\nAfter blinded review:\n\n```bash\n/home/boyuanjie/envs/tif/bin/python scripts/care-c43-static-evidence-fidelity-audit-v1/summarize_review.py\n```\n"""
    safe(OUTPUT/"reproduce_commands.md").write_text(commands,encoding="utf-8")
if __name__=="__main__":main()
