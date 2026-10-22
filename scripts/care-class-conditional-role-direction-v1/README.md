# CARE-Droid class-conditional Role direction v1

Strict 2017 source-only test on the locked P31-RF / RF-anchored DI25 / A0 Role C43-RQ line. It first audits older experiments, then evaluates global and per-Role benign/malware cosine prototypes. Positive scores always mean malware support.

The pipeline never loads 2018-2021. Prototypes are rebuilt from each outer-training split. Correction strength is selected from `{0.5,1,2}` using only strict inner-OOF source predictions. Per-fold checkpoints make reruns resumable.

Run:

```bash
bash scripts/care-class-conditional-role-direction-v1/launch_tmux.sh
tail -F artifacts/care-class-conditional-role-direction-v1.console.log
```

Final outputs include `REPORT.txt`, `final_summary.json`, prototype support/separation audits, direction probes, FP/FN audits, controls, fold results, and predictions.
