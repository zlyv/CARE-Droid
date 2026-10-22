# CARE-Droid P31-RF C43 API-identity probe source v1

This package implements only Steps 1--3 of the requested mechanism study.

- Frozen Stage-I: calibrated P31 RF.
- Frozen router: RF-anchored LR/RF/XGB/SVM MeanOpposition, foldwise Top25%.
- Source protocol: fixed 2017 five-fold OOF, 799 APKs.
- Current C43 outputs are read-only controls; no neural model is trained.
- E1 adds exact sensitive-API identity using stable canonical-signature hashes
  already present in the endpoint cache.
- Every vocabulary, APK-frequency filter and SVD is fit on the outer-train
  split only. Validation unseen APIs become UNK while Role31 remains present.
- E2 pairs and E3 paths are deliberately not implemented. `advance_decision.json`
  must pass before either is scientifically authorized.

The endpoint cache discarded raw method signatures by design. Thus exact API
identity is testable, but human-readable API names require a separate APK
re-extraction. The frozen C43 runs also did not persist per-Host attention;
this package audits every routed Host rather than fabricating attention.

Run:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-p31rf-c43-apiidentity-probe-source-v1.zip -d scripts/
bash scripts/care-p31rf-c43-apiidentity-probe-source-v1/launch_tmux.sh
tail -F artifacts/care-p31rf-c43-apiidentity-probe-source-v1.console.log
```

Main outputs:

- `REPORT.txt`
- `direction_quality.csv`
- `per_fold_direction.csv`
- `failure_taxonomy.csv`
- `fp_tp_fn_tn_metrics.csv`
- `shortcut_audit.csv`
- `nearest_neighbor_collapse.csv`
- `representation_collapse.json`
- `routed_host_audit.jsonl.gz`
- `vocabulary_fold_1.json` ... `vocabulary_fold_5.json`
- `vocabulary_audit.csv`
- `source_oof_predictions.csv`
- `advance_decision.json`
- `protocol.json`

