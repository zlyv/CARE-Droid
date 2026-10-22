# CARE-Droid LiteLocal Stage-I definition and shortcut attribution source v1

Strict 2017-only audit. It reuses `local_semantic_features_2017.npz`; APKs are not parsed again.

The audit corrects an interpretation ambiguity in the first LiteLocal report: classification metrics used the LR member (`probs[:,0]`), while routing opposition averaged LR/RF/XGB/SVM. It now persists every member probability and compares LR-only, mean-committee, and median-committee Stage-I definitions under identical five-fold OOF training.

The model/SVD/calibration seed anchor is hard-locked to the original LiteLocal experiment (`20260925 + 100000 * outer_fold`). The separate bootstrap audit seed does not affect model predictions.

It then distinguishes encoded size information from decision reliance using:

- a five-variable size-only OOF label model;
- a fold-safe FUSED63+size diagnostic model;
- paired bootstrap confidence intervals for incremental AUC;
- stabilized inverse-propensity size-balanced metrics;
- FUSED63 metrics across fold-safe size-index quintiles.

Raw size fields are diagnostic-only and never enter a proposed LiteLocal classifier or router. No future row, call graph, Stage-II model, Host, RoleGraph, or Ego is loaded.

Upload the ZIP to `/home/boyuanjie/projects/`, then run:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-litelocal-stage1-attribution-source-v1.zip -d scripts/
bash scripts/care-litelocal-stage1-attribution-source-v1/launch_tmux.sh
tail -F artifacts/care-litelocal-stage1-attribution-source-v1.console.log
```
