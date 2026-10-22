# Verified dependency map

All paths below are relative to `/home/boyuanjie/projects/tasa-droid`.
They are imported/read, never patched by this branch.

| Function | Actual original implementation |
|---|---|
| P31 SVM/committee and calibration | `scripts/care-p31-classifier-family-stage2-v1/stage1_core.py` |
| MeanOpposition and round/SHA ranking reference | `scripts/care-p31-classifier-family-stage2-v1/common.py` |
| A0 model/cache and canonical Compact76 adapter | `scripts/care-p31-classifier-family-stage2-v1/model_adapter.py` |
| Original graph-conditioned network/early stopping/BCE/optimizer | `scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py` |
| Shared/benign prototypes, Role compatibility and internal background | `scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py` |
| Direct zero-base-logit, unbounded output adapter | `scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py` |
| Source label contract (labels only; no loss override) | `scripts/care-malben-host-v2/source_label_contract.py` |

Input flow: P31 → independently OOF-calibrated LR/RF/XGB/SVM → current-SVM
MeanOpposition → Top25 mask. Expert flow: raw169 Host bag → equal-APK train
scaler + original 32/24 prototypes + Role profiles → model142 Host bag;
S7(31 nodes×3, 27 graph-global, 465 pairs, 961 inter) → original train-only graph
scaler → RoleGraph query + global24 → original Direct-C43 logit.

The actual original `compact_matrix_variant` is invoked separately with each
window's train and validation rows. Never fit global standardization on all
2788 samples. Source canonical names are frozen; unseen role columns are handled
by the existing formal fold-safe adapter, not a new feature expansion.

Original source baseline is verified against completed
`outputs/care-direct-c43-route-conditioned-training-v2/source/B0_ORIGINAL/source_predictions.csv`.
The older `artifacts/care-droid-formal-v1/formal_freeze_manifest.json` describes
RF/bounded historical assets and remains untouched. This independent branch
uses the currently requested SVM/Direct research protocol.

Model supervision units are APKs, not individual Host rows. C1 uses every
historical training APK, equal BCE, keep/fix=1/1, consistency=0. None of the
previously rejected route-weighted losses or Host evidence probes is imported.

The only C2 training change is a uniform-year quota sampler with within-year
replacement. It includes resampling variability, not an isolated proof of a
year effect. C1/C2 share train pool, background recipe, seeds, network capacity,
selected epoch counts and total gradient updates. No year weights are searched.
