# Direct-C43 route-conditioned training source Pilot

Version 2 repairs the module-name collision reported before B0 training: the generic `nested_stage1` resolved to the RF/MEAN-only helper. Family common/stage1_core/model_adapter are now loaded from explicit family file paths, and the SVM NestedStage1 uses a private module alias. Preflight instantiates the SVM helper and records module origins before preparing routes. Version 1 scripts, logs and outputs remain intact; no v1 predictions or prepared routes are reused. The experiment settings are identical to v1 apart from the independent version name.

This independent experiment tests one fixed loss-weighting rule on the existing 2017 source representation. It must not be presented as a new routing method, a new Host semantic representation, or proof of an error mechanism.

## Frozen comparison

| Arm | Weights on all outer-train APKs |
|---|---|
| B0_ORIGINAL | 1 for every APK; original inner early stopping |
| B1_ROUTE_WEIGHTED | 4 for nested-OOF Top25% route, 1 for the rest; normalized to mean 1 |
| B2_CLASS_PRIOR | Constant per true training class, matching B1 class weight mass |
| B3_CONDITIONAL_SHUFFLE | Permute B1 weights within true training class, OOF SVM predicted side, Host-count tertile |

All APKs remain in training, including empty-Host cases under the original policy. B1 does not use Stage-I correctness to assign weights. B2/B3 use training labels for their stated controls. The single 4:1 ratio is an a priori engineering choice for this Pilot, not a selected optimum; at exactly 25% it places roughly 57% of total unnormalized loss mass on routed APKs. The original minibatch normalization remains in force.

Training routing uses four-fold crossfitting inside each outer-train set, with 3-fold calibration nested inside each LR/RF/XGB/SVM fit. Top25% is ranked separately within each inner held-out cohort, following the deployment batch convention and avoiding score-scale comparison across different fits. Every training row is scored by members and calibration models excluding its label. The outer-test cohort is excluded throughout. The global source OOF committee file is checked only to confirm the unchanged deployment route; it never supplies training weights.

B0 is rerun with the original equal-BCE, seeds, backgrounds, scaler, cache order, inner early stopping and model. All 799 probabilities must differ from the frozen expert by no more than 1e-4, every class decision must match, and fixed/broken/net must be 49/17/32. Any mismatch stops before weighted C43 fits. Nested SVM must agree with the SVM column of the newly crossfitted committee to 1e-14.

B1/B2/B3 use B0's fold/seed epoch counts. This isolates weighting and holds optimizer exposure fixed; the Pilot does not test route-conditioned early stopping. Backgrounds/prototypes still use all outer-train APKs, preserving existing preprocessing. All three arms reuse the same seeds, feature input fingerprint, network capacity and deployment route. A conditional-shuffle changed-weight fraction below 10% in any fold stops the Pilot before expert fitting. There is one fixed shuffle realization; no favorable seed selection is permitted.

## Decision fixed before outcomes

The nine primary contrasts are B1 against B0/B2/B3 for end-to-end BA, routed predicted-benign FN-vs-TN AUC, and routed predicted-malware TP-vs-FP AUC. Success requires BA gain >=0.25 percentage points and each conditional AUC gain >=0.01 against all three; family-adjusted paired-bootstrap lower bounds must exceed zero. BA must improve in at least 4/5 folds. Each direction requires >=3 evaluable folds, and positive differences in >=75% of evaluable folds. B1 must retain >=49 fixed, have <=17 broken and net>=33. All conditions must pass. A one-sided result cannot justify a bidirectional claim.

APK is the resampling unit; 4000 paired bootstrap replicates preserve fold, label, route and Stage-I predicted side. Bonferroni percentile bounds use alpha/(2*9). They are approximate conditional-on-fitted-model intervals, not refitting or lineage-cluster uncertainty. The historical sequence of research trials is not corrected by this nine-contrast family. Per-seed results are sensitivity records, not additional independent APKs. One shuffle control is not an exact conditional randomization test.

If the Pilot fails, stop this branch. No ratio sweep, readout/threshold/seed search, future run or formal-model replacement is authorized by the launcher. Passing permits specification of an independent replication only.

## Installation and launch

Upload the zip to `/home/boyuanjie/projects/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
nvidia-smi
if [ -e scripts/care-direct-c43-route-conditioned-training-v2 ] ||
   [ -e outputs/care-direct-c43-route-conditioned-training-v2 ] ||
   [ -e outputs/care-direct-c43-route-conditioned-training-v2.console.log ]; then
  echo 'Existing experiment: stop and use a new version.'
else
  unzip -q ../care-direct-c43-route-conditioned-training-v2-package-r1.zip -d scripts/ &&
  RCT_GPU=1 bash scripts/care-direct-c43-route-conditioned-training-v2/launch_tmux.sh
fi
tail -F outputs/care-direct-c43-route-conditioned-training-v2.console.log
```

GPU `1` is a launch choice; set RCT_GPU to an available GPU before launch. No other user's process is stopped. The package prints the GPU inventory and records the selected logical device. It makes no runtime estimate without machine measurements. Four C43 arms use 5 folds and 3 seeds; B0 adds original early-stop fits, and routing preparation adds nested calibrated CPU fits. Routing and per-arm per-fold wall time are saved so the first completed stages provide a measured estimate.

Completion marker: `CARE_DIRECT_C43_ROUTE_CONDITIONED_DONE`. A terminal/tmux session ending alone does not establish success. Failures print `CARE_DIRECT_C43_ROUTE_CONDITIONED_FAILED`, preserve logs, write a BLOCKED decision and perform an after-asset audit when a before-manifest exists. The pipeline requires a fresh output directory and does not automatically resume or overwrite a prior attempt.

```bash
bash scripts/care-direct-c43-route-conditioned-training-v2/collect_results.sh
```

New files only go under this independent script directory and its matching outputs directory. Original scripts, APKs, formal caches/results remain read-only. The original region-cache fallback is disabled; no DEX scan or cache rebuild is allowed. Integrity manifests cover listed inputs and dependency code, not the entire server. Pipeline children use `python -B` to avoid bytecode writes into protected dependency directories. Original APKs are never installed, executed or rewritten.

## Evidence provenance

The training-target audit found all outer-train APKs, keep/fix weights 1/1 and consistency 0 in the Direct launch. The resolved label-contract wrapper contains only `v4_module.audit.future.label_of = source_label_of`. Prior Global24/A25/JointS7 and C76 residual route-conditioned experiments use different representations and routes. This Pilot evaluates the existing P31-SVM/MeanOpposition/Direct-C43 configuration; it is not a claim that route weighting is novel or that every remote historical experiment has been exhaustively excluded.
