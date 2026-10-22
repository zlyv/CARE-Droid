# CARE-Droid Multi-Year Development V1

This independent branch reuses the actual P31 classifier-family, A0 background,
HAIG role-only Host cache and RGCondAttn Direct output implementation. It adds
no network, Host feature, gate, bounded correction or searched loss weight.
2017–2021 are **development data**, not a final independent test. Historical
2017-only/future/MalEval assets remain read-only. No Git command is used.

The server's old `care-droid-formal-v1/formal_freeze_manifest.json` still names
the historical RF/bounded freeze. It is not rewritten or relabeled. This branch
uses the user-selected **SVM/Direct-C43 research baseline**, backed by the
completed Direct source and route-conditioned B0 replay assets, not that old freeze.

## Start and resume

Run from `/home/boyuanjie/projects/tasa-droid`. Initial audit creates the new
output; all subsequent commands must use `--resume`, which checks dataset
hashes, protected asset integrity and model metadata before reuse.

```bash
bash scripts/care-multiyear-development-v1/run_experiments.sh --audit-only
CUDA_VISIBLE_DEVICES=0 bash scripts/care-multiyear-development-v1/run_experiments.sh --smoke-test --resume
```

If the audit is already deployed and completed, start with `--audit-only --resume`.
The smoke uses 80 train APKs from 2017–2018, 8 validation APKs from 2019, one
original-network epoch and one seed. It is an engineering test, not a BA result.
It does not shrink the architecture or prototypes and parses no APK.

## Separate phases

```bash
bash scripts/care-multiyear-development-v1/run_experiments.sh --run-full --stage stage1 --resume
bash scripts/care-multiyear-development-v1/run_experiments.sh --run-full --stage router --resume
CUDA_VISIBLE_DEVICES=0 bash scripts/care-multiyear-development-v1/run_experiments.sh --run-full --stage stage2 --resume
bash scripts/care-multiyear-development-v1/run_experiments.sh --run-full --stage cascade --resume
bash scripts/care-multiyear-development-v1/run_experiments.sh --collect-only --resume
```

To complete one window first:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/care-multiyear-development-v1/run_experiments.sh --run-full --windows T1 --resume
```

For all T1–T4 in a persistent session (explicit user launch, never auto-started):

```bash
tmux new-session -d -s care-multiyear-development-v1 'cd /home/boyuanjie/projects/tasa-droid && CUDA_VISIBLE_DEVICES=0 bash scripts/care-multiyear-development-v1/run_experiments.sh --run-full --resume'
tail -F outputs/care_multiyear_development_v1/logs/run-*.log
```

Check `nvidia-smi` first. Use a currently free GPU, not necessarily GPU0.
No automatic device fallback changes architecture or batch size; CPU is an
explicit `--device cpu` alternative. Do not launch concurrent writers to the
same output. `last_failure.json` is diagnostic history, not the current status.

## Data and fit boundaries

Required counts: 799/793/399/397/400. Same-SHA duplicates retain their earliest
year; conflicting labels fail. No prediction-dependent removal. Missing assets,
nonfinite required arrays, dimension/Role-order mismatch fail. Existing empty
Host placeholder policy is retained and reported, not reinterpreted as benign.
P31 fractions are not row-normalized. All original global-feature standardization,
Host scalers/prototypes and S7 scalers are fitted within the training window;
only original early-stopping train holdouts choose epochs. Source/future P31
producer script hashes differ and are explicitly recorded; both resolve to the
same semantic Role mapping. No family labels are invented.

SVM and auxiliary classifiers retain original three-fold OOF sigmoid calibration.
The fitted member checkpoints include raw estimators, calibrators, train SHAs
and calibration split SHAs. Direct-C43 has no Stage-I input: its original adapter
sets the base logit to zero. A neutral constant base avoids redundant Stage-I OOF
fitting for this branch; actual smoke tests verify base-invariant probabilities.

C1 uses original equal BCE, original early stopping, 30-epoch maximum, original
3 seeds. C2 uses the same train pool, fitted backgrounds, architecture and C1
selected epoch counts; only year-uniform quota sampling replaces final training
permutation. Exactly N samples are drawn per epoch, with replacement within
each year, so gradient-step counts match. Single-year C2 reuses C1 exactly.
Saved expert states contain the NN weights, original background state and Role
profiles, Host scaler, graph scaler, args, global names and train/validation SHAs.
Original window-global construction is reproducible from those SHAs and hashed
compact cache. Model checkpoints are pickled trusted local artifacts only.

## Attribution arms

| Arm | Stage-I | Router | Expert |
|---|---|---|---|
| H0 historical descriptive | old saved SVM | old compatible committee | old saved Direct |
| U0 controlled 2017 | same-implementation retrained 2017 | matched retrained 2017 | retrained 2017 C0R |
| U1 | rolling SVM | old auxiliary members reanchored to rolling SVM | C0R |
| U2 | rolling SVM | rolling auxiliary members | C0R |
| U3 | same rolling SVM | same rolling route | rolling C1 |
| U4 | same rolling SVM | same rolling route | rolling C2 |

S1 versus S0R measures classifier data expansion. U1 includes mandatory
reanchoring, not a pure Stage-I cascade effect. U2–U1 isolates auxiliary-router
refitting; U3–U2 isolates expert expansion; U4–U3 isolates the sampler. Fixed and
Broken always use each arm's own base; they cannot be subtracted across bases
and called independent expert improvements. Budget diagnostics do not select
a new budget. Model inference computes all validation APK probabilities to
support fixed-upstream attribution; deployment remains Top25%. Its measured
inference cost is therefore full-validation inference, not selective extraction.

## Evidence boundaries

Historical baseline replay checks independent completed source B0 assets and
recomputes 799-APK cascade metrics; this is **asset replay**, not a new full
source five-fold NN training. C0R is separately retrained by the full experiment.
Bootstrap intervals are exploratory paired APK intervals conditional on fitted
models, not family-adjusted or multiplicity-corrected. Three seeds are sensitivity
checks, not independent datasets. Macro and pooled results are separate; absent
windows are never filled. Conditional AUC emits NaN and class counts when
unsupported. No new method recommendation is selected automatically.

Outputs: `dataset_audit.json`, `protocol_audit.json`, protected before/after
manifests, `baseline_replay.json`, `smoke_test.json`, year-level result CSVs,
pooled/macro comparisons, paired `ablation_results.csv`, per-APK predictions,
model checkpoints, timing, `final_report.md`, `final_decision.json`.
