# LCC conditional incremental source test v1

This separate experiment tests whether verified local-control facts add class
information to frozen Direct-C43. It does not change formal CARE-Droid, re-train
C43, tune auxiliary parameters, add a gate or load future cohorts.

## Why 799 feature rows, despite a 200-APK final route?

The passed LCC v2 audit only scanned 200 globally OOF-routed APKs. Selecting
auxiliary training examples using this global OOF route can indirectly use the
outer test labels through other folds' upstream fits. Filling the other 599
training rows with invented zero evidence is also invalid. Therefore the same
label-free v2 extractor supplies features for ALL 799 source APKs. Existing 200
scans are protected and read-only; only 599 additional scans are needed.

Each auxiliary model trains on all 639/640 outer-train APKs. Its final decisions
still apply ONLY to the unchanged 200 routed test APKs. Every other final
probability stays Stage-I. Zero-Host APKs remain in the cohort; failures are not
dropped. Exact API names stay in raw audit caches, not classifier inputs.

## Representation and selection boundary

For each fold, use its own `f*_train_hidden` and `f*_test_hidden` coordinates.
Never pool hidden coordinates from different folds as one training dataset.
Upstream training hidden states are IN-SAMPLE, not inner OOF. Consequently this
version freezes LogisticRegression(C=0.1), all features, matching, repeats and
threshold before comparison and performs NO auxiliary inner hyperparameter
search. Original C43's inner epoch selection is already frozen; its models are
not re-trained. This is a fixed-hyperparameter strict outer-OOF incremental
comparison, NOT a new fully nested hyperparameter search. Later tuning would
require upstream inner refits in a new version.

## Arms

- M0: unchanged Direct-C43 final probabilities.
- M1: frozen hidden representation with fixed logistic readout (control only).
- M2: hidden + observation coverage + 124 verified-control ratios (primary).
- M3: same dimensions/coverage/model, random placebo instead of control ratios.
- M4: same dimensions/coverage/model, label-free condition-matched shuffled ratios.
- M5: hidden + coverage only, to diagnose observation-pattern signal.

M3/M4 each have 20 frozen repeats. Their repeat 0 comparisons are pre-specified;
all repeats and their 95th-percentile performance envelopes are reported. Null
matching is approximate: source-train-fitted P31 clusters, predicted side, Host
median and UNKNOWN median. No labels enter grouping or donor selection. Roles
absent in the receiver are masked. Low shuffle mobility makes evidence
inconclusive, not an excuse to relax matching after results.

124 evidence coordinates are four edge-normalized ratios for each Role31:
any verified call, all observed calls verified, any dominating no-bypass proof,
and bypass among verified calls. These are observed proof facts, not absence
flags or benign certificates. PARTIAL and UNKNOWN do not imply unguarded
execution. Permission facts, callback guesses, full taint and new API taxonomy
are explicitly excluded.

## Statistics and stop rules

Unit: APK (799 overall, 200 routed), not Host, call, seed or null repeat. Report
fixed/broken/net, FN/FP decomposition, BA/AUC/F1, two conditional AUCs and each
outer fold. Paired bootstrap uses 2000 resamples and fixed fold/label/route/
Stage-I-side strata. Ordinary 95% intervals and Bonferroni-adjusted intervals
for ten pre-defined contrasts are exported. These are conditional fixed-model
uncertainty estimates, not re-training/family-aware population guarantees.
Previously examined source data are not a pristine confirmatory holdout.

Both conditional sides must beat readout and null95 by >=0.02, with fold and
adjusted-interval support. Integration requires net>=33, broken<=17, at least
four non-decreasing-net folds and adjusted positive BA differences versus
Direct-C43/readout/placebo/shuffle. Exact criteria are in protocol.json. Failure
stops this version. Passing only permits designing frozen diagnostics; there is
no automatic future run or formal model replacement.

## Run

Upload `care-local-control-constraint-incremental-source-v1-package-r1.zip` to
`/home/boyuanjie/projects/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
if [ -e scripts/care-local-control-constraint-incremental-source-v1 ] ||
   [ -e outputs/care-local-control-constraint-incremental-source-v1 ]; then
  echo "Experiment exists: do not overwrite; inspect or resume its own launcher"
else
  unzip -q ../care-local-control-constraint-incremental-source-v1-package-r1.zip -d scripts/ &&
  chmod +x scripts/care-local-control-constraint-incremental-source-v1/*.sh &&
  LCIP_WORKERS=4 bash scripts/care-local-control-constraint-incremental-source-v1/launch_tmux.sh
fi
tail -F outputs/care-local-control-constraint-incremental-source-v1.console.log
```

No GPU, network downloads, SDK, APK execution or Git command is needed.
Existing formal C43 `representation_store.npz`, exact replay validation and
passed LCC v2 output/caches are required. Missing inputs cause explicit BLOCKED;
the package never silently re-trains a model to replace them.

Optional path variables: LCIP_APK_INDEX, LCIP_APK_ROOTS, LCIP_ENDPOINT,
LCIP_LCC_V2_OUTPUT. Resume new scans only with identical engine/protocol hashes.
All writes are restricted to this experiment output directory. Original 799
APKs, formal caches/references and all reused parent scans are hashed before/
after; before-manifest is never overwritten on resume.

The extra 599 scans extrapolate to roughly 69 minutes of summed CPU scan time
from v2's measured 23 minutes/200, NOT a wall-time promise. Four workers can
parallelize scans; file hashes/I/O, readout/null fits and bootstrap add cost.
Actual timing is exported. A per-APK timeout/cap remains an explicit failure.

Completion marker: `CARE_LCC_INCREMENTAL_SOURCE_DONE` (including negative
research decisions). `CARE_LCC_INCREMENTAL_SOURCE_FAILED` is a technical block.

```bash
bash scripts/care-local-control-constraint-incremental-source-v1/collect_results.sh
```

Key outputs: source_metrics.csv, fold_metrics.csv, paired_intervals.csv,
null_donors.csv, fold_training_audit.csv, feature_audit.csv, final_decision.json,
REPORT.md, before/after manifests, integrity check and measured timing.
