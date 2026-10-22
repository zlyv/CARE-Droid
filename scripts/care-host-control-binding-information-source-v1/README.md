# Host-Control Binding Conditional Information Test v1

This new isolated branch tests a compact representation, not a new MIL, gate,
deeper RoleGraph or official model. The passed label-free observability gate
does not by itself establish class information. This package supplies fixed
auxiliary readouts, with full controls and conservative stopping criteria.

## Frozen comparison

- B0: original Direct-C43, never refit.
- B5: frozen hidden readout, replay of previous M1_HIDDEN_READOUT.
- B1: hidden + coverage64 + proof ratios124, replay of previous M2_HIDDEN_CONTROL.
- B2: B1 + 64 true Host-binding coordinates (primary).
- B3: B1 + 64 internally shuffled-binding coordinates, 20 repeats.
- B4: B1 + 64 random placebo coordinates, 20 repeats.

B1 and B5 must reproduce their prior routed probabilities within 1e-8 BEFORE
any B2/B3/B4 fits. All auxiliary fits use the same fixed StandardScaler + LR
(C=0.1), threshold 0.5 and all outer-train APKs. No C, projection, threshold or
seed search is permitted. The 9176-coordinate covariance from the passed audit
is reduced by one fixed, label-free signed CountSketch to 64 dimensions. No
additional APK scan, background fit, C43 training, graph construction, GPU,
network download, SDK, APK execution or Git command is needed.

For every fold, train/test hidden coordinates come from its SAME upstream
model. Training hidden states are upstream IN-SAMPLE, not inner OOF. Therefore
there is no inner selection or claim of a new fully nested fit; a later tuned
method would require separate upstream inner refits. Global OOF routing is not
used to select training rows. Unrouted test predictions remain Stage-I exactly.

## Null and representation boundary

B3 replays all 799x20 passed within-APK permutations with the exact Role set,
event-count and UNKNOWN-count matching and distinct-Host rotation. Original
124+64 aggregates remain unchanged. All previous observability joint-change
values must reproduce before sketches are accepted. Projected mobility is
reported; failure stops before fits instead of trying another projection.

B4 moments use outer-train real sketches only. Its noise is deterministic per
SHA and repeat. All coordinates are zero when every per-Role proof flag is
constant, matching the representation's structural support; this support is
already determined by the 124 ratio coordinates. B2/B3/B4 have the same input
width and auxiliary model capacity. Placebo and shuffle repeats are not APKs
or independent replicates, and no exact conditional-randomization or causal
claim is made.

This is NOT a benign certificate. Proofs concern method-local CFG structure,
not consent, data purpose, runtime execution or source-to-sink causality.
UNKNOWN/PARTIAL and missing edges remain uncertainty. Raw exact APIs and Host
pseudo-labels are excluded. A failure is scoped to this fixed compact
representation/readout, not all future Host or RoleGraph representations.

## Reporting and frozen continuation gate

Report BA/AUC/F1, fixed/broken/net with FN/FP decomposition, damage rate, two
conditional AUCs and all five folds. Unit: APK, not Host or null repeat. Primary
paired bootstrap has 2000 fold/label/route/prediction-side-stratified resamples,
ordinary 95% intervals and Bonferroni-adjusted intervals for ten differences.
Six fixed/broken/net intervals are exploratory. Report all 20 null outcomes,
not only repeat zero, and compare both directions to each null's 95% envelope.

Both sides must improve by >=0.02 over B1 and EACH null95, show at least three
positive folds out of at least four evaluable folds, and have positive adjusted
primary AUC difference lower bounds. Integration also requires net strictly
above BOTH B0/B1, broken no greater than EITHER, four nondecreasing-net folds
and three improving-net folds vs B1, plus adjusted BA comparisons and null95
support. One supported direction is PARTIALLY_SUPPORTED, not a license for
future runs. A pass only permits separately specified frozen diagnostics.

All inference conditions on fitted models and this historically examined
source development cohort. Code-family dependence and retraining variability
are not captured. Within-version multiplicity does not correct the entire
research history, and this is not pristine confirmatory testing.

## Inputs and protection

Require the passed observability directory and ALL previous 799 scans/frozen
Hosts, the previous aggregate feature/prediction matrices and protocols,
formal SVM/Direct-C43 predictions, frozen representation_store.npz and its
validated replay. Inputs are discovered from the passed audit's actual paths.
Before/after manifests protect all reused files; current hashes must also match
the passed audit. No original APK is opened; integrity claims cover reused
files only. Missing inputs block execution without excluding samples. Existing
experiment output/logs are refused; there is no overwrite/resume mode.

## Run

Upload `care-host-control-binding-information-source-v1-package-r1.zip` to
`/home/boyuanjie/projects/`. In the SERVER terminal:

```bash
cd /home/boyuanjie/projects/tasa-droid
if [ -e scripts/care-host-control-binding-information-source-v1 ] ||
   [ -e outputs/care-host-control-binding-information-source-v1 ] ||
   [ -e outputs/care-host-control-binding-information-source-v1.console.log ]; then
  echo "Experiment exists; do not overwrite."
else
  unzip -q ../care-host-control-binding-information-source-v1-package-r1.zip -d scripts/ &&
  bash scripts/care-host-control-binding-information-source-v1/launch_tmux.sh
fi
tail -F outputs/care-host-control-binding-information-source-v1.console.log
```

The upload is required BEFORE unzip. Optional HBI_PROJECT/HBI_OBSERVABILITY
override existing source project/audit locations, not cohorts or protocols.
Completion: `CARE_HOST_CONTROL_BINDING_INFORMATION_DONE`, including a negative
research decision. Technical failure: `CARE_HOST_CONTROL_BINDING_INFORMATION_FAILED`.

```bash
bash scripts/care-host-control-binding-information-source-v1/collect_results.sh
```

Measured wall costs are saved in phase_timing.json. This is cache processing,
five fixed auxiliary-fit batches and bootstrap, not a full five-fold C43 fit.
Local synthetic tests are not actual research results or a timing guarantee.
