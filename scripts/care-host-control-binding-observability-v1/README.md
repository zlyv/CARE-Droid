# Host-Control Binding Observability Audit v1

This package is a **label-free feasibility audit**, not another C43 model,
readout, gate, MIL or RoleGraph experiment. It reuses existing LCC scans and
checks whether Role-level aggregates hide heterogeneous bindings to frozen
Host coordinates. Even a pass does not prove class information or authorize
model training in this package.

## Scope and overlap

The previous incremental control experiment used APK-level 124 proof ratios
and 64 coverage values with frozen hidden readouts. Earlier HAIG work shuffled
hashed Host-call sidecars. Neither named implementation tests control facts
bound to C43 Host content while preserving all previous marginals. This is a
scoped code comparison, not an exhaustive novelty claim. Do not rerun an
equivalent historical experiment if an unindexed one is subsequently found.

## Inputs (read-only)

- Passed `outputs/care-local-control-constraint-incremental-source-v1` feature
  validation, integrity, all 799 raw scan paths, frozen pairs and proof matrix.
- Formal SVM source file: ONLY SHA, fold and routed membership are consumed.
- Source endpoint cache and `local_asset_manifest.json` Role permutation.
- Frozen `compact43_relation_source.npz`, or A0_ROLE_ONLY.npz compatibility
  cache; exact Host order, count, Role presence, endpoint sidecar and Relation31
  are checked against the retained graph. SHA joins are explicit.
- Formal `hostrel_features.py` and previous `feature_core.py`, imported with
  bytecode writing disabled. No model, hidden state or APK needs to be opened.

No SDK, GPU, new feature extraction, APK execution, network, download or Git
command is required. All writes stay in a fresh independent output directory.
Before/after SHA256 manifests protect every actual reused input. Original APKs
are not opened; integrity is not asserted for unaccessed files or the whole
project. Failed/missing APK scans block the audit, never remove samples.

## Exact synthetic control

Within each APK, exchange four proof flags only between observed frozen
Host-endpoint pairs with exactly the same Role set, event count and UNKNOWN
count, and different Host IDs. Whole flag bundles move together. Graphs and
raw traces remain unchanged. Coverage, previous proof ratios and flag
multisets must be exactly equal before/after every permutation.

Randomized distinct-Host blocks are rotated. If one Host dominates a stratum,
unmatched pairs remain fixed. Donor movement, actual flag movement and joint
descriptor movement are reported separately. Identical facts, identical Host
contexts and single-Host strata do not count as useful binding coverage.

The joint descriptor is Role-conditioned covariance of four flags with 74
frozen Host coordinates (SelfRole31, bounded Tail12, Relation31). It is only an
audit witness of non-identical joint structure, not a proposed 9176-dimensional
classifier. It excludes exact APIs, trained backgrounds and class labels.
UNKNOWN/PARTIAL never imply unguarded execution. Proofs do not establish
consent, data purpose, class direction or runtime causality. Synthetic swaps
are annotation controls, not executable or semantics-preserving APK changes.

## Frozen engineering gate

All 799 scan/context/marginal contracts and all 20 repeat invariants must pass.
In the 200 routed primary samples, require median changed APK rate >=20%, at
least four folds with median changed rate >=15%, and median >=25% of verified
pairs with changed binding. These are readiness thresholds, not significance
tests. No label-based stratification or tuning is performed. All 799 rows and
all repeats remain in denominators. A pass only allows specifying a separate
source-only information experiment. A failure stops this version.

## Run

Upload `care-host-control-binding-observability-v1-package-r1.zip` to
`/home/boyuanjie/projects/`. Then run the following in the server terminal.
The guarded block prints a message rather than exiting your interactive shell.

```bash
cd /home/boyuanjie/projects/tasa-droid
if [ -e scripts/care-host-control-binding-observability-v1 ] ||
   [ -e outputs/care-host-control-binding-observability-v1 ] ||
   [ -e outputs/care-host-control-binding-observability-v1.console.log ]; then
  echo "Directory or log already exists; do not overwrite."
else
  unzip -q ../care-host-control-binding-observability-v1-package-r1.zip -d scripts/ &&
  chmod +x scripts/care-host-control-binding-observability-v1/*.sh &&
  bash scripts/care-host-control-binding-observability-v1/launch_tmux.sh
fi
tail -F outputs/care-host-control-binding-observability-v1.console.log
```

Optional explicit input paths: HCB_PARENT, HCB_ENDPOINT, HCB_C43_CACHE,
HCB_ROLE_MANIFEST. These must refer to existing 2017 source assets; they never
permit future data, alternative Host selection or re-extraction. There is no
overwrite/resume mode; a new version is required if an existing run needs a
different implementation or protocol.

Completion marker: `CARE_HOST_CONTROL_BINDING_DONE` (including a negative
coverage decision). A technical failure prints
`CARE_HOST_CONTROL_BINDING_FAILED`, and its report records the blocking input.

```bash
bash scripts/care-host-control-binding-observability-v1/collect_results.sh
```

Key outputs: experiment_overlap_audit.json, apk_binding_audit.csv,
null_repeat_apk_metrics.csv, synthetic_binding_donors.csv (repeat 0 only),
binding_repeat_summary.csv, fold_binding_coverage.csv, binding_coverage.json,
final_decision.json, REPORT.md, before/after manifests, integrity and timing.

Runtime is CPU/disk work over existing caches, not 799 APK scans or five-fold
training. Actual time is measured; no server wall-time guarantee is made.
The package was locally self-tested but real cached-data coverage must be
obtained on the server.
