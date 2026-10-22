# Local Control Constraint Feasibility Audit (LCC-FA), stage 1

This independent source-2017 package audits all 200 frozen P31-SVM MeanOpposition Top25% APKs. It reads only SHA, fold and routed membership from the reference predictions. Labels, predicted-error groups and future cohorts are not used.

## Observation

1. Verify the signature/Role mapping/hash contract against the existing formal builder.
2. Recover the exact method and declared endpoint of every frozen Host-to-sensitive-endpoint edge. Host membership is checked against the formal `reconstruct_host_relations` implementation, including DEX-defined SDK callers irrespective of the external flag. Report full-APK event alignment separately from frozen-edge recovery.
3. Decode DEX instruction CFGs and derive postdominance plus exclusive successor reachability. Potentially throwing instructions can exit exceptionally. Catch handlers, reachable loops, unknown opcodes and malformed targets are UNKNOWN in v1.
4. Track a unique local permission return value through register moves and comparisons against zero. Require resolved framework dispatch and no bypass around the guard. Other permission candidates stay PARTIAL.

No absence of proof is called NO_GUARD. No control or permission fact is assigned a benign/malicious direction.

`protocol.json` freezes scope, limits and coverage thresholds before code scanning. Passing the control gate only supports designing a separate conditional incremental probe. Permission readiness is reported separately. This package never runs a classifier.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
LCC_WORKERS=4 bash scripts/care-local-control-constraint-feasibility-v1/launch_tmux.sh
tail -F outputs/care-local-control-constraint-feasibility-v1.console.log
```

Resume after a documented implementation/path repair using the same launch command once the prior session has exited. Cache reuse requires the same protocol hash and schema. Formal scripts, caches, APKs and historical results are protected by before/after manifests. Every write belongs to this new output directory; bytecode cache writes to imported formal modules are disabled.

## Output

- `selection_label_free.csv`, `environment_inventory.json`, `protocol.json`
- `sensitive_events.csv`, `control_proofs.csv`, `method_cfg_audit.csv`
- `frozen_edge_evidence.csv`, `missing_frozen_pairs.csv`
- `apk_coverage.csv`, `fold_coverage.csv`, `coverage.json`
- `baseline_asset_manifest_before.json`, `baseline_asset_manifest_after.json`, `baseline_integrity_check.json`
- `final_decision.json`, `REPORT.md`, `reproduce_commands.md`
- `cache/scans/*.json.gz`: DEX identities, exact invoke offsets and cited instruction evidence

Whole-APK caps do not identify which particular omitted edge was cropped; the corresponding reasons explicitly retain that uncertainty. Local proofs do not establish runtime execution, whole-program reachability, user consent, legitimate data purpose or malware status.

References for bytecode and permission contracts:

- https://source.android.com/docs/core/runtime/dalvik-bytecode
- https://developer.android.com/reference/android/content/Context#checkSelfPermission(java.lang.String)

Control-dependence and postdominance are established analysis tools. This stage measures operational observability, not publication novelty or class-direction value.
