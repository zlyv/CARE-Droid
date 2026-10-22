# CARE-Droid source recovery + Stage-II-only v1

## Mandatory preflight gates (run before any long job)

The preflight is now deliberately separate from the full 799-APK runner:

```bash
bash scripts/care-source-recovery-stage2only-v1/launch_preflight_tmux.sh
tail -f artifacts/care-source-recovery-stage2only-v1.preflight.console.log
```

It performs only two bounded tasks and then stops:

1. Audits the exact 799-row 2017 lineage against `features.csv`, records every
   duplicate row, repairs only identical concurrent-append duplicates, rewrites
   the aggregate totals from the canonical 799 rows, and enforces hard set and
   multiplicity assertions.
2. Builds a separate fixed-50 endpoint probe cache and selects the first ten
   SHA-ordered, label-independent APKs that contain a sensitive Host with a
   real incoming or outgoing Host neighbour.  It then reconstructs directed
   1-hop Ego regions and emits per-Host vectors, neighbours, roles, and edges.

Outputs:

- `source_lineage_integrity_audit.txt`
- `source_duplicate_rows.csv`
- `ego_smoke_test_summary.txt`
- `ego_smoke_test_examples.csv`

The preflight never starts OOF training, never reads 2018--2021, and never
starts full 799-APK Ego extraction.  The full runner now owns a pipeline lock;
both append-based extractors also own per-output exclusive writer locks.

After inspecting both reports, approval is explicit and hash-bound:

```bash
bash scripts/care-source-recovery-stage2only-v1/approve_preflight.sh
```

This package restores the exact 799-APK 2017 source cohort from the retained
v12 dataset, rebuilds only the missing extraction caches, and runs the
predeclared Stage-II-only ablation.

Before the long extraction, it runs a mandatory fixed-50 Host/Ego smoke audit.
The audit reconstructs regions from the same endpoint payload used downstream,
checks real nontrivial one-hop neighborhoods, exercises synthetic no-host,
single-host, and multi-host edge cases, and prints concrete methods/roles/edges.
The full 799 run is blocked until the exact passing smoke report is manually
approved.

## Storage policy

- Python/shell files stay under the project's `scripts/` directory.
- Compact CSV/JSON/report files stay under project `artifacts/`.
- Only heavy generated endpoint and Host/Ego region caches are stored under
  `/mnt/HDD1/boyuanjie/tasa-droid-storage/care-recovery/`.
- Project artifact paths for those caches are symbolic links.
- Existing non-empty paths are never overwritten by the launcher.

## Protocol

`StageIIOnly-MalBen` uses strict fixed 2017 outer five-fold OOF.  Within each
fold, all scalers, benign/malicious role prototypes, representation parameters,
and the L2 logistic classifier are fit on outer-train only.  The final feature
is `[g_i, z_mal_i-z_ben_i, h_critical_i, e_mal_i-e_ben_i]`; Global24 features,
`p_stage1`, A25/DI, and bounded correction are excluded.  Stage II is invoked
for 100% of APKs.  No 2018--2021 row is loaded.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-source-recovery-stage2only-v1.zip -d scripts/
bash scripts/care-source-recovery-stage2only-v1/launch_tmux.sh
tail -f artifacts/care-source-recovery-stage2only-v1.console.log
```

The first launch stops after the smoke audit. Review and approve it:

```bash
cat artifacts/care-source-recovery-stage2only-v1/smoke/REPORT.txt
bash scripts/care-source-recovery-stage2only-v1/approve_smoke.sh
bash scripts/care-source-recovery-stage2only-v1/launch_tmux.sh
```

Important implementation boundary: the current v4/MalBen experiment consumes
`host_values` only. The smoke audit verifies that `ego_values` are valid and
different, but it does not relabel the current experiment as Host+Ego. Adding
Ego to the model is a separate method change that requires new 2017 source-side
parameter selection.

Both extraction stages and all model stages are completion-checked and can be
resumed by rerunning the same launcher.
