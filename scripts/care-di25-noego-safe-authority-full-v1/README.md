# CARE-Droid strict nested source + frozen future safe authority v1

## Zero-byte future-asset recovery

The audited HDD cache contains all 1,989 non-empty per-APK NPZ shards, while
four historical merged outputs were replaced by zero-byte placeholders.
`run_on_server.sh` now rebuilds those merged files deterministically before any
model code runs.  It reconstructs the frozen lineage, validates every shard,
writes all outputs atomically, and requires a cross-file order/shape/offset hard
gate.  This recovery does not parse or re-extract any APK.

One long-running package with two evidence scopes:

1. strict 2017 outer five-fold OOF; every outer threshold is selected only
   inside its outer-train cohort using S3 Host-only+Stage-I-logit gate OOF;
2. source-frozen 2018-2021 temporal diagnostic. A single deployment threshold
   is selected from 2017 OOF, then frozen. Future labels are opened only by the
   final summarizer.

No Ego cache is read. Direction is opposite the frozen Stage-I decision and
magnitude is `min(abs(stage1_logit)+0.05, 2.25)`. Threshold utility is
`Fixed-1.5*Broken` with candidates 0.50..1.00. Future results cannot revise any
method choice.

If the recovered workspace lacks the old `resolved_inputs.json`, the runner
rebuilds that label-sealed manifest automatically. It reuses the complete HDD
Compact76 cache. It does not require or create `future_interz.npz`; source and
future S7 are reconstructed in memory from the audited endpoint caches.
If the old cache contains duplicate/outside rows, it is not edited. The runner
creates a new HDD lineage-exact snapshot, records every collapsed/excluded ID,
and asks the official resumable extractor to rebuild only missing or conflicting
IDs.

The recovered workspace may contain the obsolete 1991-row lineage. Before any
future preparation, the runner anchors the cohort to the exact 1989 SHA order
stored in the previously hard-gated `regions_2018_2021.npz`, joins metadata by
SHA, validates annual counts 793/399/397/400, and records every excluded row.
Recovered `artifacts/` files may also be zero-byte placeholders. Future regions
and endpoint tensors are therefore read from the original HDD asset root and
deep-validated (NPZ keys, gzip parsing, 1989 unique and identical SHA sets)
before nested training starts.

Upload into `/home/boyuanjie/projects/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-noego-safe-authority-full-v1.zip -d scripts/
bash scripts/care-di25-noego-safe-authority-full-v1/launch_tmux.sh
tail -F artifacts/care-di25-noego-safe-authority-full-v1.console.log
```

Final reports:

```bash
cat artifacts/care-di25-noego-safe-authority-full-v1/source/REPORT.txt
cat artifacts/care-di25-noego-safe-authority-full-v1/future/REPORT.txt
```
