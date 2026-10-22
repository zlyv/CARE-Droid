# CARE future local assets v2

This extraction-only package builds the frozen 2018--2021 lineage (1989 APKs:
793/399/397/400 by year), parses every APK once, and stores reusable Host,
sensitive-endpoint directed 1-hop Ego, global role-graph, and endpoint payload
assets under `/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2`.

It is resumable through one atomic NPZ shard per APK and guarded by an
exclusive writer lock. It does not fit a model, select a hyperparameter, or use
future labels to revise the 2017-frozen method. Labels are retained only as
cohort metadata for later diagnostic evaluation.

The lineage is read from the complete HDD-backed
`v12_selective_role_v3_full_2017_2021/temporal_eval_manifest.csv`; it does not
depend on the source-only `artifacts/p1-features/features.csv`. Historical
`/home/boyuanjie/datasets/...` APK paths are deterministically remapped to the
mounted `/mnt/HDD1/boyuanjie/tasa-droid-storage/datasets/...` copy and audited.
The complete manifest contains 1991 future rows. The frozen evaluable cohort is
anchored to the bundled immutable 1989-SHA reference exported from an already
reported future diagnostic before this cache build. The reference contains
793/399/397/400 APKs for 2018--2021. The extraction-status table only confirms
successful extraction for those frozen samples; its historical schema does not
contain `temporal_eval_eligible`.

If source extraction is still running, use `launch_after_source_tmux.sh` to
queue this job and avoid two Androguard parses competing for CPU/RAM/I/O.
