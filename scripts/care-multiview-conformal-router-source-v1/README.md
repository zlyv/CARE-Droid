# CARE-Droid multi-view conformal router: strict 2017 source audit

This package changes only the routing component.  It reuses the frozen
Global24 OOF probabilities and the per-APK expert probabilities from the
deduplicated bounded RGCondAttn source run.

Views:

1. all Global24 features;
2. Manifest metadata (10; the constant `manifest_parse_ok` feature is absent
   from canonical Compact76);
3. code/call size (3);
4. sensitive behavior (5, including the retained
   `context_role__boot_persistence` semantic-context signal);
5. graph composition (6).

For each fixed outer fold, the outer training set is split into proper training
and a disjoint class-conditional conformal calibration subset.  Platt scaling
uses proper-training OOF probabilities.  NLL reliability weights use a held-out
portion of those OOF predictions.  The outer validation labels are not used by
the router.

Outputs include both an exact-budget `MVCC_Top25` comparator and an operational
`MVCC_FrozenTau` route whose threshold comes only from source calibration.

Run from the repository root:

```bash
bash scripts/care-multiview-conformal-router-source-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-multiview-conformal-router-source-v1.console.log
```

Expected prerequisite:

```text
artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv
```

This experiment never loads 2018--2021 data.
