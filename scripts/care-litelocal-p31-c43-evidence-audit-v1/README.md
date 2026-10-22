# CARE-Droid P31-C43 extracted-evidence audit v1

This is a read-only strict-2017 audit of the 46 routed P31 errors and the
154 routed P31-correct controls. It does not train or invoke either stage,
change the route, search a threshold, load a future row, or consume Ego.

It answers a narrower question than the existing failure-mechanism audit:
whether the 32 wrong/zero-direction errors have uniformly benign evidence,
or whether C43-RQ is instead confirming the Stage-I prediction in both
directions. Evidence values are compared with correct APKs in the same outer
fold and with the same P31 predicted class. This avoids interpreting an
absolute distance/commonness score as a benign/malware label.

## Server commands

The zip is uploaded to `/home/boyuanjie/projects`:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-litelocal-p31-c43-evidence-audit-v1.zip -d scripts/
bash scripts/care-litelocal-p31-c43-evidence-audit-v1/run_on_server.sh

cat artifacts/care-litelocal-p31-c43-evidence-audit-v1/REPORT.txt
column -s, -t artifacts/care-litelocal-p31-c43-evidence-audit-v1/evidence_group_summary.csv
column -s, -t artifacts/care-litelocal-p31-c43-evidence-audit-v1/matched_percentile_group_summary.csv
cat artifacts/care-litelocal-p31-c43-evidence-audit-v1/hard_audit.json
```

This audit is CPU-only and intentionally does not start tmux.

## Outputs

- `REPORT.txt`: directional sanity check and Relation31 coverage.
- `routed_apk_evidence.csv`: APK-level evidence and matched-control percentiles.
- `evidence_group_summary.csv`: cohort means/medians.
- `matched_percentile_group_summary.csv`: fold/prediction-matched evidence percentiles.
- `hard_audit.json`: cohort counts and cache keys selected at runtime.

For the frozen 105-D cache, Relation31 is read from columns `[74:105]`
(original Host74 followed by Relation31). A 74-D materialized Compact43+Relation31
cache is read from `[43:74]`; every other width is contract-blocked.
