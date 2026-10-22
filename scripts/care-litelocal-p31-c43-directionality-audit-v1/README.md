# CARE-Droid P31-C43 Host directionality audit v1

This strict-2017 read-only audit asks whether the frozen Host representation
contains class direction, rather than only rarity/anomaly magnitude.

For each outer fold it constructs two symmetric reference backgrounds from the
other four folds only: benign and malware. A training APK contributes at most
one centroid per primary role, so large APKs cannot dominate merely by having
more Hosts. Each validation Host is ranked by its distance to the five nearest
benign and malware APK-role centroids.

Three spaces are audited separately:

- `CORE43`: duplicate-free Host self semantics;
- `RELATION31`: directed one-hop role relations;
- `JOINT74`: both frozen representations together.

Positive counterevidence means the class-conditional distance opposes P31's
predicted class. Results are reported for the 32 wrong/zero-direction errors,
8 bounded fixes, 6 too-weak cases, and 154 fold/prediction-matched correct
controls. This is diagnostic statistic construction, not model training or
neural inference. It does not load future rows or Ego.

Run from the repository root:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-litelocal-p31-c43-directionality-audit-v1.zip -d scripts/
bash scripts/care-litelocal-p31-c43-directionality-audit-v1/launch_tmux.sh
tail -F artifacts/care-litelocal-p31-c43-directionality-audit-v1.console.log
```

Inspect:

```bash
cat artifacts/care-litelocal-p31-c43-directionality-audit-v1/REPORT.txt
column -s, -t artifacts/care-litelocal-p31-c43-directionality-audit-v1/separability.csv
column -s, -t artifacts/care-litelocal-p31-c43-directionality-audit-v1/cohort_summary.csv
cat artifacts/care-litelocal-p31-c43-directionality-audit-v1/hard_audit.json
```

`apk_directionality.csv` contains APK-level matched-control ranks;
`routed_host_directionality.csv` contains the underlying Host-level distances.
