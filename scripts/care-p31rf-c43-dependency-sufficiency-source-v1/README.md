# CARE-Droid verified-dependency sufficiency probe v1

This is a strict 2017 source-only representation probe. It does not retrain or
modify C43 and never loads future rows.

It compares Role, Role+API identity, within-Host role co-occurrence, verified
result-centred role dependency, count-only, and prediction-stratified shuffled
dependency under the same five-fold linear-probe protocol.

The verified dependency input reuses `artifacts/p71/verified_dependency_features_2017_v3.csv`.
If it is absent, the launcher performs a resumable 32-shard extraction with two
CPU workers, then constructs P71 automatically. Completed shards are skipped on
rerun. Worker logs are stored under the experiment artifact directory.
That artifact admits only register def-use, interprocedural argument/return, and
resolved field write/read evidence. Same-method membership, same-class
membership, reachability, and co-occurrence are not dependency evidence.

Important limitation: P71 is APK-level and preserves role-pair dependency, not
exact source API identity or C43 Host ownership. Therefore a PASS authorizes a
Host-aligned exact-API re-extraction, not direct C43 integration.

Run:

```bash
bash scripts/care-p31rf-c43-dependency-sufficiency-source-v1/launch_tmux.sh
tail -F artifacts/care-p31rf-c43-dependency-sufficiency-source-v1.console.log
```

