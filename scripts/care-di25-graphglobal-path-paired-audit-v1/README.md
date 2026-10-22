# CARE-Droid GraphGlobal path paired source audit v1

This is a read-only strict-2017 audit of the four completed GraphGlobal path
arms. It performs no training and reads no 2018--2021 row.

Outputs include:

- aggregate and per-fold metrics;
- APK-level paired Q-vs-other changes;
- exact McNemar tests;
- label-stratified paired bootstrap intervals for BA, AUC, F1, Fixed,
  Broken, and Net;
- a source-freeze **candidate** under a documented rule.

Fold-direction decisions use a `1e-12` zero tolerance so exact metric ties are
not misclassified as improvements by floating-point roundoff.

The candidate is development evidence, not independent confirmation. A future
diagnostic may be executed only after explicit human approval freezes the
representation and prohibits revision from future results.

## Run

Upload the ZIP to `/home/boyuanjie/projects/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-graphglobal-path-paired-audit-v1.zip -d scripts/
bash scripts/care-di25-graphglobal-path-paired-audit-v1/run_on_server.sh
cat artifacts/care-di25-graphglobal-path-paired-audit-v1/REPORT.txt
```

