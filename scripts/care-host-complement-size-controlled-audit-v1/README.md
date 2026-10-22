# CARE-Droid Host Complement Size-Controlled Audit v1

This isolated diagnostic tests whether A/B/C/ABC Host-complement signals remain
after controlling for four predeclared size variables: Host count, method count,
graph node count and graph edge count.

For each candidate it reports strict same-year five-fold OOF results for:

1. size only;
2. size plus candidate features;
3. candidate features residualized against size using outer-train data only.

All diagnostic classifiers are fixed L2 logistic regressions. Residualization,
scaling and probe fitting use outer-train rows only. There is no hyperparameter
search, CARE-Droid replay, neural training, APK analysis, or prediction change.

Run:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-host-complement-size-controlled-audit-v1.zip -d scripts/
bash scripts/care-host-complement-size-controlled-audit-v1/launch_tmux.sh
tail -F artifacts/care-host-complement-size-controlled-audit-v1.console.log
```
