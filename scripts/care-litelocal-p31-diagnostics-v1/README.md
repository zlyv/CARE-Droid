# CARE-Droid LiteLocal P31 diagnostics v1

This package runs three isolated diagnostics for the frozen source-selected `P31 + calibrated mean committee` Stage-I:

1. strict-2017 P31-specific size-conditioned attribution;
2. frozen 2017-trained evaluation on 2018–2021, with future labels opened only by the final summarizer;
3. independent same-year five-fold OOF evaluation for each of 2018–2021.

The frozen-future model process receives only a label-free lineage. The separate same-year protocol necessarily uses each year's labels inside its training folds and is reported only as development/diagnostic OOF evidence.

Future APKs are scanned once with a resumable P31-only DEX invoke extractor. It does not instantiate Androguard `Analysis`, construct an inter-method call graph, or compute Host, RoleGraph, Ego, co-occurrence, transition, or SVD features. Raw count fields remain audit-only.

The two model jobs run concurrently after extraction. These classical models are primarily CPU-bound; the two GPUs are not required.

Upload the ZIP to `/home/boyuanjie/projects/` and run:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-litelocal-p31-diagnostics-v1.zip -d scripts/
bash scripts/care-litelocal-p31-diagnostics-v1/launch_tmux.sh
tail -F artifacts/care-litelocal-p31-diagnostics-v1.console.log
```

The final combined report is written to `artifacts/care-litelocal-p31-diagnostics-v1/REPORT.txt`.
