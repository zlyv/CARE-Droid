# CARE-Droid P31-C43 localization 2x2 source v1

Strict 2017 source-only experiment. It keeps P31 Mean Committee, fold-wise
P31 MeanOpposition DI25, C43, Relation31, the three fold-safe backgrounds,
the 113-D bounded correction interface, all-outer-train Stage-II scope, and
the 32,292-parameter capacity fixed.

The exact source code confirms that `context_entropy` is only a duplicate
diagnostic field for this non-dual-stream model. It is not a second entropy in
the eight-dimensional correction summary. Therefore this experiment does not
perform the previously proposed false "summary deduplication" ablation.

Factors:

- critical evidence: hard argmax versus attention-renormalized Top-3 pooling;
- Relation31 localization: raw relation query versus a fixed, mean-preserving
  gate based on within-APK non-relation Host evidence.

Arms:

- `L0_LOCKED`: existing C43-RQ predictions, not retrained;
- `L1_TOPK3_RAW`;
- `L2_ARGMAX_ALIGNED`;
- `L3_TOPK3_ALIGNED`.

No threshold, K, temperature, route, background, loss, or bound is searched.
No future/within-year row or Ego tensor is read.

## Run

Upload the zip to `/home/boyuanjie/projects`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-litelocal-p31-c43-localization-source-v1.zip -d scripts/
bash scripts/care-litelocal-p31-c43-localization-source-v1/launch_tmux.sh
tail -F artifacts/care-litelocal-p31-c43-localization-source-v1.console.log
```

The two marginal arms run concurrently on GPU0/GPU1. The interaction arm runs
afterward as one complete three-seed ensemble to avoid concurrent output-file
writes.

## Results

```bash
cat artifacts/care-litelocal-p31-c43-localization-source-v1/results/REPORT.txt
column -s, -t artifacts/care-litelocal-p31-c43-localization-source-v1/results/failure_mechanisms.csv
column -s, -t artifacts/care-litelocal-p31-c43-localization-source-v1/results/paired_vs_L0.csv
column -s, -t artifacts/care-litelocal-p31-c43-localization-source-v1/results/per_fold_metrics.csv
cat artifacts/care-litelocal-p31-c43-localization-source-v1/results/hard_audit.json
```
