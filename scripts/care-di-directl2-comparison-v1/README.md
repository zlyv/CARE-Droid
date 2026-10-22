# CARE-Droid DI25 × Direct-L2 comparison v1

This package performs a strict 2017 source OOF 2×2 comparison:

| Router | BoundedCorrection | Direct-L2 |
|---|---|---|
| A25 | evaluated | evaluated |
| DI25 | evaluated | evaluated |

It reuses the frozen A25 and DI25 routes, the existing bounded expert
probabilities, and the existing strict-OOF Direct-L2 expert probabilities. No
model is retrained for a particular router.

Direct-L2 is an independent Stage-II classifier:

```text
[role graph; critical Host/Ego region; attended regions; summaries]
    -> absolute malware logit -> sigmoid -> p_direct
```

For a routed APK, `p_direct` replaces `p_stage1`; it is not a residual and is
not limited by `Delta_max`. Global24 and its probability/logit are excluded
from the Direct-L2 input.

## Prerequisites

```text
artifacts/care-committee-decision-instability-router-source-v1/source_sample_scores.csv
artifacts/care-rgcondattn-direct-l2-source-v1/source_predictions.csv
```

If the Direct-L2 artifact is absent, `run_on_server.sh` invokes the established
source Direct-L2 experiment first. Future rows are not loaded in this version;
the source decision must be made before any new future Direct-L2 diagnostic is
trained.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-di-directl2-comparison-v1.zip -d scripts/
bash scripts/care-di-directl2-comparison-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-di-directl2-comparison-v1.console.log
```

Output:

```text
artifacts/care-di-directl2-comparison-source-v1/
  summary.txt
  source_2x2_metrics.csv
  source_head_pairwise.csv
  source_sample_scores.csv
  protocol.json
```

`RER@25` and `HarmfulCapture` are computed separately for each Stage-II head,
because an error repairable by BoundedCorrection need not be repairable by
Direct-L2, and vice versa. The primary comparison is DI25+Direct-L2 versus
DI25+BoundedCorrection.
