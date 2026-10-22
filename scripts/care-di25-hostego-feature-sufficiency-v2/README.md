# CARE-Droid Stage-II Feature Sufficiency Audit V2

This is a representation diagnostic, not a CARE-Droid variant.

The pipeline has three sequential hard gates:

1. manifest-driven 60/60 original-M1 checkpoint and saved-validation replay;
2. deterministic outer-train-only preprocessing reconstruction, verified against every saved validation input array;
3. fixed LogisticRegression/Ridge probes trained separately per year/fold/seed, with validation probabilities ensembled across seeds.

`R4_pre_delta` is explicitly treated as a mixed 113-D vector:

`Global24(24) + Stage-I logit(1) + graph token(32) + critical HostEgo(24) + attended HostEgo(24) + summary(8)`.

No Stage-II optimizer or neural training appears in this package. Any failed gate stops all downstream probes.

Run in tmux:

```bash
bash scripts/care-di25-hostego-feature-sufficiency-v2/launch_tmux.sh
tail -F artifacts/care-di25-hostego-feature-sufficiency-v2.console.log
```
