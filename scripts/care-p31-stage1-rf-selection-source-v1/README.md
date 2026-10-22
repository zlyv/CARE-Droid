# CARE-Droid P31 Stage-I RF selection source v1

Strict 2017 five-fold OOF comparison. It never loads a future row.

- A: locked LR Stage-I and LR-anchored route.
- B: RF Stage-I with the locked LR-anchored route (classifier-only change).
- C: RF Stage-I with RF-anchored MeanOpposition Top25%.
- D: mean committee Stage-I with mean-anchored MeanOpposition Top25%.

All end-to-end arms use capacity-matched P0-ZERO6 C43-RQ, three existing
backgrounds, all outer-train APKs, three seeds, and max_delta=2.25. RF/mean
outer-train probabilities are strict nested OOF.

Run:

```bash
unzip -o ../care-p31-stage1-rf-selection-source-v1.zip -d scripts/
bash scripts/care-p31-stage1-rf-selection-source-v1/launch_tmux.sh
tail -F artifacts/care-p31-stage1-rf-selection-source-v1.console.log
```
