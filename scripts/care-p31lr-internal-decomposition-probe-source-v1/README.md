# CARE-Droid internal decomposition source probe v1

Source-only strict-2017 diagnostic for the fixed P31-LR and LR-anchored
MeanOpposition Top25% cohort. It decomposes the internal background into:

- benign consensus;
- isolated exception;
- repeated/pervasive exception.

Role-conditioned benign percentiles and all support distributions are fitted
from outer-train only. Linear probes receive strict inner-OOF decomposition
features on outer-train. No C43 training, route change, or future row occurs.

```bash
unzip -o ../care-p31lr-internal-decomposition-probe-source-v1.zip -d scripts/
bash scripts/care-p31lr-internal-decomposition-probe-source-v1/launch_tmux.sh
tail -F artifacts/care-p31lr-internal-decomposition-probe-source-v1.console.log
```
