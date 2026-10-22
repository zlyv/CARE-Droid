# CARE-Droid Sensitive Consequence Slice v1

This source-only experiment removes the already-failed global-entry requirement. It reuses the frozen 2017 P31-RF/DI25/A0-C43 assets and the existing callsite-aware CMW graph cache.

The target representation is a bounded shortest-path projection from each frozen C43 Host to coarse external effects. Methods outside these witness paths are excluded. This is call reachability, not taint or data flow.

The pipeline first applies a label-free coverage/pruning/shortcut gate. Only if that gate passes does it run fixed five-fold, side-conditional direction probes. It never runs end-to-end correction or future diagnostics.

```bash
bash scripts/care-sensitive-consequence-slice-v1/launch_tmux.sh
tail -F outputs/care-sensitive-consequence-slice-v1.console.log
bash scripts/care-sensitive-consequence-slice-v1/collect_results.sh
```
