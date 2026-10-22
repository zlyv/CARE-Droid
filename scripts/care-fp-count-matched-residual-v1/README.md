# CARE-Droid FP Count-Matched Semantic Residual Audit v1

This source-only diagnostic follows the partially supported Sensitive Consequence Slice v1 result. It evaluates only Stage-I predicted-malware samples and asks whether normalized effect semantics distinguish FP from TP after fold-safe control for graph, Host, and effect counts.

No entry mapping, future data, end-to-end correction, or formal model modification is used.

```bash
bash scripts/care-fp-count-matched-residual-v1/launch_tmux.sh
tail -F outputs/care-fp-count-matched-residual-v1.console.log
bash scripts/care-fp-count-matched-residual-v1/collect_results.sh
```
