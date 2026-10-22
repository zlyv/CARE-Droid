# Reproduce

```bash
cd /home/boyuanjie/projects/tasa-droid
CUDA_VISIBLE_DEVICES=1 SMCIA_DEVICE=cuda bash scripts/care-direct-c43-strict-matched-audit-v1/launch_tmux.sh
tail -F outputs/care-direct-c43-strict-matched-audit-v1.console.log
bash scripts/care-direct-c43-strict-matched-audit-v1/collect_results.sh
```

Expected completion marker: `CARE_DIRECT_C43_STRICT_MATCHED_AUDIT_DONE`.

