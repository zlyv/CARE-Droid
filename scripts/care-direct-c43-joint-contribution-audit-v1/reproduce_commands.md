# Reproduce

```bash
cd /home/boyuanjie/projects/tasa-droid
CUDA_VISIBLE_DEVICES=1 JCIA_DEVICE=cuda bash scripts/care-direct-c43-joint-contribution-audit-v1/launch_tmux.sh
tail -F outputs/care-direct-c43-joint-contribution-audit-v1.console.log
bash scripts/care-direct-c43-joint-contribution-audit-v1/collect_results.sh
```

Expected completion marker: `CARE_DIRECT_C43_JOINT_AUDIT_DONE`.

