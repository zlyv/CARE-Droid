# Reproduce

```bash
cd /home/boyuanjie/projects/tasa-droid
CUDA_VISIBLE_DEVICES=1 RPF_GPU=1 bash scripts/care-direct-c43-rolepath-factorial-v1/launch_tmux.sh
tail -F outputs/care-direct-c43-rolepath-factorial-v1.console.log
bash scripts/care-direct-c43-rolepath-factorial-v1/collect_results.sh
```

Expected completion marker: `CARE_DIRECT_C43_ROLEPATH_FACTORIAL_DONE`.
