# DCE-V1 运行说明

项目：`/home/boyuanjie/projects/tasa-droid`。

先在前台用独立的 smoke 输出目录运行：

```bash
cd /home/boyuanjie/projects/tasa-droid
PYTHONDONTWRITEBYTECODE=1 DCE_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care_dce_v1_smoke_20261016 \
  CUDA_VISIBLE_DEVICES=1 /home/boyuanjie/envs/tif/bin/python scripts/care-dce-v1/main.py --smoke-test --device cuda
```

本次 smoke 已在 `outputs/care_dce_v1_smoke_20261016_r2` 通过。正式无人值守运行已启动在全新版本目录 `outputs/care_dce_v1_r3`（之前预检失败的 v1/v2 输出保持原样，没有清理）。

```bash
cd /home/boyuanjie/projects/tasa-droid
DCE_SESSION=care-dce-v1-r3 DCE_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care_dce_v1_r3 \
  CUDA_VISIBLE_DEVICES=1 bash scripts/care-dce-v1/launch_tmux.sh
tail -F outputs/care_dce_v1_r3/run.log
```

tmux 名称 `care-dce-v1-r3`；退出跟踪日志按 Ctrl-C；进入会话使用 `tmux attach -t care-dce-v1-r3`，脱离按 Ctrl-b 然后 d。实际启动配置见输出目录的 `launch_info.txt`。仅在已有 DCE 完整性清单存在时允许断点恢复，不覆盖历史资源。
