# HDBC-V1 运行与证据边界

项目：`/home/boyuanjie/projects/tasa-droid`。代码：`scripts/care-hdbc-v1/`。Python：`/home/boyuanjie/envs/tif/bin/python`。

只读源缓存和冻结模型；不打开、安装或执行APK。原RoleGraph InterZ961已含Host间有向Role统计；本实验的增量候选主要是仅DIRECT_CALL的可追溯1/2-hop路径及分跳数编码，不能把1-hop称为全新证据。

本次实际tmux名称：`care_hdbc_v1_20261017_183558`，输出目录：`/home/boyuanjie/projects/tasa-droid/outputs/care_hdbc_v1`。运行状态以输出目录中的`launch_info.txt`为准。查看：

```bash
tmux ls
HDBC_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care_hdbc_v1 bash /home/boyuanjie/projects/tasa-droid/scripts/care-hdbc-v1/status.sh
tail -n 50 /home/boyuanjie/projects/tasa-droid/outputs/care_hdbc_v1/run.log
```

进入会话使用`tmux attach -t care_hdbc_v1_20261017_183558`，脱离按Ctrl-b再按d。仅当输出目录有`asset_manifest_before.json`且未完成时，使用相同`HDBC_OUTPUT`重新执行`launch_tmux.sh`安全恢复；已有checkpoint以`xb`方式写入，不会覆盖。

```bash
cd /home/boyuanjie/projects/tasa-droid
HDBC_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care_hdbc_v1 \
  CUDA_VISIBLE_DEVICES=1 bash scripts/care-hdbc-v1/launch_tmux.sh
```

最终报告：`outputs/care_hdbc_v1/final_report.md`；判定：`final_decision.json`。2018—2021仅是冻结历史开发诊断。
