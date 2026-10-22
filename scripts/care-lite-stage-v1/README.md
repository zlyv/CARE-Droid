# CARE-LiteStage-V1

真实源APK只读扫描，2017固定五折内选择与校准。2018—2021仅在2017最终模型冻结后进行历史诊断。P31-RF保留为强基线。新源代码只在`/home/boyuanjie/projects/tasa-droid/scripts/care-lite-stage-v1/`；结果和shard只在`/home/boyuanjie/projects/tasa-droid/outputs/care_lite_stage_v1/`。

运行：`bash scripts/care-lite-stage-v1/run.sh`。

状态：`bash scripts/care-lite-stage-v1/status.sh`；日志：`tail -n 50 outputs/care_lite_stage_v1/run.log`。tmux进入：`tmux attach -t <实际session名>`；退出保持任务运行：`Ctrl+B`，再按`D`。

安全恢复：`bash scripts/care-lite-stage-v1/run.sh --resume`。成功shard不会被覆盖；阶段输出使用独占创建。若某阶段已写出部分结果后因代码错误失败，必须人工审计该阶段产物并新建版本，不强行覆盖。没有完整799源特征不报告正式OOF；未来缺失APK报告覆盖，不静默删样本。

L1 Manifest；L2 DEX元信息；L3 MethodId引用；L4 invoke扫描；L5冻结P31-RF。L3引用不意味着invoke，L4静态invoke不意味着运行时执行。MeanOpposition只用于经同一LiteStage特征重新校准的四成员和历史P31-RF；L1–L4单臂辅助对照用独立的RF不确定性Router。Stage-II专家为SCBE M0训练于同一Outer Train、等BCE且输入Stage-I logit置零，故作为路由无关的冻结共享专家；不冒充重新训练的route-conditioned专家。
