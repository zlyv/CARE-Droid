# CARE-EAD-V1

只读历史资产；2017诊断性Probe；不读取或运行APK，不修改正式模型。输出仅写入 `outputs/care_error_audit_v1/`。现存其他实验不受影响。

首次运行前需确认输出目录未被其他实验占用。`runner.py` 创建历史资产SHA256清单，每阶段前后核对。`run.sh --resume`只跳过已完成阶段；如某阶段失败并留下日志，先检查后再决定是否新建版本，绝不自动覆盖日志或结果。

查看状态：`bash scripts/care-error-audit-v1/status.sh`

进入tmux：`tmux attach -t <实际session>`；退出并保持运行：按 `Ctrl+B`，再按 `D`。

日志：`outputs/care_error_audit_v1/run.log` 与 `outputs/care_error_audit_v1/logs/<阶段>.log`。

安全恢复：`bash scripts/care-error-audit-v1/run.sh --resume`。

已有表示缓存若来自不同Direct-C43 checkpoint，或训练/测试Embedding拟合关系不对称，本版明确标记不可比较，不伪造严格OOF表示结论。未来数据仅为历史开发诊断。
