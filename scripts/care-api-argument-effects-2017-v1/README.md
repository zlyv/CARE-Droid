# CARE-Droid 静态 API 参数效果实验 v1

## 范围

只分析 2017 年原 799 个 APK。复用 Manifest135 自主路由和 W0 的原五折、
三 seed 缓存，不修改 RF、路由、W0、原型、APK 或历史输出。
这里没有动态分析、APK 安装/运行、shell 命令执行或未来年份评估。

第一版的调用范围是 Runtime.exec 的 String/String[] 重载和
ProcessBuilder.start。固定效果词表为查询、权限辅助程序、权限/所有者改变、
文件操作、包操作、网络工具、进程操作等 14 类。它们是命令形式特征，
不能直接解释为恶意规则、运行时效果或可执行文件身份。

静态解释器支持基本块内字符串/整数常量、移动、String.concat、
StringBuilder 的少数精确方法、局部 String[] 构造/赋值，以及
ProcessBuilder(String[])。没有跨方法、堆/字段、反射或加密字符串恢复；
整数主要用于数组索引和字符串拼接，尚未实现 Android 权限标志解析。
分支汇合、异常处理器和未建模操作均保守切断传播。UNKNOWN 单列，不当作无风险。
不输出原始参数、命令、URL 或字符串。

Runtime.exec(String) 按 Java StringTokenizer 分词，并不自动按 shell 解析。
只有显式 shell -c 且能解析为单条无扩展命令的情况才恢复内部类别。
复合 shell、展开、重定向和不确定值均保留未知。

## 输入

默认项目是脚本所在仓库；支持 --project、--reference、--lineage、--output。

- --reference 默认 artifacts/care_api_manifest_fusion_2017_future_v1。
  只读取其中 source_2017/source_oof_predictions.csv、
  15 个 features/w0_repr/S[1-5]_<seed>.npz，以及 source 799 个 frozen_views。
- --lineage 默认
  /mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/source_lineage_2017.csv。
  必须精确对应 799 个 SHA，包含 absolute_path 和 metadata_year/year。
  通过 absolute_path 访问服务器 APK，不要求上传数据集，不根据路径猜标签。
- 输出默认 artifacts/care_api_argument_effects_2017_v1。
  输出路径必须独立于输入，已有目录必须显式 --resume，且源码、输入和依赖指纹一致。

提取覆盖全部 799 个样本，而非仅抽取全局 OOF 路由样本充作训练集。
只解码冻结 Host 中直接调用目标 API 的方法；先前 Host 选择/裁剪之外的代码
不在本实验结论范围内。多 DEX 同 ID、未知方法和未恢复调用分别记录。
每次新扫描均在前后核对 APK 内容 SHA。缓存恢复验证内容指纹、契约和 frozen view，
不会重新读全部原始 APK；不据此声称恢复时重新核对过 APK 字节。
任何 APK 扫描失败都会停止，不删除样本，也不补造零向量。

## 两级判断

1. 无标签可行性门槛：至少 30 个参数可恢复 APK，至少 10 个在任一冻结路由
   seed 中被路由，事件恢复比例至少 20%，每个外层验证折至少 3 个可恢复 APK。
   未通过时输出 INSUFFICIENT_STATIC_INPUT，不做模型训练。
2. 通过后，在每个 W0 fold/seed 的对应坐标中训练固定 L2 offset logistic
   轻量读出。输入为 W0 113D 和有限静态特征，offset 为该 W0 logit，
   L2=0.1，阈值=0.5，无参数搜索。标准化只用当前外层训练集。

固定比较臂：

- W0：原始冻结概率，必须重放端到端 BA 0.9194653299916458。
- CODE：W0 表示的固定读出对照。
- COARSE：代码 + 覆盖率 + 13 个粗参数类别。
- VALUE：COARSE + 调用通道边际 + 参数效果边际。
- EFFECT_SHUFFLE_0..4：匹配规模、恢复状态和主通道后，在 APK 之间
  置乱效果向量。训练和验证分别置乱，不读取标签。
- BOUND：VALUE + 调用通道×效果联合坐标。
- BINDING_SHUFFLE_0..4：置乱联合坐标，保留每个接收 APK 的 VALUE 坐标。
- PLACEBO：VALUE + 训练集矩估计的无标签确定性噪声联合坐标。

COARSE 是相同调用范围内的 S1 风格类别对照，不是对旧全 Host 16D
S1 adapter 训练结果的重放。各臂使用同一个正则化读出，置乱臂与对应真实臂
维度完全一致；零填充不被宣称为有效容量匹配。CODE 控制重新拟合已有表示，
真正同维度的置乱/PLACEBO 控制额外特征拟合机会。

置乱是 APK 行绑定破坏，保留相应 cohort 的 donor 行多重集，匹配是近似的；
它不是统一重命名词表。未活跃样本保持零。任何训练/验证折的有效变化比例
不足 10%，相应方向标记 INCONCLUSIVE_NULL_MOBILITY，不能宣称该特征已被否定。

VALUE 必须同时超过 W0、CODE、COARSE 和全部 5 个效果置乱对照；
BOUND 必须先满足 VALUE 的支持条件，再超过 W0、VALUE、PLACEBO 和全部
5 个联合置乱。16 项预定比较统一 Holm 校正；配对 APK bootstrap 2000 次，
同时报告带 +1 修正的单侧 APK sign-flip p 值。支持要求 BA 差和 CI 下界
均正、Holm p<0.05、Net Fixed 优于 W0，并满足相应对照的至少 4/5 折、
至少 2/3 W0 seed 正向。置乱重复和模型 seed 不充当独立 APK。

这是对历史研究过的源域与冻结路由的条件分析。复用的路由并未在此重新做
严格嵌套，bootstrap 不包含家族相关性或重新训练整个系统的不确定性。
即使通过，也只允许设计独立复核，不能直接替换正式方法。

## 服务器运行

先将整个新目录放到服务器的 scripts/ 下。只需现有 CPU 环境，无 GPU。
原始依赖环境如果缺包，使用独立环境和 requirements.txt；启动脚本不会
自动安装或升级服务器 tif 环境。

从项目根目录执行：

    /home/boyuanjie/envs/tif/bin/python -B scripts/care-api-argument-effects-2017-v1/selftest.py
    bash scripts/care-api-argument-effects-2017-v1/run_on_server.sh

在独立 tmux 会话中执行上面的启动脚本；不要停止其他会话。
默认先做两个含目标 API 的真实 APK smoke，随后扫描 799 个 APK，
自动执行可行性判断和允许的固定读出。真实 smoke 只证明解析完成，
参数恢复数量会单独记录，不能用一个 PASS 掩盖 UNKNOWN。

已有同一版本输出、任务已经退出且需要继续时：

    bash scripts/care-api-argument-effects-2017-v1/run_on_server.sh --resume

也可以分阶段执行，并使用同一组路径参数：

    python -B scripts/care-api-argument-effects-2017-v1/run_experiment.py preflight
    python -B scripts/care-api-argument-effects-2017-v1/run_experiment.py smoke --resume
    python -B scripts/care-api-argument-effects-2017-v1/run_experiment.py extract --resume
    python -B scripts/care-api-argument-effects-2017-v1/run_experiment.py probe --resume

源码或配置改变时另建版本，不复用旧目录。提取默认为 2 个子进程，
每 APK 最长 900 秒；不提供未经实测的总耗时承诺。

关键结果：

- logs/run_status.json：真实阶段和终止状态；
- logs/progress.jsonl：追加的阶段、进度和错误记录；
- reports/extraction_coverage.csv、feasibility.json；
- reports/source_metrics.csv、W0_seed_metrics.csv、source_predictions.csv；
- reports/comparisons.csv、null_mobility.csv；
- reports/final_decision.json、final_report.md；
- integrity/before.json、after.json、outputs_sha256.json。

首次云端验证使用合成指令和本地缓存接口；真实 APK smoke、提取覆盖、
新增分类增益必须在拥有原始数据的服务器上验证，不能由本地单元测试代替。
