# OBCBC 与既有机制对照矩阵

## 读表规则

- **完整**：该论文把相应信息作为核心分析对象，并给出可执行或可评估的机制。
- **部分**：论文使用了相关信息，但分析单元、目标或覆盖范围与 OBCBC 不同。
- **否**：未发现该论文建立相应机制；这不代表论文质量不足。
- **OBCBC-R**：经第一轮新颖性审计后收窄的候选定义，即“最近第一方→第三方边界实例去卷积”。

| 工作 | 敏感行为表示 | 上下文类型 | 区分 first/third | 恢复调用边界 | activation | guard | UI | control dependency | data flow | 纯静态 | malware evidence | benign evidence | 共享库处理 | 时间评估 | 与 OBCBC-R 的重叠 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AppContext | 敏感调用及其上下文 | 入口事件、环境条件 | 否 | 跨 ICC 调用上下文，不是 ownership boundary | 完整 | 完整 | 部分 | 完整 | 否 | 是 | 完整 | 通过良恶上下文分类间接给出 | 否 | 否 | **高**：activation、guard 与敏感行为已被完整覆盖；未做共享第三方 Host 的边界实例去卷积 |
| FlowCog | source-to-sink flow | UI 文本、图像、view、guard | 否 | 否 | 完整 | 完整 | 完整 | 完整 | 完整 | 是 | 完整 | 完整：UI 是否解释具体 flow | 可观察第三方 flow，但不以边界实例归责 | 否 | **高**：UI 正当化与行为语义已覆盖；未按最近 ownership boundary 拆分同一 Host |
| CHABADA | 敏感 API 使用 | 商店描述主题 | 否 | 否 | 否 | 否 | 文本而非运行 UI | 否 | 否 | 是 | 异常偏离 | 功能一致性 | 否 | 否 | 中：行为—声明一致性重叠，但不是路径责任归属 |
| TriggerScope | 敏感方法及触发谓词 | 时间、位置、SMS 等条件 | 否 | 普通调用/控制路径，不是 ownership boundary | 部分 | 完整 | 否 | 完整 | 否 | 是 | 逻辑炸弹谓词 | 否 | 否 | 否 | 中高：guard 与路径谓词已有；责任单元不同 |
| TSOpen | TriggerScope 式敏感方法 | 条件谓词与过滤策略 | 否 | 否 | 部分 | 完整 | 否 | 完整 | 否 | 是 | trigger evidence | 否 | 否 | 否 | 中：主要构成覆盖率与选择偏差警告 |
| AppIntent | 隐私相关行为 | GUI action sequence | 否 | 入口到行为路径 | 完整 | 部分 | 完整 | 完整 | 完整 | 是 | 可疑意图 | 用户操作解释行为 | 否 | 否 | **高**：静态恢复 UI 动作到敏感行为路径；未处理 first/third 共享 Host 去卷积 |
| AsDroid | 敏感操作 | top-level function、UI text | 否 | 调用链到 top-level function | 完整 | 部分 | 完整 | 完整 | 部分 | 是 | 语义不匹配 | UI/功能匹配 | 否 | 否 | **很高**：行为归因与语义不匹配已覆盖；区别只能建立在 ownership-boundary instance 上 |
| User-trigger dependence profiling | 敏感 API | 用户输入依赖 | 否 | 入口/依赖路径 | 完整 | 否 | 用户输入 | 部分 | 完整 | 是 | 非用户触发敏感行为 | 用户触发解释 | 否 | 否 | 高：用户触发双向证据已有；未做共享依赖责任去卷积 |
| AutoPPG | 隐私收集行为 | 六类执行上下文、UI/系统/环境 | 完整（host/TPL） | host↔TPL 信息传递 | 完整 | 完整 | 完整 | 完整 | 完整 | 是 | 可疑隐私实践 | 上下文解释/隐私声明 | 完整但主要依赖类名/包归属 | 否 | **最高冲突**：同时覆盖 host/TPL、上下文和行为；未发现“同一共享 Host 按最近边界实例拆分并用于选择性纠错” |
| LibRadar | 库 API 特征 | 库身份/聚类 | 完整 | 否 | 否 | 否 | 否 | 否 | 否 | 是 | 否 | 否 | 完整 | 否 | 低机制重叠；可提供 ownership sidecar，但库身份本身被 OBCBC-R 禁止作为方向特征 |
| LibScout | 库 API/版本指纹 | 库身份与版本 | 完整 | 否 | 否 | 否 | 否 | 否 | 否 | 是 | 否 | 否 | 完整 | 否 | 低机制重叠；明确指出行为归属需求，但不恢复责任边界上下文 |
| Common Libraries | 库代码与恶意检测噪声 | 常见库/广告库 | 完整 | 否 | 否 | 否 | 否 | 否 | 否 | 是 | 过滤噪声 | 否 | 完整 | 否 | 中：证明共享库可制造捷径/误报，但方法是过滤而非责任去卷积 |
| MAPS | 隐私实践 | 第一/第三方包归属 | 完整 | 否 | 否 | 否 | 否 | 否 | 部分 | 是 | 风险实践 | 否 | 完整 | 有跨版本/规模观察 | 中高：归属已有，但依赖包命名，未到边界实例级 |
| DroidSIFT | API dependency graph | 上下文依赖图 | 否 | 调用/数据依赖 | 部分 | 部分 | 否 | 部分 | 完整 | 是 | 图异常/恶意模式 | 间接 | 未专门处理 | 否 | 中：行为图上下文已有，缺 ownership-boundary 责任单元 |
| MKLDroid | 多视图 dependency graph | API、权限、source/sink 等 | 否 | 图依赖 | 部分 | 部分 | 否 | 部分 | 完整 | 是 | malware localization | 间接 | 未专门处理 | 否 | 中：局部化已有；不做共享 Host 去卷积 |
| AMCDroid | 组件级图与特征 | 组件上下文 | 部分 | 组件/调用关系 | 部分 | 部分 | 否 | 部分 | 部分 | 是 | 恶意组件定位 | 间接 | 否 | 否 | 中：实例定位目标接近，但不是第一方→第三方边界责任 |
| Eclipse Steady / vulnerable-code reachability | 依赖中可达方法 | 应用调用上下文 | 完整 | 完整 | 否 | 否 | 否 | 调用可达性 | 否 | 是 | 漏洞可利用性/暴露 | 不适用 | 完整 | 否 | 高概念重叠：application-specific dependency reachability 已有；未用于 Android 敏感行为双向纠错 |
| MPI semantic-aware execution partitioning | provenance events | 进程内语义执行单元 | 不以 first/third 表示 | 以语义单元切分共享进程 | 完整 | 部分 | 否 | 部分 | 信息流/provenance | 动态审计，不符合后续静态边界 | 攻击证据 | 正常执行分区 | 处理共享进程混合 | 否 | 高概念重叠：共享节点去卷积已有跨域先例，限制 OBCBC-R 的广义新颖性表述 |
| HOLMES | provenance graph | 多阶段攻击语义 | 否 | provenance edge | 完整 | 部分 | 否 | 部分 | 完整 | 否 | 完整 | 基线/低噪声 | 共享系统实体 | 时间序列 | 中：提供责任链和相关事件聚合先例，不是 Android 静态边界 |
| TESSERACT | 任意 malware 特征 | 时间/空间评估约束 | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 与特征无关 | 评估框架 | 评估框架 | 可审计库偏差 | 完整 | 低机制重叠；对后续 frozen temporal protocol 约束很强 |
| Shortcut Testing | 模型对敏感编码的依赖 | 干预/反事实 | 不适用 | 否 | 否 | 否 | 否 | 否 | 否 | 可用于静态特征模型 | 识别有害 shortcut | 排除伪相关 | 可检验库身份捷径 | 视数据而定 | 低直接重叠；为 identity-zero、matched-null 提供评价原则 |
| Attention-based Deep MIL | bag/instance embedding | 弱监督实例归因 | 不适用 | 否 | 否 | 否 | 否 | 否 | 否 | 可静态使用 | 包级标签到实例贡献 | 间接 | 可把边界实例作为 instance | 否 | 中：实例级弱监督不是新颖点；OBCBC-R 不能以 attention 聚合作为主要贡献 |
| **OBCBC-R 候选** | 共享第三方敏感 Host 的边界实例 | 最近 first-party→third-party boundary、orchestrator、CFG/guard/UI、Role、distance | 完整但 identity-zero | **完整，且边界实例是分析单元** | 计划保留 | 计划保留 | 计划保留 | 计划保留 | 第一版不要求完整 taint | 是 | 责任外/自主语义的有界证据 | 与第一方责任一致的有界证据 | **同一共享 Host 按边界实例去卷积，不删除库** | 后续必须冻结后诊断 | 候选差异仅在加粗部分；其余变量均不能声称新颖 |

## 矩阵结论

1. “把 activation、guard、UI 与敏感行为结合”并不新颖，AppContext、FlowCog、AppIntent、AsDroid、AutoPPG 已分别或联合覆盖。
2. “区分第一方与第三方代码”也不新颖，LibRadar、LibScout、MAPS、AutoPPG 和供应链可达性分析均已覆盖不同层次。
3. 可以继续审计的最窄差异不是特征集合，而是**责任分析单元**：同一个共享第三方敏感 Host 在图中汇合时，是否能按最近第一方→第三方边界实例拆开，并保留各自 first-party orchestrator 的激活/guard/UI 上下文。
4. 即使这一差异在阶段 A 被判为新颖性“可成立”，也只表示尚未找到完整重复工作，不表示静态资产可构建、标签方向有效或 CARE-Droid 性能会提高。
