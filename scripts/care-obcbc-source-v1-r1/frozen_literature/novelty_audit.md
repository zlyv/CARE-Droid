# OBCBC 阶段 A 新颖性审计

## 1. 审计结论先行

第一轮的宽泛表述——“给敏感行为增加 activation、guard、UI 和第一/第三方归属，从而推断恶意或良性方向”——与既有研究发生实质性组合重叠，不能作为新贡献。最接近的冲突来自 AppContext、AsDroid、FlowCog、AppIntent 和 AutoPPG；库归属部分又分别被 LibRadar、LibScout、MAPS 和 Common Libraries 覆盖。

因此第一轮判定为：**NEEDS_REFORMULATION**。

按照预注册规则，本次只允许收窄研究问题，不增加新模块。第二轮冻结为：

> 对已经汇合到同一个第三方/共享敏感 Host 的多条第一方调用上下文，以最近的第一方→第三方调用边界**实例**作为责任单元；在不使用库身份、包名和 Host 规模的前提下，恢复每个边界实例的第一方 orchestrator、activation、guard、UI callback/resource、Role 和距离，用于判断共享 Host 所掩盖的责任上下文是否提供可弃权的双向边际纠错证据。

在已核对的原始论文中，没有发现同时满足以下四点的等价工作：

1. 同一共享第三方敏感 Host 是需要去卷积的汇合点；
2. 最近第一方→第三方**边界实例**而非库、包、API、组件或整个 APK 是基本分析单元；
3. 第一方 orchestrator 的触发/guard/UI 上下文随边界实例保留；
4. 输出进入一个保持 Stage-I、允许 abstain 的 bounded correction，而不是新的普通恶意分类器。

第二轮文献判定为：**NOVELTY_PLAUSIBLE**。这里的 “plausible” 仅表示目前未发现完整重复，不是“已经证明新颖”，更不是“实验有效”。

## 2. 十个决定性问题

### Q1. AppContext 是否已经覆盖 activation、guard 与敏感行为？

**是。** AppContext 明确定义了 activation event 和 environmental guarding factors，并在跨 ICC 的图/CFG 上关联敏感行为。因此 OBCBC 不能把“把触发和 guard 加到敏感 Host”作为新颖点。可保留的差异只能是共享第三方 Host 之前的 ownership-boundary instance 去卷积。

### Q2. FlowCog 是否已经覆盖 UI 语义对具体行为的正当化？

**是。** FlowCog 把具体 source-to-sink flow 与控制、数据、view 依赖以及 UI 文本/图像关联，判断泄露语义是否与界面描述相符。OBCBC 不能声称首次用 UI 解释敏感行为；它最多能声称把 UI 上下文绑定到特定 ownership-boundary instance，而不是绑定到已经汇合的 flow/Host。

### Q3. 既有工作是否已经区分 first-party/third-party 并恢复责任？

**部分且冲突很强。** AutoPPG 区分 host 与 TPL，建模 host↔TPL 数据传递并给出六类执行上下文；MAPS 对第一/第三方隐私实践归属；LibRadar/LibScout 提供库识别；供应链分析做应用到依赖方法的可达性。它们已经否定“所有权归属本身是新颖的”。尚未找到的是：在同一共享第三方敏感 Host 已吸收多条第一方路径时，以最近边界实例进行责任去卷积，并把它用于选择性纠错。

### Q4. 是否已有 Android malware 工作把 first-party call boundary 直接用作纠错方向？

**没有找到完全等价者，但有非常近的先例。** AsDroid/AppIntent/AppContext 已把行为追溯到 top-level function、入口或用户动作；AutoPPG 已分析 host 与 TPL 的跨边界传递。现有材料未显示它们把“最近 first-party→third-party 边界实例”作为同一共享 Host 的多实例责任单元，也未用于冻结 Stage-I 后的 bounded correction。这个否定命题受检索范围限制，后续投稿前仍需补做引文追踪和相似论文复核。

### Q5. OBCBC 是简单拼接，还是新的责任归因问题？

宽泛 OBCBC 是**简单组合**，新颖性不足。收窄后的 OBCBC-R 才是一个不同的问题：同一共享库方法/Host 是图中的多责任汇合点，传统 Host 表示把多个 first-party orchestrator 压缩成一个实例；OBCBC-R 改变分析单元，在汇合之前恢复边界实例。这不是给 Host 再拼接几个字段，而是要求一个 Host 可对应多个责任实例，并在聚合时保留 provenance。

### Q6. 可否用形式化方式说明独立机制？

设共享第三方 Host 为 `h`，进入它的第一方→第三方边界实例集合为 `B(h)={b1,...,bk}`。传统表示将其压缩为 `z(h)`；OBCBC-R 先构造 `z(bi)=(orchestrator, activation, guard, UI, role, distance)`，再通过 identity-zero 的对称聚合得到 `Z(h)={z(bi)}` 的统计。核心可证伪命题是：

- `k>1` 时，`z(h)` 会丢失不同第一方责任上下文的符号信息；
- 保留 `z(bi)` 后，在 Stage-I predicted-benign 中能够区分 FN/TN，在 predicted-malware 中能够区分 FP/TP；
- 这种增益不能由库身份、Host 数、边界数、路径数或随机重新配对解释。

若阶段 B 表明绝大多数 Host 的 `|B(h)|<=1`，或无法稳定对齐边界实例，则问题本身没有足够数据支撑，应停止。

### Q7. 为什么它适合 selective bounded correction，而不是普通分类？

OBCBC-R 只对已有 Stage-I 决策提供局部、稀疏、带不确定性的边际证据。共享 Host 可能没有可恢复边界、可能出现混合上下文，也可能同时产生良性和恶性支持。合理输出是正向、反向或 abstain，并保持 `max_delta=2.25`；这与普通 APK 分类器强制为每个样本输出类别不同。该适配性是系统设计上的合理性，不是新颖性证据。

### Q8. 它能否解释 SDK convergence、GPS/推送/统计 SDK 等现象？

**理论上可以提出解释，但本阶段不能声称已验证。** Common Libraries 表明共享库可污染恶意检测，FlowCog/AutoPPG/MAPS 表明第三方代码可产生隐私相关行为，MPI 表明共享执行实体会造成 provenance 混合。由此可推断：同一 SDK Host 的相似静态表示可能掩盖不同第一方调用责任。但具体“Baidu/GPS/推送/统计 SDK 导致 CARE-Droid 哪些错误”的说法仍是项目内经验命题，必须由阶段 B 的静态资产审计和后续 source-only probe 验证。

### Q9. 与最接近工作相比，非平凡差异是什么？

允许主张的最小差异是：

1. **多实例责任去卷积**：一个共享第三方 Host 可拆成多个最近 ownership-boundary instances；
2. **保留第一方编排上下文**：上下文属于进入库的责任边界，而不是库身份或汇合后的 Host；
3. **identity-zero null 设计**：包名、库名、API 身份、规模均不得成为方向特征；
4. **条件化双侧且可弃权**：分别检验 FN/TN 与 FP/TP，只在有证据时做 bounded correction；
5. **matched responsibility null**：边界上下文重配对、边界度数保持 shuffle、count-only 和 Host-only 必须与真实机制同容量比较。

不能主张“首次使用 activation/guard/UI”“首次区分第三方库”“首次做静态责任归因”“首次定位恶意组件”或“首次做实例级 attention”。

### Q10. 什么条件下应停止？

以下任一项成立就应停止 OBCBC-R，而不是扩大网络或修改正式基线：

- 阶段 B 无法可靠得到 first/third ownership 与最近边界；
- 共享第三方 Host 的多边界实例覆盖不足；
- 边界实例与 Host/Role/C43 attention 不能无位置猜测地对齐；
- 去除库身份后上下文大量为空或退化为路径/规模计数；
- 真实边界上下文不优于 degree-preserving shuffle、count-only 或 Host-only；
- FP-vs-TP 仍接近随机，或 FN 侧改善以 broken 增加为代价；
- 结果仅由一个 fold、一个 SDK 或少数超大 APK 驱动；
- 后续发现已有论文完整实现同一责任单元和同一用途。

## 3. 最接近的冲突工作

| 优先级 | 工作 | 冲突点 | OBCBC-R 剩余差异 | 风险 |
|---:|---|---|---|---|
| 1 | AutoPPG | host/TPL 区分、跨边界数据传递、六类上下文、UI/系统/环境触发 | 同一共享敏感 Host 的最近边界实例去卷积；选择性双侧纠错 | 极高；投稿前必须逐段对照方法与实验 |
| 2 | AsDroid | top-level function 归因、UI 文本、敏感行为语义不匹配 | ownership boundary 多实例化；共享 Host 汇合问题 | 极高；不能声称首次语义责任归因 |
| 3 | AppContext | activation、guard、跨 ICC 上下文、良恶行为分类 | ownership boundary 与 shared-host deconvolution | 高 |
| 4 | FlowCog | 具体 flow 的 UI/控制/数据语义与正当化 | 不依赖完整 taint，且上下文挂到 ownership-boundary instance | 高 |
| 5 | AppIntent / user-trigger dependence | 用户动作/输入到敏感行为的静态路径 | 共享第三方 Host 的多责任边界 | 高 |
| 6 | LibScout/LibRadar/MAPS | first/third 归属与库识别 | 库身份不进入模型；只作为构造边界的技术 sidecar | 中高 |
| 7 | Eclipse Steady | 应用到依赖方法的可达边界与应用特定暴露 | Android 敏感行为、上下文符号、双侧纠错 | 中高 |
| 8 | MPI | 共享进程/实体的语义去卷积 | Android 纯静态 ownership boundary | 中；限制广义“去卷积”新颖性 |

## 4. 声明边界

### 可以声明

- 经过原始论文审计，宽泛 OBCBC 与多项既有机制高度重叠，需要收窄。
- 尚未发现用最近 first-party→third-party 边界实例去卷积同一共享第三方敏感 Host，并将该实例级上下文用于冻结 Stage-I 后双侧 bounded correction 的完整等价工作。
- 该差异具有可证伪的静态资产条件和 matched-null 条件。

### 不可以声明

- “OBCBC 已证明新颖”；
- “OBCBC 已证明有效”；
- “首次把上下文用于 Android 恶意检测”；
- “首次结合 UI、guard 或 activation”；
- “首次区分第一方与第三方”；
- “已解决 SDK convergence 或 FP direction”；
- “文献未找到等价工作”意味着不存在未检索到的工作。

## 5. 阶段门控

- 阶段 A 文献门：**通过，但仅限 OBCBC-R 收窄定义**。
- 下一步只授权阶段 B：静态资产可行性审计，包括 ownership sidecar、最近边界恢复、多边界共享 Host 覆盖、无标签对齐与计算成本。
- 当前不授权：标签方向 probe、模型实现、五折实验、CARE-Droid 集成或 future 诊断。
- 如果阶段 B 不满足覆盖与对齐门槛，最终状态应转为 `NOT_FEASIBLE`，不能用宽泛 OBCBC 替代。
