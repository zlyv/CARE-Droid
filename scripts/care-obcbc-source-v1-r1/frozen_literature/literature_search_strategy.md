# OBCBC 文献检索策略与阶段 A 协议

## 1. 审计范围

- 审计日期：2026-10-07。
- 当前阶段：仅阶段 A（文献与新颖性审计）。
- 研究对象：Ownership-Boundary Conditioned Behavioral Context（OBCBC）。
- 研究边界：纯静态 Android 分析；本阶段没有读取服务器工程、没有运行 APK、没有修改模型、没有运行标签实验，也没有访问 2018–2021 数据。
- 判断目标：已有工作是否已经完整覆盖“在共享第三方敏感 Host 汇合前，以最近第一方→第三方调用边界实例为责任单元，保留其第一方编排上下文，并将该上下文用作选择性 bounded correction 的双向方向证据”。

## 2. 检索问题拆分

本次检索没有只搜索 “OBCBC” 这一新名称，而是把候选机制拆为五类可被既有论文独立覆盖的技术主张：

1. **敏感行为上下文**：activation event、guarding condition、CFG、控制依赖、路径谓词。
2. **功能正当化**：UI callback、view/resource/text 与具体敏感行为或信息流的对应。
3. **代码所有权**：first-party / third-party / unknown、库识别、版本识别、混淆与裁剪。
4. **责任边界**：应用代码到依赖代码的调用边界、可达性、信息传递和行为归属。
5. **机制否证**：时间偏差、规模捷径、库身份捷径、matched shuffle、弱监督实例归因。

## 3. 检索源和优先级

优先核对原始论文全文或作者版本：

- IEEE/ACM/USENIX/PETS/PMLR/Nature/Springer/Elsevier 正式页面；
- 作者或机构托管的论文 PDF；
- arXiv 仅用于作者版本或正式页面不可直接读取时的全文核对；
- DBLP/Crossref/机构书目页仅用于补 DOI、作者、venue，不用作机制结论来源。

排除：普通博客、营销网页、新闻报道、未核验二手总结和仅凭搜索摘要形成的结论。

## 4. 查询组

### 4.1 Android 上下文与用户意图

- `Android malware activation event guarding condition sensitive behavior static analysis`
- `Android UI semantics sensitive behavior control dependency`
- `Android user-trigger dependence sensitive API static`
- `Android flow-level semantics UI view dependency privacy leak`
- `Android logic bomb path predicate control dependency`

### 4.2 第三方库与代码所有权

- `Android first-party third-party code attribution sensitive API`
- `Android third-party library behavior attribution call graph`
- `library-aware Android malware detection`
- `Android common libraries malware detection false positive`
- `Android library detection obfuscation code shrinking version`

### 4.3 所有权边界和责任传播

- `first-party third-party call boundary attribution`
- `application context vulnerable library reachability`
- `host app third-party library information transfer static analysis`
- `shared infrastructure attribution call graph`
- `software supply chain blame assignment dependency reachability`

### 4.4 跨领域归因与混杂控制

- `provenance intrusion detection causal graph attribution`
- `semantic execution partitioning provenance shared process`
- `multiple instance learning instance attribution shared components`
- `shortcut testing confounder intervention`
- `graph shared hub path attribution`

## 5. 纳入标准

论文进入 `literature_inventory.csv` 至少满足一项：

- 提供与 OBCBC 某个关键变量直接对应的原始机制；
- 对“上下文是否给出恶意/良性方向”有实证；
- 对第一方/第三方归属或依赖可达性有可复用机制；
- 对共享节点导致的因果图污染、执行混合或责任混淆有可迁移机制；
- 为 temporal bias 或 shortcut/null 检验提供正式约束。

## 6. 新颖性判定方法

对每篇论文记录：输入、分析单元、上下文、是否区分所有权、是否恢复边界、输出目标、标签协议及局限。判定不采用“各模块分别有人做过，所以组合必然不新”这种机械规则，而是检查以下三层：

1. **变量层重叠**：论文是否已经使用同样的 activation、guard、UI、ownership 变量？
2. **责任单元重叠**：论文是否以“第一方→第三方边界实例”而不是 APK、库、API、flow 或顶层函数为归因单元？
3. **决策层重叠**：论文是否把该边界上下文用作 Stage-I 条件化的双向、可弃权、受界纠错证据？

只有三层均基本覆盖才判为 `LITERATURE_DUPLICATED`。如果只有变量层覆盖而责任单元和决策层未覆盖，则可以保留狭窄的新颖性主张，但必须禁止宣称“首次使用上下文/UI/库归属”。

## 7. 两轮审计

### 第一轮：宽泛 OBCBC

宽泛定义“用所有权、触发根、guard 和 UI 解释第三方敏感行为”与 AppContext、AsDroid、FlowCog、AutoPPG 和 LibScout 存在高度组合性重叠，第一轮判定为 **NEEDS_REFORMULATION**。

### 第二轮：收窄后的 OBCBC-R（Responsibility-preserving OBCBC）

冻结后的研究问题为：

> 当多个第一方编排路径在同一个第三方敏感 Host 内汇合时，能否在汇合前按“最近的、可静态证明的第一方→第三方调用边界实例”进行去卷积，并证明边界级 activation/guard/UI context 提供了超越 root-only、library deletion、ownership ratio、library identity、count-only 与 matched shuffle 的双向纠错方向信息？

该定义把新机制限定为“责任边界去卷积”，不是一般上下文提取、一般 UI 正当化、一般库识别或一般恶意分类。

## 8. 检索局限

- 本次是系统化机制审计，不是声称穷尽所有语言和数据库的正式系统综述。
- 没有找到“完全相同机制”只能支持 `NOVELTY_PLAUSIBLE`，不能证明绝对世界首创。
- 是否能从当前完整图可靠恢复 ownership、最近边界、CFG/control/UI 关联，必须在阶段 B 用无标签覆盖审计决定；阶段 A 不对此做推断。

