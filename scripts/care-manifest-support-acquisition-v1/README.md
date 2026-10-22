# Manifest支持度与补充观测路由试验

只使用旧799个source和1989个future缓存。第一层预测完全冻结，不调用第二层、不解析APK。

方法A：固定混合距离、k=5、类内10%低密度裁剪的Trust-style参考支持度。
方法B：原概率候选池内，补充31维MethodId角色存在性，用校准LR对第一层预测的反对程度筛选。
联合：用概率与支持度并集扩充候选池，再实施方法B。
对照：同算法Manifest-only LR、原概率阈值、类条件概率经验秩、同数量不确定性和随机期望。

全部选择参数固定。2017固定五折嵌套、三种子；2018—2021使用2017冻结规则逐年评价。
后缀10/25只指源域参照预算，不固定未来比例。
所有补充观测仅属于路由，不替换第一层预测。不实现EDDI的VAE或SPIREL的RL。

成本边界：缓存回放不是实际特征获取计时。报告获取比例，不声称已经实现等成本提升。
若A/B失败，不自动调整距离、阈值、候选池或运行C43。

使用环境变量MSA_OUT指向新run目录；先run.sh --smoke-test，再在tmux run.sh --resume。
结果见final_report.md、routing_seed_summary.csv、final_decision.json。
