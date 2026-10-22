# CARE-Droid SVM Direct-C43 joint contribution audit v1

This is a source-only attribution audit, not a new model. It first records which historical marginal ablations already answer Compact/HAIG, Background, Relation31, RoleGraph and attention questions. It then replays the exact frozen SVM Direct-C43 model and breaks selected cross-block correspondences only at inference time.

The within-APK arms preserve Host count and the marginal values of every permuted block. APK-level donors are one-to-one and matched within the same Stage-I prediction direction without labels. `N0_ROW_ORDER` is an exact invariance control. Twelve deterministic repeats are averaged. No future/external row, new threshold, gate, readout or separately trained arm is allowed.

This audit can localize model reliance or harmful attribution. It cannot prove causal Android program semantics. A new source ablation is allowed only when a relation passes the predeclared two-sided/per-fold diagnostic rule.

