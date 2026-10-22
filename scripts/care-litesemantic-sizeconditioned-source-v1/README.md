# LiteSemantic size-conditioned source v1

This strict-2017 experiment reuses the completed 799-row LiteSemantic cache and
does not parse APKs again. Seven raw size measurements are nuisance variables
only. They never enter a classifier or router.

For each outer fold, outer-train semantic features receive inner-OOF nuisance
predictions. Outer validation features are transformed only by a nuisance model
fit on the complete outer training partition. The fixed transform is a smooth
quantile-spline degree-2 Ridge model (`n_knots=5`, `alpha=10`).

Arms are locked RAW-L3 reference, residualized Manifest20, residualized Role62,
and residualized fused82. All new arms use identical strict OOF calibrated
committees and exact 25% DI-style routing. No future evaluation is authorized.
