# CARE unified routed residual v1

Fast, parameter-free 2017 source probe on the **fixed formal RF-DI25 routed 200 APKs**.

The primary correction is fixed before labels are evaluated:

`delta = clip(c43_delta + median(logit(LR),logit(XGB),logit(SVM)) - logit(RF), -2.25, 2.25)`

It does not train on the routed 200, does not change routing, does not load future years, and does not run GPU extraction.
Formal comparison: C43 fixed/broken/net = 11/7/+4.
