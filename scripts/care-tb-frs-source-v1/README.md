# CARE-Droid TB-FRS source-only probe v1

This package tests whether the frozen Role-constrained 32+24 prototype background loses class-direction information through hard assignment or scalar compression.

It runs 2017 strict outer-fold OOF only. It does not modify Direct-C43, run an end-to-end correction arm, load future/external data, change the Top25 route, or write formal assets.

Arms: formal Direct-C43 reference, hard four-statistic probe, soft four-statistic probe, signed first-order residual, unsigned first-order control, signed first+second order, unsigned control, APK-wise component correspondence shuffle, and capacity-only nuisance control. All learned probes receive a 16-dimensional outer-train-only projection and the same regularized logistic capacity.
