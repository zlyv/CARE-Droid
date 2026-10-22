# CARE-Droid W0 HPRN v1

This isolated experiment tests a per-Host learnable Host--Prototype relation residual while preserving the W0 Direct-C43 backbone. Stage-I is Manifest135 RF and route decisions come from the frozen autonomous Manifest error router (`P(error)>=0.5`).

R0 is the frozen W0 baseline. R1 supplies only the four existing background scalars through an equal-capacity branch. R2 supplies the correctly matched External24, Shared32, and APK-internal dominant prototype vectors. R3 shuffles each prototype binding within its APK. R4 supplies only the old scalar match value and matched prototype category. R1--R4 use the same parameterized module and W0 stays frozen.

All fitting uses 2017. Source performance is strict five-fold OOF. The full-2017 T1 models are reused without retraining on 2018--2021, which remain development diagnostics. The implementation must exactly reproduce W0 probability, Host attention, and Critical Host when the zero-initialized HPRN residual is zero.
