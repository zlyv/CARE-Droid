# Direct-C43 Stage-I error recovery audit

Read-only audit over frozen 2017 OOF and frozen 2018–2021 predictions.

It separates:

1. Stage-I errors that enter the fixed Top25% route;
2. errors the frozen Direct-C43 expert could correct under full routing;
3. fixable errors missed by routing;
4. errors Direct-C43 still gets wrong even under full routing;
5. routed failures with wrong, near-zero, or correct-but-too-weak probability movement;
6. damage to routed Stage-I-correct samples.

No model, threshold, route, or formal asset is changed. Probability movement is diagnostic and is not presented as a causal semantic explanation.
