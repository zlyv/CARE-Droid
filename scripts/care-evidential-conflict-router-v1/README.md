# CARE-Droid multi-view evidential conflict router v1

This experiment replaces probability-only post-hoc disagreement with four
genuine evidential heads.  Only routing changes: frozen Global24 predictions,
the frozen local expert, the final threshold, and the 25% Stage-II budget stay
unchanged.

## Evidence heads（证据头）

Manifest, Code/Call Scale, Sensitive Behavior, and Graph Statistics each use a
small MLP that outputs two nonnegative evidence values:

```text
e_v = softplus(MLP_v(x_v))
alpha_v = e_v + 1
p_v,c = alpha_v,c / sum(alpha_v)
u_v = 2 / sum(alpha_v)
```

The heads use a Sensoy-style evidential squared-error Bayes risk with annealed
KL regularization toward the uniform Dirichlet prior.  An ordinary probability
is not converted into evidence; evidence amount is learned by the head.

## Conflict-aware evidence fusion（冲突感知证据融合）

Each view receives a sample-specific reliability discount:

```text
r_v = exp(-gamma * u_v)
discounted_e_v = r_v * e_v
```

Pairwise conflict is computed from opposing belief masses.  The discounted
evidence is pooled, then its concentration is contracted according to conflict:

```text
U = 1 - (1 - U_ignorance) * (1 - K)
E_effective = 2 * (1 - U) / U
```

The effective total evidence is redistributed according to the pooled class
direction before the final Dirichlet opinion is formed.  Thus low evidence and
strong cross-view conflict both produce high unified uncertainty, while strong
agreement retains high effective evidence.

## Compared routes

- `A25`: frozen Global24 distance-to-boundary route;
- `EVI_Ignorance25`: learned evidence insufficiency without conflict;
- `EVI_ConflictMean25`: primary mean pairwise conflict fusion;
- `EVI_ConflictMax25`: maximum-conflict aggregation;
- `EVI_ConflictMean_NoDiscount25`: no reliability-discount ablation.

Source evaluation is strict outer-fold OOF.  The future experiment reuses the
source OOF evidential scores to fit source thresholds, trains the four final
heads on all 2017 samples, and accesses 2018-2021 labels only after scores,
routes, and final probabilities are frozen.

## Default training configuration

- hidden dimension: 32;
- dropout: 0.10;
- epochs: 300;
- learning rate: 0.001;
- weight decay: 0.0001;
- KL weight: 0.01;
- KL annealing: 100 epochs;
- reliability gamma: 1.0;
- device: automatic CUDA when available, otherwise CPU.

These values are fixed before the future diagnostic and must not be selected
using future-year labels.

## Run

Place this directory under `scripts/`, then run from the repository root:

```bash
bash scripts/care-evidential-conflict-router-v1/run_on_server.sh \
  2>&1 | tee artifacts/care-evidential-conflict-router-v1.console.log
```

The launcher prefers `/home/boyuanjie/envs/tif/bin/python`, then `python3`, and
finally `python`.  It prints `Evidential device: cuda` when the selected PyTorch
environment can access the GPU.

## Outputs

```text
artifacts/care-evidential-conflict-router-source-v1/
artifacts/care-evidential-conflict-router-future-v1/
```

The output includes routing quality, high-confidence error recall, end-to-end
metrics with the frozen local expert, per-view evidence/uncertainty/reliability,
training diagnostics, A25 overlap, and source-threshold transport diagnostics.

Package results with:

```bash
zip -r care-evidential-conflict-router-results.zip \
  artifacts/care-evidential-conflict-router-source-v1 \
  artifacts/care-evidential-conflict-router-future-v1 \
  artifacts/care-evidential-conflict-router-v1.console.log
```
