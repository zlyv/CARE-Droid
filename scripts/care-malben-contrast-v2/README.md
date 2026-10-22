# CARE-Droid role-conditioned Mal-vs-Ben evidence v2

This package changes only the prototype-based local evidence representation.
It does **not** change Global24, LR, A25, DI25, the 25% budget, the role graph,
the graph encoder, the local encoder architecture, attention equation, FixKeep,
bounded correction, the logit bound 2.25, source folds, sample order, or cache.

## What v1 actually did

`care-malben-contrast-v1` used two **class-global multi-prototype libraries**:

- 24 benign centroids fitted by `MiniBatchKMeans` over sampled benign-APK regions;
- 24 malware-associated centroids fitted the same way over sampled malware-APK
  regions.

They were not role-conditioned. A role did not have a dedicated centroid. Each
region was compared with every centroid in the corresponding class library and
used the nearest distance.

Similarity was based on RMS-Euclidean distance in the equal-APK standardized
raw region feature space:

`S_c = exp(-d_RMS(x, nearest prototype_c) / class_scale_c)`

where the scale was the training-class median nearest-prototype distance. The
vectors were **not** L2-normalized and cosine similarity was not used. Every
region retained `S_ben`, `S_mal`, `S_mal-S_ben`, and
`max(S_ben,S_mal)`. The unchanged graph-conditioned attention separately
weighted malware and benign similarity streams, and both vectors entered a
larger correction representation. All prototypes were fitted inside the
current outer-training or epoch-selection-training partition; outer validation
regions were excluded.

## What v2 changes

### Role-conditioned K=1 prototypes

The first 31 cached region columns contain semantic-role fractions. v2 assigns
one deterministic primary center role by `argmax`; ties use the smallest role
id. An all-zero synthetic region uses role `-1` and the global fallback.

For every training partition and role `r`, v2 fits:

- one benign centroid `P_ben^(r)`;
- one malware-associated centroid `P_mal^(r)`.

The fixed source-only minimum support is 20 sampled regions per role/class.
Sampling is capped per APK before centroid fitting. If one role/class has fewer
than 20 regions, only that class side falls back to its class-global centroid.
If a class-global reference were unavailable, that evidence direction would be
unavailable; with stratified source folds this last case is not expected.

Similarity remains symmetric between classes and uses the same bounded form:

`S_c = exp(-d_RMS(x,P_c^(r)) / within_class_median_scale_c^(r))`.

No class receives a manual weight. The following are saved for every region:

- `S_ben`, `S_mal`;
- `margin = S_mal-S_ben`;
- `evidence_strength = max(S_ben,S_mal)`;
- role support and fallback flags.

### Equal-capacity aggregation

The attention equation is unchanged. After attention:

`E_mal = max(0,S_mal-S_ben)` and
`E_ben = max(0,S_ben-S_mal)` for the full contrast model.

The correction head receives one vector `z_mal-z_ben`, replacing the original
single attended vector, plus the same-width scalar summary. `z_mal`, `z_ben`,
`e_mal`, and `e_ben` are saved separately for diagnosis but are not all
concatenated into the correction MLP. New evidence columns are excluded from
the Local Encoder. Consequently M1, M2, M3 and the deduplicated M0 baseline all
have exactly **33,388 trainable parameters**.

### Source-only ablations

- M0: established AnomalyBaseline at max-delta 2.25 (loaded, not retrained);
- M1: BenignOnly;
- M2: MaliciousOnly;
- M3: Role-conditioned MalBenContrast.

A25 is primary. The same OOF experts are post-hoc applied to DI25 only as a
secondary sensitivity check.

## Important implementation boundary

The current v4 loader reads `host_values` from the existing cache. Although the
cache also contains `ego_values`, the established v4 experiment did not load
them. v2 preserves that behavior to avoid changing region extraction together
with prototype evidence. Therefore this experiment is a **host-region**
prototype ablation; adding 1-hop Ego regions must be a separate future change.

The malware reference is weakly supervised: regions come from malware-labelled
APKs and can include benign functionality or library code. It is therefore
called a malware-associated reference, not ground-truth malicious functions.

## Leakage boundary

The scaler, old background statistics, role supports, class-global centroids,
role-conditioned centroids, and distance scales are refitted inside every
epoch-selection training split and every outer-training fold. No outer-
validation region contributes to any reference.

## Run source experiment in tmux

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-malben-contrast-v2.zip -d scripts/
bash scripts/care-malben-contrast-v2/launch_source_tmux.sh
```

Monitor:

```bash
tmux attach -t care-malben-v2
# detach: Ctrl-b, then d
tail -f artifacts/care-malben-contrast-v2.console.log
```

Key result:

```bash
cat artifacts/care-malben-contrast-v2/source/source_summary.txt
cat artifacts/care-malben-contrast-v2/source/rer_definition_audit.txt
cat artifacts/care-malben-contrast-v2/source/source_significance.txt
```

No future runner is included. Future evaluation is permitted only when
`source_decision.json` contains `"advance": true`; the frozen future package
must then be created without changing K, support, similarity, fallback,
aggregation, model, route, budget, or correction bound.

