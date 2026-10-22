# CARE-Droid P31-RF C43 Host-content HAIG ablation v1

Strict 2017 source-only ablation testing whether fine-grained HAIG content can
replace the coarse 31-D Role block inside each C43 Host. P31-RF, RF-anchored
MeanOpposition DI25 Top25%, fixed folds, all-outer-train Stage-II training and
`max_delta=2.25` remain unchanged.

Arms (all stored/model widths and trainable parameter counts are identical):

- `A0_ROLE_ONLY`: original Compact43 Host content; HAIG channels are zero.
- `A1_API_NO_ROLE`: Role31 is zeroed in Host content; exact API channel only.
- `A2_H3_NO_ROLE`: Role31 is zeroed; exact API + direct Host-call channel.
- `A3_ROLE_PLUS_H3`: Role31 and H3 are both present.
- `A4_H3_SHUFFLE_NO_ROLE`: A2 capacity/null control with Host-call content
  shuffled within APK and primary Role.

RoleGraph/Relation31 remain available only for Host localization. This experiment
tests removal of Role from **Host content**, not deletion of the RoleGraph.
The builder uniquely aligns HAIG and frozen C43 Hosts by exact sensitive-API
fingerprint. Ambiguous Host matches are never guessed. It stops before training
if alignment coverage is below the declared contract.

Run:

```bash
unzip -o ../care-p31rf-c43-haig-host-ablation-source-v1.zip -d scripts/
bash scripts/care-p31rf-c43-haig-host-ablation-source-v1/launch_tmux.sh
tail -F artifacts/care-p31rf-c43-haig-host-ablation-source-v1.console.log
```

