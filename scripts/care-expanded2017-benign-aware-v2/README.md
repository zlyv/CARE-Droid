# CARE-Droid expanded-2017 benign-aware V2 — isolated Phase A/B implementation

This package is deliberately limited to data audit and static feature extraction.
It does **not** claim that Stage-I, Router, Direct-C43, or end-to-end training has run.
Those stages require the extraction compatibility and coverage checks to pass first.

`audit_new_apks.py` reuses the content-verified 3989-APK read-only audit, checks
the named inputs before and after, and produces a deduplicated 2017 union. It
keeps the AndroZoo VT-count rule explicitly labelled as a proxy, not behavior
ground truth. `extract_features.py` replays P31 on an old source APK and
requires numerical agreement before new extraction. The smoke batch uses ten
new benign 2017 APKs for P31 and three for Host/endpoint construction. Full
P31 extraction covers all unique downloaded new APKs, including benign ones.
The 2017 train-eligible subset is then passed through the existing read-only
compact and endpoint builders using their official Role mapping and caps.

All generated files go to `outputs/care_expanded2017_benign_aware_v2`. Existing
APK, models, cache, source scripts and historical output are never written.
`run.sh` refuses an existing output directory; `resume.sh` uses resumable shards.
No DEX is executed, installed or rewritten. The implementation needs a smoke
run on the actual server before its compatibility can be claimed.
