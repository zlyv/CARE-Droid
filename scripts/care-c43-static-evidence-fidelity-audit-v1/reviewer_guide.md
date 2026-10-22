# Blinded semantic review guide

Review `blind_review_form.csv` without opening `blind_review_key.csv`.

For APK A versus APK B, record only statically verifiable facts. A difference in method count, graph size, package name, API frequency, or fingerprint distance is not automatically a class-direction fact.

Use `YES` for `raw_reliable_distinguishing_fact` only when a concrete DEX/Manifest fact has a defensible behavioral interpretation. Record that fact and its location. Then determine whether the fact is represented in the frozen structured/C43 input and whether the distinction survives into the hidden state or score.

Allowed review attributions are `EXTRACTION_GAP`, `REPRESENTATION_LOSS`, `STATIC_INSUFFICIENT`, `LABEL_UNCERTAIN`, and `INCONCLUSIVE`. Do not inspect labels until the form is frozen.
