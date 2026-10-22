# AndroZoo Random15K available-snapshot extraction v1

This extraction-only package freezes the currently audited AndroZoo snapshot.
It reuses the completed 2017 and routed-future Stage-II caches, then extracts
only future APKs with a valid pre-existing P31 shard that are not already in a
completed Stage-II cache. It fits no model and uses no labels for extraction.

Writes are restricted to
`outputs/care_az_random15k_available_extraction_v1/`.

