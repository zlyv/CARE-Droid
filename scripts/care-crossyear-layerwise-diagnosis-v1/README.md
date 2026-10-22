# CARE-Droid cross-year layerwise diagnosis v1

Read-only development-set diagnostic. No APK extraction, original-model edits, or formal-model change. Run `run.sh --audit-only` then `run.sh --extract-layers` then `run.sh --run-probes` then `run.sh --collect`. All writes are exclusive in the isolated output directory. A model coordinate system is identified by window and seed; no cross-window vector pooling is permitted.
