# CARE-Droid LLM Semantic-Correction Prompt V2

This isolated package rebuilds the frozen 109 source-2017 candidate prompts.
It does not import an LLM SDK, send a network request, read ground-truth labels,
or inspect FIXED/BROKEN outcomes.

V2 replaces raw size counts with within-APK proportions and replaces the
all-zero cached `role_edge_counts` with a read-only reconstruction from each
shard's `host_primary_role` and `outgoing_host_positions`.  The original shard
is never modified.  The relation is explicitly a direct Host-primary-role
projection, not runtime data flow and not ordinary-method path contraction.

Upload the zip into `/home/boyuanjie/projects/`, then unpack from the project:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-llm-semantic-correction-prompt-v2.zip -d scripts/
```

Read-only preflight:

```bash
bash scripts/care-llm-semantic-correction-prompt-v2/run_preflight_only.sh
```

Local prompt build (still no LLM call):

```bash
bash scripts/care-llm-semantic-correction-prompt-v2/run_build_on_server.sh \
  2>&1 | tee artifacts/care-llm-semantic-correction-prompt-v2.console.log
```

Do not call Qwen until the five pilot prompts and selection audit have been
manually inspected.
