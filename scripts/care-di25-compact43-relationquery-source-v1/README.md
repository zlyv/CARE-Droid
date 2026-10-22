# CARE-Droid duplicate-free Compact43 relation-query source v1

Strict 2017 five-fold OOF. H0 is a locked reference. Two capacity-matched
candidate arms are trained concurrently on two GPUs:

- C43_COMPACT: `SelfRole31 + frozen Tail12`, no stored duplicate role block.
- C43_RELATION_QUERY: identical Compact43 correction representation; directed
  neighbor-role31 can affect Host attention only.

Both candidates instantiate the relation projection, so their trainable
parameter counts are equal. Relation features never enter prototype distance,
region values, summaries, or the delta head. Three backgrounds are built in
the Compact43 space with role distance weight 2, mathematically preserving
the duplicated H0 role-distance contribution without storing duplicate columns.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-compact43-relationquery-source-v1.zip -d scripts/
bash scripts/care-di25-compact43-relationquery-source-v1/launch_tmux.sh
tail -F artifacts/care-di25-compact43-relationquery-source-v1.console.log
```
