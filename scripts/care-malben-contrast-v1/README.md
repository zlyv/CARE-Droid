# CARE-Droid malicious-evidence vs benign-explanation v1

This is the first controlled modification of Stage II. It changes only the
local evidence representation.

For each outer/inner training partition, it builds two training-only reference
libraries from Host/Ego region features:

- benign prototypes from regions sampled from benign APKs;
- malware-associated prototypes from regions sampled from malware APKs.

Each region obtains benign similarity, malware similarity, a signed contrast,
and reference strength. The established graph-conditioned attention equation
is unchanged. Its weights separately aggregate a malicious-evidence vector and
a benign-explanation vector; both enter the same FixKeep bounded-correction head.

Important limitation: only APK labels are available. Therefore the malware
prototype library is *malware-associated*, not a ground-truth collection of
malicious functions. This limitation must be stated in the paper.

## Server run (tmux)

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o care-malben-contrast-v1.zip -d scripts/
bash scripts/care-malben-contrast-v1/launch_tmux.sh
```

Monitor:

```bash
tmux attach -t care-malben-v1
# detach: Ctrl-b, then d
tail -f artifacts/care-malben-contrast-v1.console.log
```

Final source report:

```bash
cat artifacts/care-malben-contrast-v1/comparison/REPORT.txt
```

No future-year runner is included in v1. A future diagnostic should be created
only if the source-side decision is frozen first.
