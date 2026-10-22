# CARE-Droid expanded 2017 cross-year v1

Current implemented stage: Phase 0 read-only dataset audit. It verifies the bounded
AndroZoo candidate against its selected/source manifests, content SHA, exact DEX
identity, named historical manifests, labels and year fields. It writes only to the
fresh output namespace. This package deliberately does **not** implement or run
training before the audit and compatibility smoke-test gates pass.

Run in tmux: `bash scripts/care-expanded2017-crossyear-v1/run.sh`.

The source label is an AndroZoo VT detection-count proxy: 0 benign, >=4 malware;
1–3 excluded. It is not a manually adjudicated behavior label. Year is from
`dex_date`, not directory name alone.
