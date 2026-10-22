#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("artifacts/care-litelocal-p31-c43-localization-source-v1")
P31_ROOT = Path("artifacts/care-litelocal-p31-c43-source-v1")
SCRIPTS = Path("scripts")
HERE = Path(__file__).resolve().parent
COMPACT = SCRIPTS / "care-di25-compact43-relationquery-source-v1"
GRAPHGLOBAL = SCRIPTS / "care-di25-graphglobal-path-source-v1"
SOURCE_VENDOR = SCRIPTS / "care-stage2only-anomaly-v1/vendor"
for path in (
    HERE, COMPACT, GRAPHGLOBAL, SOURCE_VENDOR, SCRIPTS,
    SCRIPTS / "care-rolegraph-conditioned-v4-script",
    SCRIPTS / "care-rgcondattn-dedup-bounded-source-v1",
    SCRIPTS / "care-malben-host-v2",
):
    if str(path.resolve()) not in sys.path:
        sys.path.insert(0, str(path.resolve()))

REQUIRED = (
    P31_ROOT / "preflight/p31_reference/source_predictions.csv",
    P31_ROOT / "arms/P31_C43_RQ/source_predictions.csv",
    Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
    Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz"),
    COMPACT / "compact_models.py", COMPACT / "compact_background.py",
    GRAPHGLOBAL / "graphglobal_path_models.py", GRAPHGLOBAL / "source_label_contract.py",
    SCRIPTS / "care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
    SCRIPTS / "care-litelocal-p31-c43-source-v1/nested_p31.py",
)
EXPECTED_DEPENDENCY_SHA256 = {
    COMPACT / "compact_models.py": "6e82acd4db52c06d41e1958c52c50ca4d6cdc49ffbffc79e363045cc570b18f7",
    COMPACT / "compact_background.py": "a62c5d22f0b9582cc38f2e98e80b1d986977ab708a113a6870a60cfe00b8345b",
    GRAPHGLOBAL / "graphglobal_path_models.py": "91c8676706fd2d6e0df5175a98144fafdb75a466704be2777672840da336c8cd",
    SCRIPTS / "care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py": "4668446af73bd2853af88713ed7fb3e86ce20647280c3087406441570c4a951f",
}


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    missing = [str(path) for path in REQUIRED if not path.is_file() or path.stat().st_size < 100]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- " + "\n- ".join(missing))
    changed = [str(path) for path, expected in EXPECTED_DEPENDENCY_SHA256.items()
               if digest(path) != expected]
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen model dependency changed:\n- " +
                           "\n- ".join(changed))
    reference = read(REQUIRED[0])
    ids = [row["sha256"].strip().casefold() for row in reference]
    folds = [int(float(row["fold"])) for row in reference]
    route = [int(float(row["P31_DI25__routed"])) for row in reference]
    if len(reference) != 799 or len(set(ids)) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort changed")
    fold_sizes = {value: folds.count(value) for value in range(1, 6)}
    route_sizes = {value: sum(r for r, f in zip(route, folds) if f == value)
                   for value in range(1, 6)}
    if fold_sizes != {1:160, 2:160, 3:160, 4:160, 5:159}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: fold sizes changed")
    if sum(route) != 200 or route_sizes != {1:40, 2:40, 3:40, 4:40, 5:40}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31-DI25 route changed")
    from localization_models import ARM_SPECS, make_localization_class
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    args = SimpleNamespace(max_delta=2.25, graph_dim=32, embedding_dim=24,
                           hidden_dim=48, graph_heads=4, graph_layers=2,
                           dropout=.2)
    counts, signatures = {}, {}
    for arm in ARM_SPECS:
        model = make_localization_class(v4, arm)(24, "RGCondAttn", args)
        counts[arm] = sum(parameter.numel() for parameter in model.parameters())
        signatures[arm] = [(name, tuple(value.shape)) for name, value in model.state_dict().items()
                           if not name.endswith("switch")]
    if set(counts.values()) != {32292} or len({repr(value) for value in signatures.values()}) != 1:
        raise RuntimeError("BLOCKED_BY_CONTRACT: capacity mismatch " + repr(counts))
    runtime = "\n".join((HERE / name).read_text(encoding="utf-8")
                          for name in ("run_arm.py", "localization_models.py"))
    forbidden = ("future_interz.npz", "future_regions_2018_2021.npz",
                 "endpoint_centered_tensor_cache_2018_2021.jsonl.gz",
                 "within_year_router_inputs.csv")
    if any(token in runtime for token in forbidden):
        raise RuntimeError("BLOCKED_BY_CONTRACT: forbidden future dependency")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "dependency_sha256.txt").write_text(
        "\n".join(f"{digest(path)}  {path}" for path in REQUIRED[4:]) + "\n",
        encoding="utf-8",
    )
    print("P31_C43_LOCALIZATION_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 route_P31_DI25=200 route_per_fold=40,40,40,40,40")
    print("parameter_count_all_arms=32292 correction_width=113 summary_width=8")
    print("L0=reuse_locked_not_retrained candidates=" + ",".join(ARM_SPECS))
    print("future_rows_loaded=0 ego_consumed=false")


if __name__ == "__main__":
    main()
