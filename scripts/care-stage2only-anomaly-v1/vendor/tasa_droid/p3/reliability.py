"""Training-fold-only A3-MVP reliability evidence and pilot components.

The cache contains hashed node IDs and compact role IDs, so reliability can be
fit without reopening an APK.  This module deliberately exposes the pilot
definitions in the output audit.  It must not be used as a formal result until
the definitions and values are copied into the resolved protocol config.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .a3_mvp import A3_MVP_COMPONENTS


@dataclass(frozen=True)
class PilotReliabilityDefinition:
    """Explicit, auditable pilot definitions for D/T/one-minus-F.

    D is shrunk distinct-APK support, T is smoothed temporal-year coverage,
    and one-minus-F is smoothed semantic-role mapping coverage.  These are
    pilot definitions only; formal training must resolve them by the V6
    nested-validation protocol.
    """

    alpha_prior: float = 1.0
    kappa_support: float = 20.0
    kappa_context: float = 20.0
    r_min: float = 0.10
    r_max: float = 0.95

    def validate(self) -> None:
        if self.alpha_prior <= 0.0:
            raise ValueError("alpha_prior must be positive")
        if self.kappa_support <= 0.0 or self.kappa_context <= 0.0:
            raise ValueError("kappa values must be positive")
        if not (0.0 <= self.r_min <= self.r_max <= 1.0):
            raise ValueError("reliability bounds must satisfy 0 <= r_min <= r_max <= 1")


def _clean_row(row: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lstrip("\ufeff"): value for key, value in row.items()}


def read_split_manifest(path: Path) -> dict[str, dict[str, set[str]]]:
    """Return split -> train/test SHA sets; excluded rows are never used."""

    result: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"train": set(), "test": set()}
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = _clean_row(raw)
            split_name = row.get("split_name", "")
            sha256 = row.get("sha256", "").casefold()
            partition = row.get("partition", "").casefold()
            if not split_name or not sha256:
                continue
            if partition in {"train", "test"}:
                result[split_name][partition].add(sha256)
    return dict(result)


def read_cache(path: Path) -> dict[str, dict]:
    """Load successful compact cache rows keyed by SHA256."""

    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "OK":
                continue
            sha256 = str(row.get("sha256", "")).casefold()
            if not sha256:
                continue
            if sha256 in result:
                raise ValueError(f"duplicate cache SHA256: {sha256}")
            result[sha256] = row
    return result


def _node_id(value) -> str:
    return str(int(value))


def validate_active_components(active_components: Iterable[str] | None) -> tuple[str, ...]:
    """Validate and normalize the reliability components used by a protocol."""

    components = tuple(A3_MVP_COMPONENTS if active_components is None else active_components)
    if not components:
        raise ValueError("at least one reliability component is required")
    if len(set(components)) != len(components):
        raise ValueError("reliability components must be unique")
    unknown = [name for name in components if name not in A3_MVP_COMPONENTS]
    if unknown:
        raise ValueError(f"unknown reliability components: {unknown}")
    return components


def _iter_node_observations(row: Mapping) -> Iterable[tuple[str, bool]]:
    payload = row.get("payload") or {}
    node_ids = payload.get("node_ids") or []
    role_ids = payload.get("node_role_ids") or []
    if len(node_ids) != len(role_ids):
        raise ValueError("cache node_ids and node_role_ids length mismatch")
    for node_id, roles in zip(node_ids, role_ids):
        yield _node_id(node_id), bool(roles)


def _binary_label(row: Mapping) -> int:
    value = str(row.get("metadata_label", row.get("label", ""))).casefold()
    if value in {"malicious", "malware", "1", "true"}:
        return 1
    if value in {"benign", "0", "false"}:
        return 0
    raise ValueError(f"unknown training label: {value!r}")


def _pilot_components(
    support_count: int,
    year_count: int,
    occurrence_count: int,
    mapped_occurrence_count: int,
    train_year_count: int,
    definition: PilotReliabilityDefinition,
    active_components: tuple[str, ...],
) -> dict[str, float]:
    alpha = definition.alpha_prior
    d_value = (support_count + alpha) / (
        support_count + alpha + definition.kappa_support
    )
    t_value = (year_count + alpha) / (
        max(1, train_year_count) + 2.0 * alpha
    )
    one_minus_f = (mapped_occurrence_count + alpha) / (
        occurrence_count + alpha + definition.kappa_context
    )
    values = {
        "D": max(0.0, min(1.0, d_value)),
        "T": max(0.0, min(1.0, t_value)),
        "one_minus_F": max(0.0, min(1.0, one_minus_f)),
    }
    return {name: values[name] for name in active_components}


def fit_training_fold_components(
    cache_rows: Mapping[str, Mapping],
    train_shas: set[str],
    definition: PilotReliabilityDefinition,
    active_components: Iterable[str] | None = None,
) -> dict:
    """Fit pilot components using only ``train_shas``.

    The returned map is keyed by hashed node ID.  Test rows are not inspected
    by this function, so unseen test nodes can only use the explicit global
    training prior supplied in the result metadata.
    """

    definition.validate()
    active_components = validate_active_components(active_components)
    rows = [cache_rows[sha] for sha in sorted(train_shas) if sha in cache_rows]
    years = {
        str(row.get("metadata_year", ""))
        for row in rows
        if str(row.get("metadata_year", ""))
    }
    support = defaultdict(int)
    years_by_node: dict[str, set[str]] = defaultdict(set)
    occurrences = defaultdict(int)
    mapped_occurrences = defaultdict(int)
    class_document_frequency = {
        0: defaultdict(int),
        1: defaultdict(int),
    }
    training_class_row_count = {0: 0, 1: 0}
    global_occurrences = 0
    global_mapped = 0
    for row in rows:
        year = str(row.get("metadata_year", ""))
        label = _binary_label(row)
        training_class_row_count[label] += 1
        for node_id, mapped in _iter_node_observations(row):
            support[node_id] += 1
            class_document_frequency[label][node_id] += 1
            occurrences[node_id] += 1
            years_by_node[node_id].add(year)
            global_occurrences += 1
            if mapped:
                mapped_occurrences[node_id] += 1
                global_mapped += 1

    components = {
        node_id: _pilot_components(
            support[node_id],
            len(years_by_node[node_id]),
            occurrences[node_id],
            mapped_occurrences[node_id],
            len(years),
            definition,
            active_components,
        )
        for node_id in sorted(support)
    }
    global_prior = _pilot_components(
        0,
        0,
        global_occurrences,
        global_mapped,
        len(years),
        definition,
        active_components,
    )
    return {
        "components_by_node_id": components,
        # Distinct-APK document frequency is retained for optional
        # train-only prevalence weighting.  The cache contains each hashed
        # node at most once per APK, so ``support`` is a document frequency,
        # not a raw call count.
        "document_frequency_by_node_id": dict(support),
        "class_document_frequency_by_node_id": {
            node_id: {
                "benign": class_document_frequency[0][node_id],
                "malicious": class_document_frequency[1][node_id],
            }
            for node_id in sorted(support, key=int)
        },
        "training_class_row_count": {
            "benign": training_class_row_count[0],
            "malicious": training_class_row_count[1],
        },
        "global_training_prior": global_prior,
        "node_count": len(components),
        "training_row_count": len(rows),
        "training_years": sorted(years),
        "training_occurrence_count": global_occurrences,
        "training_mapped_occurrence_count": global_mapped,
        "definition_status": "PILOT_PROVISIONAL_NOT_FORMAL",
        "reliability_components": list(active_components),
    }


def component_rows(evidence: Mapping) -> Iterable[dict]:
    for node_id, components in sorted(
        evidence["components_by_node_id"].items(), key=lambda item: int(item[0])
    ):
        yield {
            "node_id": node_id,
            **components,
            "component_source": "TRAINING_FOLD_ONLY",
        }
