#!/usr/bin/env python3
"""Fail-closed source label contract for CARE-Droid experiments."""


def source_label_of(row) -> int:
    value = str(row.get("metadata_label", "")).strip().casefold()
    if value == "benign":
        return 0
    if value in {"malicious", "malware"}:
        return 1
    raise RuntimeError(f"Unknown source metadata_label: {value!r}")


def install_source_label_contract(v4_module) -> None:
    """Install only for the imported CARE dependency graph in this process."""
    v4_module.audit.future.label_of = source_label_of
