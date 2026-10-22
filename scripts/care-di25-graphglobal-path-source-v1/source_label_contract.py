#!/usr/bin/env python3
"""Compatibility layer for the recovered source metadata labels."""


def source_label_of(row) -> int:
    value = str(row.get("metadata_label", "")).strip().casefold()
    if value == "benign":
        return 0
    if value in {"malicious", "malware"}:
        return 1
    raise RuntimeError(f"Unknown source metadata_label: {value!r}")


def install_source_label_contract(v4_module) -> None:
    v4_module.audit.future.label_of = source_label_of

