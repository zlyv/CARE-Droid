from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_PACK_STATES = {"mixed_or_packed", "packed", "unpacked", "unknown"}
ALLOWED_PURPOSES = {"feasibility_only", "formal_candidate_after_p0", "exploratory"}


@dataclass(frozen=True)
class DataSource:
    source_id: str
    enabled: bool
    purpose: str
    declared_pack_state: str
    apk_roots: tuple[str, ...]
    list_roots: tuple[str, ...]


@dataclass(frozen=True)
class ScanRules:
    extensions: tuple[str, ...]
    year_regex: str
    label_rules: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    dataset_root: Path
    sources: tuple[DataSource, ...]
    scan: ScanRules


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        value = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YAML 配置需要 PyYAML。请先执行: python -m pip install -e ."
            ) from exc
        value = yaml.safe_load(text)
    else:
        raise ValueError(f"不支持的配置格式: {path.suffix}; 请使用 .yaml/.yml/.json")
    if not isinstance(value, dict):
        raise ValueError("配置文件顶层必须是对象")
    return value


def _expand_path(value: str, base: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = base / expanded
    return expanded.resolve(strict=False)


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve(strict=True)
    raw = _load_mapping(config_path)

    dataset_root_raw = raw.get("dataset_root")
    if not isinstance(dataset_root_raw, str) or not dataset_root_raw.strip():
        raise ValueError("dataset_root 必须是非空路径")
    dataset_root = _expand_path(dataset_root_raw, config_path.parent)

    source_values = raw.get("sources")
    if not isinstance(source_values, list) or not source_values:
        raise ValueError("sources 必须是非空列表")

    sources: list[DataSource] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(source_values):
        if not isinstance(item, dict):
            raise ValueError(f"sources[{index}] 必须是对象")
        source_id = str(item.get("id", "")).strip()
        if not source_id:
            raise ValueError(f"sources[{index}].id 不能为空")
        if source_id in seen_ids:
            raise ValueError(f"重复的数据源 id: {source_id}")
        seen_ids.add(source_id)

        purpose = str(item.get("purpose", "exploratory"))
        if purpose not in ALLOWED_PURPOSES:
            raise ValueError(f"{source_id}: 不支持的 purpose={purpose}")
        pack_state = str(item.get("declared_pack_state", "unknown"))
        if pack_state not in ALLOWED_PACK_STATES:
            raise ValueError(f"{source_id}: 不支持的 declared_pack_state={pack_state}")

        apk_roots = item.get("apk_roots", [])
        list_roots = item.get("list_roots", [])
        if not isinstance(apk_roots, list) or not all(isinstance(x, str) for x in apk_roots):
            raise ValueError(f"{source_id}.apk_roots 必须是字符串列表")
        if not apk_roots and bool(item.get("enabled", True)):
            raise ValueError(f"{source_id}: 已启用的数据源必须至少包含一个 apk_root")
        if not isinstance(list_roots, list) or not all(isinstance(x, str) for x in list_roots):
            raise ValueError(f"{source_id}.list_roots 必须是字符串列表")

        sources.append(
            DataSource(
                source_id=source_id,
                enabled=bool(item.get("enabled", True)),
                purpose=purpose,
                declared_pack_state=pack_state,
                apk_roots=tuple(apk_roots),
                list_roots=tuple(list_roots),
            )
        )

    scan_raw = raw.get("scan", {})
    if not isinstance(scan_raw, dict):
        raise ValueError("scan 必须是对象")
    extensions = scan_raw.get("extensions", [".apk"])
    if not isinstance(extensions, list) or not all(isinstance(x, str) for x in extensions):
        raise ValueError("scan.extensions 必须是字符串列表")
    normalized_extensions = tuple(
        sorted({x.lower() if x.startswith(".") else f".{x.lower()}" for x in extensions})
    )
    year_regex = str(
        scan_raw.get("year_regex", r"(?<![0-9])(2017|2018|2019|2020|2021)(?![0-9])")
    )
    label_raw = scan_raw.get("label_rules", {})
    if not isinstance(label_raw, dict):
        raise ValueError("scan.label_rules 必须是对象")
    label_rules: dict[str, tuple[str, ...]] = {}
    for label, patterns in label_raw.items():
        if not isinstance(patterns, list) or not all(isinstance(x, str) for x in patterns):
            raise ValueError(f"scan.label_rules.{label} 必须是字符串列表")
        label_rules[str(label)] = tuple(patterns)

    return ProjectConfig(
        path=config_path,
        dataset_root=dataset_root,
        sources=tuple(sources),
        scan=ScanRules(
            extensions=normalized_extensions,
            year_regex=year_regex,
            label_rules=label_rules,
        ),
    )


def resolve_data_path(config: ProjectConfig, value: str) -> Path:
    return _expand_path(value, config.dataset_root)


def config_summary(config: ProjectConfig) -> dict[str, Any]:
    return {
        "config": str(config.path),
        "dataset_root": str(config.dataset_root),
        "dataset_root_exists": config.dataset_root.is_dir(),
        "sources": [
            {
                "id": source.source_id,
                "enabled": source.enabled,
                "purpose": source.purpose,
                "declared_pack_state": source.declared_pack_state,
                "apk_roots": [
                    {
                        "configured": value,
                        "resolved": str(resolve_data_path(config, value)),
                        "exists": resolve_data_path(config, value).is_dir(),
                    }
                    for value in source.apk_roots
                ],
                "list_roots": [
                    {
                        "configured": value,
                        "resolved": str(resolve_data_path(config, value)),
                        "exists": resolve_data_path(config, value).is_dir(),
                    }
                    for value in source.list_roots
                ],
            }
            for source in config.sources
        ],
    }

