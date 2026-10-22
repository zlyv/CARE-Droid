from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml


@dataclass(frozen=True)
class RoleSpec:
    role: str
    flow_types: Set[str]
    sensitive_arguments: tuple[int, ...]
    source_value_kind: Optional[str]


class BehaviorTaxonomy:
    def __init__(self, roles: Dict[str, RoleSpec]):
        self.roles = roles

    @classmethod
    def load(cls, path: str | Path) -> "BehaviorTaxonomy":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw_roles = data.get("roles", {})
        roles: Dict[str, RoleSpec] = {}

        for role, spec in raw_roles.items():
            spec = spec or {}

            source_value = spec.get("source_value") or {}

            roles[role] = RoleSpec(
                role=role,
                flow_types=set(spec.get("flow_types", [])),
                sensitive_arguments=tuple(
                    int(x) for x in spec.get("sensitive_arguments", [])
                ),
                source_value_kind=source_value.get("kind"),
            )

        return cls(roles)

    def get(self, role: str) -> Optional[RoleSpec]:
        return self.roles.get(role)

    def is_source(self, role: str) -> bool:
        spec = self.get(role)
        return bool(spec and "source" in spec.flow_types)

    def is_transform(self, role: str) -> bool:
        spec = self.get(role)
        return bool(spec and "transform" in spec.flow_types)

    def is_sink(self, role: str) -> bool:
        spec = self.get(role)
        return bool(spec and "sink" in spec.flow_types)

    def sensitive_argument_indices(self, role: str) -> List[int]:
        spec = self.get(role)
        if not spec:
            return []
        return list(spec.sensitive_arguments)

    def source_value_kind(self, role: str) -> Optional[str]:
        spec = self.get(role)
        if not spec:
            return None
        return spec.source_value_kind
