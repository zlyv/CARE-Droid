from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


STATUS_VERIFIED_SENSITIVE = "VERIFIED_SENSITIVE_DEPENDENCY"
STATUS_RESOLVED_NONSENSITIVE = "RESOLVED_NONSENSITIVE_ORIGIN"
STATUS_UNRESOLVED = "UNRESOLVED"


@dataclass
class DependencyStep:
    method_id: str
    instruction_offset: int
    step_type: str

    api_signature: Optional[str] = None
    semantic_role: Optional[str] = None
    register: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerifiedDependencyEvidence:
    apk_id: str

    sink_role: str
    sink_api: str
    sink_method: str
    sink_argument_index: int

    source_role: Optional[str] = None
    source_api: Optional[str] = None
    source_method: Optional[str] = None

    transform_roles: List[str] = field(default_factory=list)

    dependency_types: List[str] = field(default_factory=list)

    interprocedural_depth: int = 0

    status: str = STATUS_UNRESOLVED
    unresolved_reason: Optional[str] = None

    trace: List[DependencyStep] = field(default_factory=list)

    analysis_states: int = 0
    analysis_seconds: float = 0.0

    def is_verified_sensitive(self) -> bool:
        return self.status == STATUS_VERIFIED_SENSITIVE

    def is_resolved_nonsensitive(self) -> bool:
        return self.status == STATUS_RESOLVED_NONSENSITIVE

    def is_unresolved(self) -> bool:
        return self.status == STATUS_UNRESOLVED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
